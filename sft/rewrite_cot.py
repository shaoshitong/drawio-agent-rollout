"""Rewrite teacher reasoning chronologically; never expose the current answer."""

import argparse
import fcntl
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .causal_cot import (POLICY, anchors, answer_fields, digest, fragments,
                         mechanical_feedback, message_key, payload, rewrite_history, restore_source_commitments)
from .check_prefix_chains import Reader
from .data import load_manifest
from .history import HISTORY_POLICY
from .observation_context import ObservationContext, POLICY as CONTEXT_POLICY, apply_observations
from .store import Store


class IncompleteRewrite(ValueError):
    pass


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def select_rows(data):
    accepted = load_manifest(data / "curriculum.jsonl")
    metadata = {row["episode"]: {key: row[key] for key in (
        "snapshot_root", "paper_group", "split", "final_components", "final_vertices",
        "final_edges", "final_image_cells", "episode_order", "chat_template_kwargs",
        "archived_final_render_verified", "fidelity_verified")} for row in accepted}
    rows = []
    for row in load_manifest(data / "sft-all-decisions.jsonl"):
        if not row.get("supervision_eligible", True):
            continue
        rows.append({**row, **metadata[row["episode"]],
                     "history_policy": HISTORY_POLICY,
                     "sample_id": f"{row['episode']}:{row['call_id']}"})
    return sorted(rows, key=lambda row: (row["episode_order"], row["call_id"]))


def parse_judge(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    result = json.loads(text)
    return result["faithful"] is True and not result["missing"] and not result["unsupported"], result


class Rewriter:
    def __init__(self, args):
        from transformers import AutoTokenizer

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.store = Store(args.output / "archive")
        self.lock = threading.Lock()
        self.started = time.time()
        self.counts = defaultdict(int)
        self.selection = getattr(args, "selection", {})
        self.context = ObservationContext(self) if args.rewrite_observations else None
        self.state = None
        if getattr(args, "state_prefix_model", None):
            from transformers import AutoProcessor
            from .state_context import StateContext

            processor = AutoProcessor.from_pretrained(args.state_prefix_model, local_files_only=True)
            self.state = StateContext(processor, ObservationContext(self))

    @lru_cache(maxsize=8192)
    def count(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def request(self, body, folder, scope):
        reference = self.store.put(body)
        started = time.time()
        record = {**scope, "started": started, "request_ref": reference,
                  "request_sha256": digest(body), "current_answer_available": False,
                  "future_fragments_available": False}
        try:
            request = Request(self.args.endpoint, data=json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=600) as response:
                result = json.load(response)
            record.update(response_ref=self.store.put(result), usage=result.get("usage"))
            choice = result["choices"][0]
            content = choice["message"].get("content")
            if choice["finish_reason"] != "stop" or not content or content.lstrip().startswith(("<think>", "</think>")):
                raise IncompleteRewrite("Incomplete local rewrite/judge response")
            return choice["message"]["content"]
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            record["error"] = str(error)
            if isinstance(error, HTTPError):
                record["http_error"] = error.read().decode(errors="replace")[:3000]
            raise
        finally:
            record["ended"] = time.time()
            with (folder / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            self.store.metadata("last_request", record["ended"])

    def fragment(self, part, previous, folder, sample_id, request_ref):
        source = part["text"]
        scope = {"sample_id": sample_id, "source_start": part["start"],
                 "source_end": part["end"], "source_sha256": digest(source),
                 "archived_request_ref": request_ref, "previous_sha256": digest(previous)}
        if self.count(source) <= 96:
            return source, {**scope, "outcome": "already_short", "attempts": []}
        if not part["boundary_safe"]:
            return source, {**scope, "outcome": "source_retained_at_identifier_boundary", "attempts": []}
        feedback, proposal, attempts = None, None, []
        known = previous
        for attempt in range(2):
            body = payload(previous, source, feedback=feedback, candidate=proposal,
                           target_tokens=max(48, round(self.count(source) * 0.35)), seed=42 + attempt)
            try:
                generated = self.request(body, folder, {**scope, "operation": "compress", "attempt": attempt})
                candidate = json.loads(generated)["compressed_reasoning"]
                candidate, restored = restore_source_commitments(source, candidate)
                proposal = candidate
                checks = mechanical_feedback(source, candidate, known)
                shorter = self.count(candidate) < self.count(source)
                judge_text = None
                if shorter and not any(checks.values()):
                    judge_body = payload(previous, source, operation="judge", candidate=candidate)
                    judge_text = self.request(judge_body, folder, {**scope, "operation": "judge", "attempt": attempt})
            except (IncompleteRewrite, json.JSONDecodeError, KeyError) as error:
                feedback = str(error)
                attempts.append({"passed": False, "error": feedback})
                continue
            try:
                faithful, judge = parse_judge(judge_text) if judge_text else (False, {"not_run": "mechanical_check_failed"})
            except (KeyError, ValueError, TypeError):
                faithful, judge = False, {"invalid_judge_json": judge_text}
            passed = faithful and not any(checks.values()) and shorter
            attempts.append({"candidate": candidate, "checks": checks, "judge": judge,
                             "restored_source_clauses": restored,
                             "omitted_source_anchors": sorted(anchors(source) - anchors(candidate)),
                             "shorter": shorter, "passed": passed})
            if passed:
                return candidate, {**scope, "outcome": "compressed", "attempts": attempts}
            feedback = json.dumps({"checks": checks, "judge": judge, "must_be_shorter": not shorter})
        return source, {**scope, "outcome": "source_retained_after_failed_verification", "attempts": attempts}

    def decision(self, row, reader, prior, folder):
        selection = self.selection.get(row["sample_id"])
        if selection and not selection["keep"]:
            proof = {"rejection": selection["reason"], "offline_selection_only": True,
                     "policy": self.args.selection_policy}
            return {"row": {**row, "state_rejection": proof},
                    "proof": {"sample_id": row["sample_id"], **proof, "at": time.time()}, "key": None}
        source_request = reader.value(row["request_ref"])
        state = None
        if self.state:
            from .state_prefix import snapshot_before_request

            files, snapshot = snapshot_before_request(row, reader.store)
            messages, state_proof = self.state.build(source_request, files,
                lambda name: reader.store.raw(files[name]), row["stage"], row["request_ref"], reader.store.raw)
            if messages is None:
                return {"row": {**row, "state_rejection": state_proof},
                        "proof": {"sample_id": row["sample_id"], **state_proof, "at": time.time()}, "key": None}
            state = {**state_proof, "snapshot": snapshot, "source_request_ref": row["request_ref"], "messages": messages}
        source_response = reader.value(row["response_ref"])
        original = source_response["choices"][0]["message"]
        if state:
            from .action_visibility import qualify_target

            formatted = {**source_request, "messages": state["messages"]}
            target_check = qualify_target(self.state.processor, formatted, source_response, state["files_sha256"],
                lambda name: reader.store.raw(files[name]), state["input_tokens"],
                mutation=self.selection.get(row["sample_id"], {}).get("mutation", True))
            if "rejection" in target_check:
                return {"row": {**row, "state_rejection": target_check},
                        "proof": {"sample_id": row["sample_id"], **target_check, "at": time.time()}, "key": None}
        history, matches = rewrite_history(source_request["messages"], prior)
        observations = []
        if self.context:
            history, observations = self.context.rewrite(history, row["request_ref"])
        source = original.get("reasoning_content") or ""
        parts = list(fragments(source, self.tokenizer, self.args.block_tokens))
        if "".join(part["text"] for part in parts) != source:
            raise ValueError("Reasoning segmentation is not lossless")
        compact, audits = [], []
        reused = message_key(original) in prior
        for part in [] if reused else parts:
            text, audit = self.fragment(part, "".join(compact), folder, row["sample_id"], row["request_ref"])
            compact.append(text + "\n" if audit["outcome"] == "compressed" else text)
            audits.append(audit)
        reasoning = prior[message_key(original)] if reused else "".join(compact)
        if state:
            compact_response = deepcopy(source_response)
            compact_response["choices"][0]["message"]["reasoning_content"] = reasoning
            target_check = qualify_target(self.state.processor, formatted, compact_response, {}, None,
                state["input_tokens"], mutation=False, include_reasoning=True)
            if "rejection" in target_check:
                return {"row": {**row, "state_rejection": target_check},
                        "proof": {"sample_id": row["sample_id"], **target_check, "at": time.time()}, "key": None}
        changes = [{"index": index, "source_sha256": digest(before.get("reasoning_content") or ""),
                    "reasoning_content": after["reasoning_content"]}
                   for index, (before, after) in enumerate(zip(source_request["messages"], history))
                   if before.get("reasoning_content") != after.get("reasoning_content")]
        overlay = {"policy": POLICY, "request_ref": row["request_ref"], "response_ref": row["response_ref"],
                   "history": changes, "current_source_sha256": digest(source),
                   "current_reasoning": reasoning, "observations": observations,
                   "context_policy": CONTEXT_POLICY if self.context else None, "state_prefix": state}
        proof = {"sample_id": row["sample_id"], "original_reasoning_tokens": self.count(source),
                 "rewritten_reasoning_tokens": self.count(reasoning), "fragments": len(parts),
                 "reused_prior_reasoning": reused,
                 "compressed_fragments": sum(a["outcome"] == "compressed" for a in audits),
                 "fallback_fragments": sum(a["outcome"].startswith("source_retained") for a in audits),
                 "answer_fields_sha256": digest(answer_fields(original)), "answer_fields_unchanged": True,
                 "observation_summaries": len(observations),
                 "history_changes_match_observations": all(answer_fields(a) == answer_fields(b)
                     for a, b in zip(apply_observations(source_request["messages"], observations), history)), **matches,
                 "audit_ref": self.store.put(audits), "at": time.time()}
        if state:
            proof["state_prefix"] = {key: value for key, value in state.items() if key != "messages"}
        self.store.metadata(row["sample_id"], proof)
        if state:
            row = {**row, "state_prefix_input_tokens": state["input_tokens"], "state_prefix_policy": state["policy"]}
        return {"row": {**row, "reasoning_rewrite": {"root": str(self.store.root.resolve()),
                         "ref": self.store.put(overlay)}}, "proof": proof, "key": message_key(original)}

    def episode(self, episode, rows):
        folder = self.args.output / "episodes" / episode
        folder.mkdir(parents=True, exist_ok=True)
        reader, prior = Reader(rows[0]["snapshot_root"]), {}
        for row in rows:
            path = folder / (str(row["call_id"]) + ".json")
            if path.exists():
                result = json.loads(path.read_text())
                if any(result["row"][key] != row[key] for key in ("request_ref", "response_ref")):
                    raise ValueError("Resume source references changed")
            else:
                result = self.decision(row, reader, prior, folder)
                atomic(path, result)
            key = result["key"]
            if key is not None:
                overlay = self.store.get(result["row"]["reasoning_rewrite"]["ref"])
                if key in prior and prior[key] != overlay["current_reasoning"]:
                    raise ValueError("Ambiguous rewrite for an identical archived message")
                prior[key] = overlay["current_reasoning"]
            with self.lock:
                proof = result["proof"]
                self.counts["decisions"] += 1
                outcome = "selection_excluded" if proof.get("offline_selection_only") else "state_rejected"
                self.counts[outcome if key is None else "accepted"] += 1
                for name in ("original_reasoning_tokens", "rewritten_reasoning_tokens", "compressed_fragments",
                             "fallback_fragments", "unmatched_history_messages"):
                    self.counts[name] += proof.get(name, 0)
                atomic(self.args.output / "progress.json", {"at": time.time(), "started": self.started,
                    "phase": "rewriting", "source_commit": self.args.source_commit, **self.counts})
                print(json.dumps({"episode": episode, **proof}), flush=True)
        atomic(folder / "completed.json", {"episode": episode, "decisions": len(rows), "at": time.time()})


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "model", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:32527/v1/chat/completions")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block-tokens", type=int, default=768)
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument("--rewrite-observations", action="store_true")
    context_group.add_argument("--state-prefix-model", type=Path)
    parser.add_argument("--selection-plan", type=Path,
                        help="Full-corpus offline qualification and sampling receipts")
    parser.add_argument("--sample-decisions", type=int, default=0,
                        help="Frozen random train-only calibration; zero processes the full corpus")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lease = (args.output / "worker.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease.seek(0)
    lease.truncate()
    lease.write(str(os.getpid()) + "\n")
    lease.flush()
    rows = select_rows(args.data)
    if args.selection_plan:
        from .prepare_selection import POLICY as selection_policy, load_selection

        if not args.state_prefix_model:
            parser.error("--selection-plan requires --state-prefix-model")
        args.selection = load_selection(args.selection_plan, rows)
        args.selection_policy = selection_policy
    if args.sample_decisions:
        rows = random.Random(20261122).sample([row for row in rows if row["split"] == "train"], args.sample_decisions)
        rows.sort(key=lambda row: (row["episode_order"], row["call_id"]))
    plan = {"policy": POLICY, "source_commit": args.source_commit, "rows": len(rows),
            "source_index_sha256": digest(rows), "workers": args.workers, "block_tokens": args.block_tokens,
            "model": str(args.model), "endpoint": args.endpoint, "current_answer_available": False,
            "future_fragments_available": False, "training_started": False}
    plan.update(calibration_only=bool(args.sample_decisions), selection_seed=20261122)
    plan["context_policy"] = CONTEXT_POLICY if args.rewrite_observations else None
    plan["state_prefix_model"] = str(args.state_prefix_model) if args.state_prefix_model else None
    if args.selection_plan:
        plan["selection_plan_sha256"] = digest(json.loads(args.selection_plan.read_text()))
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing output belongs to another source or rewrite policy")
    atomic(plan_path, plan)
    with urlopen(args.endpoint.split("/v1/")[0] + "/get_server_info", timeout=30) as response:
        server = json.load(response)
    if server["allow_auto_truncate"] is not False or server["tp_size"] != 4:
        raise ValueError("Rewriter requires TP4 and explicit rejection of oversized prompts")
    atomic(args.output / "server-contract.json", {key: server[key] for key in (
        "model_path", "context_length", "tp_size", "allow_auto_truncate", "version")})
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode"]].append(row)
    rewriter, failures = Rewriter(args), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(rewriter.episode, episode, group): episode for episode, group in groups.items()}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as error:
                failure = {"episode": pending[future], "type": type(error).__name__, "error": str(error)}
                failures.append(failure)
                atomic(args.output / "failures.json", failures)
                print(json.dumps(failure), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} episodes incomplete; completed decisions are resumable")
    results = [json.loads((args.output / "episodes" / row["episode"] / (str(row["call_id"]) + ".json")).read_text())["row"]
               for row in rows]
    if args.selection_plan:
        from .prepare_selection import finalize_rows

        results = finalize_rows(results)
    with (args.output / "rewritten.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in results)
    atomic(args.output / "completed.json", {**plan, "at": time.time(), "counts": dict(rewriter.counts),
           "retained_after_phase_check": sum(not row.get("state_rejection") for row in results)})


if __name__ == "__main__":
    main()
