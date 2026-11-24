"""Stage a completed, gold-blind DSH rollout for the causal SFT rewriter."""

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from .audit_transitions import event_calls
from .check_prefix_chains import Reader
from .data import load_manifest
from .gateway import parse_reply
from .history import HISTORY_POLICY
from .prepare_trajectories import read_jsonl
from .state_context import StateContext
from .state_prefix import POLICY, resolve_images
from .store import Store, canonical


def artifact_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def check_artifacts(run, source, seed):
    protocol = json.loads((run / "protocol.json").read_text())
    completed = json.loads((run / "completed.json").read_text())
    case = run / "case"
    reference = artifact_sha(case / "reference.png")
    if (source["split"] != "train" or seed["seed_id"] != source["sample_id"]
            or seed["split"] != "train" or seed["copied_files"]["reference.png"] != reference):
        raise ValueError("Rollout does not match the frozen training reference")
    if (not completed["completed"] or protocol["reference_sha256"] != reference
            or protocol["state_prefix_policy"] != POLICY or protocol["build_after_tools"]
            or any(protocol[key] is not False for key in ("gold_available", "gold_feedback", "mcp"))):
        raise ValueError("Rollout lacks an autonomous, completed gold-blind record")
    renders = [json.loads((run / f"render{stage}.json").read_text()) for stage in (1, 2)]
    for stage, render in enumerate(renders, 1):
        if (not render["renderer"].startswith("official_drawio") or render["unresolved"]
                or not (render["vertices"] and render["edges"])
                or render["source_sha256"] != artifact_sha(case / f"round{stage}.drawio")
                or render["png_sha256"] != artifact_sha(case / f"round{stage}.png")):
            raise ValueError(f"Round {stage} does not have an intact official render")
    review = json.loads((case / "review1.json").read_bytes())
    if (review["status"] not in {"FIX", "PASS"} or not completed["reviewer_session"]
            or (review["status"] == "FIX" and renders[0]["source_sha256"] == renders[1]["source_sha256"])):
        raise ValueError("The frozen review or its corresponding revision is missing")
    return reference


def native_calls(run):
    tools = {}
    for path in (run / "generator").glob("round*/events.jsonl"):
        for event in event_calls(path):
            ident = event["callId"]
            if ident in tools or event["result"] is None:
                raise ValueError("DSH tool event has no unique recorded result: " + ident)
            tools[ident] = event
    return tools


@lru_cache(maxsize=1)
def journal_arguments(run):
    import zstandard

    arguments = {}
    for path in (run / "generator/home/.dsh/sessions").rglob("session.v4.jsonl.zstd"):
        with zstandard.open(path, "rt") as stream:
            for event in map(json.loads, stream):
                if event["type"] == "tool/call":
                    data = event["data"]
                    arguments[data["callId"]] = json.loads(data["arguments"])
    return arguments


def check_call(run, event, call):
    function = call["function"]
    arguments = json.loads(function["arguments"])
    if event["tool"] != function["name"]:
        raise ValueError("Tool name differs from DSH execution: " + call["id"])
    if event["input"] != arguments and journal_arguments(run).get(call["id"]) != arguments:
        raise ValueError("Tool arguments differ from DSH execution: " + call["id"])


def capture(run, state):
    records = read_jsonl(run / "generator-context-requests.jsonl")
    replies = {row["response_to"]: row for row in records if "response_to" in row}
    reader = Reader(run / "context")
    tools = native_calls(run)
    for row in (row for row in records if "raw_request" in row):
        path = run / "generator-context-requests-raw" / Path(row["raw_request"]).name
        raw = path.with_suffix("").with_suffix(".response.bin").read_bytes()
        reply, usage = parse_reply(raw)
        response = replies[row["timestamp"]]
        choices = reply["choices"]
        if (response["status"] != 200 or response["stream_error"] or b"data: [DONE]" not in raw
                or len(choices) != 1 or choices[0]["finish_reason"] not in ("tool_calls", "stop")):
            raise ValueError("Incomplete model reply: " + path.name)
        for call in choices[0]["message"].get("tool_calls", []):
            check_call(run, tools.pop(call["id"]), call)
        proof = row["state_prefix"]
        files = reader.store.get(proof["snapshot_ref"])
        stage = 2 if "review1.json" in files else 1
        source = reader.value(proof["source_request_ref"])
        rebuilt, checked = state.build(source, files, lambda name: reader.store.raw(files[name]),
                                       stage, proof["source_request_ref"], reader.store.raw)
        request = json.loads(path.read_bytes())
        if (proof["policy"] != POLICY or proof["history_thinking_messages"] != 0
                or request["messages"] != rebuilt or request["tools"] != source["tools"]
                or checked["input_tokens"] != proof["input_tokens"]):
            raise ValueError("Stored prefix does not reproduce the live request: " + path.name)
        yield (row["timestamp"], stage, source, files, reply, raw, usage, reader.store)
    if tools:
        raise ValueError("DSH executed tools outside the captured provider requests")


def stage_rollout(run, source, seed, state, output):
    reference = check_artifacts(run, source, seed)
    captured = list(capture(run, state))
    if {item[1] for item in captured} != {1, 2}:
        raise ValueError("Both generation stages must be present")
    output.mkdir(parents=True)
    archive = Store(output / "archive")
    rows = []
    excluded = {"reasoning_rewrite", "state_prefix_input_tokens", "state_prefix_policy",
                "prompt_token_sha256", "token_count", "prompt_tokens", "completion_tokens",
                "image_count", "has_reasoning", "tool_calls", "has_content", "tokenization"}
    template = {key: value for key, value in source.items() if key not in excluded}
    for timestamp, stage, request, files, reply, raw, usage, original in captured:
        request["messages"] = resolve_images(request["messages"], original.raw)
        request_ref = archive.put(request)
        artifacts = archive.put({name: archive.blob(original.raw(ref)) for name, ref in files.items()})
        number = archive.reserve(source["episode"], "generator", stage, request_ref, artifacts, 0, 0, 0)
        response_ref = archive.put(reply)
        archive.finish(number, response_ref, archive.blob(raw), 200, usage, 0)
        archive.db.execute("update calls set started=? where id=?", (timestamp, number))
        rows.append({**template, "sample_id": f"{source['episode']}:local-{number}",
                     "call_id": number, "stage": stage, "decision_order": number - 1,
                     "request_ref": request_ref, "response_ref": response_ref,
                     "snapshot_root": str(output.resolve()), "history_policy": HISTORY_POLICY,
                     "archived_final_render_verified": True, "fidelity_verified": False})
    archive.db.commit()
    (output / "source.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    return {"decisions": len(rows), "reference_sha256": reference,
            "replay_identical": True, "training_manifest_modified": False}


def main():
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rollout", "baseline", "seed-receipt", "state-prefix-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    seed = json.loads(args.seed_receipt.read_text())
    source = next(row for row in load_manifest(args.baseline, "train") if row["sample_id"] == seed["seed_id"])
    state = StateContext(AutoProcessor.from_pretrained(args.state_prefix_model, local_files_only=True))
    print(json.dumps(stage_rollout(args.rollout, source, seed, state, args.output)))


if __name__ == "__main__":
    main()
