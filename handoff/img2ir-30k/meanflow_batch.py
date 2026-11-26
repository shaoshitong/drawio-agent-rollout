"""Generate canonical IR or editable Draw.io over an image manifest."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import sleep

from meanflow_two_stage import CODEX_CHATGPT, PROVIDERS


LOG_LOCK = Lock()


def load_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def status(folder, ir_only=False):
    receipt = folder / "report.json"
    if not receipt.is_file():
        return "interrupted" if folder.exists() and not ir_only else "pending"
    try:
        report = json.loads(receipt.read_text())
    except json.JSONDecodeError:
        return "pending" if ir_only else "interrupted"
    checks_ok = report.get("compiler_checks", {}).get("recompilation_byte_stable", False)
    valid_ir = checks_ok and report.get("ir_complete") and (folder / "ir.json").is_file()
    if ir_only:
        return "complete_canonical_ir" if valid_ir and report.get("source") == "canonical_ir" else "failed"
    xml = (folder / "reconstruction.drawio").is_file()
    if xml and valid_ir:
        return "complete_" + report["source"]
    return "xml_only" if xml and checks_ok else "failed"


def run_case(row, args, route):
    name = row["case_id"]
    case = args.out / name
    prior = status(case, args.ir_only)
    if prior != "pending":
        return prior, None
    runner = Path(__file__).with_name("meanflow_two_stage.py")
    stage1, stage2 = route
    command = [sys.executable, str(runner), row["image"], "--compiler", str(args.compiler),
               "--out", str(case), "--stage1", stage1, "--stage2", stage2,
               "--ir-max", str(args.ir_max), "--xml-max", str(args.xml_max),
               "--ir-format", args.ir_format,
               "--deepseek-auth", args.deepseek_auth, "--hy-thinking", args.hy_thinking,
               "--deepseek-thinking", args.deepseek_thinking]
    if args.ir_only:
        command.append("--ir-only")
    env = os.environ.copy()
    if "openai" not in route:
        env.pop("OPENAI_API_KEY", None)
    process = subprocess.run(command, capture_output=True, text=True, env=env)
    result = status(case, args.ir_only)
    record = {"case_id": name, "status": result, "stage1": stage1, "stage2": None if args.ir_only else stage2}
    unavailable = None
    if (case / "report.json").is_file():
        report = json.loads((case / "report.json").read_text())
        for provider in route:
            field = f"{provider}_unavailable"
            if reason := report.get(field):
                record[field] = reason
                unavailable = (provider, "turn_failed" if provider == "tcodex" and reason is True else reason)
                break
    if result in {"interrupted", "pending"}:
        record["exit_code"] = process.returncode
    with LOG_LOCK:
        print(json.dumps(record), flush=True)
    if unavailable and (
        unavailable[1] == "rate_limit"
        or args.ir_only and (unavailable[1] == "transport" or unavailable == ("tcodex", "turn_failed"))
    ):
        sleep(45)
    return result, unavailable


def replay_cached_endings(rows, args, remaining):
    if not rows or remaining == 0:
        return 0

    sys.path.insert(0, str(args.compiler.resolve()))
    from drawio.compiler import ir
    from drawio.compiler.compiler import compile_bytes
    from drawio.compiler.emit import emit
    from drawio.compiler.source import CompileError
    from drawio.compiler.syntax import parse_style
    from meanflow_two_stage import (check_canonical_canvas, compile_visual_ir, normalize_final_ir_record,
                                    read_ir_lines, require_editable, write_json)

    recovered = 0
    ledger = args.out / ".ir-requests"
    for row in rows:
        case = args.out / row["case_id"]
        path = case / "report.json"
        prior = json.loads(path.read_text())
        if prior.get("ir_error") != "Incomplete stage 1 IR" or prior.get("ir_format") != "jsonl":
            continue
        request = prior["ir_request"]
        digest = row["sha256"]
        cached = ledger / f"{digest}.json"
        if not cached.is_file():
            continue
        reply = json.loads(cached.read_text())
        if reply["request"] != request or (case / "stage1.txt").read_bytes() != reply["text"].encode("utf-8"):
            raise ValueError(f"Inconsistent cached response for {row['case_id']}")
        payload, normalized = normalize_final_ir_record(reply["text"], request["finish"])
        if not normalized:
            continue
        try:
            draft, complete = read_ir_lines(payload, False, ir, parse_style, strict=True)
            if not complete:
                continue
            result = require_editable(compile_visual_ir(draft, ir, emit, compile_bytes))
            with Path(row["image"]).open("rb") as image:
                header = image.read(24)
                image.seek(0)
                if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR" or (
                    file_digest(image, "sha256").hexdigest() != digest
                ):
                    raise OSError(f"Source PNG differs from its manifest for {row['case_id']}")
            canvas = tuple(int.from_bytes(header[i:i+4], "big") for i in (16, 20))
            check_canonical_canvas(result.ir_json, canvas)
        except (ValueError, TypeError, KeyError, ArithmeticError, CompileError, ir.IRError):
            continue

        prior["prior_ir_error"] = prior.pop("ir_error")
        prior.update(source="canonical_ir", ir_complete=True, normalized_end_record=True, ir_response_reused=True,
                     compiler_checks=result.report["checks"], image_sha256=digest,
                     canonical_ir_sha256=result.report["ir_sha256"])
        write_json(case / "ir.json", result.ir_json)
        write_json(path, prior)
        recovered += 1
        if recovered == remaining:
            break
    return recovered


def run_pending(pending, args, routes, remaining=None, tcodex_fallback=False):
    queue = iter(pending)
    complete = 0
    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        idle = list(reversed(routes))
        active = {}
        exhausted = False
        disabled_routes = set()
        fallback_slots = 0
        hy_access_errors = 0
        while True:
            if fallback_slots and not any("openai" in route for route in active.values()):
                idle.extend([("tcodex", "tcodex")] * fallback_slots)
                print(json.dumps({"tcodex_started": fallback_slots, "reason": "openai_quota"}), flush=True)
                fallback_slots = 0
            while idle and not exhausted and (remaining is None or complete + len(active) < remaining):
                case = next(queue, None)
                if case is None:
                    exhausted = True
                else:
                    route = idle.pop()
                    active[pool.submit(run_case, case, args, route)] = route
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                route = active.pop(future)
                result, unavailable = future.result()
                if unavailable == ("hy", "access"):
                    hy_access_errors += 1
                elif "hy" in route and result.startswith("complete_"):
                    hy_access_errors = 0
                if unavailable:
                    provider, reason = unavailable
                    stop = reason in {"access", "quota", "turn_failed"}
                    if args.ir_only and unavailable == ("tcodex", "turn_failed"):
                        stop = False
                    if unavailable == ("hy", "access"):
                        stop = hy_access_errors >= 3
                    if stop and provider not in disabled_routes:
                        disabled_routes.add(provider)
                        print(json.dumps({f"{provider}_disabled": reason}), flush=True)
                        idle = [slot for slot in idle if not disabled_routes.intersection(slot)]
                        if unavailable == ("openai", "quota") and tcodex_fallback:
                            fallback_slots = routes.count(("openai", "openai"))
                if not disabled_routes.intersection(route):
                    idle.append(route)
                if result.startswith("complete_"):
                    complete += 1


def summarize(rows, root, ir_only=False):
    counts = Counter()
    accepted = []
    for row in rows:
        folder = root / row["case_id"]
        result = "held_out" if row.get("held_out") else status(folder, ir_only)
        counts[result] += 1
        if result.startswith("complete_"):
            paths = {"ir": str(folder / "ir.json"), "report": str(folder / "report.json")}
            if ir_only:
                paths["ir_source"] = "canonical_ir"
            else:
                paths.update(xml_source=result.removeprefix("complete_"), xml=str(folder / "reconstruction.drawio"))
            accepted.append({**row, **paths})
    path = root / "accepted.jsonl"
    tmp = root / "accepted.jsonl.tmp"
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted))
    tmp.replace(path)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--hy-jobs", type=int, default=0, help="Slots using HY for both stages")
    parser.add_argument("--deepseek-jobs", type=int, default=0, help="Slots using DeepSeek for both stages")
    parser.add_argument("--openai-jobs", type=int, default=0, help="Slots using OpenAI Sol medium for both stages")
    parser.add_argument("--tcodex-jobs", type=int, default=0, help="Slots using tcodex DeepSeek v4.1 Flash for both stages")
    parser.add_argument("--codex-chatgpt-jobs", type=int, default=0, help="IR-only slots using Codex ChatGPT Sol medium")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target-complete", type=int, help="Stop at this total, including completed cases")
    parser.add_argument("--ir-max", type=int, default=12000)
    parser.add_argument("--xml-max", type=int, default=16000)
    parser.add_argument("--ir-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--ir-only", action="store_true", help="Collect canonical IR with one request per image")
    parser.add_argument("--stage1", choices=PROVIDERS, default="hy")
    parser.add_argument("--stage2", choices=PROVIDERS, default="deepseek")
    parser.add_argument("--deepseek-auth", choices=("account", "api-key"), default="account")
    parser.add_argument("--hy-thinking", choices=("none", "low"), default="none")
    parser.add_argument("--deepseek-thinking", choices=("none", "low"), default="none")
    args = parser.parse_args()
    if args.jobs < 1 or (args.limit is not None and args.limit < 0):
        parser.error("jobs must be positive and limit must be nonnegative")
    if min(args.hy_jobs, args.deepseek_jobs, args.openai_jobs, args.tcodex_jobs, args.codex_chatgpt_jobs) < 0:
        parser.error("provider job counts must be nonnegative")
    if not args.ir_only and (args.codex_chatgpt_jobs or CODEX_CHATGPT in (args.stage1, args.stage2)):
        parser.error("The Codex ChatGPT route is available for IR-only requests")
    if args.target_complete is not None and args.target_complete < 0:
        parser.error("target-complete must be nonnegative")
    args.out = args.out.resolve()
    rows = load_rows(args.manifest)
    keys = [row["case_id"] for row in rows]
    if len(keys) != len(set(keys)) or any(not key or not key.replace("-", "").isalnum() for key in keys):
        parser.error("case_id values must be unique names")
    if any(not Path(row["image"]).is_file() for row in rows):
        parser.error("The manifest contains unavailable input images")
    args.out.mkdir(parents=True, exist_ok=True)
    routes = ([("hy", "hy")] * args.hy_jobs + [("deepseek", "deepseek")] * args.deepseek_jobs +
              [("openai", "openai")] * args.openai_jobs + [("tcodex", "tcodex")] * args.tcodex_jobs +
              [(CODEX_CHATGPT, CODEX_CHATGPT)] * args.codex_chatgpt_jobs)
    with (args.out / "batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A scheduler is already running for this output directory")
        pending, failed, complete = [], [], 0
        for row in rows:
            if row.get("held_out"):
                continue
            result = status(args.out / row["case_id"], args.ir_only)
            if result == "pending":
                pending.append(row)
            elif result.startswith("complete_"):
                complete += 1
            elif args.ir_only and result == "failed":
                failed.append(row)
        remaining = None if args.target_complete is None else max(0, args.target_complete - complete)
        if recovered := replay_cached_endings(failed, args, remaining):
            print(json.dumps({"cached_ir_replayed": recovered}), flush=True)
            if remaining is not None:
                remaining -= recovered
        default_route = (args.stage1, args.stage1 if args.ir_only else args.stage2)
        run_pending(pending[:args.limit], args, routes or [default_route] * args.jobs,
                    remaining, tcodex_fallback=bool(args.openai_jobs and not args.tcodex_jobs))
        counts = summarize(rows, args.out, args.ir_only)
        complete = sum(n for k, n in counts.items() if k.startswith("complete_"))
        summary = {"status": dict(counts), "complete": complete}
        if args.target_complete is not None:
            summary["target_complete"] = args.target_complete
            summary["shortfall"] = max(0, args.target_complete - complete)
        print(json.dumps(summary), flush=True)
        return 3 if summary.get("shortfall", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
