"""Audit repeated reference crops and severe exact repetitions in reasoning."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def reasoning_loop(text, threshold=16):
    """Detect long literal repetition; code, arguments and answers are not inputs."""
    width = 200
    if len(text) < threshold * width:
        return None
    starts = list(range(0, len(text) - width + 1, width)) + [len(text) - width]
    for start in starts:
        block = text[start:start + width]
        occurrences = text.count(block)
        if occurrences >= threshold:
            return {"threshold": threshold, "occurrences": occurrences,
                    "block_characters": width, "reasoning_characters": len(text),
                    "block_sha256": sha256(block.encode()).hexdigest()}
    return None


def crop_rectangles(command):
    try:
        argv = shlex.split(command)
        if len(argv) != 3 or Path(argv[0]).name not in ("python", "python3") or argv[1] != "-c":
            return None
        tree = ast.parse(argv[2])
        images, crops, rectangles, saved = set(), set(), [], set()
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom) and statement.module == "PIL":
                if [(name.name, name.asname) for name in statement.names] != [("Image", None)]:
                    return None
                continue
            target = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                if not isinstance(statement.targets[0], ast.Name):
                    return None
                target, call = statement.targets[0].id, statement.value
            elif isinstance(statement, ast.Expr):
                call = statement.value
            else:
                return None
            if not isinstance(call, ast.Call) or call.keywords:
                return None
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                if target or any(not isinstance(value, ast.Constant) for value in call.args):
                    return None
                continue
            if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
                return None
            owner, method = call.func.value.id, call.func.attr
            if owner == "Image" and method == "open" and target:
                if len(call.args) != 1 or ast.literal_eval(call.args[0]) != "/case/reference.png":
                    return None
                images.add(target)
            elif owner in images and method == "crop" and target:
                box = ast.literal_eval(call.args[0]) if len(call.args) == 1 else None
                if not isinstance(box, tuple) or len(box) != 4 or any(type(x) not in (int, float) for x in box):
                    return None
                crops.add(target)
                rectangles.append(box)
            elif owner in crops and method == "save" and target is None:
                path = Path(ast.literal_eval(call.args[0])) if len(call.args) == 1 else Path()
                if path.parent != Path("/case") or path.suffix != ".png" or path.name == "reference.png":
                    return None
                saved.add(owner)
            else:
                return None
        return rectangles if rectangles and crops == saved else None
    except (SyntaxError, ValueError, TypeError):
        return None


def audit(path, threshold):
    calls, seen, streak, trigger, total = {}, set(), 0, None, 0
    raw = path.read_bytes()
    for line in raw.decode().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "tool_call":
            total += 1
            calls[event["callId"]] = (total, event)
        if event.get("type") != "tool_result" or event["callId"] not in calls:
            continue
        index, call = calls[event["callId"]]
        boxes = crop_rectangles(call.get("input", {}).get("command", "")) if call.get("tool") == "bash" else None
        duplicate = event.get("status") == "completed" and boxes and set(boxes) <= seen
        streak = streak + 1 if duplicate else 0
        if boxes:
            seen.update(boxes)
        if streak >= threshold and trigger is None:
            trigger = {"tool_call_index": index, "call_id": event["callId"],
                       "consecutive_duplicate_crops": streak, "distinct_rectangles": len(seen)}
    return {"threshold": threshold, "trigger": trigger, "observed_tool_calls": total,
            "final_duplicate_streak": streak, "distinct_rectangles": len(seen),
            "events_sha256": sha256(raw).hexdigest()}


def stop_actor(run, report):
    launch = json.loads((run / "launch.json").read_text())
    driver = launch["pid"]
    rows = subprocess.run(["ps", "--ppid", str(driver), "-o", "pid="], capture_output=True, text=True)
    for value in rows.stdout.split():
        pid = int(value)
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
        except FileNotFoundError:
            continue
        if Path(argv[0]).name != "bwrap" or "--unshare-all" not in argv or str((run / "case").resolve()) not in argv:
            continue
        receipt = {"outcome": "HOST_DETECTED_SELF_LOOP", "natural_model_end": False,
                   "completed_reconstruction": False, "quality_score": None,
                   "exploratory_posthoc_rule": True, "raw_trace_retained": True,
                   "training_modified": False, "at": time.time(), "actor_pid": pid,
                   "driver_pid": driver, "censored_at": report["trigger"], "audit": report}
        (run / "host-loop-outcome.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.kill(pid, signal.SIGTERM)
        print(json.dumps({"run": str(run), **receipt}), flush=True)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        for run in args.run:
            if (run / "host-loop-outcome.json").exists():
                continue
            for path in sorted((run / "generator").glob("round*/events.jsonl")):
                report = audit(path, args.threshold)
                if not args.enforce:
                    print(json.dumps({"run": str(run), "stage": path.parent.name, **report}), flush=True)
                elif report["trigger"] and not list((run / "case").glob("*.drawio")):
                    stop_actor(run, report)
                    break
        if not args.watch:
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
