"""Observe an existing SFT job without changing or restarting it."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def children(pid):
    path = Path(f"/proc/{pid}/task/{pid}/children")
    return path.read_text().split() if path.exists() else []


def worker_status(pid, py_spy):
    result = subprocess.run(
        [str(py_spy), "dump", "--pid", str(pid), "--nonblocking", "--locals"],
        capture_output=True, text=True, timeout=15,
    )
    status = {"pid": int(pid), "reader_exit": result.returncode}
    text = result.stdout
    frame = re.search(r"Trainer\._run_epoch[^\n]*\n(.*?)(?=\n    \S|\Z)", text, re.S)
    if frame:
        for field in ("step", "update_step", "i", "num_batches"):
            value = re.search(rf"\n\s+{field}: (\d+)", frame[1])
            if value:
                status[field] = int(value[1])
        status["phase"] = "backward" if "_engine_run_backward" in text else "training"
    frame = re.search(r"FixedLoss\.evaluate[^\n]*\n(.*?)(?=\n    \S|\Z)", text, re.S)
    if frame:
        status["phase"] = "fixed_evaluation"
        for field in ("step", "index"):
            value = re.search(rf"\n\s+{field}: (\d+)", frame[1])
            if value:
                status[f"evaluation_{field}"] = int(value[1])
    return status


def snapshot(run, log, py_spy):
    launch = read_json(run / "launch.json")
    cmdline = Path(f"/proc/{launch['pid']}/cmdline")
    live = cmdline.exists() and str(run).encode() in cmdline.read_bytes()
    text = log.read_text(errors="replace") if log.exists() else ""
    totals = re.findall(r"Total steps = (\d+)", text)
    total = int(totals[-1]) if totals else None
    observed = [int(step) for step in re.findall(rf"(\d+)/{total}\b", text)] if total else []
    baseline = read_json(run / "fixed-loss/baseline.json")
    exit_path = run / "exit-status.txt"
    if not exit_path.exists():
        exit_path = run / "exit-status"
    exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
    state = read_json(run / "trainer_state.json")
    update = read_json(run / "adapter-update.json")
    workers = []
    if live and py_spy.is_file():
        for child in children(launch["pid"]):
            for pid in children(child):
                workers.append(worker_status(pid, py_spy))
    return {
        "at": time.time(), "run": str(run), "source_commit": launch.get("source_commit", "unpublished"),
        "launcher_live": live, "exit_code": exit_code, "total_optimizer_updates": total,
        "observed_optimizer_updates": max([0, state.get("global_step", 0), *observed]),
        "workers": workers,
        "baseline": {key: value for key, value in baseline.get("metrics", {}).items()
                     if key.endswith("/overall")},
        "first_weight_update_step": update.get("first_update_step"),
        "checkpoints": sorted(path.name for path in run.glob("checkpoint-*")),
        "fixed_evaluation_failures": [path.name for path in (run / "fixed-loss").glob("failure*.json")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--interval", type=float, default=45)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    base = run.parents[1]
    log = args.log or base / "logs" / f"{run.name}.log"
    py_spy = base / "tools/py-spy/bin/py-spy"
    while True:
        report = snapshot(run, log, py_spy)
        with (run / "observer.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        if args.once or report["exit_code"] is not None:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
