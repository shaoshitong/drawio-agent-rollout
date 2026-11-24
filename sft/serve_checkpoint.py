"""Serve one scheduler-owned checkpoint for an external identical DSH harness."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

from .evaluate_checkpoint import ready, save, start_model, stop_model


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "out", "stop-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=32645)
    parser.add_argument("--sampling-preset", default="precise-coding")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    server, log = start_model(args)
    signal.signal(signal.SIGTERM, lambda *_: args.stop_file.touch())
    try:
        ready(server, args.port, timeout=900)
        save(args.out / "ready.json", {"pid": os.getpid(), "server_pid": server.pid,
             "job_id": os.environ.get("SLURM_JOB_ID"), "at": time.time()})
        while server.poll() is None and not args.stop_file.exists():
            time.sleep(5)
        if server.poll() not in (None, 0):
            raise RuntimeError("Owned inference server exited: " + str(server.returncode))
    finally:
        stop_model(server, log)


if __name__ == "__main__":
    main()
