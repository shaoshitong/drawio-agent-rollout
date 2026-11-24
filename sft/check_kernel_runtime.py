"""Compare two isolated runtimes on one GPU before full-context capacity tests."""

import argparse
from copy import copy
import json
from pathlib import Path

from .check_activation_offload import compare, run_trial, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "candidate-root", "source", "runtime", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    indices = config["gate_indices"]
    candidate_args = copy(args)
    candidate_args.root = args.candidate_root
    code = 1
    try:
        baseline = run_trial(args, config, "baseline", False, indices[:2])
        candidate = run_trial(candidate_args, config, "candidate", False, indices[:2])
        result = compare(baseline, candidate)
        result.update(baseline_runtime=str(args.root), candidate_runtime=str(args.candidate_root),
                      same_gpu_allocation=True, external_activation_offloading=False)
        write_json(args.output / "comparison.json", result)
        if not result["passed"]:
            raise ValueError("Kernel runtime exceeds the existing optimizer parity thresholds")
        run_trial(candidate_args, config, "capacity", False, indices)
        code = 0
    finally:
        (args.output / "exit-code").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
