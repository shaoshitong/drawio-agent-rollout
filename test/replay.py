"""Replay the frozen upstream acceptance suites against this package on Linux.

The data bundle supplies the original runners and fixtures; neither is replaced
by smaller synthetic tests. Generated import adapters are written to --out.
"""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def package_digest():
    digest = sha256()
    for path in sorted((ROOT / "drawio").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def adapters(data, output, suite):
    folder = output.parent / (output.name + "-runner")
    folder.mkdir(parents=True, exist_ok=True)
    canonical = suite in ("corpus", "variants", "controls")
    modules = (
        {
            "compiler": "drawio.compiler.compiler",
            "source": "drawio.compiler.source",
            "cli": "drawio.compiler.cli",
            "native_verify": "drawio.compiler.native_verify",
        }
        if canonical
        else {
            "cli": "drawio.compare.cli",
            "compare": "drawio.compare.compare",
            "model": "drawio.compare.model",
        }
    )
    modules["runtime"] = "drawio.runtime.editor"
    fingerprint = package_digest()
    for name, module in modules.items():
        body = f"# package sha256: {fingerprint}\nfrom {module} import *\n"
        if name == "compiler":
            body += """
from drawio.compiler.compiler import compile_bytes as current_compile, dump
from baseline.compiler import compile_bytes as baseline_compile, dump as baseline_dump

def compile_bytes(data, self_check=True):
    current = current_compile(data, self_check=self_check)
    original = baseline_compile(data, self_check=self_check)
    current_artifacts = (current.xml, dump(current.ir_json), dump(current.source_map), dump(current.report))
    original_artifacts = (original.xml, baseline_dump(original.ir_json),
                          baseline_dump(original.source_map), baseline_dump(original.report))
    if current_artifacts != original_artifacts:
        raise AssertionError("Canonical XML, IR, source map or report differs from the frozen baseline")
    return current
"""
        (folder / (name + ".py")).write_text(body)
    for name in ("batch.py", "controls.py", "acceptance.py", "generalization.py"):
        shutil.copyfile(data / "validation" / name, folder / name)
    return folder


def boundaries(data, output):
    specification = importlib.util.spec_from_file_location(
        "frozen_boundaries", data / "validation/boundary_equivalence.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    sys.path[:0] = [str(ROOT), str(data)]
    from drawio.compiler.compiler import compile_bytes, dump
    from drawio.compiler.source import CompileError
    from baseline.compiler import compile_bytes as old_compile, dump as old_dump
    from baseline.source import CompileError as OldError

    differences, valid, invalid = [], 0, 0
    for name, blob in module.fixtures().items():
        records = []
        for compile_one, serialize, error_type in (
            (old_compile, old_dump, OldError),
            (compile_bytes, dump, CompileError),
        ):
            try:
                result = compile_one(blob)
                records.append(
                    {
                        "success": [
                            sha256(value).hexdigest()
                            for value in (
                                result.xml,
                                serialize(result.ir_json),
                                serialize(result.source_map),
                                serialize(result.report),
                            )
                        ]
                    }
                )
            except error_type as error:
                records.append(
                    {"error": [item.to_json() for item in error.diagnostics], "message": str(error)}
                )
        valid += "success" in records[0]
        invalid += "error" in records[0]
        if records[0] != records[1]:
            differences.append(name)
    result = {"cases": valid + invalid, "valid": valid, "invalid": invalid, "differences": differences}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return int(bool(differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("corpus", "variants", "controls", "boundaries", "generalization", "acceptance"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.out.resolve()
    if args.suite == "boundaries":
        return boundaries(data, output)
    folder = adapters(data, output, args.suite)
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(ROOT), str(data))),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    if args.suite in ("corpus", "variants", "controls"):
        command = [
            sys.executable,
            str(folder / "batch.py"),
            str(data / (args.suite + ".jsonl")),
            "--root",
            str(data),
            "--out",
            str(output),
            "--workers",
            str(args.workers),
        ]
        if args.suite != "corpus":
            rows = [json.loads(line) for line in (data / (args.suite + ".jsonl")).read_text().splitlines()]
            bases = {row.get("case", row["id"]): str(data / row["base"]) for row in rows}
            path = folder / "bases.jsonl"
            path.write_text(
                "".join(json.dumps({"id": name, "xml": value}) + "\n" for name, value in bases.items())
            )
            command += ["--bases", str(path)]
    elif args.suite == "generalization":
        command = [
            sys.executable,
            str(folder / "generalization.py"),
            "check",
            str(data / "generalization/sources_100"),
            str(data / "generalization/variants_100"),
            str(output),
            "--workers",
            str(args.workers),
            "--one-per-source",
        ]
    else:
        command = [sys.executable, str(folder / "acceptance.py"), "--out", str(output)]
    result = subprocess.run(command, env=environment)
    if args.suite == "controls":
        return subprocess.run(
            [sys.executable, str(folder / "controls.py"), "check", str(data / "controls.jsonl"), str(output)],
            env=environment,
        ).returncode
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
