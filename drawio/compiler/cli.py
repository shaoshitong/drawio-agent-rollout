"""drawio-compile: compile one draw.io file to canonical editable XML through a graph IR."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from .compiler import COMPILER_VERSION, compile_bytes, compile_ir, dump
from .ir_jsonl import to_ir_jsonl
from .source import CompileError

OUTPUT = "canonical.drawio"
JSONL_OUTPUT = "ir.jsonl"
INPUT_FORMATS = {"xml": None, "ir-json": "ir_json", "ir-jsonl": "jsonl"}


def write_outputs(folder, result):
    if result.xml is not None:
        (folder / OUTPUT).write_bytes(result.xml)
    (folder / "ir.json").write_bytes(dump(result.ir_json))
    (folder / "source_map.json").write_bytes(dump(result.source_map))
    (folder / "report.json").write_bytes(dump(result.report))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="drawio-compile", description=__doc__)
    parser.add_argument(
        "source", type=Path, help="a .drawio file or bare mxGraphModel XML, or IR with --input-format"
    )
    parser.add_argument("--out", type=Path, required=True, help="new directory for the outputs")
    parser.add_argument(
        "--input-format",
        choices=sorted(INPUT_FORMATS),
        default="xml",
        help="xml (default), drawio-semantic-ir/2 JSON or compact JSON Lines IR",
    )
    parser.add_argument("--no-xml", action="store_true", help=f"do not write {OUTPUT}")
    parser.add_argument(
        "--emit-jsonl", action="store_true", help=f"also write the canonical compact JSONL IR as {JSONL_OUTPUT}"
    )
    parser.add_argument(
        "--verify-native",
        action="store_true",
        help="also open source and output in the pinned draw.io editor and require equal "
        "pixels and editing behaviour (needs the lab runtime)",
    )
    parser.add_argument("--version", action="version", version=COMPILER_VERSION)
    args = parser.parse_args(argv)
    if args.no_xml and args.input_format == "xml":
        parser.error("--no-xml needs an IR --input-format; XML input always compiles to XML")
    if args.verify_native and args.input_format != "xml":
        parser.error("--verify-native compares XML source and output; it needs xml input")
    native = jsonl = None
    try:
        if args.out.exists():
            raise OSError(f"output directory already exists: {args.out}")
        data = args.source.read_bytes()
        if args.input_format == "xml":
            result = compile_bytes(data)
            if args.emit_jsonl:
                jsonl, _ = to_ir_jsonl(result)
        else:
            result = compile_ir(
                data,
                input_format=INPUT_FORMATS[args.input_format],
                emit_xml=not args.no_xml,
                emit_jsonl=args.emit_jsonl,
            )
            jsonl = result.ir_jsonl
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".drawio-compile-", dir=args.out.parent) as staging:
            folder = Path(staging) / "result"
            folder.mkdir()
            write_outputs(folder, result)
            if jsonl is not None:
                (folder / JSONL_OUTPUT).write_bytes(jsonl.encode("utf-8"))
            if args.verify_native:
                from .native_verify import verify_files

                native = verify_files(data, result, folder)
                (folder / "native.json").write_bytes(dump(native))
            folder.rename(args.out)
    except CompileError as error:
        for diagnostic in error.diagnostics:
            print(f"drawio-compile: {args.source}: {diagnostic}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"drawio-compile: {error}", file=sys.stderr)
        return 2
    written = result.xml is not None
    summary = {
        "output": str(args.out / OUTPUT) if written else None,
        "sha256": result.report["output"]["sha256"] if written else None,
        "pages": result.report["pages"],
        "input_bytes": len(data),
        "output_bytes": len(result.xml) if written else None,
        "warnings": sum(d["severity"] == "warning" for d in result.report["diagnostics"]),
    }
    if jsonl is not None:
        summary["jsonl"] = str(args.out / JSONL_OUTPUT)
    if native is not None:
        summary["native_equal"] = native["equal"]
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if native is None or native["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
