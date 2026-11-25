"""The compilation pipeline: unpack, parse, build IR, normalize, emit, re-read.

The native entry starts from IR instead of XML: decode IR JSON or compact
JSONL (or adapt an external schema), build and verify the IR, then share the
normalization, emission and self-checks of the XML path.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from operator import itemgetter

from .adapter import adapt_ir
from .build import build
from .emit import emit
from . import ir
from .ir_json import decode, read_ir_json
from . import ir_jsonl
from .normalize import normalize
from .source import CompileError, Diagnostic, unpack

COMPILER_VERSION = "drawio-semantic-compiler/1.1.1"
INPUT_FORMATS = ("jsonl", "ir_json", "adapted")


@dataclass
class Compilation:
    xml: bytes
    document: ir.Document
    ir_json: dict
    source_map: dict
    report: dict


@dataclass
class IRCompilation:
    document: ir.Document
    ir_json: dict
    source_map: dict
    report: dict
    diagnostics: list
    xml: bytes | None = None
    ir_jsonl: str | None = None


def settle(document, origins):
    """Stage 4 on a built IR: record the source draw order, then normalize."""
    for page, origin in zip(document.pages, origins, strict=True):
        for index, cell in enumerate(page.walk()):
            origin[cell.uid]["draw_index"] = index
    return normalize(document, origins)


def front(data):
    """Stages 1-4. Returns the normalized IR and everything recorded on the way."""
    source = unpack(data)
    document, origins, diagnostics, representation = build(source)
    before = [len(page.cells) for page in document.pages]
    rules, audit = settle(document, origins)
    encodings = [page.encoding for page in source.pages]
    return document, origins, diagnostics, representation, rules, audit, encodings, before


def xml_checks(xml, ir_json):
    """The emitted file must read back as the same IR and compile to itself."""
    rebuilt, _, _, repeated_representation, repeated_rules, *_ = front(xml)
    return {
        "emitted_xml_rebuilds_ir": rebuilt.to_json() == ir_json,
        "recompilation_byte_stable": emit(rebuilt) == xml,
        "emitted_xml_is_fixed_point": not (repeated_representation or repeated_rules),
    }


def require(checks):
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise CompileError(
            Diagnostic("error", "self_check", "compiler self-check failed: " + ", ".join(failed))
        )


def source_map(document, origins, encodings, before):
    pages = []
    for number, (page, origin, encoding, cells_before) in enumerate(
        zip(document.pages, origins, encodings, before, strict=True), 1
    ):
        origin_points = ir.page_origins(page)
        for cell in page.cells.values():
            position = origin_points[cell.uid]
            if position is not None and cell.kind in (ir.VERTEX, ir.FRAME):
                origin[cell.uid]["page_xy"] = [float(position[0]), float(position[1])]
        cells = sorted(origin.values(), key=itemgetter("root_child_index"))
        pages.append(
            {
                "page": number,
                "name": page.name,
                "source_encoding": encoding,
                "source_cells": cells_before,
                "cells": cells,
            }
        )
    return {"schema": "drawio-semantic-source-map/1", "pages": pages}


def digest(data):
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data)}


def compile_bytes(data, self_check=True):
    """Compile draw.io XML bytes to canonical XML, IR, source map and report."""
    document, origins, diagnostics, representation, rules, audit, encodings, before = front(data)
    xml = emit(document)
    ir_json = document.to_json()
    checks = {"ir_verified": True}
    if self_check:
        checks.update(xml_checks(xml, ir_json))
        require(checks)
    report = {
        "schema": "drawio-semantic-compile-report/1",
        "compiler": COMPILER_VERSION,
        "ir": ir.IR_VERSION,
        "input": digest(data),
        "output": digest(xml),
        "ir_sha256": sha256(dump(ir_json)).hexdigest(),
        "pages": len(document.pages),
        "stages": [
            "unpack",
            "lex_and_parse",
            "build_and_verify_ir",
            "normalize_ir",
            "emit_xml",
            "reread_and_recompile",
        ],
        "checks": checks,
        "representation_changes": representation,
        "rules_applied": rules,
        "reducibility_audit": audit,
        "diagnostics": [d.to_json() for d in diagnostics],
    }
    return Compilation(xml, document, ir_json, source_map(document, origins, encodings, before), report)


def read(source, input_format, adapter, allow_incomplete):
    """Stage 1 of the native entry: decode or adapt to a verified IR."""
    if adapter is not None or input_format == "adapted":
        if input_format not in (None, "adapted") or adapter is None:
            raise CompileError(
                Diagnostic("error", "input_format", 'an adapter goes with input_format="adapted"')
            )
        adapted = adapt_ir(source, adapter)
        decoded = decode(adapted.document, encoding="adapted")
        decoded.diagnostics[:0] = adapted.diagnostics
        return decoded
    if input_format == "jsonl":
        return ir_jsonl.read_ir_jsonl(source, allow_incomplete=allow_incomplete)
    if input_format == "ir_json":
        return read_ir_json(source)
    raise CompileError(
        Diagnostic(
            "error",
            "input_format",
            f"input_format must be one of {list(INPUT_FORMATS)} (or pass adapter=), not {input_format!r}",
        )
    )


def compile_ir(
    source,
    *,
    input_format=None,
    emit_xml=True,
    emit_jsonl=False,
    self_check=True,
    allow_incomplete=False,
    allow_lossy_jsonl=False,
    adapter=None,
):
    """Compile IR to the canonical IR, and optionally canonical XML and compact JSONL.

    ``input_format`` is ``"jsonl"`` (compact JSON Lines), ``"ir_json"``
    (``drawio-semantic-ir/2``) or ``"adapted"`` with ``adapter`` (an
    ``IRAdapter`` or a registered name). ``source`` is bytes, text or, for IR
    JSON and adapters, an object. The canonical IR and XML equal what
    :func:`compile_bytes` gives for ``emit`` of the same IR. ``self_check``
    proves that by re-reading the emitted XML, whether or not it is returned,
    and that emitted JSONL reads back as the same IR.
    """
    decoded = read(source, input_format, adapter, allow_incomplete)
    document, origins = decoded.document, decoded.origins
    before = [len(page.cells) for page in document.pages]
    rules, audit = settle(document, origins)
    ir_json = document.to_json()
    diagnostics = list(decoded.diagnostics)
    checks = {"ir_verified": True}
    xml = emit(document) if emit_xml or self_check else None
    if self_check:
        checks.update(xml_checks(xml, ir_json))
    text, problems = None, []
    if emit_jsonl:
        text, problems = ir_jsonl.project(document, "warning" if allow_lossy_jsonl else "error")
        if problems and not allow_lossy_jsonl:
            raise CompileError(problems + diagnostics)
        diagnostics.extend(problems)
        if self_check and not problems:
            try:
                rebuilt = ir_jsonl.reread(text)
                checks["emitted_jsonl_rebuilds_ir"] = rebuilt.to_json() == ir_json
                checks["jsonl_byte_stable"] = ir_jsonl.project(rebuilt)[0] == text
            except CompileError:
                checks["emitted_jsonl_rebuilds_ir"] = False
    require(checks)
    stages = ["adapt" if decoded.encoding == "adapted" else "decode_" + decoded.encoding.replace("-", "_")]
    stages += ["build_and_verify_ir", "normalize_ir"]
    if xml is not None:
        stages.append("emit_xml")
    if self_check:
        stages.append("reread_and_recompile")
    if emit_jsonl:
        stages.append("emit_jsonl")
        if "emitted_jsonl_rebuilds_ir" in checks:
            stages.append("reread_jsonl")
    if isinstance(source, str):
        source = source.encode("utf-8")
    report = {
        "schema": "drawio-semantic-compile-report/1",
        "compiler": COMPILER_VERSION,
        "ir": ir.IR_VERSION,
        "input": {"format": decoded.encoding, **(digest(source) if isinstance(source, bytes) else {})},
        "output": digest(xml) if emit_xml else None,
        "ir_jsonl_output": digest(text.encode("utf-8")) if text is not None else None,
        "ir_jsonl_lossless": not problems if emit_jsonl else None,
        "ir_sha256": sha256(dump(ir_json)).hexdigest(),
        "pages": len(document.pages),
        "stages": stages,
        "checks": checks,
        "representation_changes": decoded.representation,
        "rules_applied": rules,
        "reducibility_audit": audit,
        "diagnostics": [d.to_json() for d in diagnostics],
    }
    encodings = [decoded.encoding] * len(document.pages)
    return IRCompilation(
        document,
        ir_json,
        source_map(document, origins, encodings, before),
        report,
        diagnostics,
        xml if emit_xml else None,
        text,
    )


def compile_ir_jsonl(data, *, emit_xml=True, emit_jsonl=True, self_check=True, allow_incomplete=False):
    """Compile compact JSONL; by default also returns its canonical JSONL."""
    return compile_ir(
        data,
        input_format="jsonl",
        emit_xml=emit_xml,
        emit_jsonl=emit_jsonl,
        self_check=self_check,
        allow_incomplete=allow_incomplete,
    )


def compile_ir_json(data, *, emit_xml=True, emit_jsonl=False, self_check=True, allow_lossy_jsonl=False):
    """Compile ``drawio-semantic-ir/2`` JSON."""
    return compile_ir(
        data,
        input_format="ir_json",
        emit_xml=emit_xml,
        emit_jsonl=emit_jsonl,
        self_check=self_check,
        allow_lossy_jsonl=allow_lossy_jsonl,
    )


def dump(value):
    return (json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
