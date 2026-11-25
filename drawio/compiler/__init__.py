"""Canonical draw.io compiler with editable graph IR."""

from .adapter import AdaptContext, AdaptedIR, IRAdapter, adapt_ir, get_adapter, register_adapter
from .compiler import (
    COMPILER_VERSION,
    Compilation,
    IRCompilation,
    compile_bytes,
    compile_ir,
    compile_ir_json,
    compile_ir_jsonl,
)
from .ir_json import read_ir_json
from .ir_jsonl import emit_ir_jsonl, read_ir_jsonl, to_ir_jsonl
from .source import CompileError, Diagnostic, Loc

__all__ = [
    "COMPILER_VERSION",
    "AdaptContext",
    "AdaptedIR",
    "Compilation",
    "CompileError",
    "Diagnostic",
    "IRAdapter",
    "IRCompilation",
    "Loc",
    "adapt_ir",
    "compile_bytes",
    "compile_ir",
    "compile_ir_json",
    "compile_ir_jsonl",
    "emit_ir_jsonl",
    "get_adapter",
    "read_ir_json",
    "read_ir_jsonl",
    "register_adapter",
    "to_ir_jsonl",
]
