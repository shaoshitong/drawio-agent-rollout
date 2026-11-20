"""Canonical draw.io compiler with editable graph IR."""

from .compiler import COMPILER_VERSION, compile_bytes
from .source import CompileError

__all__ = ["COMPILER_VERSION", "CompileError", "compile_bytes"]
