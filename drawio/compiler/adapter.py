"""Native entry: adapters from external graph schemas onto ``drawio-semantic-ir/2``.

An adapter maps a foreign document to an IR/2 object and states the space
its geometry is measured in. The compiler accepts exactly what the editor
needs to place every cell: stable IDs, one parent per cell, vertex boxes and a
position for every edge end. It does not lay out graphs, so a topology-only
input is rejected with ``adapter_missing_geometry`` rather than guessed.
Page-pixel geometry is converted to the IR's parent-local pixels exactly.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from . import ir
from .ir_json import is_list, number
from .source import CompileError, Diagnostic, Loc

PARENT_LOCAL, PAGE = "parent-local-px", "page-px"
COORDINATE_SPACES = (PARENT_LOCAL, PAGE)


@dataclass(frozen=True)
class AdaptContext:
    """What the compiler tells an adapter about the IR it expects."""

    adapter: str
    ir_version: str = ir.IR_VERSION
    coordinate_spaces: tuple = COORDINATE_SPACES


@dataclass
class AdaptedIR:
    document: dict  # a drawio-semantic-ir/2 object; "ir" may be left out
    coordinate_space: str  # one of COORDINATE_SPACES
    diagnostics: list = field(default_factory=list)


@runtime_checkable
class IRAdapter(Protocol):
    name: str

    def adapt(self, source, context: AdaptContext) -> AdaptedIR: ...


_ADAPTERS = {}


def register_adapter(adapter, *, replace=False):
    """Make an adapter available by its ``name``. Returns the adapter."""
    if not isinstance(adapter, IRAdapter) or not isinstance(adapter.name, str) or not adapter.name:
        raise TypeError("an adapter has a non-empty name and an adapt(source, context) method")
    if adapter.name in _ADAPTERS and not replace:
        raise ValueError(f"an adapter named {adapter.name!r} is already registered")
    _ADAPTERS[adapter.name] = adapter
    return adapter


def get_adapter(name):
    try:
        return _ADAPTERS[name]
    except KeyError:
        known = ", ".join(sorted(_ADAPTERS)) or "none"
        raise CompileError(
            Diagnostic("error", "adapter_unknown", f"no adapter named {name!r}; registered: {known}")
        ) from None


def adapt_ir(source, adapter):
    """Run an adapter (or the one registered under a name) and check its output.

    Returns an :class:`AdaptedIR` in parent-local pixels whose document is
    ready for the IR/2 decoder.
    """
    if isinstance(adapter, str):
        adapter = get_adapter(adapter)
    if not isinstance(adapter, IRAdapter):
        raise TypeError("expected an IRAdapter or the name of a registered one")
    adapted = adapter.adapt(source, AdaptContext(adapter.name))
    if not isinstance(adapted, AdaptedIR) or not isinstance(adapted.document, dict):
        raise CompileError(
            Diagnostic("error", "adapter_output", f"adapter {adapter.name!r} did not return an AdaptedIR document")
        )
    errors = []

    def error(code, message, page=None, path=(), cell_id=None, attribute=None):
        loc = Loc(page=page, path="/" + "/".join(str(p) for p in path))
        errors.append(Diagnostic("error", code, message, loc, cell_id, attribute))

    if adapted.coordinate_space not in COORDINATE_SPACES:
        error(
            "adapter_coordinate_space",
            f"coordinate space {adapted.coordinate_space!r} is not one of {list(COORDINATE_SPACES)}",
        )
    document = deepcopy(adapted.document)
    document.setdefault("ir", ir.IR_VERSION)
    pages = document.get("pages")
    for index, page in enumerate(pages if is_list(pages) else ()):
        cells = page.get("cells") if isinstance(page, dict) else None
        for position, cell in enumerate(cells if is_list(cells) else ()):
            if isinstance(cell, dict):
                check_cell(cell, index + 1, ["pages", index, "cells", position], error)
        if adapted.coordinate_space == PAGE and is_list(cells):
            to_parent_local(cells, index + 1, ["pages", index, "cells"], error)
    adapter_errors = [d for d in adapted.diagnostics if d.severity == "error"]
    if errors or adapter_errors:
        raise CompileError(errors + list(adapted.diagnostics))
    diagnostics = list(adapted.diagnostics)
    if adapted.coordinate_space == PAGE:
        diagnostics.append(
            Diagnostic("info", "adapter_coordinate_space", f"geometry converted from {PAGE} to {PARENT_LOCAL}")
        )
    return AdaptedIR(document, PARENT_LOCAL, diagnostics)


def check_cell(cell, page, path, error):
    """Stable IDs and an explicit position for everything the editor places."""
    cell_id = cell.get("id")
    if not isinstance(cell_id, str) or not cell_id:
        error("adapter_missing_id", "every cell needs a stable, non-empty string ID", page, path=[*path, "id"])
        cell_id = None
    kind = cell.get("kind")
    geometry = cell.get("geometry")
    geometry = geometry if isinstance(geometry, dict) else {}
    if kind == ir.VERTEX:
        missing = [key for key in ("x", "y", "width", "height") if geometry.get(key) is None]
        if missing:
            error(
                "adapter_missing_geometry",
                "vertex has no " + ", ".join(missing) + "; the compiler does not lay out graphs",
                page,
                path=[*path, "geometry"],
                cell_id=cell_id,
                attribute="geometry",
            )
    elif kind == ir.EDGE:
        for end, point in (("source", "source_point"), ("target", "target_point")):
            if cell.get(end) is None and geometry.get(point) is None:
                error(
                    "adapter_missing_geometry",
                    f"edge has neither a {end} terminal nor a {point}",
                    page,
                    path=[*path, "geometry"],
                    cell_id=cell_id,
                    attribute=point,
                )


def to_parent_local(cells, page, path, error):
    """Rewrite page-pixel positions as offsets from each parent's origin."""
    by_id = {c.get("id"): c for c in cells if isinstance(c, dict) and isinstance(c.get("id"), str)}
    origins = {}

    def origin(cell, seen=()):
        """Page position of a cell's local origin, or None if it is not defined."""
        cell_id = cell.get("id")
        if cell_id in origins:
            return origins[cell_id]
        geometry = cell.get("geometry") if isinstance(cell.get("geometry"), dict) else None
        kind = cell.get("kind")
        if kind in (ir.ROOT, ir.LAYER):
            result = (0, 0)
        elif kind == ir.EDGE or cell_id in seen or (geometry is not None and geometry.get("relative")):
            result = None
        elif geometry is not None and (geometry.get("x") is not None or geometry.get("y") is not None):
            result = tuple(number(geometry.get(key, 0)) for key in ("x", "y"))
            result = None if None in result else result
        else:  # a frame without geometry sits on its parent's origin
            parent = by_id.get(cell.get("parent"))
            result = None if parent is None else origin(parent, (*seen, cell_id))
        origins[cell_id] = result
        return result

    for cell in cells:
        if isinstance(cell, dict):
            origin(cell)

    for position, cell in enumerate(cells):
        if not isinstance(cell, dict) or not isinstance(cell.get("geometry"), dict):
            continue
        geometry = cell["geometry"]
        where = [*path, position, "geometry"]
        parent = by_id.get(cell.get("parent"))
        if parent is None:
            continue  # the decoder reports the unresolved parent
        if geometry.get("relative") and cell.get("kind") != ir.EDGE:
            error(
                "adapter_coordinate_space",
                "relative geometry has no page-pixel position; adapt it in parent-local-px",
                page,
                path=where,
                cell_id=cell.get("id"),
                attribute="relative",
            )
            continue
        base = origin(parent)
        if base is None:
            error(
                "adapter_coordinate_space",
                "the parent has no page-pixel origin (an edge, a cycle or invalid geometry)",
                page,
                path=where,
                cell_id=cell.get("id"),
                attribute="parent",
            )
            continue

        def local(pair):
            values = [number(v) for v in pair] if is_list(pair) and len(pair) == 2 else None
            if values is None or None in values:
                return pair  # left for the decoder to report
            return [values[0] - base[0], values[1] - base[1]]

        if cell.get("kind") == ir.EDGE:
            for key in ("source_point", "target_point"):
                if geometry.get(key) is not None:
                    geometry[key] = local(geometry[key])
            if is_list(geometry.get("points")):
                geometry["points"] = [local(p) for p in geometry["points"]]
            if not geometry.get("relative"):
                if geometry.get("x") is not None or geometry.get("y") is not None:
                    x, y = local([geometry.get("x", 0), geometry.get("y", 0)])
                    geometry["x"], geometry["y"] = x, y
                bounds = geometry.get("alternate_bounds")
                if is_list(bounds) and len(bounds) == 4:
                    geometry["alternate_bounds"] = [*local(bounds[:2]), *bounds[2:]]
        else:
            x, y = local([geometry.get("x", 0), geometry.get("y", 0)])
            geometry["x"], geometry["y"] = x, y
            bounds = geometry.get("alternate_bounds")
            if is_list(bounds) and len(bounds) == 4:
                geometry["alternate_bounds"] = [*local(bounds[:2]), *bounds[2:]]
