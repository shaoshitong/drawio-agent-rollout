"""Native entry: the compact JSON Lines IR the visual model writes and reads.

One record per line::

    {"canvas":[800,600]}
    {"id":"a","k":"v","g":[20,20,110,45],"l":"Input","s":"rounded=1"}
    {"id":"e1","k":"e","f":"a","t":"b","w":[[160,42]]}
    {"end":true}

The reader creates the root ``0`` and layer ``1``, maps each record to
``drawio-semantic-ir/2`` and decodes that, so a JSONL graph means exactly
what the same graph written as IR JSON means. Every diagnostic names the line
of the record in the original text, also inside a Markdown fence.

The projection writes the canonical JSONL of a canonical IR. JSONL holds one
page with one layer of vertices and edges; anything else is reported as a
``jsonl_lossy_projection`` instead of being dropped silently.
"""

from copy import deepcopy
from decimal import Decimal
import json
import re

from . import ir
from .ir_json import Rejected, decode, loads, number, read_ir_json
from .normalize import normalize
from .source import LIMIT, CompileError, Diagnostic, Loc
from .syntax import format_number, format_style, parse_style

ROOT_ID, LAYER_ID = "0", "1"
PAGE_NAME = "Page 1"
CANVAS = ("pageWidth", "pageHeight")
DEFAULT_CANVAS = (Decimal(827), Decimal(1169))  # the editor's default A4 page
FIELDS = {
    "v": frozenset({"id", "k", "p", "l", "s", "g"}),
    "e": frozenset({"id", "k", "p", "l", "s", "f", "t", "sp", "tp", "w"}),
}
_OPEN = re.compile(r"\s*```\s*(?:jsonl|ndjson|json)?\s*\Z", re.I)
_CLOSE = re.compile(r"\s*```\s*\Z")


class Reader:
    def __init__(self, allow_incomplete):
        self.allow_incomplete = allow_incomplete
        self.errors, self.diagnostics = [], []

    def note(self, severity, code, message, loc, cell_id=None, attribute=None):
        if severity == "error" and self.allow_incomplete and code in ("jsonl_end_missing", "jsonl_fence"):
            severity = "warning"
        diagnostics = self.errors if severity == "error" else self.diagnostics
        diagnostics.append(Diagnostic(severity, code, message, loc, cell_id, attribute))

    def lines(self, data, allow_fences):
        """Yield (loc, text) for each non-blank line of the JSONL body."""
        if isinstance(data, (bytes, bytearray)):
            if len(data) > LIMIT:
                raise CompileError(Diagnostic("error", "input_too_large", f"JSONL exceeds {LIMIT} bytes"))
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as error:
                raise CompileError(
                    Diagnostic("error", "jsonl_encoding", f"JSONL is not UTF-8: {error.reason}", Loc(byte=error.start))
                ) from None
        if not isinstance(data, str):
            raise CompileError(Diagnostic("error", "jsonl_encoding", "JSONL must be bytes or text"))
        if len(data.encode("utf-8")) > LIMIT:
            raise CompileError(Diagnostic("error", "input_too_large", f"JSONL exceeds {LIMIT} bytes"))
        lines, start, offsets = data.split("\n"), 0, []
        for line in lines:
            offsets.append(start)
            start += len(line.encode("utf-8")) + 1
        filled = [i for i, line in enumerate(lines) if line.strip()]
        body = filled
        if allow_fences and filled and _OPEN.match(lines[filled[0]]):
            first, last = filled[0], filled[-1]
            if len(filled) > 1 and _CLOSE.match(lines[last]):
                body = filled[1:-1]
            else:
                body = filled[1:]
                self.note("error", "jsonl_fence", "the Markdown fence is not closed", Loc(first + 1, 1, offsets[first], 1))
        for index in body:
            line = lines[index]
            column = len(line) - len(line.lstrip()) + 1
            yield Loc(index + 1, column, offsets[index] + len(line[: column - 1].encode("utf-8")), 1), line

    def records(self, data, allow_fences):
        """Parse the lines into (loc, record) for valid JSON objects."""
        lines = list(self.lines(data, allow_fences))
        for position, (loc, line) in enumerate(lines):
            try:
                record = loads(line)
            except json.JSONDecodeError as error:
                if self.allow_incomplete and position == len(lines) - 1:
                    self.note("warning", "jsonl_truncated", f"the last line is cut off: {error.msg}", loc)
                    continue
                prefix = line[: error.pos]
                at = Loc(loc.line, error.pos + 1, loc.byte - loc.column + 1 + len(prefix.encode("utf-8")), 1)
                self.note("error", "jsonl_json", f"malformed JSON: {error.msg}", at)
                continue
            except Rejected as error:
                code = "jsonl_duplicate_key" if error.code == "json_duplicate_key" else error.code
                self.note("error", code, str(error), loc)
                continue
            except (ValueError, RecursionError) as error:
                self.note("error", "jsonl_json", f"malformed JSON: {error}", loc)
                continue
            if not isinstance(record, dict):
                self.note("error", "jsonl_record_type", f"a record is a JSON object, not {type(record).__name__}", loc)
                continue
            yield loc, record

    def read(self, data, allow_fences):
        canvas = canvas_loc = end = None
        rows = []  # (loc, IR/2 cell record)
        for loc, record in self.records(data, allow_fences):
            if end is not None:
                self.note("error", "jsonl_end_not_last", f"a record follows the end record of line {end.line}", loc)
            elif "canvas" in record:
                if canvas_loc is not None:
                    self.note("error", "jsonl_canvas", f"a second canvas record; the first is on line {canvas_loc.line}", loc)
                elif rows:
                    self.note("error", "jsonl_canvas", "the canvas record must be the first record", loc)
                else:
                    canvas_loc, canvas = loc, self.canvas(record, loc)
            elif "end" in record:
                if record != {"end": True}:
                    self.note("error", "jsonl_record", 'the end record is exactly {"end": true}', loc)
                end = loc
            else:
                if canvas_loc is None and not rows:
                    self.note("error", "jsonl_canvas", "the first record must be the canvas record", loc)
                cell = self.cell(record, loc)
                if cell is not None:
                    rows.append((loc, cell))
        if canvas_loc is None and not any(d.code == "jsonl_canvas" for d in self.errors):
            self.note("error", "jsonl_canvas", "the canvas record is missing", Loc(1, 1, 0, 1))
        if end is None:
            self.note("error", "jsonl_end_missing", 'the graph does not finish with {"end": true}', rows[-1][0] if rows else canvas_loc)
        if self.errors:
            raise CompileError(self.errors + self.diagnostics)
        return canvas, canvas_loc, rows

    def canvas(self, record, loc):
        value = record["canvas"]
        if record.keys() != {"canvas"} or not isinstance(value, list) or len(value) != 2:
            self.note("error", "jsonl_canvas", 'the canvas record is {"canvas": [width, height]}', loc)
            return None
        size = [number(v) for v in value]
        if any(v is None or v <= 0 for v in size):
            self.note("error", "jsonl_canvas", f"canvas width and height must be positive finite numbers: {value!r}", loc)
            return None
        return size

    def cell(self, record, loc):
        """Check one cell record and map it to an IR/2 cell. Returns None when it is invalid."""
        before = len(self.errors)
        cell_id = record.get("id") if isinstance(record.get("id"), str) else None

        def bad(code, message, attribute=None):
            self.note("error", code, message, loc, cell_id, attribute)

        kind = record.get("k")
        if kind not in FIELDS:
            if kind is None:
                bad("jsonl_missing_field", "missing field 'k'", "k")
            else:
                bad("jsonl_kind", f"k must be 'v' (vertex) or 'e' (edge), not {kind!r}", "k")
            allowed = FIELDS["v"] | FIELDS["e"]
        else:
            allowed = FIELDS[kind]
        for key in sorted(record.keys() - allowed):
            bad("jsonl_unknown_field", f"unknown field {key!r}" + (f" for k={kind!r}" if kind in FIELDS else ""), key)
        for key in ("id", "g") if kind == "v" else ("id",):
            if key not in record:
                bad("jsonl_missing_field", f"missing field {key!r}", key)
        for key in ("id", "p", "f", "t", "l", "s"):
            if key in record and not isinstance(record[key], str):
                bad("jsonl_field_type", f"{key} must be a string", key)
        if cell_id == "":
            bad("jsonl_field_type", "id must not be empty", "id")
        elif cell_id in (ROOT_ID, LAYER_ID):
            bad("jsonl_reserved_id", f"ID {cell_id!r} is reserved for the root and the layer", "id")

        def coordinates(key, value, size):
            if not isinstance(value, list) or len(value) != size:
                bad("invalid_geometry", f"{key} must be a list of {size} numbers", key)
                return None
            result = [number(v) for v in value]
            if None in result:
                bad("invalid_number", f"{key} holds a value that is not a finite number: {value!r}", key)
                return None
            return result

        geometry = {}
        if kind == "v" and "g" in record:
            g = coordinates("g", record["g"], 4)
            if g is not None:
                if g[2] < 0 or g[3] < 0:
                    bad("negative_size", "vertex width and height must not be negative", "g")
                geometry = dict(zip(("x", "y", "width", "height"), g, strict=True))
        elif kind == "e":
            geometry = {"relative": True}
            for key, name in (("sp", "source_point"), ("tp", "target_point")):
                if key in record:
                    geometry[name] = coordinates(key, record[key], 2)
            if "w" in record:
                if isinstance(record["w"], list):
                    geometry["points"] = [coordinates("w", point, 2) for point in record["w"]]
                else:
                    bad("invalid_geometry", "w must be a list of [x, y] waypoints", "w")
        if len(self.errors) > before:
            return None
        cell = {"id": cell_id, "kind": "vertex" if kind == "v" else "edge", "parent": record.get("p", LAYER_ID)}
        if "l" in record:
            cell["label"] = record["l"]
        if "s" in record:
            cell["style"] = [list(token) for token in parse_style(record["s"])]
        cell["geometry"] = geometry
        for key, name in (("f", "source"), ("t", "target")):
            if key in record:
                cell[name] = record[key]
        return cell


def read_ir_jsonl(data, *, allow_incomplete=False, allow_fences=True):
    """Decode compact JSONL into a verified IR, not yet normalized.

    ``allow_incomplete`` accepts a graph cut off before its end record (or
    inside its last line, or before a closing fence) with a warning.
    """
    reader = Reader(allow_incomplete)
    canvas, canvas_loc, rows = reader.read(data, allow_fences)
    width, height = (format_number(v) for v in canvas)
    cells = [{"id": ROOT_ID, "kind": "root"}, {"id": LAYER_ID, "kind": "layer", "parent": ROOT_ID}]
    cells.extend(cell for _, cell in rows)
    payload = {
        "ir": ir.IR_VERSION,
        "pages": [{"name": PAGE_NAME, "settings": [["pageWidth", width], ["pageHeight", height]], "cells": cells}],
    }
    locs = [canvas_loc, canvas_loc, *(loc for loc, _ in rows)]

    def locate(page, cell, path):
        return canvas_loc if cell is None else locs[cell]

    decoded = decode(payload, encoding="ir-jsonl", locate=locate)
    decoded.diagnostics[:0] = reader.diagnostics
    return decoded


def json_number(value):
    """A coordinate as the JSON number that prints as ``format_number`` prints it."""
    as_float = float(value)
    return int(as_float) if as_float == int(as_float) and abs(as_float) < 1e16 else as_float


def dump(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def project(document, severity="error"):
    """Write the canonical JSONL of a canonical IR.

    Returns (text, diagnostics): one ``jsonl_lossy_projection`` diagnostic of
    ``severity`` per construct JSONL cannot hold. The text is a best effort
    when there are any: the first page, its vertices and edges.
    """
    problems = []

    def lossy(page, cell, attribute, message):
        problems.append(
            Diagnostic(
                severity,
                "jsonl_lossy_projection",
                f"compact JSONL cannot hold {message}",
                Loc(page=page),
                cell.name if cell is not None else None,
                attribute,
            )
        )

    for extra, _ in enumerate(document.pages[1:], 2):
        lossy(extra, None, None, "more than one page")
    if document.attrs:
        lossy(None, None, "attributes", "document attributes")
    if document.extras:
        lossy(None, None, "extensions", "document extensions")
    page = document.pages[0]
    if page.name != PAGE_NAME:
        lossy(1, None, "name", f"the page name {page.name!r}; it reads back as {PAGE_NAME!r}")
    if page.anchor is not None:
        lossy(1, None, "id", "a page ID")
    if page.attrs:
        lossy(1, None, "attributes", "page attributes")
    if page.extras or page.model_extras:
        lossy(1, None, "extensions", "page or model extensions")
    settings = dict(page.model_attrs)
    for key in sorted(settings.keys() - set(CANVAS)):
        lossy(1, None, key, f"the model setting {key!r}")
    canvas = []
    for key, default in zip(CANVAS, DEFAULT_CANVAS, strict=True):
        value = number(settings.get(key))
        if value is None or value <= 0 or format_number(value) != settings[key]:
            lossy(1, None, key, f"{key}={settings.get(key)!r}; the canvas is a positive canonical number")
            value = default
        canvas.append(json_number(value))
    rows = [{"canvas": canvas}]

    root = page.cells[page.root]
    first = page.cells[root.children[0]] if root.children else None
    layer = first if first is not None and first.kind == ir.LAYER else None
    if root.name != ROOT_ID:
        lossy(1, root, "id", f"the root ID {root.name!r}; the root is {ROOT_ID!r}")
    if layer is None:
        lossy(1, root, None, "a page whose first root child is not a layer")
    elif layer.name != LAYER_ID:
        lossy(1, layer, "id", f"the layer ID {layer.name!r}; the layer is {LAYER_ID!r}")
    names = {root.uid: ROOT_ID}
    if layer is not None:
        names[layer.uid] = LAYER_ID
    taken = {cell.name for cell in page.cells.values()} | {ROOT_ID, LAYER_ID}
    fresh = (f"n{i}" for i in range(len(page.cells) * 2 + 2))
    origins = ir.page_origins(page)
    for cell in page.walk():
        if cell.uid in names:
            continue
        if cell.kind not in (ir.VERTEX, ir.EDGE):
            lossy(1, cell, "kind", f"a {cell.kind}" + (" besides the first layer" if cell.kind == ir.LAYER else ""))
            continue
        name = cell.name
        if name in (ROOT_ID, LAYER_ID):
            lossy(1, cell, "id", f"the ID {name!r} on a {cell.kind}")
            name = next(n for n in fresh if n not in taken)
            taken.add(name)
        names[cell.uid] = name
    for cell in (root, layer):
        if cell is not None:
            check_implied(cell, lossy)
    for cell in page.walk():
        if cell.kind in (ir.VERTEX, ir.EDGE):
            rows.append(record(page, cell, names, origins, lossy))
    rows.append({"end": True})
    return "".join(dump(row) + "\n" for row in rows), problems


def check_implied(cell, lossy):
    """The root and the layer are implied by JSONL and carry nothing."""
    for attribute, present in (
        ("label", cell.label is not None),
        ("user_object", cell.wrapper is not None),
        ("style", cell.style is not None),
        ("geometry", cell.geometry is not None),
        ("attributes", bool(cell.attrs)),
        ("extensions", bool(cell.extras)),
    ):
        if present:
            lossy(1, cell, attribute, f"{attribute} on the {cell.kind}")
    check_options(cell, lossy)


def check_options(cell, lossy):
    for key, default in ir.CELL_OPTIONS:
        if getattr(cell, key) != default:
            lossy(1, cell, key, f"{key}={str(getattr(cell, key)).lower()}")


def record(page, cell, names, origins, lossy):
    """The JSONL record of one vertex or edge."""
    vertex = cell.kind == ir.VERTEX
    row = {"id": names[cell.uid], "k": "v" if vertex else "e"}
    parent = cell.parent
    while parent not in names:  # lossy: hoist out of cells JSONL does not hold
        parent = page.cells[parent].parent
    if names[parent] != LAYER_ID:
        row["p"] = names[parent]
    shift = None
    if parent != cell.parent and origins[cell.parent] is not None and origins[parent] is not None:
        shift = (origins[cell.parent][0] - origins[parent][0], origins[cell.parent][1] - origins[parent][1])
    if cell.label is not None:
        row["l"] = cell.label
    if cell.wrapper is not None:
        lossy(1, cell, "user_object", "a user object")
    if cell.style is not None:
        row["s"] = format_style(cell.style)
        if parse_style(row["s"]) != cell.style:
            lossy(1, cell, "style", f"the style tokens {cell.style!r}")
    check_options(cell, lossy)
    if cell.attrs:
        lossy(1, cell, "attributes", "cell attributes")
    if cell.extras:
        lossy(1, cell, "extensions", "cell extensions")
    g = cell.geometry
    if g is not None and (g.attrs or g.extras):
        lossy(1, cell, "geometry", "geometry attributes or extensions")
    if g is not None and g.alternate_bounds is not None:
        lossy(1, cell, "alternate_bounds", "alternate bounds")
    if g is not None and g.offset is not None:
        lossy(1, cell, "offset", "a geometry offset")
    move = (lambda p: p) if shift is None else (lambda p: (p[0] + shift[0], p[1] + shift[1]))
    if vertex:
        if g is None:
            lossy(1, cell, "geometry", "a vertex without geometry")
            g = ir.Geometry()
        if g.relative:
            lossy(1, cell, "relative", "relative vertex geometry")
        for key in ("source_point", "target_point", "points"):
            if getattr(g, key):
                lossy(1, cell, key, f"{key} on a vertex")
        width, height = g.width, g.height
        if width < 0 or height < 0:
            lossy(1, cell, "geometry", "a negative vertex size")
            width, height = max(Decimal(0), width), max(Decimal(0), height)
        x, y = move((g.x, g.y))
        row["g"] = [json_number(v) for v in (x, y, width, height)]
        return row
    if g is None:
        lossy(1, cell, "geometry", "an edge without geometry")
        g = ir.Geometry(relative=True)
    if not g.relative:
        lossy(1, cell, "relative", "edge geometry that is not relative")
    for key in ("x", "y", "width", "height"):
        if getattr(g, key):
            lossy(1, cell, key, f"edge geometry {key}")
    for key, end in (("f", cell.source), ("t", cell.target)):
        if end is not None:
            if end in names:
                row[key] = names[end]
            else:
                lossy(1, cell, "source" if key == "f" else "target", "a terminal JSONL does not hold")
    for key, value in (("sp", g.source_point), ("tp", g.target_point)):
        if value is not None:
            row[key] = [json_number(v) for v in move(value)]
    if g.points:
        row["w"] = [[json_number(v) for v in move(p)] for p in g.points]
    return row


def reread(text):
    """The canonical IR a JSONL text compiles to."""
    decoded = read_ir_jsonl(text)
    normalize(decoded.document, decoded.origins)
    return decoded.document


def canonical(document_or_ir):
    """A normalized copy of a document, an IR JSON object or anything with a ``.document``."""
    if isinstance(document_or_ir, dict):
        document = read_ir_json(document_or_ir).document
    else:
        document = deepcopy(getattr(document_or_ir, "document", document_or_ir))
        if not isinstance(document, ir.Document):
            raise TypeError("expected an ir.Document, a compilation or an IR JSON object")
    origins = [{cell.uid: {"original_id": cell.name} for cell in page.cells.values()} for page in document.pages]
    normalize(document, origins)
    return document


def to_ir_jsonl(document_or_ir, *, lossless=True):
    """Project an IR to canonical compact JSONL. Returns (text, diagnostics).

    With ``lossless`` any construct JSONL cannot hold is an error and raises
    ``CompileError``; otherwise it is a warning and the text is a best effort.
    """
    document = canonical(document_or_ir)
    text, problems = project(document, "error" if lossless else "warning")
    if not problems:
        try:
            same = reread(text).to_json() == document.to_json()
        except CompileError:
            same = False
        if not same:
            problems.append(
                Diagnostic(
                    "error" if lossless else "warning",
                    "jsonl_lossy_projection",
                    "the compact JSONL does not read back as the same IR",
                )
            )
    if lossless and problems:
        raise CompileError(problems)
    return text, problems


def emit_ir_jsonl(document_or_ir, *, lossless=True):
    """:func:`to_ir_jsonl` as UTF-8 bytes. Returns (data, diagnostics)."""
    text, diagnostics = to_ir_jsonl(document_or_ir, lossless=lossless)
    return text.encode("utf-8"), diagnostics
