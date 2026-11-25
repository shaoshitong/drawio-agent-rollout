"""Native entry: decode and verify ``drawio-semantic-ir/2`` JSON into the graph IR.

The decoder accepts what :meth:`ir.Document.to_json` writes and gives it the
meaning the XML path would: the result is the IR that ``build`` reads from
``emit`` of the same document. Facts the IR derives from content are derived
here the same way: ID roles from styles and user objects, anchored cell IDs
and kept page IDs from the ``data:`` links in the document. A stated value
that disagrees with the content is an error, as is anything ``emit`` could not
write back to the same IR.
"""

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import json
import math
import re

from . import ir
from . import syntax
from .build import CELL_FIELDS, FILE_METADATA, GEOMETRY_FIELDS, ID_PLACEHOLDER, VIEW_STATE, escape_text
from .build import fragment, geometry_role
from .emit import emit
from .source import LIMIT, CompileError, Diagnostic, Element, Loc, RuleLog, lex

DOCUMENT_KEYS = frozenset({"ir", "pages", "attributes", "extensions"})
PAGE_KEYS = frozenset({"name", "id", "settings", "cell_ids_are_anchors", "cells", "attributes", "extensions"})
CELL_KEYS = frozenset(
    {
        "id",
        "kind",
        "parent",
        "label",
        "user_object",
        "style",
        "geometry",
        "source",
        "target",
        "connectable",
        "visible",
        "collapsed",
        "attributes",
        "extensions",
        "id_roles",
    }
)
GEOMETRY_KEYS = frozenset(
    {
        "x",
        "y",
        "width",
        "height",
        "relative",
        "offset",
        "source_point",
        "target_point",
        "points",
        "alternate_bounds",
        "attributes",
        "extensions",
    }
)
USER_OBJECT_KEYS = frozenset({"element", "data", "extensions"})
# The XML role of each structured geometry field, as <mxGeometry> children carry it.
GEOMETRY_ROLES = {
    "source_point": "sourcePoint",
    "target_point": "targetPoint",
    "points": "points",
    "offset": "offset",
    "alternate_bounds": "alternateBounds",
}
ACTION_LINK, PAGE_LINK = b"data:action/json", b"data:page/id,"

_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_NAME = re.compile(r"(?:[^\W\d]|:)[\w.:\-·]*\Z")
# Characters XML 1.0 cannot carry, so emitted XML could not hold them.
_NOT_XML = re.compile("[\0-\10\13\14\16-\37" + chr(0xD800) + "-" + chr(0xDFFF) + chr(0xFFFE) + chr(0xFFFF) + "]")


@dataclass
class DecodedIR:
    """A verified, not yet normalized IR with what was recorded while reading it."""

    document: ir.Document
    origins: list  # per page: uid -> origin record, as ``build`` keeps them
    diagnostics: list
    representation: list  # representation changes, as ``build`` reports them
    encoding: str  # the source encoding named in the source map


class Rejected(ValueError):
    """A JSON text uses a construct the strict reader does not accept."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Rejected("json_duplicate_key", f"key {key!r} appears twice in one object")
        result[key] = value
    return result


def _constant(name):
    raise Rejected("invalid_number", f"{name} is not a finite number")


def loads(text):
    """``json.loads`` that keeps decimals exact and rejects duplicate keys, NaN and infinities."""
    return json.loads(text, parse_float=Decimal, parse_constant=_constant, object_pairs_hook=_unique)


def number(value):
    """A finite coordinate as the editor reads its canonical text, or None.

    Accepts JSON numbers and numeric strings, never booleans. The value is
    rounded through ``format_number`` exactly as emitting and re-reading XML
    rounds it, so ``-0`` becomes ``0`` and ``10.0``, ``1e1`` and ``"10"`` agree.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if not _NUMBER.match(value):
            return None
        value = Decimal(value)
    elif isinstance(value, float):
        value = Decimal(repr(value))
    elif isinstance(value, int):
        value = Decimal(value)
    elif not isinstance(value, Decimal):
        return None
    if not value.is_finite() or not math.isfinite(float(value)):
        return None
    return Decimal(syntax.format_number(value))


def valid_text(value):
    return isinstance(value, str) and not _NOT_XML.search(value)


@lru_cache(maxsize=4096)
def _valid_xml_name(value):
    if not value or _NOT_XML.search(value):
        return False
    if _NAME.match(value):
        return True
    try:
        lex(f"<{value}/>".encode("utf-8"))
        return True
    except CompileError:
        return False


def valid_name(value):
    return isinstance(value, str) and _valid_xml_name(value)


def is_list(value):
    return isinstance(value, (list, tuple))


def path_loc(page, cell, path):
    return Loc(page=page, path="/" + "/".join(str(part) for part in path))


class Decoder:
    def __init__(self, locate):
        self.locate = locate
        self.errors, self.diagnostics, self.changes = [], [], RuleLog()
        self.origins = []
        self.page_number = self.cell_index = None

    def note(self, severity, code, message, path, cell_id=None, attribute=None):
        loc = self.locate(self.page_number, self.cell_index, path)
        diagnostics = self.errors if severity == "error" else self.diagnostics
        diagnostics.append(Diagnostic(severity, code, message, loc, cell_id, attribute))

    def error(self, code, message, path, cell_id=None, attribute=None):
        self.note("error", code, message, path, cell_id, attribute)

    def fields(self, record, path, allowed, required=(), cell_id=None):
        """Check an object's keys. Returns False when it is not an object at all."""
        if not isinstance(record, dict):
            self.error("ir_field_type", "expected a JSON object", path, cell_id)
            return False
        for key in sorted(record.keys() - allowed, key=str):
            self.error("ir_field", f"unknown field {key!r}", [*path, key], cell_id, str(key))
        for key in required:
            if record.get(key) is None:
                self.error("ir_field", f"missing field {key!r}", path, cell_id, key)
        return True

    def text(self, record, key, path, cell_id=None, name=False):
        """An optional string field; None when absent or invalid."""
        value = record.get(key)
        if value is None:
            return None
        if not (valid_name(value) if name else valid_text(value)):
            what = "an XML name" if name else "a string of XML characters"
            self.error("ir_field_type", f"{key} must be {what}", [*path, key], cell_id, key)
            return None
        return value

    def flag(self, record, key, default, path, cell_id=None):
        value = record.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            self.error("ir_field_type", f"{key} must be true or false", [*path, key], cell_id, key)
            return default
        return value

    def number(self, value, path, cell_id, attribute):
        result = number(value)
        if result is None:
            self.error("invalid_number", f"{value!r} is not a finite number", path, cell_id, attribute)
            return Decimal(0)
        return result

    def numbers(self, value, size, path, cell_id, attribute):
        if not is_list(value) or len(value) != size:
            self.error("invalid_geometry", f"{attribute} must hold {size} numbers", path, cell_id, attribute)
            return None
        return tuple(self.number(v, [*path, i], cell_id, attribute) for i, v in enumerate(value))

    def pairs(self, value, path, cell_id=None, reserved=frozenset(), dropped=frozenset(), rule=None):
        """Sorted (name, value) attributes. Names in ``dropped`` are left out as ``rule``."""
        if value is None:
            return ()
        if not is_list(value):
            self.error("ir_field_type", "expected a list of [name, value] pairs", path, cell_id)
            return ()
        result, seen, removed = [], set(), []
        for index, pair in enumerate(value):
            where = [*path, index]
            if not is_list(pair) or len(pair) != 2 or not valid_text(pair[1]):
                self.error("ir_field_type", "expected a [name, value] pair of strings", where, cell_id)
                continue
            key = pair[0]
            if not valid_name(key):
                self.error("invalid_name", f"{key!r} is not an XML attribute name", where, cell_id)
                continue
            elif key in seen:
                self.error("duplicate_attribute", f"attribute {key!r} is given twice", where, cell_id, key)
            elif key in reserved:
                self.error(
                    "reserved_attribute", f"{key!r} is a field of the IR, not an attribute", where, cell_id, key
                )
            elif key in dropped:
                removed.append(key)
            else:
                result.append((key, pair[1]))
            seen.add(key)
        if removed:
            self.changes.add(rule, self.page_number, sorted(removed))
        return tuple(sorted(result))

    def extensions(self, value, path, cell_id=None, reject=None, text=False):
        """Canonical XML fragments carried through. ``reject`` names content XML would read otherwise."""
        if value is None:
            return ()
        if not is_list(value):
            self.error("ir_field_type", "extensions must be a list of XML fragments", path, cell_id)
            return ()
        for index, item in enumerate(value):
            where = [*path, index]
            problem = None
            if not valid_text(item):
                problem = "is not a string of XML characters"
            else:
                try:
                    children = [c for c in lex(b"<x>" + item.encode("utf-8") + b"</x>").children]
                except CompileError:
                    children = None
                if children is None or len(children) != 1:
                    problem = "is not one well-formed XML element" + (" or text" if text else "")
                elif isinstance(children[0], Element):
                    if fragment(children[0]) != item:
                        problem = "is not in canonical form " + repr(fragment(children[0]))
                    elif reject is not None:
                        problem = reject(children[0])
                elif not text or not item.strip() or escape_text(children[0]) != item:
                    problem = "is not a canonical text extension" if text else "is not an XML element"
            if problem is not None:
                self.error("ir_extension", f"extension {problem}", where, cell_id)
        return tuple(value)

    def document(self, payload):
        if not self.fields(payload, [], DOCUMENT_KEYS, ("ir", "pages")):
            raise CompileError(self.errors)
        if "ir" in payload and payload["ir"] != ir.IR_VERSION:
            self.error("ir_version", f"expected ir {ir.IR_VERSION!r}, found {payload['ir']!r}", ["ir"])
        pages = payload.get("pages")
        if pages is not None and (not is_list(pages) or not pages):
            self.error("no_pages", "pages must be a non-empty list", ["pages"])
            pages = None
        attrs = self.pairs(
            payload.get("attributes"), ["attributes"], dropped=FILE_METADATA, rule="drop_file_metadata"
        )
        extras = self.extensions(
            payload.get("extensions"),
            ["extensions"],
            reject=lambda e: "would be read as a page" if e.tag == "diagram" else None,
        )
        decoded = [self.page(number, record) for number, record in enumerate(pages or (), 1)]
        self.page_number = self.cell_index = None
        if self.errors:
            raise CompileError(self.errors + self.diagnostics)
        document = ir.Document(attrs, extras, decoded)
        try:
            ir.verify(document, named=True)
        except ir.IRError as error:
            raise CompileError(Diagnostic("error", "ir_invalid", str(error))) from None
        self.links(document, [record.get("cell_ids_are_anchors") for record in pages])
        if self.errors:
            raise CompileError(self.errors + self.diagnostics)
        return document

    def links(self, document, stated):
        """Anchored cell IDs and kept page IDs, decided from the document as ``build`` decides them."""
        text = emit(document)
        pinned, keep_page_ids = ACTION_LINK in text, PAGE_LINK in text
        for number, (page, claim) in enumerate(zip(document.pages, stated, strict=True), 1):
            self.page_number = number
            path = ["pages", number - 1, "cell_ids_are_anchors"]
            if claim is not None and not isinstance(claim, bool):
                self.error("ir_field_type", "cell_ids_are_anchors must be true or false", path)
            elif claim is not None and claim != pinned:
                self.error(
                    "pinned_mismatch",
                    f"cell_ids_are_anchors is {str(claim).lower()} but the document "
                    + ("has" if pinned else "has no")
                    + " custom action links that address cells by ID",
                    path,
                )
            page.pinned = pinned
            if pinned:
                self.note(
                    "info",
                    "cell_ids_are_anchors",
                    "the page has custom action links that address cells by ID; cell IDs keep their names",
                    path,
                )
            if page.anchor is not None and not keep_page_ids:
                page.anchor = None
                self.changes.add("drop_unreferenced_page_id", number)
        self.page_number = None

    def page(self, number, record):
        self.page_number, self.cell_index = number, None
        path = ["pages", number - 1]
        if not self.fields(record, path, PAGE_KEYS, ("cells",)):
            return None
        name = self.text(record, "name", path)
        anchor = self.text(record, "id", path)
        settings = self.pairs(
            record.get("settings"), [*path, "settings"], dropped=VIEW_STATE, rule="drop_view_state"
        )
        attrs = self.pairs(record.get("attributes"), [*path, "attributes"], reserved={"id", "name"})
        extensions = record.get("extensions") or {}
        where = [*path, "extensions"]
        if self.fields(extensions, where, {"page", "model"}):
            extras = self.extensions(
                extensions.get("page"),
                [*where, "page"],
                reject=lambda e: "would be read as the page model" if e.tag == "mxGraphModel" else None,
            )
            model_extras = self.extensions(
                extensions.get("model"),
                [*where, "model"],
                reject=lambda e: "would be read as the cell list" if e.tag == "root" else None,
            )
        else:
            extras = model_extras = ()
        records = record.get("cells")
        if records is not None and not is_list(records):
            self.error("ir_field_type", "cells must be a list", [*path, "cells"])
            records = None
        cells, refs, by_id, origin = {}, {}, {}, {}
        for index, raw in enumerate(records or ()):
            self.cell_index = index
            decoded = self.cell(index, raw, [*path, "cells", index])
            if decoded is None:
                continue
            cell, ref = decoded
            if cell.name in by_id:
                self.error(
                    "duplicate_id",
                    f"cell ID {cell.name!r} is already used on this page",
                    [*path, "cells", index, "id"],
                    cell.name,
                    "id",
                )
                continue
            by_id[cell.name] = index
            cells[index], refs[index] = cell, ref
            origin[index] = {
                "original_id": cell.name,
                "element": cell.wrapper or "mxCell",
                "root_child_index": index + 1,
                "location": self.locate(number, index, [*path, "cells", index]).to_json(),
            }
            if cell.id_roles:
                origin[index]["id_roles"] = list(cell.id_roles)
        self.origins.append(origin)
        root = self.link(cells, refs, by_id, path)
        self.cell_index = None
        observed = {}
        for cell in cells.values():
            for role in cell.id_roles:
                observed[role] = observed.get(role, 0) + 1
        for role, count in sorted(observed.items()):
            self.note(
                "info",
                "cell_id_is_observed",
                f"{count} cell ID(s) on this page are observed by the editor as {role} and keep their names",
                path,
            )
        return ir.Page(name, anchor, attrs, settings, extras, model_extras, cells, root, False)

    def link(self, cells, refs, by_id, path):
        """Resolve parents and terminals into one tree. Returns the root uid."""
        roots = [uid for uid, cell in cells.items() if cell.kind == ir.ROOT]
        parentless = [uid for uid in cells if refs[uid]["parent"] is None]
        for uid in sorted(set(roots[1:]) | (set(parentless) - set(roots[:1]))):
            self.cell_index = uid
            cell = cells[uid]
            message = "a page has one root cell" if cell.kind == ir.ROOT else "only the root has no parent"
            self.error("model_root", message, [*path, "cells", uid], cell.name)
        if not roots:
            self.cell_index = None
            self.error("model_root", "the page has no root cell", [*path, "cells"])
        for uid, cell in cells.items():
            self.cell_index = uid
            where = [*path, "cells", uid]
            ref = refs[uid]["parent"]
            if ref is not None:
                if cell.kind == ir.ROOT:
                    self.error("model_root", "the root cell has no parent", [*where, "parent"], cell.name, "parent")
                elif ref not in by_id:
                    self.error(
                        "unresolved_parent",
                        f"parent {ref!r} is not a cell of this page",
                        [*where, "parent"],
                        cell.name,
                        "parent",
                    )
                elif by_id[ref] == uid:
                    self.error("parent_cycle", "cell is its own parent", [*where, "parent"], cell.name, "parent")
                else:
                    cell.parent = by_id[ref]
                    cells[cell.parent].children.append(uid)
            for end in ("source", "target"):
                ref = refs[uid][end]
                if ref is None:
                    continue
                if cell.kind != ir.EDGE:
                    self.error(
                        "terminal_on_non_edge", f"only edges can have a {end} terminal", [*where, end], cell.name, end
                    )
                elif ref not in by_id:
                    self.error(
                        "dangling_terminal",
                        f"{end} terminal {ref!r} is not a cell of this page",
                        [*where, end],
                        cell.name,
                        end,
                    )
                else:
                    setattr(cell, end, by_id[ref])
        if not roots:
            return -1
        root = roots[0]
        reached, stack = set(), [root]
        while stack:
            uid = stack.pop()
            reached.add(uid)
            stack.extend(cells[uid].children)
        for uid, cell in cells.items():
            self.cell_index = uid
            where = [*path, "cells", uid]
            if uid not in reached:
                if cell.parent is not None:
                    self.error(
                        "parent_cycle",
                        "cell is not connected to the root through its parents",
                        [*where, "parent"],
                        cell.name,
                        "parent",
                    )
                continue
            if cell.kind == ir.ROOT:
                continue
            under_root = cell.parent == root
            if cell.kind == ir.LAYER and not under_root:
                self.error("ir_kind", "a layer must be a child of the root", [*where, "kind"], cell.name, "kind")
            elif cell.kind == ir.FRAME and under_root:
                self.error(
                    "ir_kind", "a frame directly under the root is a layer", [*where, "kind"], cell.name, "kind"
                )
        return root

    def cell(self, uid, record, path):
        """Decode one cell record. Returns (cell, references) or None."""
        cell_id = record.get("id") if isinstance(record, dict) else None
        cell_id = cell_id if isinstance(cell_id, str) else None
        if not self.fields(record, path, CELL_KEYS, ("id", "kind"), cell_id):
            return None
        if "id" in record and (not valid_text(record["id"]) or not record["id"]):
            self.error("ir_field_type", "id must be a non-empty string", [*path, "id"], None, "id")
            return None
        kind = record.get("kind")
        if kind is not None and kind not in ir.KINDS:
            self.error("ir_kind", f"unknown kind {kind!r}", [*path, "kind"], cell_id, "kind")
        if cell_id is None or kind not in ir.KINDS:
            return None
        cell = ir.Cell(uid, kind, name=cell_id)
        refs = {key: self.text(record, key, path, cell_id) for key in ("parent", "source", "target")}
        cell.label = self.text(record, "label", path, cell_id)
        if cell.label == "":
            cell.label = None
            self.changes.add("omit_empty_label", self.page_number)
        self.user_object(cell, record.get("user_object"), [*path, "user_object"])
        self.style(cell, record.get("style"), [*path, "style"])
        if record.get("geometry") is not None:
            cell.geometry = self.geometry(record["geometry"], [*path, "geometry"], cell_id, kind)
        for key, default in ir.CELL_OPTIONS:
            setattr(cell, key, self.flag(record, key, default, path, cell_id))
        cell.attrs = self.pairs(record.get("attributes"), [*path, "attributes"], cell_id, reserved=CELL_FIELDS)
        cell.extras = self.extensions(
            record.get("extensions"),
            [*path, "extensions"],
            cell_id,
            reject=lambda e: (
                "would be read as the cell geometry"
                if e.tag == "mxGeometry" and e.get("as") == "geometry"
                else None
            ),
        )
        roles = []
        if syntax.id_seeds_sketch(cell.style):
            roles.append(ir.SKETCH_SEED)
        if (
            cell.wrapper is not None
            and ("placeholders", "1") in cell.data
            and any(ID_PLACEHOLDER in text for text in (cell.label or "", *(value for _, value in cell.data)))
        ):
            roles.append(ir.PLACEHOLDER)
        cell.id_roles = tuple(roles)
        stated = record.get("id_roles")
        if stated is not None:
            if not is_list(stated) or not all(role in ir.ID_ROLES for role in stated):
                self.error(
                    "invalid_id_roles",
                    f"id_roles must list roles from {list(ir.ID_ROLES)}",
                    [*path, "id_roles"],
                    cell_id,
                    "id_roles",
                )
            elif sorted(set(stated)) != sorted(roles) or len(stated) != len(set(stated)):
                self.error(
                    "id_roles_mismatch",
                    f"id_roles {list(stated)} disagree with the roles the content gives the ID: {roles}",
                    [*path, "id_roles"],
                    cell_id,
                    "id_roles",
                )
        return cell, refs

    def user_object(self, cell, value, path):
        if value is None or not self.fields(value, path, USER_OBJECT_KEYS, ("element",), cell.name):
            return
        element = self.text(value, "element", path, cell.name, name=True)
        if element == "mxCell":
            self.error("ir_field_type", "a user object element cannot be <mxCell>", [*path, "element"], cell.name)
        elif element is not None:
            cell.wrapper = element
        cell.data = self.pairs(value.get("data"), [*path, "data"], cell.name, reserved={"id", "label"})
        cell.wrapper_extras = self.extensions(
            value.get("extensions"),
            [*path, "extensions"],
            cell.name,
            reject=lambda e: "would be read as the cell of the user object" if e.tag == "mxCell" else None,
            text=True,
        )

    def style(self, cell, value, path):
        if value is None:
            return
        if not is_list(value) or not all(
            is_list(token)
            and len(token) == 2
            and (token[0] is None or valid_text(token[0]))
            and valid_text(token[1])
            for token in value
        ):
            self.error("ir_field_type", "style must be a list of [key or null, value] pairs", path, cell.name)
            return
        tokens = tuple((key, text) for key, text in value)
        if syntax.parse_style(syntax.format_style(tokens)) != tokens:
            self.error(
                "invalid_style",
                f"style tokens do not survive the style syntax: {syntax.format_style(tokens)!r}",
                path,
                cell.name,
                "style",
            )
            return
        if not tokens:
            self.changes.add("omit_empty_style", self.page_number)
            return
        cell.style = tokens
        for key, text in tokens:
            if key in syntax.COLOR_KEYS and not syntax.valid_color(text):
                self.note(
                    "warning",
                    "invalid_color",
                    f"{key}={text!r} is not a colour the editor can paint",
                    path,
                    cell.name,
                    "style",
                )

    def geometry(self, value, path, cell_id, kind):
        geometry = ir.Geometry()
        if not self.fields(value, path, GEOMETRY_KEYS, (), cell_id):
            return geometry
        for key in ("x", "y", "width", "height"):
            if value.get(key) is not None:
                setattr(geometry, key, self.number(value[key], [*path, key], cell_id, key))
        if kind == ir.VERTEX and (geometry.width < 0 or geometry.height < 0):
            self.note("warning", "negative_size", "vertex has a negative width or height", path, cell_id)
        geometry.relative = self.flag(value, "relative", False, path, cell_id)
        for key, size in (("offset", 2), ("source_point", 2), ("target_point", 2), ("alternate_bounds", 4)):
            if value.get(key) is not None:
                setattr(geometry, key, self.numbers(value[key], size, [*path, key], cell_id, key))
        points = value.get("points")
        if points is not None:
            if is_list(points):
                decoded = [self.numbers(p, 2, [*path, "points", i], cell_id, "points") for i, p in enumerate(points)]
                geometry.points = tuple(p for p in decoded if p is not None)
            else:
                self.error("invalid_geometry", "points must be a list of [x, y]", [*path, "points"], cell_id)
        geometry.attrs = self.pairs(value.get("attributes"), [*path, "attributes"], cell_id, reserved=GEOMETRY_FIELDS)
        taken = {role for key, role in GEOMETRY_ROLES.items() if getattr(geometry, key)}

        def reject(element):
            if geometry_role(element) is not None:
                return f"would be read as geometry {geometry_role(element)}; give it as that field"
            if element.get("as") in taken:
                return f"repeats the geometry role {element.get('as')!r}"
            return None

        geometry.extras = self.extensions(value.get("extensions"), [*path, "extensions"], cell_id, reject=reject)
        return geometry


def decode(payload, *, encoding="ir-json", locate=None):
    """Decode an IR/2 object. ``locate(page, cell_index, path)`` places diagnostics."""
    decoder = Decoder(locate or path_loc)
    document = decoder.document(payload)
    return DecodedIR(
        document, decoder.origins, decoder.diagnostics, list(decoder.changes.entries.values()), encoding
    )


def read_ir_json(data):
    """Decode ``drawio-semantic-ir/2`` from UTF-8 bytes, text or an already parsed object."""
    if isinstance(data, (bytes, bytearray)):
        if len(data) > LIMIT:
            raise CompileError(Diagnostic("error", "input_too_large", f"IR JSON exceeds {LIMIT} bytes"))
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as error:
            raise CompileError(
                Diagnostic("error", "ir_encoding", f"IR JSON is not UTF-8: {error.reason}", Loc(byte=error.start))
            ) from None
    if isinstance(data, str):
        if len(data.encode("utf-8")) > LIMIT:
            raise CompileError(Diagnostic("error", "input_too_large", f"IR JSON exceeds {LIMIT} bytes"))
        try:
            data = loads(data)
        except json.JSONDecodeError as error:
            loc = Loc(error.lineno, error.colno, len(error.doc[: error.pos].encode("utf-8")))
            raise CompileError(Diagnostic("error", "json_syntax", error.msg, loc)) from None
        except Rejected as error:
            raise CompileError(Diagnostic("error", error.code, str(error))) from None
        except (ValueError, RecursionError) as error:
            raise CompileError(Diagnostic("error", "json_syntax", str(error))) from None
    return decode(data)
