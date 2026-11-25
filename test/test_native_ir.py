"""Native IR entry: IR/2 JSON, compact JSONL, adapters and the CLI."""

from decimal import Decimal
import json
from pathlib import Path

import pytest

from drawio.compiler import (
    AdaptedIR,
    CompileError,
    IRCompilation,
    adapt_ir,
    compile_bytes,
    compile_ir,
    compile_ir_json,
    compile_ir_jsonl,
    emit_ir_jsonl,
    read_ir_json,
    read_ir_jsonl,
    register_adapter,
    to_ir_jsonl,
)
from drawio.compiler import ir
from drawio.compiler.cli import main
from drawio.compiler.emit import emit
from drawio.compiler.syntax import parse_style


def mxfile(*cells, model='pageWidth="800" pageHeight="600"', name="Page 1", extra=""):
    body = '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells)
    return (
        f'<mxfile><diagram name="{name}"><mxGraphModel {model}><root>{body}</root>'
        f"</mxGraphModel></diagram>{extra}</mxfile>"
    ).encode()


def vertex(cid, x, y, w=80, h=40, parent="1", label=None, style=None, extra=""):
    value = f' value="{label}"' if label is not None else ""
    style = f' style="{style}"' if style is not None else ""
    return (
        f'<mxCell id="{cid}"{value}{style} vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>{extra}</mxCell>'
    )


def edge(cid, source=None, target=None, parent="1", style=None, body=""):
    ends = "".join(f' {k}="{v}"' for k, v in (("source", source), ("target", target)) if v is not None)
    style = f' style="{style}"' if style is not None else ""
    return (
        f'<mxCell id="{cid}"{style} edge="1" parent="{parent}"{ends}>'
        f'<mxGeometry relative="1" as="geometry">{body}</mxGeometry></mxCell>'
    )


SIMPLE = mxfile(
    vertex("a", 20, 20, 110, 45, label="Input", style="rounded=1;fillColor=#dae8fc"),
    vertex("b", 170, 20, 100, 45, label="Output"),
    edge("e1", "a", "b", style="endArrow=block", body='<Array as="points"><mxPoint x="160" y="42.5"/></Array>'),
    edge("e2", "b", None, body='<mxPoint x="400" y="300" as="targetPoint"/>'),
)

RICH = (
    b'<mxfile host="x" modified="y" compressed="false">'
    b'<diagram id="p1" name="Main"><mxGraphModel dx="10" dy="20" grid="1" gridSize="10" pageWidth="850">'
    b'<root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="L2" value="Back" parent="0" visible="0"/>'
    b'<mxCell id="g" value="Group" style="swimlane;startSize=20" vertex="1" parent="1" collapsed="1">'
    b'<mxGeometry x="100" y="100" width="300" height="200" as="geometry">'
    b'<mxRectangle x="100" y="100" width="120" height="20" as="alternateBounds"/></mxGeometry></mxCell>'
    b'<mxCell id="f" parent="g"><mxGeometry x="15" y="25" as="geometry"/></mxCell>'
    b'<mxCell id="in" value="Inner" vertex="1" parent="f" connectable="0">'
    b'<mxGeometry x="10" y="10" width="60" height="30" as="geometry"/></mxCell>'
    b'<mxCell id="port" style="ellipse" vertex="1" parent="in">'
    b'<mxGeometry x="1" y="0.5" width="10" height="10" relative="1" as="geometry">'
    b'<mxPoint x="-5" y="-5" as="offset"/></mxGeometry></mxCell>'
    b'<object label="Obj %id%" placeholders="1" tooltip="t" id="obj">'
    b'<mxCell style="sketch=1;curveFitting=1" vertex="1" parent="L2" custom="k">'
    b'<mxGeometry x="500" y="50" width="80" height="80" as="geometry" foo="bar"/>'
    b'<extra a="1"/></mxCell></object>'
    b'<mxCell id="e" value="label" edge="1" parent="1" source="port" target="obj">'
    b'<mxGeometry x="-0.2" y="10" relative="1" as="geometry"><mxPoint x="1" y="2" as="sourcePoint"/>'
    b'<Array as="points"><mxPoint x="300" y="400"/><mxPoint x="320" y="420"/></Array>'
    b'<mxPoint x="5" y="5" as="offset"/></mxGeometry></mxCell>'
    b'<mxCell id="lbl" value="edge label" style="edgeLabel" vertex="1" connectable="0" parent="e">'
    b'<mxGeometry x="-0.5" relative="1" as="geometry"><mxPoint as="offset"/></mxGeometry></mxCell>'
    b"</root><extension x='1'/></mxGraphModel><pageExtra/></diagram>"
    b'<diagram name="Second"><mxGraphModel><root><mxCell id="r"/><mxCell id="l" parent="r"/>'
    b'<mxCell id="v" value="two" vertex="1" parent="l"><mxGeometry x="1.5" y="2.25" width="3" height="4" '
    b'as="geometry"/></mxCell></root></mxGraphModel></diagram><fileExtra/></mxfile>'
)

PINNED = mxfile(
    vertex("first", 10, 10, label="go", extra=""),
    vertex("second", 100, 10),
    '<UserObject label="link" link="data:action/json,{&quot;actions&quot;:[{&quot;toggle&quot;:'
    '{&quot;cells&quot;:[&quot;second&quot;]}}]}" id="btn"><mxCell vertex="1" parent="1">'
    '<mxGeometry x="10" y="100" width="60" height="30" as="geometry"/></mxCell></UserObject>',
)


def jsonl(*rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


GRAPH = jsonl(
    {"canvas": [800, 600]},
    {"id": "a", "k": "v", "g": [20, 20, 110, 45], "l": "Input", "s": "rounded=1;fillColor=#dae8fc"},
    {"id": "b", "k": "v", "g": [170, 20, 100, 45], "l": "Output"},
    {"id": "e1", "k": "e", "f": "a", "t": "b", "s": "endArrow=block", "w": [[160, 42.5]]},
    {"end": True},
)


def codes(error):
    return [d.code for d in error.value.diagnostics if d.severity == "error"]


def reject(text, **options):
    with pytest.raises(CompileError) as error:
        read_ir_jsonl(text, **options)
    return error


# Reference protocol (meanflow_two_stage.read_ir_lines / compile_visual_ir), kept as the parity oracle.
def reference_read_ir_lines(text):
    lines = text.splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    width, height = records[0]["canvas"]
    cells = [{"id": "0", "kind": "root"}, {"id": "1", "kind": "layer", "parent": "0"}]
    for row in records[1:-1]:
        is_vertex = row["k"] == "v"
        cell = {"id": str(row["id"]), "kind": "vertex" if is_vertex else "edge", "parent": str(row.get("p", "1"))}
        if row.get("l"):
            cell["label"] = row["l"]
        if row.get("s"):
            cell["style"] = [list(token) for token in parse_style(row["s"])]
        if is_vertex:
            cell["geometry"] = dict(zip(("x", "y", "width", "height"), row["g"], strict=True))
        else:
            for source, target in (("f", "source"), ("t", "target")):
                if row.get(source) is not None:
                    cell[target] = str(row[source])
            cell["geometry"] = {"relative": True}
            for source, target in (("sp", "source_point"), ("tp", "target_point"), ("w", "points")):
                if row.get(source) is not None:
                    cell["geometry"][target] = row[source]
        cells.append(cell)
    settings = [["pageWidth", str(width)], ["pageHeight", str(height)]]
    return {"ir": ir.IR_VERSION, "pages": [{"name": "Page 1", "settings": settings, "cells": cells}]}


def reference_compile_visual_ir(payload):
    def point(value):
        return tuple(Decimal(str(n)) for n in value)

    pages = []
    for raw in payload["pages"]:
        records = raw["cells"]
        uids = {str(record["id"]): uid for uid, record in enumerate(records)}
        cells = {}
        for uid, record in enumerate(records):
            coords = record.get("geometry")
            geometry = None
            if coords is not None:
                numbers = {k: Decimal(str(coords.get(k, 0))) for k in ("x", "y", "width", "height")}
                ends = {k: point(coords[k]) for k in ("source_point", "target_point", "offset") if k in coords}
                geometry = ir.Geometry(
                    **numbers,
                    **ends,
                    relative=bool(coords.get("relative", False)),
                    points=tuple(point(p) for p in coords.get("points", [])),
                )
            style = tuple(tuple(item) for item in record.get("style", [])) or None
            refs = {k: uids[str(record[k])] for k in ("parent", "source", "target") if record.get(k) is not None}
            cells[uid] = ir.Cell(
                uid=uid,
                kind=record["kind"],
                name=str(record["id"]),
                style=style,
                label=record.get("label") or None,
                geometry=geometry,
                **refs,
            )
        roots = []
        for cell in cells.values():
            if cell.parent is None:
                roots.append(cell.uid)
            else:
                cells[cell.parent].children.append(cell.uid)
        settings = tuple((name, str(value)) for name, value in raw.get("settings", []))
        pages.append(ir.Page(raw.get("name"), None, (), settings, (), (), cells, roots[0]))
    document = ir.Document((), (), pages)
    ir.verify(document, named=True)
    return compile_bytes(emit(document))


# Compact JSONL


def test_jsonl_compiles_like_the_reference_protocol():
    result = compile_ir_jsonl(GRAPH)
    reference = reference_compile_visual_ir(reference_read_ir_lines(GRAPH))
    assert isinstance(result, IRCompilation)
    assert result.ir_json == reference.ir_json
    assert result.xml == reference.xml
    assert result.ir_jsonl == to_ir_jsonl(reference)[0]
    assert all(result.report["checks"].values())
    assert {"emitted_xml_rebuilds_ir", "emitted_jsonl_rebuilds_ir", "jsonl_byte_stable"} <= result.report["checks"].keys()


def test_jsonl_reference_parity_with_containers_and_free_ends():
    text = jsonl(
        {"canvas": [640, 480]},
        {"id": "box", "k": "v", "g": [10, 10, 300, 200], "l": "Box", "s": "swimlane"},
        {"id": "x", "k": "v", "p": "box", "g": [20, 40, 50.5, 30], "l": "X"},
        {"id": "y", "k": "v", "p": "box", "g": [120, 40, 50, 30]},
        {"id": "loose", "k": "e", "sp": [0, 0], "tp": [100, 100]},
        {"id": "inner", "k": "e", "p": "box", "f": "x", "t": "y", "w": [[90, 55]], "l": "go"},
        {"id": "half", "k": "e", "f": "y", "tp": [600, 400]},
        {"end": True},
    )
    result = compile_ir_jsonl(text)
    reference = reference_compile_visual_ir(reference_read_ir_lines(text))
    assert result.ir_json == reference.ir_json
    assert result.xml == reference.xml


def test_jsonl_canonical_output_matches_the_training_format():
    result = compile_ir_jsonl(GRAPH)
    rows = [json.loads(line) for line in result.ir_jsonl.splitlines()]
    assert rows[0] == {"canvas": [800, 600]}
    assert rows[-1] == {"end": True}
    assert rows[1] == {"id": "2", "k": "v", "l": "Input", "s": "fillColor=#dae8fc;rounded=1", "g": [20, 20, 110, 45]}
    assert rows[3] == {"id": "4", "k": "e", "s": "endArrow=block", "f": "2", "t": "3", "w": [[160, 42.5]]}
    assert isinstance(rows[3]["w"][0][1], float) and isinstance(rows[1]["g"][0], int)
    data, diagnostics = emit_ir_jsonl(result)
    assert data == result.ir_jsonl.encode() and diagnostics == []


def test_jsonl_fixed_points():
    first = compile_ir_jsonl(GRAPH)
    second = compile_ir_jsonl(first.ir_jsonl)
    assert second.ir_jsonl == first.ir_jsonl
    assert second.ir_json == first.ir_json and second.xml == first.xml
    via_xml = compile_bytes(first.xml)
    assert via_xml.ir_json == first.ir_json and via_xml.xml == first.xml
    assert to_ir_jsonl(via_xml)[0] == first.ir_jsonl


def test_jsonl_line_locations_and_source_map():
    text = '\n{"canvas":[10,10]}\n\n{"id":"a","k":"v","g":[0,0,1,1]}\n{"end":true}\n'
    decoded = read_ir_jsonl(text)
    result = compile_ir_jsonl(text)
    cells = result.source_map["pages"][0]["cells"]
    assert cells[2]["original_id"] == "a" and cells[2]["location"]["line"] == 4
    assert cells[2]["location"]["byte"] == text.encode().index(b'{"id"')
    assert result.source_map["pages"][0]["source_encoding"] == "ir-jsonl"
    assert decoded.document.pages[0].name == "Page 1"


@pytest.mark.parametrize("language", ["jsonl", "ndjson", "json", "JSONL", ""])
def test_fenced_jsonl_keeps_original_line_numbers(language):
    fenced = f"```{language}\n" + GRAPH + "```\n"
    assert compile_ir_jsonl(fenced).ir_json == compile_ir_jsonl(GRAPH).ir_json
    broken = f"\n```{language}\n" + GRAPH.replace('"t": "b"', '"t": "zz"') + "```"
    error = reject(broken)
    (diagnostic,) = error.value.diagnostics
    assert diagnostic.code == "dangling_terminal" and diagnostic.loc.line == 6
    assert diagnostic.cell_id == "e1" and diagnostic.attribute == "target"


def test_fences_can_be_refused():
    error = reject("```jsonl\n" + GRAPH + "```\n", allow_fences=False)
    assert codes(error)[0] == "jsonl_json"
    assert error.value.diagnostics[0].loc.line == 1


def test_unclosed_fence():
    assert "jsonl_fence" in codes(reject("```jsonl\n" + GRAPH))
    decoded = read_ir_jsonl("```jsonl\n" + GRAPH, allow_incomplete=True)
    assert [d.code for d in decoded.diagnostics if d.severity == "warning"] == ["jsonl_fence"]


def test_missing_end_marker():
    text = GRAPH.replace('{"end": true}\n', "")
    error = reject(text)
    assert codes(error) == ["jsonl_end_missing"]
    assert error.value.diagnostics[0].loc.line == 4
    decoded = read_ir_jsonl(text, allow_incomplete=True)
    assert [d.code for d in decoded.diagnostics if d.severity == "warning"] == ["jsonl_end_missing"]
    assert len(decoded.document.pages[0].cells) == 5


def test_truncated_last_line_is_tolerated_only_when_incomplete_is_allowed():
    text = GRAPH.replace('{"end": true}\n', '{"id": "c", "k": "v", "g": [1, 2')
    assert codes(reject(text)) == ["jsonl_json", "jsonl_end_missing"]
    decoded = read_ir_jsonl(text, allow_incomplete=True)
    assert {d.code for d in decoded.diagnostics} >= {"jsonl_truncated", "jsonl_end_missing"}


def test_records_after_end():
    error = reject(GRAPH + '{"id":"late","k":"v","g":[0,0,1,1]}\n')
    assert codes(error) == ["jsonl_end_not_last"]
    assert error.value.diagnostics[0].loc.line == 6


@pytest.mark.parametrize(
    "canvas, line",
    [
        ('{"canvas":[800]}', 1),
        ('{"canvas":[0,600]}', 1),
        ('{"canvas":[-1,600]}', 1),
        ('{"canvas":[800,"wide"]}', 1),
        ('{"canvas":[true,600]}', 1),
        ('{"canvas":[1e999,600]}', 1),
        ('{"canvas":[800,600],"extra":1}', 1),
        ('{"canvas":800}', 1),
    ],
)
def test_bad_canvas(canvas, line):
    error = reject(canvas + "\n" + GRAPH.split("\n", 1)[1])
    assert codes(error) == ["jsonl_canvas"]
    assert error.value.diagnostics[0].loc.line == line


def test_missing_and_duplicate_canvas():
    body = GRAPH.split("\n", 1)[1]
    error = reject(body)
    assert codes(error)[0] == "jsonl_canvas" and error.value.diagnostics[0].loc.line == 1
    error = reject('{"canvas":[1,1]}\n{"canvas":[2,2]}\n' + body)
    assert codes(error) == ["jsonl_canvas"] and error.value.diagnostics[0].loc.line == 2


def test_canvas_must_be_first():
    rows = GRAPH.splitlines()
    error = reject("\n".join([rows[1], rows[0], *rows[2:]]))
    assert codes(error) == ["jsonl_canvas", "jsonl_canvas"]


@pytest.mark.parametrize(
    "line, code",
    [
        ('{"id":"c","k":"v","g":[0,0,1,1]', "jsonl_json"),
        ("[1,2]", "jsonl_record_type"),
        ('"text"', "jsonl_record_type"),
        ('{"id":"c","id":"d","k":"v","g":[0,0,1,1]}', "jsonl_duplicate_key"),
        ('{"id":"c","k":"x","g":[0,0,1,1]}', "jsonl_kind"),
        ('{"id":"c","g":[0,0,1,1]}', "jsonl_missing_field"),
        ('{"id":"c","k":"v"}', "jsonl_missing_field"),
        ('{"k":"v","g":[0,0,1,1]}', "jsonl_missing_field"),
        ('{"id":"c","k":"v","g":[0,0,1,1],"z":1}', "jsonl_unknown_field"),
        ('{"id":"c","k":"v","g":[0,0,1,1],"f":"a"}', "jsonl_unknown_field"),
        ('{"id":"c","k":"e","g":[0,0,1,1]}', "jsonl_unknown_field"),
        ('{"id":7,"k":"v","g":[0,0,1,1]}', "jsonl_field_type"),
        ('{"id":"c","k":"v","g":[0,0,1,1],"l":5}', "jsonl_field_type"),
        ('{"id":"c","k":"v","g":[0,0,1,1],"p":null}', "jsonl_field_type"),
        ('{"id":"0","k":"v","g":[0,0,1,1]}', "jsonl_reserved_id"),
        ('{"id":"1","k":"e"}', "jsonl_reserved_id"),
        ('{"id":"a","k":"v","g":[0,0,1,1]}', "duplicate_id"),
        ('{"id":"c","k":"v","p":"nowhere","g":[0,0,1,1]}', "unresolved_parent"),
        ('{"id":"c","k":"e","f":"nowhere"}', "dangling_terminal"),
        ('{"id":"c","k":"e","t":"nowhere"}', "dangling_terminal"),
        ('{"id":"c","k":"v","p":"c","g":[0,0,1,1]}', "parent_cycle"),
        ('{"id":"c","k":"v","g":[0,0,NaN,1]}', "invalid_number"),
        ('{"id":"c","k":"v","g":[0,0,Infinity,1]}', "invalid_number"),
        ('{"id":"c","k":"v","g":[0,0,1e400,1]}', "invalid_number"),
        ('{"id":"c","k":"v","g":[0,true,1,1]}', "invalid_number"),
        ('{"id":"c","k":"v","g":[0,"x",1,1]}', "invalid_number"),
        ('{"id":"c","k":"v","g":[0,null,1,1]}', "invalid_number"),
        ('{"id":"c","k":"v","g":[0,0,1]}', "invalid_geometry"),
        ('{"id":"c","k":"v","g":"0,0,1,1"}', "invalid_geometry"),
        ('{"id":"c","k":"e","sp":[0]}', "invalid_geometry"),
        ('{"id":"c","k":"e","tp":[0,1,2]}', "invalid_geometry"),
        ('{"id":"c","k":"e","w":[[0,1],[2]]}', "invalid_geometry"),
        ('{"id":"c","k":"e","w":[0,1]}', "invalid_geometry"),
        ('{"id":"c","k":"e","sp":[0,false]}', "invalid_number"),
        ('{"id":"c","k":"v","g":[0,0,-1,1]}', "negative_size"),
        ('{"id":"c","k":"v","g":[0,0,1,-0.5]}', "negative_size"),
        ('{"id":"c","k":"v","g":[0,0,1,1],"s":"fillColor=#fff;=;x"}', None),
    ],
)
def test_record_errors_name_their_line(line, code):
    rows = GRAPH.splitlines()
    text = "\n".join([*rows[:3], line, *rows[3:]]) + "\n"
    if code is None:
        read_ir_jsonl(text)
        return
    error = reject(text)
    assert set(codes(error)) == {code}
    assert error.value.diagnostics[0].loc.line == 4
    assert error.value.diagnostics[0].loc.page == 1


def test_parent_cycle_across_records():
    error = reject(
        jsonl(
            {"canvas": [10, 10]},
            {"id": "a", "k": "v", "p": "b", "g": [0, 0, 1, 1]},
            {"id": "b", "k": "v", "p": "a", "g": [0, 0, 1, 1]},
            {"end": True},
        )
    )
    assert codes(error) == ["parent_cycle", "parent_cycle"]
    assert [d.loc.line for d in error.value.diagnostics] == [2, 3]


def test_several_errors_are_reported_together():
    error = reject(
        jsonl(
            {"canvas": [10, 10]},
            {"id": "0", "k": "v", "g": [0, 0, 1, 1]},
            {"id": "b", "k": "q"},
        )
    )
    assert codes(error) == ["jsonl_reserved_id", "jsonl_kind", "jsonl_end_missing"]


def test_edges_without_any_end_are_valid_draw_io():
    result = compile_ir_jsonl(jsonl({"canvas": [10, 10]}, {"id": "e", "k": "e"}, {"end": True}))
    assert result.ir_json["pages"][0]["cells"][2] == {"id": "2", "kind": "edge", "parent": "1", "geometry": {"relative": True}}


def test_empty_label_and_style_are_absent():
    result = compile_ir_jsonl(jsonl({"canvas": [10, 10]}, {"id": "a", "k": "v", "g": [0, 0, 1, 1], "l": "", "s": ""}, {"end": True}))
    cell = result.ir_json["pages"][0]["cells"][2]
    assert "label" not in cell and "style" not in cell
    rules = {entry["rule"] for entry in result.report["representation_changes"]}
    assert {"omit_empty_label", "omit_empty_style"} <= rules


def test_jsonl_accepts_bytes_and_rejects_bad_utf8():
    assert compile_ir_jsonl(GRAPH.encode()).xml == compile_ir_jsonl(GRAPH).xml
    with pytest.raises(CompileError) as error:
        read_ir_jsonl(b'{"canvas":[1,1]}\n\xff\n')
    assert codes(error) == ["jsonl_encoding"]


# Versioned IR JSON


@pytest.mark.parametrize("name", ["SIMPLE", "RICH", "PINNED"])
def test_ir_json_round_trip_is_byte_identical(name):
    data = globals()[name]
    reference = compile_bytes(data)
    for source in (reference.ir_json, json.dumps(reference.ir_json), json.dumps(reference.ir_json).encode()):
        result = compile_ir_json(source)
        assert result.ir_json == reference.ir_json
        assert result.xml == reference.xml
    assert compile_bytes(result.xml).xml == reference.xml
    assert all(result.report["checks"].values())


def test_rich_document_keeps_editing_semantics():
    result = compile_ir_json(compile_bytes(RICH).ir_json)
    page = result.ir_json["pages"][0]
    cells = {cell["id"]: cell for cell in page["cells"]}
    kinds = [cell["kind"] for cell in page["cells"]]
    assert len(result.ir_json["pages"]) == 2 and kinds.count("layer") == 2
    assert "frame" not in kinds  # the transparent frame folds into the group
    assert page["id"] is None  # nothing links to the page ID
    assert dict(page["settings"]) == {"grid": "1", "gridSize": "10", "pageWidth": "850"}
    assert page["extensions"] == {"page": ["<pageExtra/>"], "model": ['<extension x="1"/>']}
    assert result.ir_json["extensions"] == ["<fileExtra/>"]
    sketch = cells["obj"]
    assert sketch["id_roles"] == ["sketch_seed", "placeholder"]
    assert sketch["user_object"]["data"] == [["placeholders", "1"], ["tooltip", "t"]]
    assert sketch["attributes"] == [["custom", "k"]] and sketch["extensions"] == ['<extra a="1"/>']
    assert sketch["geometry"]["attributes"] == [["foo", "bar"]]
    group = next(c for c in page["cells"] if c.get("label") == "Group")
    assert group["collapsed"] is True and group["geometry"]["alternate_bounds"] == ["100", "100", "120", "20"]
    inner = next(c for c in page["cells"] if c.get("label") == "Inner")
    assert inner["connectable"] is False and inner["geometry"]["x"] == "25" and inner["geometry"]["y"] == "35"
    port = next(c for c in page["cells"] if c.get("style") == [[None, "ellipse"]])
    assert port["geometry"]["relative"] is True and port["geometry"]["offset"] == ["-5", "-5"]
    edge_cell = next(c for c in page["cells"] if c["kind"] == "edge")
    assert edge_cell["source"] == port["id"] and edge_cell["target"] == "obj"
    assert edge_cell["geometry"]["points"] == [["300", "400"], ["320", "420"]]
    label = next(c for c in page["cells"] if c.get("label") == "edge label")
    assert label["parent"] == edge_cell["id"] and "offset" not in label["geometry"]
    assert result.ir_json["pages"][1]["cells"][0]["id"] == "0"


def test_page_id_kept_when_linked():
    data = RICH.replace(b'value="label"', b'value="see data:page/id,p1"')
    reference = compile_bytes(data)
    assert reference.ir_json["pages"][0]["id"] == "p1"
    result = compile_ir_json(reference.ir_json)
    assert result.xml == reference.xml


def test_unlinked_page_id_is_dropped_as_build_drops_it():
    payload = compile_bytes(SIMPLE).ir_json
    payload["pages"][0]["id"] = "stale"
    result = compile_ir_json(payload)
    assert result.ir_json["pages"][0]["id"] is None
    assert "drop_unreferenced_page_id" in {e["rule"] for e in result.report["representation_changes"]}


def test_pinned_ids_are_detected_and_validated():
    reference = compile_bytes(PINNED)
    assert reference.ir_json["pages"][0]["cell_ids_are_anchors"] is True
    names = [cell["id"] for cell in reference.ir_json["pages"][0]["cells"]]
    assert names == ["0", "1", "first", "second", "btn"]
    payload = json.loads(json.dumps(reference.ir_json))
    del payload["pages"][0]["cell_ids_are_anchors"]
    assert compile_ir_json(payload).xml == reference.xml
    payload["pages"][0]["cell_ids_are_anchors"] = False
    with pytest.raises(CompileError) as error:
        compile_ir_json(payload)
    assert codes(error) == ["pinned_mismatch"]
    simple = compile_bytes(SIMPLE).ir_json
    simple["pages"][0]["cell_ids_are_anchors"] = True
    with pytest.raises(CompileError) as error:
        compile_ir_json(simple)
    assert codes(error) == ["pinned_mismatch"]


def test_id_roles_are_derived_and_validated():
    payload = compile_bytes(RICH).ir_json
    obj = next(c for c in payload["pages"][0]["cells"] if c["id"] == "obj")
    del obj["id_roles"]
    assert compile_ir_json(payload).xml == compile_bytes(RICH).xml
    obj["id_roles"] = ["sketch_seed"]
    with pytest.raises(CompileError) as error:
        compile_ir_json(payload)
    assert codes(error) == ["id_roles_mismatch"]
    obj["id_roles"] = ["sketchy"]
    with pytest.raises(CompileError) as error:
        compile_ir_json(payload)
    assert codes(error) == ["invalid_id_roles"]


def base_payload(*cells):
    return {
        "ir": ir.IR_VERSION,
        "pages": [
            {
                "name": "Page 1",
                "settings": [["pageWidth", "100"], ["pageHeight", "100"]],
                "cells": [{"id": "0", "kind": "root"}, {"id": "1", "kind": "layer", "parent": "0"}, *cells],
            }
        ],
    }


def box(cid="a", **extra):
    return {"id": cid, "kind": "vertex", "parent": "1", "geometry": {"x": 0, "y": 0, "width": 10, "height": 10}, **extra}


def ir_error(payload):
    with pytest.raises(CompileError) as error:
        compile_ir_json(payload)
    return error


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda p: p.update(ir="drawio-semantic-ir/1"), "ir_version"),
        (lambda p: p.update(pages=[]), "no_pages"),
        (lambda p: p.update(extra=1), "ir_field"),
        (lambda p: p["pages"][0].pop("cells"), "ir_field"),
        (lambda p: p["pages"][0]["cells"].append(box("a")), "duplicate_id"),
        (lambda p: p["pages"][0]["cells"].append({"id": "r2", "kind": "root"}), "model_root"),
        (lambda p: p["pages"][0]["cells"].pop(0), "model_root"),
        (lambda p: p["pages"][0]["cells"].append(box("b", parent="zz")), "unresolved_parent"),
        (lambda p: p["pages"][0]["cells"].append({"id": "e", "kind": "edge", "parent": "1", "source": "zz"}), "dangling_terminal"),
        (lambda p: p["pages"][0]["cells"].append(box("b", source="a")), "terminal_on_non_edge"),
        (lambda p: p["pages"][0]["cells"].extend([box("b", parent="c"), box("c", parent="b")]), "parent_cycle"),
        (lambda p: p["pages"][0]["cells"].append(box("b", kind="blob")), "ir_kind"),
        (lambda p: p["pages"][0]["cells"].append({"id": "L", "kind": "layer", "parent": "a"}), "ir_kind"),
        (lambda p: p["pages"][0]["cells"].append({"id": "F", "kind": "frame", "parent": "0"}), "ir_kind"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(x="NaN"), "invalid_number"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(x=float("inf")), "invalid_number"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(x=True), "invalid_number"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(x="1e400"), "invalid_number"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(points=[[1]]), "invalid_geometry"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(offset=[1, 2, 3]), "invalid_geometry"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(depth=1), "ir_field"),
        (lambda p: p["pages"][0]["cells"][2].update(style="rounded=1"), "ir_field_type"),
        (lambda p: p["pages"][0]["cells"][2].update(style=[["a;b", "1"]]), "invalid_style"),
        (lambda p: p["pages"][0]["cells"][2].update(style=[["a", "1"], [None, ""]]), "invalid_style"),
        (lambda p: p["pages"][0]["cells"][2].update(label="bad\x01"), "ir_field_type"),
        (lambda p: p["pages"][0]["cells"][2].update(visible="no"), "ir_field_type"),
        (lambda p: p["pages"][0]["cells"][2].update(attributes=[["parent", "x"]]), "reserved_attribute"),
        (lambda p: p["pages"][0]["cells"][2].update(attributes=[["k", "1"], ["k", "2"]]), "duplicate_attribute"),
        (lambda p: p["pages"][0]["cells"][2].update(attributes=[["1bad", "x"]]), "invalid_name"),
        (lambda p: p["pages"][0]["cells"][2].update(extensions=["<a>"]), "ir_extension"),
        (lambda p: p["pages"][0]["cells"][2].update(extensions=['<a  b="1" />']), "ir_extension"),
        (lambda p: p["pages"][0]["cells"][2].update(extensions=['<mxGeometry as="geometry"/>']), "ir_extension"),
        (lambda p: p["pages"][0]["cells"][2]["geometry"].update(extensions=['<mxPoint x="1" as="offset"/>']), "ir_extension"),
        (lambda p: p["pages"][0]["cells"][2].update(user_object={"element": "mxCell"}), "ir_field_type"),
        (lambda p: p["pages"][0]["cells"][2].update(user_object={"element": "object", "data": [["label", "x"]]}), "reserved_attribute"),
        (lambda p: p.update(extensions=["<diagram/>"]), "ir_extension"),
    ],
)
def test_ir_json_integrity(mutate, code):
    payload = base_payload(box())
    mutate(payload)
    assert code in codes(ir_error(payload))


def test_ir_json_diagnostics_locate_the_field():
    payload = base_payload(box(), {"id": "e", "kind": "edge", "parent": "1", "target": "ghost"})
    (diagnostic,) = ir_error(payload).value.diagnostics
    assert diagnostic.loc.page == 1 and diagnostic.loc.path == "/pages/0/cells/3/target"
    assert diagnostic.cell_id == "e" and diagnostic.attribute == "target"
    assert diagnostic.to_json()["location"] == {"page": 1, "path": "/pages/0/cells/3/target"}
    assert "at page 1 /pages/0/cells/3/target (cell 'e')" in str(diagnostic)


def test_ir_json_text_errors():
    with pytest.raises(CompileError) as error:
        read_ir_json('{"ir": "drawio-semantic-ir/2",\n "pages": [}')
    assert codes(error) == ["json_syntax"] and error.value.diagnostics[0].loc.line == 2
    with pytest.raises(CompileError) as error:
        read_ir_json('{"ir": "x", "ir": "y", "pages": []}')
    assert codes(error) == ["json_duplicate_key"]
    with pytest.raises(CompileError) as error:
        read_ir_json('{"ir": "drawio-semantic-ir/2", "pages": [NaN]}')
    assert codes(error) == ["invalid_number"]


def test_ir_json_negative_size_warns_like_build():
    payload = base_payload(box(geometry={"x": 0, "y": 0, "width": -5, "height": 10}))
    result = compile_ir_json(payload)
    assert "negative_size" in [d.code for d in result.diagnostics if d.severity == "warning"]
    xml = compile_bytes(result.xml)
    assert xml.xml == result.xml


# Equivalence and sensitivity


def canonical(text):
    result = compile_ir_jsonl(text)
    return result.ir_json, result.xml


def graph(*rows, canvas=(200, 100)):
    return jsonl({"canvas": list(canvas)}, *rows, {"end": True})


A = {"id": "a", "k": "v", "g": [0, 0, 40, 20], "l": "A", "s": "rounded=1;fillColor=#fff;baseStyle;b=2;a=1"}
B = {"id": "b", "k": "v", "g": [100, 0, 40, 20], "l": "B"}
E = {"id": "e", "k": "e", "f": "a", "t": "b"}


def test_equivalent_inputs_converge():
    base = canonical(graph(A, B, E))
    renamed = graph({**A, "id": "first"}, {**B, "id": "second"}, {**E, "id": "x", "f": "first", "t": "second"})
    permuted = graph({**A, "s": "fillColor=#fff;rounded=1;baseStyle;a=1;b=2"}, B, E)
    overwritten = graph({**A, "s": "fillColor=#000;rounded=1;fillColor=#fff;baseStyle;b=2;a=1"}, B, E)
    numbers = graph({**A, "g": [0.0, -0.0, 4e1, "20"]}, {**B, "g": ["100", 0, "40.0", 2e1]}, E, canvas=(2e2, "100"))
    for variant in (renamed, permuted, overwritten, numbers):
        assert canonical(variant) == base
    as_json = compile_ir_json(json.loads(json.dumps(base[0]).replace('"40"', "40.0")))
    assert (as_json.ir_json, as_json.xml) == base


@pytest.mark.parametrize(
    "variant",
    [
        graph(A, B, {**E, "f": "b", "t": "a"}),  # reversed edge
        graph(B, A, E),  # draw order
        graph(A, B, {"id": "e", "k": "e", "f": "a", "tp": [100, 10]}),  # bound vs free end
        graph({**A, "g": [0, 0, 200, 100]}, {**B, "p": "a", "g": [100, 0, 40, 20]}, E),  # editable group parent
        graph({**A, "s": "baseStyle;rounded=1;fillColor=#fff;b=2;a=1"}, B, E),  # key moved across a stylename
    ],
)
def test_semantic_changes_stay_distinct(variant):
    base = canonical(graph(A, B, E))
    ir_json, xml = canonical(variant)
    assert ir_json != base[0] and xml != base[1]


# Lossless JSONL boundary


def lossy(document, construct):
    with pytest.raises(CompileError) as error:
        to_ir_jsonl(document)
    diagnostics = error.value.diagnostics
    assert diagnostics and all(d.code == "jsonl_lossy_projection" for d in diagnostics)
    assert any(construct in (d.attribute, d.cell_id) or construct in d.message for d in diagnostics), diagnostics
    text, warnings = to_ir_jsonl(document, lossless=False)
    assert warnings and all(d.severity == "warning" for d in warnings)
    read_ir_jsonl(text)  # the best effort still reads back
    return diagnostics


def test_xml_compiled_simple_graph_projects_losslessly():
    reference = compile_bytes(SIMPLE)
    text, diagnostics = to_ir_jsonl(reference)
    assert diagnostics == []
    again = compile_ir_jsonl(text)
    assert again.ir_json == reference.ir_json and again.xml == reference.xml


def test_rich_document_reports_every_lossy_construct():
    diagnostics = lossy(compile_bytes(RICH), "more than one page")
    attributes = {d.attribute for d in diagnostics}
    assert {"name", "grid", "extensions", "collapsed", "connectable", "relative", "offset", "alternate_bounds"} <= attributes
    assert {"user_object", "attributes", "geometry", "kind", "x", "y"} <= attributes
    assert all(d.loc.page in (None, 1, 2) for d in diagnostics)
    assert any(d.loc.page == 2 for d in diagnostics)
    assert any(d.cell_id == "obj" and d.attribute == "user_object" for d in diagnostics)


@pytest.mark.parametrize(
    "data, construct",
    [
        (SIMPLE.replace(b'name="Page 1"', b'name="Other"'), "name"),
        (SIMPLE.replace(b'pageHeight="600"', b'pageHeight="600" grid="0"'), "grid"),
        (SIMPLE.replace(b'pageHeight="600"', b""), "pageHeight"),
        (SIMPLE.replace(b'pageWidth="800"', b'pageWidth="800.50"'), "pageWidth"),
        (mxfile(vertex("a", 0, 0, extra='<x/>')), "extensions"),
        (mxfile(vertex("a", 0, 0).replace('vertex="1"', 'vertex="1" visible="0"')), "visible"),
        (mxfile(vertex("a", 0, 0, w=-1)), "geometry"),
        (mxfile('<mxCell id="a" vertex="1" parent="1"/>'), "geometry"),
        (mxfile('<mxCell id="e" edge="1" parent="1"/>'), "geometry"),
        (mxfile('<mxCell id="e" edge="1" parent="1"><mxGeometry x="1" relative="1" as="geometry"/></mxCell>'), "x"),
        (mxfile('<mxCell id="e" edge="1" parent="1"><mxGeometry as="geometry"/></mxCell>'), "relative"),
        (mxfile(vertex("a", 0, 0), '<mxCell id="b" vertex="1" parent="a"><mxGeometry x="0.5" relative="1" as="geometry"/></mxCell>'), "relative"),
        (mxfile(vertex("a", 0, 0).replace('as="geometry"/>', 'as="geometry"><mxPoint x="1" as="sourcePoint"/></mxGeometry>')), "source_point"),
        (mxfile('<object label="x" k="v" id="a"><mxCell vertex="1" parent="1"><mxGeometry as="geometry"/></mxCell></object>'), "user_object"),
        (mxfile('<mxCell id="F" style="x" parent="1"><mxGeometry x="5" as="geometry"/></mxCell>'), "kind"),
        (mxfile('<mxCell id="L2" parent="0"/>'), "kind"),
        (mxfile(edge("e", "a"), vertex("a", 0, 0)).replace(b'<mxCell id="0"/>', b'<mxCell id="0" value="root"/>'), "label"),
        (mxfile(vertex("a", 0, 0)).replace(b"<mxfile>", b'<mxfile theme="dark">'), "attributes"),
        (mxfile(vertex("a", 0, 0), extra="<foo/>"), "extensions"),
        (b'<mxfile><diagram name="Page 1"><mxGraphModel pageWidth="10" pageHeight="10"><root><mxCell id="r"/>'
         b'<mxCell id="1" parent="r"/><mxCell id="a" value="data:action/json" vertex="1" parent="1">'
         b'<mxGeometry width="1" height="1" as="geometry"/></mxCell></root></mxGraphModel></diagram></mxfile>', "id"),
    ],
)
def test_lossy_constructs_are_diagnosed(data, construct):
    lossy(compile_bytes(data), construct)


def test_lossy_projection_through_compile_ir():
    payload = compile_bytes(RICH).ir_json
    with pytest.raises(CompileError) as error:
        compile_ir_json(payload, emit_jsonl=True)
    assert "jsonl_lossy_projection" in codes(error)
    result = compile_ir_json(payload, emit_jsonl=True, allow_lossy_jsonl=True)
    assert result.report["ir_jsonl_lossless"] is False
    assert result.xml == compile_bytes(RICH).xml and result.ir_jsonl.endswith('{"end":true}\n')
    assert "emitted_jsonl_rebuilds_ir" not in result.report["checks"]


def test_lossy_projection_hoists_out_of_frames_exactly():
    data = mxfile(
        '<mxCell id="F" style="x" parent="1"><mxGeometry x="5" y="7" as="geometry"/></mxCell>',
        vertex("a", 10, 10, parent="F"),
    )
    text, _ = to_ir_jsonl(compile_bytes(data), lossless=False)
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[1]["g"][:2] == [15, 17] and "p" not in rows[1]


def test_sketch_style_roles_survive_jsonl():
    data = mxfile(vertex("keep", 10, 10, style="sketch=1"), vertex("x", 50, 50))
    reference = compile_bytes(data)
    assert reference.ir_json["pages"][0]["cells"][2]["id"] == "keep"
    text, diagnostics = to_ir_jsonl(reference)
    assert diagnostics == [] and '"id":"keep"' in text
    assert compile_ir_jsonl(text).xml == reference.xml


def test_to_ir_jsonl_accepts_raw_documents_and_ir_json():
    decoded = read_ir_jsonl(GRAPH)
    expected = compile_ir_jsonl(GRAPH).ir_jsonl
    assert to_ir_jsonl(decoded)[0] == expected
    assert to_ir_jsonl(decoded.document)[0] == expected
    assert to_ir_jsonl(compile_ir_jsonl(GRAPH).ir_json)[0] == expected
    assert to_ir_jsonl(decoded)[0] == expected  # the input is not modified


# External adapters


class Graph:
    """A toy external schema: flat nodes with absolute positions and parent links."""

    def __init__(self, name, space="page-px"):
        self.name, self.space = name, space

    def adapt(self, source, context):
        cells = [{"id": "0", "kind": "root"}, {"id": "1", "kind": "layer", "parent": "0"}]
        for node in source["nodes"]:
            cell = {"id": node.get("key"), "kind": "vertex", "parent": node.get("in", "1"), "label": node["text"]}
            if "box" in node:
                cell["geometry"] = dict(zip(("x", "y", "width", "height"), node["box"], strict=True))
            cells.append(cell)
        for link in source["links"]:
            geometry = {"relative": True}
            if "points" in link:
                geometry["points"] = link["points"]
            if "to_xy" in link:
                geometry["target_point"] = link["to_xy"]
            cells.append(
                {
                    "id": link["key"],
                    "kind": "edge",
                    "parent": "1",
                    "source": link["from"],
                    "target": link.get("to"),
                    "geometry": geometry,
                }
            )
        page = {"name": "Page 1", "settings": [["pageWidth", "400"], ["pageHeight", "300"]], "cells": cells}
        return AdaptedIR({"pages": [page]}, self.space)


GRAPH_SOURCE = {
    "nodes": [
        {"key": "outer", "text": "Outer", "box": [100, 100, 200, 150]},
        {"key": "inner", "text": "Inner", "in": "outer", "box": [130, 140, 50, 30]},
        {"key": "deep", "text": "Deep", "in": "inner", "box": [140, 150, 10, 10]},
    ],
    "links": [{"key": "l1", "from": "inner", "to": "deep", "points": [[150, 260]]}],
}


def test_adapter_converts_page_px_to_parent_local():
    adapter = register_adapter(Graph("toy-graph"), replace=True)
    adapted = adapt_ir(GRAPH_SOURCE, "toy-graph")
    assert adapted.coordinate_space == "parent-local-px"
    cells = {c["id"]: c for c in adapted.document["pages"][0]["cells"]}
    assert [cells["inner"]["geometry"][k] for k in ("x", "y")] == [30, 40]
    assert [cells["deep"]["geometry"][k] for k in ("x", "y")] == [10, 10]
    assert GRAPH_SOURCE["nodes"][1]["box"] == [130, 140, 50, 30]  # the source is not modified
    for given in (adapter, "toy-graph"):
        result = compile_ir(GRAPH_SOURCE, input_format="adapted", adapter=given)
        assert result.report["input"]["format"] == "adapted"
    result = compile_ir(GRAPH_SOURCE, adapter=adapter)
    local = adapt_ir(GRAPH_SOURCE, Graph("toy-local", "parent-local-px")).document
    for cell in local["pages"][0]["cells"]:
        if cell["id"] == "inner":
            cell["geometry"].update(x=30, y=40)
        if cell["id"] == "deep":
            cell["geometry"].update(x=10, y=10)
    assert compile_ir_json(local).xml == result.xml
    page_xy = {c["original_id"]: c.get("page_xy") for c in result.source_map["pages"][0]["cells"]}
    assert page_xy["deep"] == [140.0, 150.0]
    ids = [c["id"] for c in result.ir_json["pages"][0]["cells"]]
    assert ids == ["0", "1", "2", "3", "4", "5"]
    origin = {c["original_id"]: c["id"] for c in result.source_map["pages"][0]["cells"]}
    assert origin["deep"] == "4" and origin["l1"] == "5"  # stable mapping from external IDs


def test_adapter_rejects_topology_only_input():
    source = {
        "nodes": [{"key": "a", "text": "A"}, {"key": "b", "text": "B", "box": [0, 0, 10, 10]}],
        "links": [{"key": "l", "from": "a", "to": None}],
    }
    with pytest.raises(CompileError) as error:
        compile_ir(source, input_format="adapted", adapter=Graph("toy-topology"))
    diagnostics = error.value.diagnostics
    assert [d.code for d in diagnostics] == ["adapter_missing_geometry", "adapter_missing_geometry"]
    assert [(d.cell_id, d.attribute) for d in diagnostics] == [("a", "geometry"), ("l", "target_point")]
    source["nodes"][0]["box"] = [0, 0, 10, 10]
    source["links"][0]["to_xy"] = [50, 50]
    assert compile_ir(source, adapter=Graph("toy-topology")).xml


def test_adapter_rejects_unknown_space_missing_ids_and_bad_hierarchy():
    with pytest.raises(CompileError) as error:
        compile_ir(GRAPH_SOURCE, adapter=Graph("toy-space", "screen"))
    assert codes(error) == ["adapter_coordinate_space"]
    source = json.loads(json.dumps(GRAPH_SOURCE))
    del source["nodes"][0]["key"]
    with pytest.raises(CompileError) as error:
        compile_ir(source, adapter=Graph("toy-ids"))
    assert "adapter_missing_id" in codes(error)
    source = json.loads(json.dumps(GRAPH_SOURCE))
    source["nodes"][0]["in"] = "ghost"
    with pytest.raises(CompileError) as error:
        compile_ir(source, adapter=Graph("toy-parent"))
    assert "unresolved_parent" in codes(error)


def test_adapter_page_px_needs_absolute_geometry():
    class Relative(Graph):
        def adapt(self, source, context):
            adapted = super().adapt(source, context)
            adapted.document["pages"][0]["cells"][3]["geometry"]["relative"] = True
            return adapted

    with pytest.raises(CompileError) as error:
        compile_ir(GRAPH_SOURCE, adapter=Relative("toy-relative"))
    assert codes(error) == ["adapter_coordinate_space", "adapter_coordinate_space"]


def test_adapter_registry_and_input_format_contract():
    with pytest.raises(CompileError) as error:
        compile_ir(GRAPH_SOURCE, input_format="adapted", adapter="no-such-adapter")
    assert codes(error) == ["adapter_unknown"]
    with pytest.raises(CompileError) as error:
        compile_ir(GRAPH_SOURCE, input_format="jsonl", adapter=Graph("x"))
    assert codes(error) == ["input_format"]
    with pytest.raises(CompileError) as error:
        compile_ir(GRAPH, input_format="ir-jsonl")
    assert codes(error) == ["input_format"]
    with pytest.raises(CompileError) as error:
        compile_ir(GRAPH_SOURCE, input_format="adapted")
    assert codes(error) == ["input_format"]
    register_adapter(Graph("toy-once"), replace=True)
    with pytest.raises(ValueError):
        register_adapter(Graph("toy-once"))
    with pytest.raises(TypeError):
        register_adapter(object())


# Public API contract


def test_compile_ir_options():
    result = compile_ir(GRAPH, input_format="jsonl", emit_xml=False, self_check=False)
    assert result.xml is None and result.ir_jsonl is None and result.report["output"] is None
    assert result.report["checks"] == {"ir_verified": True}
    result = compile_ir(GRAPH, input_format="jsonl", emit_jsonl=True)
    assert result.ir_jsonl and result.report["ir_jsonl_output"]["bytes"] == len(result.ir_jsonl.encode())
    assert result.report["stages"] == [
        "decode_ir_jsonl",
        "build_and_verify_ir",
        "normalize_ir",
        "emit_xml",
        "reread_and_recompile",
        "emit_jsonl",
        "reread_jsonl",
    ]
    assert result.report["input"]["sha256"] and result.report["pages"] == 1


def test_compile_bytes_contract_is_unchanged():
    result = compile_bytes(SIMPLE)
    assert list(result.report) == [
        "schema",
        "compiler",
        "ir",
        "input",
        "output",
        "ir_sha256",
        "pages",
        "stages",
        "checks",
        "representation_changes",
        "rules_applied",
        "reducibility_audit",
        "diagnostics",
    ]
    assert list(result.report["checks"]) == [
        "ir_verified",
        "emitted_xml_rebuilds_ir",
        "recompilation_byte_stable",
        "emitted_xml_is_fixed_point",
    ]
    with pytest.raises(CompileError) as error:
        compile_bytes(b'<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="9"/></root></mxGraphModel>')
    assert str(error.value.diagnostics[0]).startswith("error[unresolved_parent] at page 1 line 1 column ")


# CLI


def run(tmp_path, capsys, *args):
    code = main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, json.loads(out) if out else None, err


def test_cli_xml_contract(tmp_path, capsys):
    source = tmp_path / "input.drawio"
    source.write_bytes(SIMPLE)
    code, summary, _ = run(tmp_path, capsys, source, "--out", tmp_path / "out")
    reference = compile_bytes(SIMPLE)
    assert code == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["canonical.drawio", "ir.json", "report.json", "source_map.json"]
    assert (tmp_path / "out" / "canonical.drawio").read_bytes() == reference.xml
    assert json.loads((tmp_path / "out" / "ir.json").read_bytes()) == reference.ir_json
    assert list(summary) == ["output", "sha256", "pages", "input_bytes", "output_bytes", "warnings"]
    assert summary["sha256"] == reference.report["output"]["sha256"]
    code, _, err = run(tmp_path, capsys, source, "--out", tmp_path / "out")
    assert code == 2 and "already exists" in err


def test_cli_xml_with_jsonl(tmp_path, capsys):
    source = tmp_path / "input.drawio"
    source.write_bytes(SIMPLE)
    code, summary, _ = run(tmp_path, capsys, source, "--out", tmp_path / "out", "--emit-jsonl")
    assert code == 0 and summary["jsonl"] == str(tmp_path / "out" / "ir.jsonl")
    assert (tmp_path / "out" / "ir.jsonl").read_text() == to_ir_jsonl(compile_bytes(SIMPLE))[0]
    with pytest.raises(SystemExit):
        main([str(source), "--out", str(tmp_path / "x"), "--no-xml"])
    capsys.readouterr()


def test_cli_ir_jsonl(tmp_path, capsys):
    source = tmp_path / "graph.jsonl"
    source.write_text(GRAPH)
    code, summary, _ = run(tmp_path, capsys, source, "--out", tmp_path / "out", "--input-format", "ir-jsonl", "--emit-jsonl")
    expected = compile_ir_jsonl(GRAPH)
    assert code == 0 and summary["pages"] == 1
    assert (tmp_path / "out" / "canonical.drawio").read_bytes() == expected.xml
    assert (tmp_path / "out" / "ir.jsonl").read_text() == expected.ir_jsonl
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["input"]["format"] == "ir-jsonl"


def test_cli_ir_json_without_xml(tmp_path, capsys):
    source = tmp_path / "graph.json"
    source.write_text(json.dumps(compile_bytes(SIMPLE).ir_json))
    code, summary, _ = run(tmp_path, capsys, source, "--out", tmp_path / "out", "--input-format", "ir-json", "--no-xml")
    assert code == 0 and summary["output"] is None and summary["sha256"] is None
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["ir.json", "report.json", "source_map.json"]
    assert json.loads((tmp_path / "out" / "ir.json").read_text()) == compile_bytes(SIMPLE).ir_json


def test_cli_reports_located_diagnostics(tmp_path, capsys):
    source = tmp_path / "graph.jsonl"
    source.write_text(GRAPH.replace('"t": "b"', '"t": "ghost"'))
    code, summary, err = run(tmp_path, capsys, source, "--out", tmp_path / "out", "--input-format", "ir-jsonl")
    assert code == 2 and summary is None and not (tmp_path / "out").exists()
    assert "error[dangling_terminal] at page 1 line 4 column 1 (cell 'e1')" in err
    with pytest.raises(SystemExit):
        main([str(source), "--out", str(tmp_path / "x"), "--input-format", "ir-jsonl", "--verify-native"])
    capsys.readouterr()


def test_module_entry_point_is_unchanged():
    assert (Path(__file__).resolve().parents[1] / "drawio" / "compiler" / "__main__.py").read_text() == (
        "from .cli import main\n\nraise SystemExit(main())\n"
    )


# Review-driven regression tests (Unicode XML names, adapter errors, malformed JSON, non-relative page-px edges)


def test_unicode_combining_xml_name_roundtrips_through_ir_json():
    xml = mxfile('<mxCell id="a" vertex="1" parent="1" foo\u0301="v"><mxGeometry width="10" height="10" as="geometry"/></mxCell>')
    compiled = compile_bytes(xml)
    again = compile_ir_json(compiled.ir_json, emit_xml=True, self_check=True)
    assert again.xml == compiled.xml and again.ir_json == compiled.ir_json


def test_adapter_error_diagnostic_stops_compilation():
    from drawio.compiler import Diagnostic

    class FailingAdapter:
        name = "failing-adapter"

        def adapt(self, source, context):
            return AdaptedIR(
                document=compile_bytes(SIMPLE).ir_json,
                coordinate_space="parent-local-px",
                diagnostics=[Diagnostic("error", "adapter_custom_error", "upstream schema invalid")],
            )

    with pytest.raises(CompileError) as error:
        compile_ir({}, input_format="adapted", adapter=FailingAdapter())
    assert "adapter_custom_error" in codes(error)


def test_unhashable_attribute_key_and_huge_int_raise_compile_error():
    payload = compile_bytes(SIMPLE).ir_json
    payload["attributes"] = [[["not", "hashable"], "v"]]
    with pytest.raises(CompileError) as error:
        compile_ir_json(payload)
    assert "invalid_name" in codes(error)

    huge = '{"ir":"drawio-semantic-ir/2","pages":[{"cells":[{"id":"0","kind":"root","x":' + ("9" * 5000) + "}]}]}"
    with pytest.raises(CompileError) as error:
        compile_ir_json(huge)
    assert "json_syntax" in codes(error)


def test_adapter_converts_non_relative_edge_xy_in_page_px():
    class EdgeAdapter:
        name = "edge-page-px"

        def adapt(self, source, context):
            return AdaptedIR(
                document={
                    "pages": [
                        {
                            "name": "Page 1",
                            "settings": [["pageWidth", "400"], ["pageHeight", "300"]],
                            "cells": [
                                {"id": "0", "kind": "root"},
                                {"id": "1", "kind": "layer", "parent": "0"},
                                {"id": "box", "kind": "vertex", "parent": "1", "geometry": {"x": 100, "y": 50, "width": 200, "height": 150}},
                                {
                                    "id": "e1",
                                    "kind": "edge",
                                    "parent": "box",
                                    "geometry": {
                                        "x": 120,
                                        "y": 70,
                                        "source_point": [110, 60],
                                        "target_point": [150, 90],
                                    },
                                },
                            ],
                        }
                    ]
                },
                coordinate_space="page-px",
            )

    adapted = adapt_ir({}, EdgeAdapter())
    edge_cell = {c["id"]: c for c in adapted.document["pages"][0]["cells"]}["e1"]
    assert edge_cell["geometry"]["x"] == 20 and edge_cell["geometry"]["y"] == 20
    assert edge_cell["geometry"]["source_point"] == [10, 10]
    assert edge_cell["geometry"]["target_point"] == [50, 40]
