"""Generate canonical IR in one vision request, or request XML in a second stage."""

import argparse
import base64
from contextlib import contextmanager
from decimal import Decimal
import fcntl
from hashlib import sha256
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
from shutil import which
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic


IR_PROMPT = """Turn the source image into a compact editable diagram IR. Return one minified JSON object on one line.
Use this schema:
{"ir":"drawio-semantic-ir/2","pages":[{"name":"Page 1","settings":[["pageWidth","800"],["pageHeight","600"]],"cells":[
{"id":"0","kind":"root"},{"id":"1","kind":"layer","parent":"0"},
{"id":"n1","kind":"vertex","parent":"1","label":"Visible label","style":[["rounded","1"],["fillColor","#dae8fc"]],"geometry":{"x":20,"y":20,"width":110,"height":45}},
{"id":"n2","kind":"vertex","parent":"1","label":"Other node","geometry":{"x":180,"y":20,"width":95,"height":45}},
{"id":"e1","kind":"edge","parent":"1","source":"n1","target":"n2","geometry":{"relative":true,"points":[[160,42]]}}]}]}
Replace the example with the actual diagram. Map topology, containers, text, and arrows first, then approximate styles.
Use the source image dimensions for pageWidth/pageHeight and parent-local pixel coordinates.
IDs must be unique per page. List parents before children, and siblings from back to front.
Use vertex geometry for shapes, text and editable containers. Waypoints are optional. For loose edges, provide
geometry.source_point and geometry.target_point when the corresponding source or target cell ID is missing.
Use short IDs and keep only distinguishing style overrides. Omit default styles and empty fields.
For dense repeated decorations or matrix entries, use a compact editable label or just the important entries.
Keep meaningful nodes, labels, and arrows. Prefer a complete coarse graph to a truncated detailed graph.
Produce the complete minified JSON now."""

XML_PROMPT = """Recreate the source image as a single complete editable Draw.io XML document.
Use the image for visual truth and the supplied IR for layout, IDs, hierarchy, labels and edge endpoints.
If the IR is missing or incomplete, use the image directly. Approximate low-priority details if needed.
Emit only <mxfile><diagram><mxGraphModel><root>...</root></mxGraphModel></diagram></mxfile>.
Use native editable shapes, text and edges. Every edge has geometry with relative="1" and as="geometry".
Use unique IDs, escape all XML attributes, and close every tag. Finish the complete diagram in this reply."""

CODEX_CHATGPT = "codex_chatgpt"
CODEX_CLI = (os.environ.get("IMG2IR_CODEX_BIN") or which("codex")
             or "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")
PROVIDERS = (CODEX_CHATGPT,)
SHORT_REASONING_PROMPT = (
    "\nKeep the reasoning brief. Make one pass over the main groups, labels, and arrow directions. "
    "Estimate positions directly, then emit the full JSON Lines."
)

IR_LINES_PROMPT = """Describe the source image as an editable Draw.io graph. Return compact JSON Lines.
Write one independent JSON object per line. Use the following record forms. These values are examples only:
{"canvas":[800,600]}
{"id":"a","k":"v","g":[20,20,110,45],"l":"Input","s":"rounded=1;fillColor=#dae8fc"}
{"id":"b","k":"v","g":[170,20,100,45],"l":"Output"}
{"id":"e1","k":"e","f":"a","t":"b","s":"endArrow=block"}
{"end":true}

Write the actual canvas width and height first. The importer creates the root and layer. Keep IDs unique,
short, and distinct from 0 and 1. 'v' is a vertex, 'e' is an edge. 'g' holds x,y,width,height in parent-local pixels.
Use 'p' for the parent ID when inside a container. 'l' holds the editable label, and optional 's' holds native Draw.io style.
For an edge, 'f' and 't' point to node IDs. A free edge end can use 'sp' or 'tp' instead, with [x,y];
optional 'w' holds a list of [x,y] waypoints. Draw containers before their children and preserve visual stacking.
Map boxes, labels, hierarchy and arrows first. Simplify repetitions and keep style strings short.
Finish with the end record. Produce the actual graph now."""


class ProviderUnavailable(RuntimeError):
    def __init__(self, provider, host, status, reason):
        detail = "is unavailable" if status is None else f"returned HTTP {status}"
        super().__init__(f"{host} {detail} ({reason})")
        self.provider = provider
        self.reason = reason


class ProviderTransportError(RuntimeError):
    reason = "transport"

    def __init__(self, provider, message):
        super().__init__(message)
        self.provider = provider


class MissingIRResponse(RuntimeError):
    """The image already has a durable dispatch record but no recoverable response."""


def codex_chatgpt_env(env):
    return {name: value for name, value in env.items() if name not in {
        "OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
        "HY_API_KEY", "DEEPSEEK_API_KEY",
    }}


def codex_chatgpt_available(env):
    result = subprocess.run([str(CODEX_CLI), "login", "status"], env=codex_chatgpt_env(env),
                            capture_output=True, text=True, timeout=10)
    return result.returncode == 0 and "Logged in using ChatGPT" in result.stdout + result.stderr


def codex_chatgpt_completion(prompt, args):
    model = "gpt-6.2-sol"
    effort = "medium"
    started = monotonic()
    with TemporaryDirectory(prefix="img2ir-codex-") as directory:
        command = [str(CODEX_CLI), "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--cd", directory, "--model", model,
                   "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
                   "-c", f'model_reasoning_effort="{effort}"']
        for feature in ("shell_tool", "unified_exec", "view_image", "hooks", "tool_suggest",
                        "multi_agent", "skill_search", "apps", "image_generation"):
            command.extend(("--disable", feature))
        command.extend(("--json", "--image", str(args.image.resolve()), "--", "-"))
        try:
            result = subprocess.run(command, input=prompt + "\nReturn the JSON Lines as the final answer. Do not invoke tools.",
                                    env=codex_chatgpt_env(os.environ), capture_output=True, text=True,
                                    timeout=args.timeout + 40)
        except subprocess.TimeoutExpired:
            raise ProviderTransportError(CODEX_CHATGPT, "The Codex vision turn timed out") from None

    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode:
        errors = (result.stderr + "\n" + "\n".join(
            json.dumps(event) for event in events if event["type"] in {"error", "turn.failed"}
        )).lower()
        if any(term in errors for term in ("usage limit", "insufficient_quota", "usage_limit_reached")):
            reason = "quota"
        elif any(term in errors for term in ("rate limit", "429 too many requests")):
            reason = "rate_limit"
        elif any(term in errors for term in ("401 unauthorized", "403 forbidden", "not logged in", "does not support")):
            reason = "access"
        elif any(term in errors for term in ("stream disconnected", "connection", "timed out")):
            raise ProviderTransportError(CODEX_CHATGPT, "The Codex vision connection failed")
        else:
            reason = "turn_failed"
        raise ProviderUnavailable(CODEX_CHATGPT, "Codex ChatGPT", None, reason)

    if any("model" in line.lower() and any(word in line.lower() for word in ("unknown", "unsupported", "fallback"))
           for line in result.stderr.splitlines()):
        raise ProviderUnavailable(CODEX_CHATGPT, "Codex ChatGPT", None, "access")
    turns = [event for event in events if event["type"] == "turn.completed"]
    starts = sum(event["type"] == "turn.started" for event in events)
    items = [event["item"] for event in events if event["type"] == "item.completed"]
    messages = [item["text"] for item in items if item["type"] == "agent_message"]
    if (starts != 1 or len(turns) != 1 or len(messages) != 1
            or any(item["type"] not in {"reasoning", "agent_message"} for item in items)):
        raise ValueError("Codex did not return one tool-free vision turn")
    return messages[0].strip(), {
        "provider": CODEX_CHATGPT, "model": model, "requested_model": model, "reasoning_effort": effort,
        "finish": "completed", "seconds": round(monotonic() - started, 2), "usage": turns[0].get("usage", {}),
        "completed_turns": 1, "tool_calls": 0,
    }


def completion(provider, prompt, img, maximum, args):
    if provider != CODEX_CHATGPT:
        raise ValueError("This runner accepts only the Codex ChatGPT route")
    return codex_chatgpt_completion(prompt, args)


def unwrap(text, language):
    match = re.fullmatch(rf"```(?:{language})?\s*\n(.*?)\n```", text, flags=re.S | re.I)
    return match[1] if match else text


def xml_body(text):
    # Some models prepend a sentence even when the document is complete.
    documents = re.findall(r"<mxfile\b[^>]*>.*?</mxfile\s*>", text, flags=re.S)
    if len(documents) == 1:
        return documents[0]
    if len(documents) > 1:
        raise ValueError("Expected one Draw.io document")
    return unwrap(text, "xml")


def read_ir(text, truncated):
    text = unwrap(text, "json")
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        if not truncated:
            raise
    # Recover complete cells from a single page cut off at the output limit.
    match = re.search(r'"cells"\s*:\s*\[', text)
    if match is None:
        raise ValueError("Truncated IR has no cell list")
    partial = json.loads(text[:match.end()] + "]}]}")
    pos = match.end()
    decoder = json.JSONDecoder()
    cells = partial["pages"][0]["cells"]
    while pos < len(text):
        suffix = text[pos:].lstrip()
        pos += len(text[pos:]) - len(suffix)
        if suffix.startswith(","):
            pos += 1
            continue
        if not suffix.startswith("{"):
            break
        try:
            cell, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        cells.append(cell)
    if not cells:
        raise ValueError("No complete cells in truncated IR")
    return partial, False


def normalize_final_ir_record(text, finish):
    if finish in {"stop", "completed"}:
        prefix, newline, last = unwrap(text, "jsonl|ndjson|json").rstrip().rpartition("\n")
        if newline and last.strip() == '{"id":"end":true}':
            return prefix + newline + '{"end":true}', True
    return text, False


def read_ir_lines(text, truncated, ir, parse_style, *, strict=False):
    lines = unwrap(text, "jsonl|ndjson|json").splitlines()
    records = []
    malformed = False
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            malformed = True
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("Expected a canvas record and cell records")
    if strict and records[0].keys() != {"canvas"}:
        raise ValueError("Expected the first record to contain only the canvas")
    width, height = records[0]["canvas"]
    complete = not truncated and not malformed and records[-1] == {"end": True}
    cells = [{"id": "0", "kind": "root"}, {"id": "1", "kind": "layer", "parent": "0"}]
    body = records[1:-1] if records[-1] == {"end": True} else records[1:]
    for row in body:
        if row["k"] not in ("v", "e"):
            raise ValueError(f"Unknown cell kind {row['k']!r}")
        vertex = row["k"] == "v"
        if strict:
            common = {"id", "k", "p", "l", "s"}
            geometry = {"g"} if vertex else {"f", "t", "sp", "tp", "w"}
            if unsupported := row.keys() - common - geometry:
                raise ValueError(f"Unsupported fields for {row['k']}: {', '.join(sorted(unsupported))}")
        cell = {"id": str(row["id"]), "kind": "vertex" if vertex else "edge", "parent": str(row.get("p", "1"))}
        if row.get("l"):
            cell["label"] = row["l"]
        if row.get("s"):
            cell["style"] = [list(token) for token in parse_style(row["s"])]
        if vertex:
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
    return {"ir": ir.IR_VERSION, "pages": [{"name": "Page 1", "settings": settings, "cells": cells}]}, complete


def compile_visual_ir(payload, ir, emit, compile_bytes):
    """Map the model-facing subset of IR/2 into the compiler's native objects."""
    if payload["ir"] != ir.IR_VERSION:
        raise ValueError("IR version mismatch")

    def point(value):
        if len(value) != 2:
            raise ValueError("Expected a coordinate pair")
        return tuple(Decimal(str(n)) for n in value)

    pages = []
    for raw in payload["pages"]:
        records = raw["cells"]
        uids = {str(record["id"]): uid for uid, record in enumerate(records)}
        if len(uids) != len(records):
            raise ValueError("Duplicate cell ID")
        cells = {}
        for uid, record in enumerate(records):
            coords = record.get("geometry")
            geometry = None
            if coords is not None:
                number = {k: Decimal(str(coords.get(k, 0))) for k in ("x", "y", "width", "height")}
                endpoints = {k: point(coords[k]) for k in ("source_point", "target_point", "offset") if k in coords}
                geometry = ir.Geometry(**number, **endpoints, relative=bool(coords.get("relative", False)),
                                       points=tuple(point(p) for p in coords.get("points", [])))
            style = tuple(tuple(item) for item in record.get("style", [])) or None
            refs = {k: uids[str(record[k])] for k in ("parent", "source", "target") if record.get(k) is not None}
            cells[uid] = ir.Cell(uid=uid, kind=record["kind"], name=str(record["id"]), style=style,
                                 label=record.get("label") or None, geometry=geometry, **refs)
        roots = []
        for cell in cells.values():
            if cell.parent is None:
                roots.append(cell.uid)
            else:
                cells[cell.parent].children.append(cell.uid)
        if len(roots) != 1:
            raise ValueError("Each page must contain one root")
        allowed_settings = {"pageWidth", "pageHeight", "page", "grid", "gridSize", "guides"}
        settings = tuple((name, str(value)) for name, value in raw.get("settings", []) if name in allowed_settings)
        pages.append(ir.Page(raw.get("name"), None, (), settings, (), (), cells, roots[0]))
    doc = ir.Document((), (), pages)
    ir.verify(doc, named=True)
    return compile_bytes(emit(doc))


def require_editable(result):
    if not any(cell.kind == "vertex" for page in result.document.pages for cell in page.cells.values()):
        raise ValueError("The diagram needs at least one editable vertex")
    return result


def require_self_contained_images(payload):
    from PIL import Image

    mime_types = {"image/png": "PNG", "image/jpeg": "JPEG", "image/gif": "GIF", "image/webp": "WEBP"}
    for page in payload["pages"]:
        for cell in page["cells"]:
            style = dict(cell.get("style") or [])
            if style.get("html") == "1" and re.search(r"<\s*img\b", cell.get("label") or "", re.I):
                raise ValueError(f"Use an editable shape or embedded image style instead of HTML images in {cell['id']}")
            if "image" not in style and style.get("shape") != "image":
                continue
            header, _, data = (style.get("image") or "").partition(",")
            mime_type = header.removeprefix("data:").removesuffix(";base64")
            if not header.startswith("data:") or mime_type not in mime_types or not data:
                raise ValueError(f"Unresolved image asset in cell {cell['id']}")
            try:
                with Image.open(BytesIO(base64.b64decode(data, validate=True))) as image:
                    if image.format != mime_types[mime_type]:
                        raise ValueError("Wrong embedded image format")
                    image.verify()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                raise ValueError(f"Invalid embedded image asset in cell {cell['id']}") from error


def check_canonical_canvas(payload, canvas):
    require_self_contained_images(payload)
    page, = payload["pages"]
    settings = dict(page["settings"])
    if canvas is not None and tuple(int(settings[k]) for k in ("pageWidth", "pageHeight")) != canvas:
        raise ValueError("Input image and IR canvas mismatch")
    for cell in page["cells"]:
        if cell["kind"] == "vertex":
            geometry = cell.get("geometry") or {}
            if any(Decimal(str(geometry.get(axis, 0))) < 0 for axis in ("width", "height")):
                raise ValueError(f"Negative vertex width or height in cell {cell['id']}")


def sync_directory(folder):
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def write_report(out, report):
    write_json(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


@contextmanager
def output_dir(folder, ir_only):
    folder.mkdir(parents=True, exist_ok=ir_only)
    if not ir_only:
        yield
        return
    with (folder / ".ir.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def request_ir_once(args, prompt, img, image_hash):
    ledger = args.out.parent / ".ir-requests"
    ledger.mkdir(exist_ok=True)
    claim = ledger / f"{image_hash}.dispatch"
    cached = ledger / f"{image_hash}.json"
    if cached.is_file():
        saved = json.loads(cached.read_text())
        return saved["text"], saved["request"], saved.get("reasoning_content", ""), True
    try:
        with claim.open("x") as stream:
            stream.write(args.stage1 + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise MissingIRResponse("An IR request may already have been sent for this PNG") from None
    sync_directory(ledger)
    sync_directory(ledger.parent)
    try:
        text, request = completion(args.stage1, prompt, img, args.ir_max, args)
    except OSError as error:
        raise ProviderTransportError(args.stage1, str(error)) from None
    reasoning = request.pop("reasoning_content", "")
    saved = {"text": text, "request": request}
    if request["provider"] in {"hy", "tcodex"} and request.get("reasoning_effort") == "low":
        request["reasoning_required"] = True
        if reasoning.strip():
            request["reasoning_sha256"] = sha256(reasoning.encode("utf-8")).hexdigest()
            saved["reasoning_content"] = reasoning
    write_json(cached, saved)
    return text, request, reasoning, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--compiler", type=Path, required=True, help="drawio-agent-rollout checkout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage1", choices=PROVIDERS, default=CODEX_CHATGPT)
    parser.add_argument("--stage2", choices=PROVIDERS, default=CODEX_CHATGPT)
    parser.add_argument("--deepseek-auth", choices=("account", "api-key"), default="account")
    parser.add_argument("--hy-thinking", choices=("none", "low"), default="none")
    parser.add_argument("--deepseek-thinking", choices=("none", "low"), default="none")
    parser.add_argument("--ir-format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--ir-only", action="store_true", help="Stop after the first request and save canonical IR")
    parser.add_argument("--ir-max", type=int, default=12000)
    parser.add_argument("--xml-max", type=int, default=16000)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not args.ir_only or args.stage1 != CODEX_CHATGPT or args.stage2 != CODEX_CHATGPT:
        parser.error("This handoff runs only the Codex ChatGPT IR route")
    sys.path.insert(0, str(args.compiler.resolve()))
    from drawio.compiler import ir
    from drawio.compiler.compiler import compile_bytes
    from drawio.compiler.emit import emit
    from drawio.compiler.source import CompileError
    from drawio.compiler.syntax import parse_style

    media_type = mimetypes.guess_type(args.image.name)[0]
    if media_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        parser.error("Expected a PNG, JPEG, WebP or GIF")
    raw_image = args.image.read_bytes()
    img = (media_type, base64.b64encode(raw_image).decode())
    ir_prompt = IR_LINES_PROMPT if args.ir_format == "jsonl" else IR_PROMPT
    canvas = None
    if raw_image.startswith(b"\x89PNG\r\n\x1a\n") and raw_image[12:16] == b"IHDR":
        width = int.from_bytes(raw_image[16:20], "big")
        height = int.from_bytes(raw_image[20:24], "big")
        canvas = width, height
        ir_prompt += f"\nThe input canvas is exactly {width} by {height} pixels."
    known_errors = (OSError, RuntimeError, ValueError, TypeError, KeyError, ArithmeticError, CompileError, ir.IRError)
    image_hash = sha256(raw_image).hexdigest() if args.ir_only else None
    with output_dir(args.out, args.ir_only):
        return generate(args, ir_prompt, img, canvas, image_hash, ir, emit, compile_bytes,
                        parse_style, known_errors)


def generate(args, ir_prompt, img, canvas, image_hash, ir, emit, compile_bytes, parse_style, known_errors):
    report, draft, fallback = {}, None, None
    try:
        if args.ir_only:
            self_contained_prompt = ir_prompt + (
                "\nDraw pictorial blocks using editable native Draw.io shapes and text. "
                "Approximate photos or icons with shapes or concise labels. "
                "Do not emit image or shape=image styles, HTML image tags, or asset filenames and URLs."
            )
            if ((args.stage1 == "hy" and args.hy_thinking != "none")
                    or (args.stage1 == "tcodex" and args.deepseek_thinking != "none")):
                self_contained_prompt += SHORT_REASONING_PROMPT
            report["native_shapes_prompt"] = True
            text, request, reasoning, reused = request_ir_once(
                args, self_contained_prompt, img, image_hash
            )
            report["ir_request"] = request
            report["ir_response_reused"] = reused
            if request.get("reasoning_required"):
                if not reasoning.strip():
                    raise ValueError("Missing provider reasoning")
                if sha256(reasoning.encode("utf-8")).hexdigest() != request["reasoning_sha256"]:
                    raise ValueError("Reasoning differs from the saved response")
                (args.out / "reasoning.txt").write_text(reasoning, encoding="utf-8")
        else:
            text, report["ir_request"] = completion(args.stage1, ir_prompt, img, args.ir_max, args)
        (args.out / "stage1.txt").write_text(text)
        finish = report["ir_request"]["finish"]
        truncated = finish in {"length", "max_tokens"}
        if args.ir_only and finish == "max_output_tokens":
            truncated = True
        report["ir_format"] = args.ir_format
        if args.ir_format == "jsonl":
            payload = text
            if args.ir_only:
                payload, normalized = normalize_final_ir_record(text, finish)
                if normalized:
                    report["normalized_end_record"] = True
            draft, report["ir_complete"] = read_ir_lines(payload, truncated, ir, parse_style, strict=args.ir_only)
        else:
            draft, report["ir_complete"] = read_ir(text, truncated)
        if args.ir_only and (truncated or not report["ir_complete"]):
            report["ir_complete"] = False
            raise ValueError("Incomplete stage 1 IR")
        fallback = require_editable(compile_visual_ir(draft, ir, emit, compile_bytes))
        saved = fallback.ir_json if args.ir_only else draft
        if args.ir_only:
            check_canonical_canvas(saved, canvas)
        write_json(args.out / "ir.json", saved)
    except known_errors as error:
        report["ir_error"] = str(error)[:350]
        if isinstance(error, (ProviderUnavailable, ProviderTransportError)):
            report[f"{error.provider}_unavailable"] = error.reason
        if isinstance(error, (MissingIRResponse, ProviderTransportError)):
            report["ir_attempt_state"] = "ambiguous"
    if args.ir_only:
        complete = fallback is not None and "ir_error" not in report
        report["ir_complete"] = complete
        report["source"] = "canonical_ir" if complete else "failed"
        if complete:
            report["compiler_checks"] = fallback.report["checks"]
            report["image_sha256"] = image_hash
            report["canonical_ir_sha256"] = fallback.report["ir_sha256"]
        write_report(args.out, report)
        return 0 if complete else 1
    try:
        if args.stage1 == args.stage2 and report.get(f"{args.stage1}_unavailable"):
            raise RuntimeError(f"Stage 2 skipped after {args.stage1} could not complete stage 1")
        ir_hint = json.dumps(draft, ensure_ascii=False, separators=(",", ":")) if fallback is not None else "unavailable"
        if fallback is not None and not report["ir_complete"]:
            ir_hint = "Partial graph. Fill missing content using the image.\n" + ir_hint
        text, report["xml_request"] = completion(args.stage2, XML_PROMPT + "\n\nIR\n" + ir_hint,
                                                 img, args.xml_max, args)
        (args.out / "stage2.txt").write_text(text)
        diagram = require_editable(compile_bytes(xml_body(text).encode()))
        report["source"] = "model_xml"
    except known_errors as error:
        report["xml_error"] = str(error)[:350]
        if isinstance(error, ProviderUnavailable):
            report[f"{error.provider}_unavailable"] = error.reason
        diagram = fallback
        report["source"] = "compiled_ir" if fallback else "failed"
    if diagram is not None:
        (args.out / "reconstruction.drawio").write_bytes(diagram.xml)
        report["vertices"] = sum(c.kind == "vertex" for p in diagram.document.pages for c in p.cells.values())
        report["edges"] = sum(c.kind == "edge" for p in diagram.document.pages for c in p.cells.values())
        report["compiler_checks"] = diagram.report["checks"]
    write_report(args.out, report)
    return 0 if diagram else 1


if __name__ == "__main__":
    raise SystemExit(main())
