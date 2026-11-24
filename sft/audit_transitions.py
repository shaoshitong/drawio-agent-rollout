"""Locate repeated crop actions and construction attempts without executing model code."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def event_calls(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    results = {row["callId"]: row for row in events if row["type"] == "tool_result"}
    return [{**row, "result": results.get(row["callId"])}
            for row in events if row["type"] == "tool_call"]


def archived_calls(root, episode, stage):
    from .data import open_store

    store = open_store(root)
    with sqlite3.connect("file:" + str(root / "archive/ledger.sqlite") + "?mode=ro", uri=True) as connection:
        rows = connection.execute("select id,response from calls where episode=? and stage=? "
            "and role='generator' and response is not null and status=200 and error is null "
            "order by id", (episode, stage)).fetchall()
    calls = []
    for number, reference in rows:
        message = store.get(reference)["choices"][0]["message"]
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            calls.append({"callId": call["id"], "source_decision": number,
                "tool": call["function"]["name"],
                "input": json.loads(arguments) if isinstance(arguments, str) else arguments})
    return calls


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def crop_operation(command):
    box = re.search(r"\.crop\(\s*(\([^()]+\))\s*\)", command)
    source = re.search(r"Image\.open\(['\"]([^'\"]+)['\"]\)", command)
    destination = re.search(r"\.save\(['\"]([^'\"]+)['\"]\)", command)
    if not box or not source or re.search(r"\.(resize|rotate|transpose|thumbnail)\(", command):
        return None
    try:
        coordinates = ast.literal_eval(box[1])
    except (ValueError, SyntaxError):
        return None
    if not isinstance(coordinates, tuple) or len(coordinates) != 4:
        return None
    if not all(isinstance(value, int) for value in coordinates):
        return None
    return {"source": source[1], "box": coordinates,
            "destination": destination[1] if destination else None}


def analyze(calls):
    rows, first_crop, outputs, viewed = [], {}, {}, set()
    longest, current, previous, first_xml = [], [], None, None
    repeat_run, longest_repeat, crop_markers = [], [], 0
    for index, call in enumerate(calls, 1):
        arguments = call["input"]
        command = arguments.get("command", "")
        crop_markers += bool(re.search(r"\.crop\s*\(", command))
        crop = crop_operation(command)
        signature = digest([crop["source"], crop["box"]]) if crop else None
        repeat_from = first_crop.get(signature) if signature else None
        repeat_run = repeat_run + [index] if repeat_from else []
        if len(repeat_run) > len(longest_repeat):
            longest_repeat = repeat_run.copy()
        if signature:
            first_crop.setdefault(signature, index)
            outputs[index] = crop["destination"]
        if call["tool"] == "read_image":
            path = arguments.get("file_path")
            matching = [number for number, destination in outputs.items() if destination == path]
            if matching:
                viewed.add(matching[-1])
        current = current + [index] if signature and signature == previous else [index] if signature else []
        previous = signature
        if len(current) > len(longest):
            longest = current.copy()
        text = json.dumps(arguments, ensure_ascii=False)
        native_path = Path(arguments.get("file_path", "")).suffix in (".drawio", ".xml")
        xml_attempt = bool(re.search(r"<(?:mxfile|mxGraphModel|mxCell)(?:\s|>)", text)) and (
            call["tool"] in ("write", "edit") and native_path
            or call["tool"] == "bash" and ".drawio" in command and any(
                marker in command for marker in ("write(", "write_text(", "cat ")))
        if xml_attempt and first_xml is None:
            first_xml = index
        result = call.get("result")
        result_text = str(result.get("result", "")) if result else ""
        exit_code = re.search(r"\[exit code: (-?\d+)\]", result_text)
        rows.append({"tool_index": index, "call_id": call["callId"],
            "source_decision": call.get("source_decision"), "tool": call["tool"],
            "description": arguments.get("description"), "arguments_sha256": digest(arguments),
            "crop": crop, "same_source_box_first_seen": repeat_from,
            "xml_write_attempt": xml_attempt, "result_status": result.get("status") if result else None,
            "result_exit_code": int(exit_code[1]) if exit_code else None,
            "official_render_succeeded": "official_drawio_Graph.getSvg" in result_text,
            "possible_input_preview_cut": any(isinstance(value, str) and len(value) == 8192
                                               for value in arguments.values()),
            "result_sha256": digest(result.get("result")) if result else None})
    return {"tools": len(rows), "tool_counts": dict(Counter(row["tool"] for row in rows)),
        "crop_marker_calls": crop_markers, "literal_crop_calls": len(outputs),
        "distinct_literal_source_boxes": len(first_crop),
        "crop_calls_with_no_later_read_image": sorted(set(outputs) - viewed),
        "first_xml_write_attempt": first_xml, "longest_same_source_box_run": longest,
        "longest_repeated_source_boxes_run": longest_repeat,
        "failed_process_calls": [row["tool_index"] for row in rows if row["result_exit_code"] not in (None, 0)],
        "successful_official_render_calls": [row["tool_index"] for row in rows if row["official_render_succeeded"]],
        "possible_input_preview_cut_calls": [row["tool_index"] for row in rows if row["possible_input_preview_cut"]],
        "scope": "Literal PIL source+box equality, not pixel/content novelty or proof of useless work. "
                 "Missing read_image does not exclude stdout-based measurement. XML markers are attempts, not compilation success. "
                 "CLI events may contain previews; they are not certified full model prefixes.",
        "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", type=Path)
    source.add_argument("--archive-root", type=Path)
    parser.add_argument("--episode")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calls = event_calls(args.events) if args.events else archived_calls(args.archive_root, args.episode, args.stage)
    report = analyze(calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "rows"}
    for key, value in summary.items():
        if isinstance(value, list):
            summary[key] = {"count": len(value), "first": value[0] if value else None,
                            "last": value[-1] if value else None}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
