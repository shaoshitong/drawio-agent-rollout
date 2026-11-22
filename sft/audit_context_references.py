"""Flag exact quoted evidence lost by a fixed history-only context projection."""

import argparse
from collections import Counter
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import time

from .audit_context_budget import clear_context, result_anchors, template_messages
from .check_prefix_chains import Reader
from .data import load_manifest


QUOTES = re.compile(r'`([^`\n]{24,512})`|"([^"\n]{24,512})"|“([^”\n]{24,512})”')


def quoted_gaps(before, after, reasoning):
    before, after = " ".join(before.split()), " ".join(after.split())
    fragments = {" ".join(next(group for group in match.groups() if group).split())
                 for match in QUOTES.finditer(reasoning)}
    supported = [fragment for fragment in fragments if fragment in before]
    missing = [fragment for fragment in supported if fragment not in after]
    return {"quoted_fragments": len(fragments), "supported_before": len(supported),
            "missing_after": [{"sha256": sha256(fragment.encode()).hexdigest(), "characters": len(fragment)}
                              for fragment in sorted(missing)]}


def fragment_sources(messages, fragment):
    """Return raw-source provenance and age, without exporting quoted text."""
    from .context_projection import message_text, strip_history_thinking
    from .audit_context_budget import tool_result

    assistants = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    sources = []
    for index, message in enumerate(messages):
        visible = message_text(message)
        sections = [("reasoning", message.get("reasoning_content") or ""),
                    ("tool_text" if tool_result(message) else message["role"] + "_visible", visible)]
        if message["role"] == "assistant" and not isinstance(message.get("reasoning_content"), str):
            stripped = message_text(strip_history_thinking([message])[0])
            if stripped != visible:
                sections[0] = ("reasoning", visible.split("</think>", 1)[0])
                sections[1] = ("assistant_visible", stripped)
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            sections.append(("tool_arguments", arguments if isinstance(arguments, str) else json.dumps(arguments)))
        for kind, text in sections:
            if fragment in " ".join(text.split()):
                sources.append({"message_index": index, "kind": kind,
                                "assistant_age": len(assistants) - assistants.index(index)
                                if kind == "reasoning" and index in assistants else None})
    return sources


def trace_projection(args):
    """Explain flags against a frozen projection; targets never select its inputs."""
    from transformers import AutoProcessor
    from .context_projection import project_history, strip_history_thinking

    captured = args.trace_projection.read_text()
    lines = captured[:captured.rfind("\n") + 1].splitlines()
    audit = [json.loads(line) for line in lines if line.strip()]
    rows = json.loads(args.manifest.read_text())
    if args.accepted_manifest:
        rows += load_manifest(args.accepted_manifest)
    indexed = {row["sample_id"]: row for row in rows}
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    readers, traced = {}, []
    for record in audit:
        flags = record.get("history_thinking_reference_audit", {}).get("missing_after", [])
        if not flags:
            continue
        row = indexed[record["sample_id"]]
        root = row["snapshot_root"]
        if root not in readers:
            readers[root] = Reader(root)
        reader = readers[root]
        request = reader.value(row["request_ref"])
        response = reader.value(row["response_ref"])["choices"][0]["message"]
        prompt = template_messages(request["messages"])
        candidate = strip_history_thinking(prompt)
        options = {**row.get("chat_template_kwargs", {}), "tools": request.get("tools", []),
                   "tokenize": False, "add_generation_prompt": True}
        stripped_text = " ".join(processor.apply_chat_template(candidate, **options).split())
        if record["stage"] == 2:
            candidate = strip_history_thinking(project_history(prompt, record["keep_results"])[0])
        after = processor.apply_chat_template(candidate, **options)
        if sha256(after.encode()).hexdigest() != record["prompt_after_sha256"]:
            raise ValueError("Frozen projection replay differs: " + row["sample_id"])
        text = (response.get("reasoning_content") or "") + "\n" + (response.get("content") or "")
        for call in response.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            text += "\n" + (arguments if isinstance(arguments, str) else json.dumps(arguments))
        fragments = {" ".join(next(part for part in match.groups() if part).split())
                     for match in QUOTES.finditer(text)}
        wanted = {flag["sha256"] for flag in flags}
        matches = []
        normalized = " ".join(after.split())
        for fragment in sorted(fragments):
            digest = sha256(fragment.encode()).hexdigest()
            if digest not in wanted:
                continue
            sources = fragment_sources(prompt, fragment)
            encoded = json.dumps(fragment, ensure_ascii=False)[1:-1]
            matches.append({"sha256": digest, "characters": len(fragment), "sources": sources,
                "raw_nonreasoning_match": any(source["kind"] != "reasoning" for source in sources),
                "remaining_raw_sources": fragment_sources(candidate, fragment),
                "json_escaped_after_thinking_strip": encoded in stripped_text,
                "html_escaped_after_thinking_strip": html.escape(fragment) in stripped_text,
                "json_escaped_in_candidate": encoded in normalized,
                "double_json_escaped_in_candidate": json.dumps(encoded, ensure_ascii=False)[1:-1] in normalized,
                "html_escaped_in_candidate": html.escape(fragment) in normalized,
                "contains_path": bool(re.search(r'/[\w.-]+/|\.(?:drawio|png|py|json)\b', fragment)),
                "contains_geometry_numbers": len(re.findall(r'\b\d+(?:\.\d+)?\b', fragment)) >= 2,
                "contains_xml": bool(re.search(r'</?[A-Za-z][^>]*>', fragment))})
        if {item["sha256"] for item in matches} != wanted:
            raise ValueError("Cannot reproduce target quote flags: " + row["sample_id"])
        traced.append({"sample_id": row["sample_id"], "split": row["split"],
                       "closing_reply": record["closing_reply"], "projection_stage": record["stage"],
                       "image_count": row.get("image_count"), "flags": matches})
    result = {"at": time.time(), "audited_decisions_snapshot": len(audit), "flagged_decisions": len(traced),
              "by_split": dict(Counter(row["split"] for row in traced)), "records": traced,
              "audit_snapshot_sha256": sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
              "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "raw_reasoning_exported": False, "targets_choose_projection": False, "training_modified": False,
              "limitations": ["Raw-source matches may include reasoning hidden by the native template.",
                  "Exact missing text does not establish unsupported semantics; images/files/derivation may suffice."]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sources.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


def audit(args):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    rows = json.loads(args.manifest.read_text())
    readers, records = {}, []
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "decisions.jsonl").open("w") as stream:
        for row in rows:
            root = row["snapshot_root"]
            if root not in readers:
                readers[root] = Reader(root)
            reader = readers[root]
            request = reader.value(row["request_ref"])
            prompt = template_messages(request["messages"])
            protected = set().union(*result_anchors(prompt).values())
            candidate = clear_context(prompt, keep_results=args.keep_results, protected=protected)
            options = {**row.get("chat_template_kwargs", {}), "tools": request.get("tools", []),
                       "tokenize": False, "add_generation_prompt": True}
            before = processor.apply_chat_template(prompt, **options)
            after = processor.apply_chat_template(candidate, **options)
            # The future target is used only to flag audit risks, never to construct the projection.
            message = reader.value(row["response_ref"])["choices"][0]["message"]
            record = {"sample_id": row["sample_id"], "split": row["split"],
                      "prompt_before_sha256": sha256(before.encode()).hexdigest(),
                      "prompt_after_sha256": sha256(after.encode()).hexdigest(),
                      **quoted_gaps(before, after, message.get("reasoning_content") or "")}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            if len(records) % 25 == 0:
                print(json.dumps({"measured": len(records), "total": len(rows)}), flush=True)
    summary = {"at": time.time(), "records": len(records), "keep_results": args.keep_results,
               "quoted_fragments": sum(row["quoted_fragments"] for row in records),
               "supported_before": sum(row["supported_before"] for row in records),
               "missing_after": sum(len(row["missing_after"]) for row in records),
               "decisions_with_gap": sum(bool(row["missing_after"]) for row in records),
               "gap_examples": [row["sample_id"] for row in records if row["missing_after"]][:12],
               "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
               "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
               "limitations": ["Literal quotations of24-512characters only; no semantic entailment test.",
                   "Images, external files, paraphrases and short references may provide unmeasured support.",
                   "A missing quote is a review flag; no flags do not certify sufficiency."],
               "target_used_to_choose_projection": False, "training_modified": False,
               "raw_reasoning_exported": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--keep-results", type=int, default=16)
    parser.add_argument("--trace-projection", type=Path)
    parser.add_argument("--accepted-manifest", type=Path)
    args = parser.parse_args()
    trace_projection(args) if args.trace_projection else audit(args)


if __name__ == "__main__":
    main()
