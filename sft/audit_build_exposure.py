"""Audit which observations and construction actions the SFT sampler exposes."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from .check_prefix_chains import Reader
from .data import load_manifest
from .state_prefix import ERROR, text


PROGRESS = {"code_file_changed", "native_xml_changed", "candidate_render_changed"}
READ_TOOLS = {"read_image", "read"}


def phases(reader):
    ledger = reader.store.root / "ledger.sqlite"
    groups = defaultdict(list)
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for row in connection.execute(
            "select episode,stage,id,request,artifacts from calls "
            "where role='generator' and request is not null and artifacts is not null order by id"
        ):
            groups[row[:2]].append(row[2:])
    return groups


def outcome(reader, reference, ids):
    results = {}
    for message_ref in reversed(reader.node(reader.field(reference, "messages"))[1]):
        fields = dict(reader.node(message_ref)[1])
        if "tool_call_id" not in fields:
            continue
        ident = reader.value(fields["tool_call_id"])
        if ident in ids and reader.value(fields["role"]) == "tool":
            results[ident] = reader.value(message_ref)
        if len(results) == len(ids):
            return results
    return results


def read_signature(calls, files, results):
    signatures = []
    for call in calls:
        function = call["function"]
        args = function["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        path = args.get("file_path") or args.get("path") or ""
        digest = files.get(path.removeprefix("/case/"))
        result = results.get(call["id"])
        if not digest or not result or ERROR.search(text(result)) or result.get("is_error"):
            return []
        signature = json.dumps([function["name"], args, digest, result["content"]], sort_keys=True)
        signatures.append(sha256(signature.encode()).hexdigest())
    return signatures


def audit_phase(reader, archived, rows, effects, parts, consumed):
    locations = {number: index for index, (number, _, _) in enumerate(archived)}
    seen_reads, details = set(), []
    for row in rows:
        number, ident = row["call_id"], row["sample_id"]
        index = locations[number]
        effect = effects[ident]
        before = effect["xml_present_before"]
        message = reader.value(row["response_ref"])["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        kind = effect["artifact_effect"]
        signatures = []
        if before is False and calls and all(c["function"]["name"] in READ_TOOLS for c in calls):
            if index + 1 < len(archived):
                files = reader.value(archived[index][2])
                results = outcome(reader, archived[index + 1][1], {c["id"] for c in calls})
                signatures = read_signature(calls, files, results)
        duplicate = bool(signatures and set(signatures) <= seen_reads)
        seen_reads.update(signatures)
        predecessor = index + 1 < len(archived) and effects.get(
            f'{row["episode"]}:{archived[index + 1][0]}', {}).get("artifact_effect") in PROGRESS
        if kind in PROGRESS:
            category = kind
        elif not calls and (message.get("content") or "").strip() and index == len(archived) - 1:
            category = "terminal_reply"
        elif before is not False:
            category = "post_xml_or_unknown"
        elif duplicate:
            category = "exact_prior_read"
        elif signatures:
            category = "new_read_signature"
        else:
            category = "other_pre_xml"
        details.append({
            "id": ident, "episode": row["episode"], "stage": row["stage"],
            "category": category, "archive_decision_ordinal": index + 1,
            "thinking_tokens": parts[ident]["partition_tokens"].get("reasoning", 0),
            "target_tokens": row["completion_tokens"], "seen_through_218": ident in consumed,
            "exact_read_signatures": signatures if duplicate else [],
            "candidate_for_downsampling": duplicate and kind == "no_case_file_change"
                and not predecessor and not (message.get("content") or "").strip(),
        })
    return details


def totals(rows):
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["category"]].update(decisions=1, thinking_tokens=row["thinking_tokens"],
            target_tokens=row["target_tokens"], downsampling_candidates=int(row["candidate_for_downsampling"]))
    return {key: dict(value) for key, value in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "effects", "partitions", "consumed", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()
    effects = {row["sample_id"]: row for row in load_manifest(args.effects)}
    parts = {row["sample_id"]: row for row in load_manifest(args.partitions)}
    consumed = {row["sample_id"] for row in load_manifest(args.consumed)}
    groups = defaultdict(list)
    for row in load_manifest(args.manifest, "train"):
        groups[(row["snapshot_root"], row["episode"], row["stage"])].append(row)
    readers, indexes, details = {}, {}, []
    for position, ((root, episode, stage), rows) in enumerate(sorted(groups.items()), 1):
        if root not in readers:
            readers[root] = Reader(root)
            indexes[root] = phases(readers[root])
        details.extend(audit_phase(readers[root], indexes[root][(episode, stage)],
            sorted(rows, key=lambda row: row["call_id"]), effects, parts, consumed))
        if position % 25 == 0:
            print(json.dumps({"phases": position, "decisions": len(details)}), flush=True)
    report = {
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "decisions": len(details), "phases": len(groups),
        "train": totals(details), "seen_through_218": totals(row for row in details if row["seen_through_218"]),
        "model_calls": 0, "active_manifest_changed": False,
    }
    with (args.out / "decisions.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in details)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
