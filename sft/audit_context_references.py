"""Flag exact quoted evidence lost by a fixed history-only context projection."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from .audit_context_budget import clear_context, result_anchors, template_messages
from .check_prefix_chains import Reader


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
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
