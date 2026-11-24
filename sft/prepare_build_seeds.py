"""Export pre-construction training states for gold-blind local rollouts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .check_prefix_chains import Reader
from .data import load_manifest
from .history import strip_history_thinking
from .state_context import StateContext
from .state_prefix import resolve_images, snapshot_before_request, tool_cycles


def earliest_seeds(manifest, eligibility):
    source = {row["sample_id"]: row for row in load_manifest(manifest, "train")}
    chosen = {}
    for item in load_manifest(eligibility):
        if item["sample_id"] not in source or item["category"] != "requires_causal_build_redo":
            continue
        row = source[item["sample_id"]]
        if row["stage"] != 1 or row["request_ref"] != item["request_ref"]:
            raise ValueError("Eligibility does not match the archived first-stage request")
        previous = chosen.get(row["episode"])
        if previous is None or row["call_id"] < previous["call_id"]:
            chosen[row["episode"]] = row
    return sorted(chosen.values(), key=lambda row: (row["final_components"], row["episode"]))


def seed(row, reader, state, folder):
    request = reader.value(row["request_ref"])
    files, snapshot = snapshot_before_request(row, reader.store)
    if any(Path(name).suffix == ".drawio" for name in files):
        raise ValueError("Seed already includes a candidate XML")
    read_file = lambda name: reader.store.raw(files[name])
    messages, proof = state.build(request, files, read_file, 1, row["request_ref"], reader.store.raw)
    if messages is None or messages != strip_history_thinking(messages):
        raise ValueError("Seed is not a valid history-free-thinking prefix")
    folder.mkdir()
    copied = {}
    for name, digest in sorted(files.items()):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid archived workspace path")
        raw = read_file(name)
        if sha256(raw).hexdigest() != digest:
            raise ValueError("Export would change the archived file")
        destination = folder / "case" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        copied[name] = digest
    # This is the only request sent to the model. Provenance stays in the receipt.
    body = {"messages": messages, "tools": request["tools"]}
    raw = json.dumps(body, ensure_ascii=False).encode()
    (folder / "request.json").write_bytes(raw)
    inherited = resolve_images(strip_history_thinking(request["messages"]), reader.store.raw)
    past = json.dumps([message for cycle in tool_cycles(inherited) for message in cycle],
                      ensure_ascii=False).encode()
    (folder / "past.json").write_bytes(past)
    receipt = {"seed_id": row["sample_id"], "episode": row["episode"], "split": row["split"],
        "components_for_queue_order": row["final_components"], "request_sha256": sha256(raw).hexdigest(),
        "past_sha256": sha256(past).hexdigest(),
        "archived_request_ref": row["request_ref"], "snapshot": snapshot, "copied_files": copied,
        "input_tokens": proof["input_tokens"], "state_policy": proof["policy"],
        "prior_tool_cycles": len(tool_cycles(request["messages"])), "history_thinking_messages": 0,
        "archived_current_response_read": False, "gold_available": False}
    (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "eligibility", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    state = StateContext(AutoProcessor.from_pretrained(args.model, local_files_only=True))
    rows = earliest_seeds(args.manifest, args.eligibility)
    args.out.mkdir()
    receipts = []
    for row in rows[:args.count]:
        folder = args.out / row["episode"]
        receipt = seed(row, Reader(row["snapshot_root"]), state, folder)
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    (args.out / "summary.json").write_text(json.dumps({
        "eligible_train_episodes": len(rows), "model_calls": 0, "seeds": receipts,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
