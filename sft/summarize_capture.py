"""Rebuild stream and image summaries from immutable raw OpenAI capture files."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from rollout.trace_proxy import images, response_summary


def summarize(directory, *, audit=(), tokenizer=None):
    seen = set()
    requests = {Path(row["raw_request"]).name: row for row in audit if row.get("raw_request")}
    for path in sorted(directory.glob("*.request.json")):
        body = path.read_bytes()
        request = json.loads(body)
        image_inputs = list(images(request))
        hashes = {item["sha256"] for item in image_inputs if item.get("inline")}
        response = path.with_name(path.name.replace(".request.json", ".response.bin"))
        raw = response.read_bytes() if response.exists() else b""
        complete = b"data: [DONE]" in raw
        # A killed looping response still contains valid SSE deltas before its incomplete tail.
        result = response_summary([raw], with_arguments=True, with_reasoning=True)
        usage = result.get("usage") or {}
        state = requests.get(path.name, {}).get("state_prefix") or {}
        counts = None
        if tokenizer:
            counts = {"reasoning": len(tokenizer.encode(result["reasoning_text"], add_special_tokens=False)),
                      "content": len(tokenizer.encode(result["content_text"], add_special_tokens=False)),
                      "tool_arguments": sum(len(tokenizer.encode(tool.get("arguments", ""),
                          add_special_tokens=False)) for tool in result["tool_calls"].values()),
                      "scope": "Separately retokenized text, not original decoder token counts"}
        yield {"request": str(path), "request_sha256": sha256(body).hexdigest(),
            "model": request.get("model"), "requested_max_tokens": request.get("max_tokens"),
            "prompt_tokens": usage.get("prompt_tokens", state.get("input_tokens")),
            "prompt_token_source": "provider_usage" if "prompt_tokens" in usage else
                "archived_state_prefix" if "input_tokens" in state else "unavailable",
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "retokenized": counts,
            "state_prefix": state,
            "messages": len(request.get("messages", [])), "image_inputs": image_inputs,
            "new_image_hashes": sorted(hashes - seen), "response_bytes": len(raw),
            "response_sha256": sha256(raw).hexdigest(),
            "complete_stream": complete, "response": result}
        seen.update(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="Original request/response audit JSONL for exact input counts")
    parser.add_argument("--tokenizer", type=Path, help="Optional local tokenizer; no model forward is run")
    args = parser.parse_args()
    audit = [json.loads(line) for line in args.audit.read_text().splitlines() if line.strip()] if args.audit else []
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = list(summarize(args.directory, audit=audit, tokenizer=tokenizer))
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"requests": len(rows), "complete_streams": sum(row["complete_stream"] for row in rows),
                      "new_image_inputs": sum(len(row["new_image_hashes"]) for row in rows)}))


if __name__ == "__main__":
    main()
