"""Rebuild stream and image summaries from immutable raw OpenAI capture files."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from rollout.trace_proxy import images, response_summary


def summarize(directory):
    seen = set()
    for path in sorted(directory.glob("*.request.json")):
        body = path.read_bytes()
        request = json.loads(body)
        image_inputs = list(images(request))
        hashes = {item["sha256"] for item in image_inputs if item.get("inline")}
        response = path.with_name(path.name.replace(".request.json", ".response.bin"))
        raw = response.read_bytes() if response.exists() else b""
        complete = b"data: [DONE]" in raw
        yield {"request": str(path), "request_sha256": sha256(body).hexdigest(),
            "messages": len(request.get("messages", [])), "image_inputs": image_inputs,
            "new_image_hashes": sorted(hashes - seen), "response_bytes": len(raw),
            "complete_stream": complete,
            "response": response_summary([raw], with_arguments=True) if complete else None}
        seen.update(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(summarize(args.directory))
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"requests": len(rows), "complete_streams": sum(row["complete_stream"] for row in rows),
                      "new_image_inputs": sum(len(row["new_image_hashes"]) for row in rows)}))


if __name__ == "__main__":
    main()
