"""Find conservative native-template prefix-chain candidates without decoding images."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

from .data import open_store


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def content_parts(content):
    if content is None:
        return "", ()
    if isinstance(content, str):
        return content.strip(), ()
    text, images = [], []
    for part in content:
        if part.get("type") == "text":
            text.append(part["text"])
        elif part.get("type") == "image_url":
            image = part["image_url"]["url"]
            if not isinstance(image, tuple) or image[0] != "image_blob":
                raise ValueError("Unarchived image URL")
            images.append(image)
            text.append("<|vision_start|><|image_pad|><|vision_end|>")
        else:
            raise ValueError("Unsupported content part: " + str(part.get("type")))
    return "".join(text).strip(), tuple(images)


class Reader:
    def __init__(self, root):
        self.store = open_store(root)
        self.messages = {}

    @lru_cache(maxsize=2048)
    def node(self, reference):
        return json.loads(self.store.raw(reference))

    def field(self, reference, key):
        return dict(self.node(reference)[1])[key]

    def value(self, reference):
        node = self.node(reference)
        if node[0] == "dict":
            return {key: self.value(value) for key, value in node[1]}
        if node[0] == "list":
            return [self.value(value) for value in node[1]]
        if node[0] == "image":
            return ("image_blob", node[1], node[2])
        return node[1]

    def message(self, reference):
        if reference in self.messages:
            return self.messages[reference]
        message = self.value(reference)
        role = message["role"]
        content, images = content_parts(message.get("content"))
        reasoning = ""
        if role == "assistant":
            if isinstance(message.get("reasoning_content"), str):
                reasoning = message["reasoning_content"].strip()
            elif "</think>" in content:
                reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
                content = content.split("</think>")[-1].lstrip("\n")
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments", {})
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            calls.append({"name": function["name"], "arguments": arguments})
        payload = {"role": role, "content": content, "tool_calls": calls}
        without_reasoning = digest(payload)
        payload["reasoning_content"] = reasoning
        result = {"role": role, "digest": digest(payload), "without_reasoning": without_reasoning,
                  "real_user": role == "user" and not
                  (content.startswith("<tool_response>") and content.endswith("</tool_response>")),
                  "images": images}
        self.messages[reference] = result
        return result

    def context(self, row):
        reference = row["request_ref"]
        refs = self.node(self.field(reference, "messages"))[1]
        messages = [self.message(ref) for ref in refs]
        users = [index for index, message in enumerate(messages) if message["real_user"]]
        last_user = users[-1] if users else None
        if last_user is None:
            raise ValueError("No real user message")
        projected = tuple(message["without_reasoning"] if message["role"] == "assistant" and index <= last_user
                          else message["digest"] for index, message in enumerate(messages))
        return {"refs": refs, "projected": projected, "last_user": (last_user, projected[last_user]),
                "images": tuple(image for message in messages for image in message["images"]),
                "tools": self.field(reference, "tools")}

    def completion(self, row):
        choices = self.node(self.field(row["response_ref"], "choices"))[1]
        return self.message(self.field(choices[0], "message"))["digest"]


def link_reasons(previous, current, response, previous_row, row):
    length = len(previous["refs"])
    reasons = []
    if previous["projected"] != current["projected"][:length]:
        reasons.append("projected_message_prefix_changed")
    if previous["last_user"] != current["last_user"]:
        reasons.append("last_real_user_changed")
    if previous["images"] != current["images"]:
        reasons.append("image_blob_sequence_changed")
    if previous["tools"] != current["tools"]:
        reasons.append("tools_changed")
    if len(current["refs"]) <= length or current["projected"][length] != response:
        reasons.append("previous_completion_not_appended")
    if previous_row.get("image_count") != row.get("image_count"):
        reasons.append("prepared_image_count_changed")
    return reasons


def summarize(chains, break_reasons, all_reasons, counters, elapsed):
    lengths = Counter(len(chain) for chain in chains)
    original = sum(row["token_count"] for chain in chains for row in chain)
    retained = sum(chain[-1]["token_count"] for chain in chains)
    return {"candidate_chains": len(chains), "decisions": sum(size * count for size, count in lengths.items()),
            "chain_length_histogram": dict(sorted(lengths.items())), "max_chain": max(lengths, default=0),
            "original_processed_tokens": original, "end_of_chain_processed_tokens": retained,
            "potential_input_token_reduction_factor": original / retained if retained else None,
            "primary_break_reasons": dict(break_reasons), "all_break_reasons": dict(all_reasons),
            "checks": dict(counters), "elapsed_seconds": elapsed,
            "scope": "Conservative projected-prefix candidates, not certified merges. No token/BPE/mRoPE or pixel proof.",
            "image_policy": "Same archived image-blob sequence, no base64/image decoding; image preprocessing not executed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-episodes", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["episode"]].append(row)
    chains, readers, failures = [], {}, []
    break_reasons, all_reasons, counters = Counter(), Counter(), Counter()
    started = time.monotonic()
    for number, (episode, decisions) in enumerate(episodes.items(), 1):
        if args.limit_episodes and number > args.limit_episodes:
            break
        decisions.sort(key=lambda row: row["call_id"])
        previous = previous_row = response = None
        chain = []
        for row in decisions:
            root = str(args.manifest.parent / row["snapshot_root"])
            if root not in readers:
                readers[root] = Reader(root)
            reader = readers[root]
            try:
                context = reader.context(row)
                current_response = reader.completion(row)
            except (ValueError, KeyError, TypeError) as error:
                context = current_response = None
                counters["unsupported_projection"] += 1
                if len(failures) < 16:
                    failures.append({"sample_id": row["sample_id"], "error": str(error)})
            reasons = []
            if previous_row is not None:
                counters["links_examined"] += 1
                reasons = ["unsupported_projection"] if previous is None or context is None else \
                    link_reasons(previous, context, response, previous_row, row)
                if not reasons:
                    counters["candidate_links"] += 1
                    counters["raw_message_prefix_identical"] += previous["refs"] == context["refs"][:len(previous["refs"])]
                else:
                    break_reasons[reasons[0]] += 1
                    all_reasons.update(reasons)
            if reasons and chain:
                chains.append(chain)
                chain = []
            chain.append(row)
            previous, previous_row, response = context, row, current_response
        if chain:
            chains.append(chain)
        counters["episodes"] += 1
        if number % 25 == 0:
            print(json.dumps(summarize(chains, break_reasons, all_reasons, counters,
                                       time.monotonic() - started)), flush=True)
    report = summarize(chains, break_reasons, all_reasons, counters, time.monotonic() - started)
    report.update(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  projection_failures=failures, source="prepared eligible trajectory manifest",
                  longest_chains=[{"episode": chain[0]["episode"], "length": len(chain),
                                   "first": chain[0]["sample_id"], "last": chain[-1]["sample_id"]}
                                  for chain in sorted(chains, key=len, reverse=True)[:16]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
