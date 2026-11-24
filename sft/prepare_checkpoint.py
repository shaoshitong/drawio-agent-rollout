"""Prepare a hard-linked full-checkpoint export with the original base preprocessing."""

import argparse
import json
import math
import os
from pathlib import Path
import struct

from .merge_lora import sha256


PREPROCESSING = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja", "preprocessor_config.json",
    "video_preprocessor_config.json", "processor_config.json",
}


def tensor_header(path):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        return {name: value for name, value in json.loads(stream.read(length)).items() if name != "__metadata__"}


def prepare(base, checkpoint, output):
    tensors = tensor_header(checkpoint / "model.safetensors")
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {name: value for shard in sorted(set(index.values()))
                for name, value in tensor_header(base / shard).items()}
    missing = sorted(expected.keys() - tensors.keys())
    if any(not name.startswith("mtp.") for name in missing) or tensors.keys() - expected.keys():
        raise ValueError("Full checkpoint differs from the base model's required tensor names")
    if any(value["shape"] != expected[name]["shape"] for name, value in tensors.items()):
        raise ValueError("Full checkpoint changes a base tensor shape")
    output.mkdir(parents=True, exist_ok=False)
    sources = {path.name: path for path in base.iterdir() if path.is_file() and path.name in PREPROCESSING}
    sources.update({name: checkpoint / name for name in
                    ("model.safetensors", "config.json", "generation_config.json", "trainer_state.json", "full-phase.json")})
    files = {}
    for name, source in sources.items():
        os.link(source, output / name)
        files[name] = {"bytes": source.stat().st_size, "sha256": sha256(source)}
        print(json.dumps({"prepared": name, "bytes": files[name]["bytes"]}), flush=True)
    record = {"global_step": json.loads((checkpoint / "trainer_state.json").read_text())["global_step"],
              "checkpoint": str(checkpoint), "base_preprocessing": str(base), "files": files,
              "tensors": len(tensors), "parameters": sum(math.prod(value["shape"]) for value in tensors.values()),
              "weight_dtypes": sorted({value["dtype"] for value in tensors.values()}),
              "omitted_optional_mtp_tensors": missing, "optimizer_included": False}
    (output / "inference-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.base, args.checkpoint, args.output)
    print(json.dumps({key: record[key] for key in ("global_step", "tensors", "parameters", "weight_dtypes")}), flush=True)


if __name__ == "__main__":
    main()
