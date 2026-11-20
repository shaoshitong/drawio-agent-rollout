"""Lazily resolve immutable teacher decisions into multimodal SFT examples."""

from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path

if __package__:
    from .export_sft import materialize as teacher_example
    from .store import Store
else:
    from export_sft import materialize as teacher_example
    from store import Store


def load_manifest(path, split=None):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [row for row in rows if split is None or row["split"] == split]


def token_sha256(ids):
    return sha256(ids.detach().cpu().numpy().astype("<i8", copy=False).tobytes()).hexdigest()


class ReadOnlyStore(Store):
    """Bound cached object bytes while leaving immutable snapshots untouched."""

    def __init__(self, root, cache_bytes=32 * 1024**2):
        root = Path(root)
        self.root = root / "archive" if (root / "archive").is_dir() else root
        self.objects = self.root / "objects"
        if not self.objects.is_dir():
            raise FileNotFoundError(self.objects)
        self.cache, self.cache_bytes, self.used_bytes = OrderedDict(), cache_bytes, 0

    def raw(self, reference):
        if reference in self.cache:
            self.cache.move_to_end(reference)
            return self.cache[reference]
        value = super().raw(reference)
        if len(value) <= self.cache_bytes // 4:
            while self.used_bytes + len(value) > self.cache_bytes:
                self.used_bytes -= len(self.cache.popitem(last=False)[1])
            self.cache[reference] = value
            self.used_bytes += len(value)
        return value


def open_store(root):
    return ReadOnlyStore(root)


def materialize(row, base_dir, store=None):
    root = Path(base_dir) / row["snapshot_root"]
    example = teacher_example(store or open_store(root), row)
    if not row.get("supervision_eligible", True) or not example["supervision_eligible"]:
        raise ValueError("Interrupted decision is not an SFT target: " + row["sample_id"])
    for message in example["prompt"] + example["completion"]:
        if message.get("content") is None:
            message["content"] = ""
    tools = example["tools"]
    if isinstance(tools, str):
        tools = json.loads(tools)
    if not isinstance(tools, list):
        raise TypeError("Archived tools must be a JSON list")
    return {
        "prompt": example["prompt"],
        "completion": example["completion"],
        "images": example["images"],
        "sample_id": row["sample_id"],
        "kind": "teacher_decision",
        "chat_template_kwargs": {**row.get("chat_template_kwargs", {}), "tools": tools},
    }


class ManifestDataset:
    """Keep references in memory; load each decision and its images on demand."""

    def __init__(self, manifest, split=None, store_path=None):
        self.path = Path(manifest).resolve()
        self.rows = load_manifest(self.path, split)
        self.store_path = Path(store_path) if store_path else None
        self.stores = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        root = self.store_path or self.path.parent / row["snapshot_root"]
        key = str(root)
        if key not in self.stores:
            self.stores[key] = open_store(root)
        return materialize(row, self.path.parent, self.stores[key])
