"""Read the existing, bounded SFT comparison export without copying its data.

The source contains ten selected training decisions from eight episodes. These
are original and rewritten supervision pairs, not model-generated checkpoint
outputs. In particular, this export predates the current state-v3 curriculum.
"""

import base64
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote


SFT_EXPORT = Path(
    "/Users/shaoshitong/Documents/deepseek-harness/default-workspace/"
    "sft-forward-compare/samples.json.gz"
)
SFT_EXPORTER = SFT_EXPORT.with_name("extract.py")
REMOTE_SOURCE = (
    "/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121/runs/"
    "causal-cot-20261122/production-v2/episodes"
)
BAND_LABELS = {"small": "小规模", "mid": "中规模", "large": "大规模"}


@lru_cache(maxsize=1)
def _read_export(modified_ns, size):
    """Use file metadata as a cache key so a replaced export is reread."""
    del modified_ns, size
    with gzip.open(SFT_EXPORT, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("SFT comparison export is missing its samples list")
    ids = [f"{row['episode']}:{row['call_id']}" for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("SFT comparison export contains duplicate decision IDs")
    return data


def _export():
    info = SFT_EXPORT.stat()
    return _read_export(info.st_mtime_ns, info.st_size)


def _category(row):
    calls = row.get("old_target_tool_calls", [])
    names = {call.get("name", "") for call in calls}
    arguments = "\n".join(str(call.get("arguments", "")) for call in calls)
    if re.search(r"\.drawio|<mxfile|<mxGraphModel", arguments, re.I):
        return "xml_related"
    if re.search(r"crop|裁图", arguments, re.I):
        return "crop_related"
    if names & {"read_image", "view_image"}:
        return "image_read"
    return "other_tool" if calls else "assistant_only"


def _catalog_row(row):
    identity = f"{row['episode']}:{row['call_id']}"
    original_tokens = row.get("old_input_tokens")
    rewritten_tokens = row.get("new_input_tokens")
    return {
        "id": identity,
        "title": (
            f"{BAND_LABELS.get(row.get('band'), row.get('band', ''))}"
            f" · {row.get('final_components', '—')} 组件"
            f" · 决策 #{row['call_id']}"
        ),
        "category": _category(row),
        "episode": row["episode"],
        "call_id": row["call_id"],
        "stage": row.get("stage"),
        "step": None,
        "split": row.get("split"),
        "band": row.get("band"),
        "final_components": row.get("final_components"),
        "original_tokens": original_tokens,
        "rewritten_tokens": rewritten_tokens,
        "original_reasoning_tokens": row.get("old_reasoning_tokens"),
        "rewritten_reasoning_tokens": row.get("new_reasoning_tokens"),
        "original_message_count": len(row.get("old_messages", [])),
        "rewritten_message_count": len(row.get("new_messages", [])),
        "source": "production-v2 · own-state-prefix-v2 · 10 条既有导出",
        "current_curriculum": False,
        "comparison_kind": "supervision_rewrite",
    }


def get_sft_catalog():
    """Return small, stable catalog rows, without images or message bodies."""
    return [_catalog_row(row) for row in _export()["samples"]]


def _sample(identity):
    for row in _export()["samples"]:
        if f"{row['episode']}:{row['call_id']}" == identity:
            return row
    raise KeyError(identity)


def _tool_call(call):
    function = call.get("function") or {}
    result = {
        "function": {
            "name": function.get("name") or call.get("name") or "unknown",
            "arguments": function.get("arguments", call.get("arguments", "")),
        }
    }
    if call.get("id") is not None:
        result["id"] = call["id"]
    return result


def _message(message):
    result = {
        "role": message.get("role") or "unknown",
        "content": message.get("text", message.get("content", "")),
        "reasoning": message.get("reasoning", ""),
        "tool_calls": [_tool_call(call) for call in message.get("tool_calls", [])],
    }
    if message.get("tool_call_id") is not None:
        result["tool_call_id"] = message["tool_call_id"]
    if message.get("images"):
        result["image_metadata"] = message["images"]
    return result


def _side(row, prefix):
    calls = row.get(prefix + "_target_tool_calls", [])
    tool_calls_exported = prefix == "old" or bool(calls)
    return {
        "messages": [_message(message) for message in row.get(prefix + "_messages", [])],
        "reasoning": row.get(prefix + "_target", ""),
        "content": "",
        "content_available": False,
        "tool_calls": [_tool_call(call) for call in calls],
        "tool_calls_available": tool_calls_exported,
        "answer_fields_unchanged_proof": row.get("proof", {}).get("answer_fields_unchanged"),
        "input_tokens": row.get(prefix + "_input_tokens"),
        "reasoning_tokens": row.get(prefix + "_reasoning_tokens"),
        "image_count": row.get(prefix + "_image_count"),
    }


def get_sft_detail(identity):
    """Return complete text present in the export for one supervision pair."""
    row = _sample(identity)
    generated = _export().get("generated_at")
    images = []
    for key, title in (("reference_image", "参考图"), ("render_image", "既有渲染")):
        image = row.get(key)
        if image:
            images.append({
                "id": key,
                "title": title,
                "url": f"/api/sft/{quote(identity, safe='')}/images/{key}",
                "mime": image["mime"],
                "original": image.get("original"),
                "display": image.get("out"),
                "sha256": image.get("sha256"),
                "source": image.get("name"),
            })
    return {
        **_catalog_row(row),
        "original": _side(row, "old"),
        "rewritten": _side(row, "new"),
        "description": (
            "同一教师决策的原始历史输入与重写后的监督输入，并列展示导出中保存的完整消息、"
            "当前 reasoning 和工具参数。这些是监督数据对，不是训练后模型的生成结果。"
        ),
        "provenance": {
            "local_export": str(SFT_EXPORT),
            "exporter": str(SFT_EXPORTER),
            "remote_source": REMOTE_SOURCE,
            "generated_at": generated,
            "generated_at_utc": (
                datetime.fromtimestamp(generated, timezone.utc).isoformat()
                if generated is not None else None
            ),
            "prefix": row.get("prefix", {}),
            "proof": row.get("proof", {}),
            "render_source": row.get("render_source"),
            "render_label": row.get("render_label"),
            "limits": [
                "既有的 10 个选定决策、8 条 episode，不能作为全量数据的代表性评估。",
                "导出使用 own-state-prefix-v2，早于当前 state-v3 curriculum。",
                "导出未保存当前目标的 content 字段，空字段表示未导出。",
                "导出的新目标工具数组均为空，尽管保留证明为 true，须标为未导出而非无工具动作。",
                "历史消息的工具返回 call ID 未保留，不能凭相邻位置断言配对。",
                "图片为已有的长边不超过 900px 的 WebP 预览，非原始全分辨率图。",
                "只有参考图和末次渲染预览，历史中的其他图片仅保留标签。",
                "history 与 observations 辅助列表在原导出中最多保留 60 项。",
            ],
        },
        "images": images,
        "history_rewrites": row.get("history", []),
        "observations": row.get("observations", []),
    }


def get_sft_image(identity, key):
    """Return a verified local preview as (bytes, MIME type)."""
    if key not in ("reference_image", "render_image"):
        raise KeyError(key)
    image = _sample(identity).get(key)
    if not image:
        raise KeyError(key)
    raw = base64.b64decode(image["b64"], validate=True)
    if image.get("sha256") and hashlib.sha256(raw).hexdigest() != image["sha256"]:
        raise ValueError("SFT image preview checksum mismatch")
    return raw, image["mime"]
