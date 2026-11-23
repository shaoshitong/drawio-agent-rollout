"""Compute the claude2 timing metric T from a stamped gate-1 run."""
import json, re, statistics, sys
from pathlib import Path

out, manifest = Path(sys.argv[1]), Path(sys.argv[2])
stamps = []
for line in (out / "train-stamped.log").read_text(errors="replace").splitlines():
    m = re.match(r"^(\d+\.\d+) \{'loss':", line)
    if m:
        stamps.append(float(m.group(1)))
rows = [json.loads(l) for l in manifest.read_text().splitlines()]
mean_tokens = statistics.mean(r["token_count"] for r in rows)
per_update = 32
batches = {}
for path in sorted((out / "check-run").glob("batches-rank-*.jsonl")):
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        batches[rec["batch"]] = batches.get(rec["batch"], 0) + sum(s["sequence_tokens"] for s in rec["samples"])
result = {"loss_steps": len(stamps), "mean_manifest_tokens": mean_tokens,
          "mean_tokens_per_update": mean_tokens * per_update}
if len(stamps) >= 2:
    step2_seconds = stamps[1] - stamps[0]
    tokens_update2 = sum(v for k, v in batches.items() if 9 <= k <= 16) or None
    result["step2_seconds"] = step2_seconds
    result["tokens_update2"] = tokens_update2
    if tokens_update2:
        T = step2_seconds * (mean_tokens * per_update) / tokens_update2 / 60
        result["T_minutes"] = T
        band = "fixed" if T <= 5 else ("bs2ga4" if T <= 9 else "downsample")
        result["band"] = band
print(json.dumps(result, ensure_ascii=False, indent=1))
