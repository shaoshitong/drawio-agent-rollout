"""Export every training-loss observation from an SFT log."""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile


def read_metrics(log):
    rows = []
    for line in log.read_text().replace("\r", "\n").splitlines():
        if not line.startswith("{'loss':"):
            continue
        item = ast.literal_eval(line)
        row = {key: float(item[key]) for key in ("loss", "grad_norm", "learning_rate")}
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"Nonfinite training metric at observation {len(rows) + 1}")
        row["step"] = int(item.get("step", len(rows) + 1))
        rows.append(row)
    if not rows:
        raise ValueError(f"No training-loss observations in {log}")
    for index, row in enumerate(rows):
        row["moving_average_5"] = (
            sum(item["loss"] for item in rows[index - 4:index + 1]) / 5
            if index >= 4 else None
        )
    return rows


def draw_curve(rows, output, stem, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(12.8, 6.4), layout="constrained")
    axis.plot([row["step"] for row in rows], [row["loss"] for row in rows],
              color="#2563eb", linewidth=1.7, marker="o", markersize=4,
              label="Training loss: every update")
    smooth = [row for row in rows if row["moving_average_5"] is not None]
    axis.plot([row["step"] for row in smooth], [row["moving_average_5"] for row in smooth],
              color="#b45309", linewidth=2.4, linestyle="--", label="Trailing 5-update mean")
    first, last = rows[0]["step"], rows[-1]["step"]
    axis.set(xlim=(first, max(first + 1, last)),
             ylim=(0, max(1.65, max(row["loss"] for row in rows) * 1.1)),
             xlabel="Optimizer update", ylabel="Training cross-entropy loss",
             title=f"{title} | all {len(rows)} completed updates")
    interval = max(5, math.ceil((last - first) / 12 / 5) * 5)
    ticks = range(math.ceil(first / interval) * interval, last + 1, interval)
    axis.set_xticks(sorted({first, last, *ticks}))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower left", frameon=False)
    axis.annotate(f"{rows[-1]['loss']:.3f}", xy=(last, rows[-1]["loss"]),
                  xytext=(-8, 14), textcoords="offset points", ha="right")
    figure.supxlabel("All raw observations are shown. Different batches per update.", fontsize=10)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=180)
    plt.close(figure)


def export(log, output, title, total_steps):
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / "mpl-cache")
    rows = read_metrics(log)
    stem = f"loss-steps-{rows[0]['step']}-{rows[-1]['step']}"
    with (output / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss", "moving_average_5",
                                                    "grad_norm", "learning_rate"])
        writer.writeheader()
        writer.writerows(rows)
    report = {"generated_at": time.time(), "completed_updates": rows[-1]["step"],
              "total_updates": total_steps, "source": str(log.resolve()), "loss_rows": rows}
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    draw_curve(rows, output, stem, title)
    names = [f"{stem}.{suffix}" for suffix in ("png", "svg", "pdf", "csv", "json")]
    files = {name: {"bytes": (output / name).stat().st_size,
                    "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
             for name in names}
    manifest = {"completed_updates": rows[-1]["step"], "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output / "loss-curves.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in [*names, "manifest.json"]:
            bundle.write(output / name, name)
    return {"completed_updates": rows[-1]["step"], "stem": stem,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Trajectory SFT")
    parser.add_argument("--total-steps", type=int)
    args = parser.parse_args()
    print(json.dumps(export(args.log, args.output, args.title, args.total_steps)))


if __name__ == "__main__":
    main()
