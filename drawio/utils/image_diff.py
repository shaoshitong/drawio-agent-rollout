"""Locate image differences without gold XML. Bounds use target PNG pixels."""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def on_white(path):
    with Image.open(path) as source:
        im = source.convert("RGBA")
    white = Image.new("RGBA", im.size, "white")
    white.alpha_composite(im)
    return white.convert("RGB")


def registration(target, candidate, radius=8):
    # Match the dominant extent with a uniform resize. Fit a small translation
    # using blurred ink, so the alignment is less sensitive to a one-pixel stroke.
    axis = int(target.height > target.width)
    scale = target.size[axis] / candidate.size[axis]
    src = candidate.resize(tuple(max(1, round(v * scale)) for v in candidate.size), Image.Resampling.LANCZOS)
    w, h = target.size
    target_ink = 1 - np.array(target.convert("L"), dtype=np.float32) / 255
    src_ink = 1 - np.array(src.convert("L"), dtype=np.float32) / 255
    a = gaussian_filter(target_ink, 2)[::2, ::2]
    b = gaussian_filter(src_ink, 2)
    fit = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            padded = np.zeros((h, w), dtype=np.float32)
            x0, y0, x1, y1 = max(0, dx), max(0, dy), min(w, src.width + dx), min(h, src.height + dy)
            if x1 > x0 and y1 > y0:
                padded[y0:y1, x0:x1] = b[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
            score = float(np.mean((a - padded[::2, ::2]) ** 2))
            fit.append((score, abs(dx) + abs(dy), dx, dy))
    best = min(fit)
    return dict(
        dx=best[2],
        dy=best[3],
        scale=scale,
        axis=axis,
        radius=radius,
        fit_ink_mse=best[0],
        fit_ink_mse_at_zero_shift=next(z[0] for z in fit if z[2:] == (0, 0)),
        touches_search_boundary=max(abs(best[2]), abs(best[3])) == radius,
    )


def features(target, candidate, transform):
    frame = Image.new("RGB", target.size, "white")
    src = candidate.resize(
        tuple(max(1, round(v * transform["scale"])) for v in candidate.size), Image.Resampling.LANCZOS
    )
    frame.paste(src, (transform["dx"], transform["dy"]))
    a, b = (np.asarray(im, dtype=np.float32) / 255 for im in (target, frame))
    fa, fb = np.any(a < 0.88, axis=2), np.any(b < 0.88, axis=2)
    err = np.mean((a - b) ** 2, axis=2)
    # Use lightly blurred error only for ranking; raw MSE is still reported.
    smooth_a, smooth_b = (gaussian_filter(arr, (0.8, 0.8, 0)) for arr in (a, b))
    smooth = np.mean((smooth_a - smooth_b) ** 2, axis=2)
    return frame, err, smooth, fa, fb


def score(bbox, err, smooth, fa, fb):
    x0, y0, x1, y1 = bbox
    sl = (slice(y0, y1), slice(x0, x1))
    a, b = fa[sl], fb[sl]
    union = a | b
    coverage = float(union.mean())
    smooth_fg = float(smooth[sl][union].mean()) if union.any() else 0.0
    return dict(
        bbox=list(bbox),
        mse=float(err[sl].mean()),
        foreground_mse=float(err[sl][union].mean()) if union.any() else 0.0,
        ranking_score=smooth_fg * coverage**0.5,
        foreground_coverage=coverage,
        target_only_ink_px=int((a & ~b).sum()),
        candidate_only_ink_px=int((b & ~a).sum()),
        foreground_iou=float((a & b).sum() / max(1, union.sum())),
    )


def overlap(a, b):
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    area = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
    return area / max(1, min((x1 - x0) * (y1 - y0), (u1 - u0) * (v1 - v0)))


def positions(n, window, stride):
    return sorted(set(range(0, max(1, n - window + 1), stride)) | {max(0, n - window)})


def locate(w, h, metrics):
    proposals = []
    for window, stride in [(48, 24), (80, 40)]:
        for y in positions(h, window, stride):
            for x in positions(w, window, stride):
                item = score((x, y, min(w, x + window), min(h, y + window)), *metrics)
                if item["foreground_coverage"] >= 0.08 and item["ranking_score"] > 1e-6:
                    proposals.append(item)
    selected = []
    for item in sorted(proposals, key=lambda p: -p["ranking_score"]):
        if all(overlap(item["bbox"], old["bbox"]) < 0.25 for old in selected):
            selected.append(dict(id=f"P{len(selected) + 1}", **item))
            if len(selected) >= 5:
                break
    return selected
