"""Synthetic variant generator — owner: Member 3.

Ported from tools/generate_synthetic_dev.py (Member 3, commit 4b4b6d7) into the pipeline, so that
`python -m src.main ...` generates the synthetic dev set in one command.

Contract with the pipeline:
    generate_dev_variants(ctx, data) -> DataFrame, one row per generated image, with GENERATED_COLUMNS.
    Images are written to ctx.out("generated/<image_id>.jpg").

What is generated, from every original with split == "dev" (never from holdout):
  * night / glare / fog / blur at severity 0, 1, 2 (src/common/corruptions.py, parameter tables cited there)
  * one "erase" variant: one of the 3 largest labelled objects is in-painted away, its label is KEPT
    (this is the label-invalidating edit of usability rule 3)
Glare: the centre is chosen here (half of the time on a labelled object, half off every object) and
written to `params`, so usability rule 2 ("glare centre inside a box") can be checked afterwards.
Deterministic: every image gets its own seed derived from ctx.seed; same run twice -> same sha256.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..common import corruptions
from ..common.context import RunContext

GENERATED_COLUMNS = ["image_id", "path", "parent_image", "transform", "severity", "seed", "params"]
TRANSFORMS = ["night", "glare", "fog", "blur"]
SEVERITIES = [0, 1, 2]


def inside_any_box(cx: float, cy: float, boxes: pd.DataFrame) -> bool:
    if boxes.empty:
        return False
    return bool(((boxes.x <= cx) & (cx <= boxes.x + boxes.w) & (boxes.y <= cy) & (cy <= boxes.y + boxes.h)).any())


def glare_centre(rng: np.random.Generator, boxes: pd.DataFrame, W: int, H: int, on_object: bool) -> tuple[float, float]:
    if on_object and len(boxes):
        b = boxes.iloc[int(rng.integers(len(boxes)))]
        return float(b.x + b.w / 2), float(b.y + b.h / 2)
    for _ in range(200):  # a point outside every labelled box
        cx, cy = float(rng.uniform(0.05, 0.95) * W), float(rng.uniform(0.05, 0.6) * H)
        if not inside_any_box(cx, cy, boxes):
            return cx, cy
    return float(W * 0.5), float(H * 0.05)


def erase_object(img: np.ndarray, box) -> np.ndarray:
    """Remove one object with OpenCV in-painting (Telea); the label stays in objects.csv."""
    H, W = img.shape[:2]
    pad = 3
    x0, y0 = max(0, int(box.x) - pad), max(0, int(box.y) - pad)
    x1, y1 = min(W, int(box.x + box.w) + pad), min(H, int(box.y + box.h) + pad)
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    out = cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask, 7, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def generate_dev_variants(ctx: RunContext, data: dict) -> pd.DataFrame:
    originals = data["originals"]
    dev = originals[originals.split == "dev"]
    objects = data["objects"]
    ctx.stats["n_dev_originals"] = len(dev)
    rows: list[dict] = []
    for _, o in dev.iterrows():
        pid = str(o.image_id)
        img = np.asarray(Image.open(ctx.input_dir / str(o.path)).convert("RGB"))
        H, W = img.shape[:2]
        boxes = objects[objects.image_id == pid]
        jobs: list[tuple[str, int, dict]] = [(t, s, {}) for t in TRANSFORMS for s in SEVERITIES]
        if len(boxes):
            jobs.append(("erase", 1, {}))
        for t, s, params in jobs:
            seed = int(ctx.seed * 100_000 + len(rows))
            rng = np.random.default_rng(seed)
            if t == "glare" and s > 0:
                on = bool(rng.random() < 0.5)
                cx, cy = glare_centre(rng, boxes, W, H, on)
                params = {"center": [round(cx, 1), round(cy, 1)], "radius_frac": 0.35,
                          "strength": corruptions.GLARE_STRENGTH[s]}
                new = corruptions.glare(img, severity=s, center=(cx, cy))
            elif t == "erase":
                big = boxes.assign(a=boxes.w * boxes.h).sort_values("a", ascending=False).head(3)
                victim = big.iloc[int(rng.integers(len(big)))]
                params = {"erased_obj_id": str(victim.obj_id), "method": "cv2.inpaint TELEA r=7 pad=3"}
                new = erase_object(img, victim)
            else:
                new = corruptions.apply(t, img, severity=s, seed=seed)
            iid = f"syn_{pid}_{t}_s{s}"
            rel = f"generated/{iid}.jpg"
            Image.fromarray(new).save(ctx.out(rel), quality=92)
            rows.append({"image_id": iid, "path": rel, "parent_image": pid, "transform": t, "severity": s,
                         "seed": seed, "params": json.dumps(params)})
    return pd.DataFrame(rows, columns=GENERATED_COLUMNS)
