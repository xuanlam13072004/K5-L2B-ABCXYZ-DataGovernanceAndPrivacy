"""Quality gate — owner: Member 4.

Contract with the pipeline:
    predict(ctx, data) -> DataFrame with one row per image in data["holdout"] and columns
        image_id, pred_unusable (0/1), score (higher = review first), reason (short text)

Usability rule (frozen, fixtures/t03_synthetic/README_usability_rule.md) — unusable if ANY of:
  1. severity == 2 for night / fog / blur / glare
  2. glare severity >= 1 whose centre falls inside a labelled object box
  3. erase removed a labelled object but its label is still listed

The holdout does NOT say which transform / severity was applied: compare each image with its parent
(data["originals"], column parent_image) and use the parent's boxes (data["objects"]; same 320x180 size).
Thresholds come from ctx.param("thresholds") — tune on the dev gold set, commit BEFORE running holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.context import RunContext
from ..common.image_ops import (
    bbox_erase_metrics,
    estimate_glare_peak,
    laplacian_variance,
    load_rgb,
    ssim_score,
    to_gray,
)

OUTPUT_COLUMNS = ["image_id", "pred_unusable", "score", "reason"]

DEFAULT_THRESHOLDS = {
    "night_sev2_mean_ratio": 0.14,
    "night_sev1_mean_ratio": 0.40,
    "blur_sev2_lap_ratio": 0.0032,
    "blur_sev2_std_ratio": 0.895,
    "blur_sev1_lap_ratio": 0.06,
    "fog_sev2_std_ratio": 0.60,
    "fog_sev2_lap_ratio": 0.25,
    "fog_sev1_std_ratio": 0.82,
    "glare_sev2_strength": 0.75,
    "glare_sev1_strength": 0.40,
    "erase_max_std": 3.5,
    "erase_max_color_std": 6.0,
    "erase_min_parent_std": 12.0,
    "erase_min_mad": 10.0,
}


def evaluate_single_image(
    img: np.ndarray,
    parent_img: np.ndarray,
    boxes: pd.DataFrame,
    thresholds: dict | None = None,
) -> dict:
    """Evaluate a single transformed image against its parent original and object boxes."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    g = to_gray(img)
    pg = to_gray(parent_img)
    diff = g - pg

    mean_ratio = float(g.mean() / max(pg.mean(), 1e-5))
    std_ratio = float(g.std() / max(pg.std(), 1e-5))
    lap_val = laplacian_variance(img)
    plap_val = laplacian_variance(parent_img)
    lap_ratio = float(lap_val / max(plap_val, 1e-5))
    ssim_val = ssim_score(img, parent_img)
    p99_diff = float(np.percentile(diff, 99))

    # 1. Check Erase (object removed, label kept)
    # Only check when global image is not heavily darkened/blurred globally
    is_global_unchanged = (0.90 <= mean_ratio <= 1.10) and (lap_ratio > 0.50)
    if is_global_unchanged and len(boxes) > 0:
        for _, box in boxes.iterrows():
            bm = bbox_erase_metrics(img, parent_img, float(box.x), float(box.y), float(box.w), float(box.h))
            if (
                bm["std"] < th["erase_max_std"]
                and bm["color_std"] < th["erase_max_color_std"]
                and bm["parent_std"] > th["erase_min_parent_std"]
                and bm["mad"] > th["erase_min_mad"]
            ):
                score = float(np.clip(0.92 + 0.08 * (1.0 - bm["std"] / th["erase_max_std"]), 0.90, 0.99))
                return {
                    "pred_unusable": 1,
                    "score": round(score, 4),
                    "reason": "object_removed_label_kept",
                    "ssim": round(ssim_val, 4),
                }

    # 2. Check Night (darkening + sensor noise)
    if mean_ratio < th["night_sev1_mean_ratio"]:
        if mean_ratio < th["night_sev2_mean_ratio"]:
            score = float(np.clip(0.85 + 0.15 * (1.0 - mean_ratio / th["night_sev2_mean_ratio"]), 0.80, 0.99))
            return {"pred_unusable": 1, "score": round(score, 4), "reason": "severity_2_night", "ssim": round(ssim_val, 4)}
        score = float(np.clip(0.35 * (1.0 - (mean_ratio - th["night_sev2_mean_ratio"]) / (th["night_sev1_mean_ratio"] - th["night_sev2_mean_ratio"])), 0.15, 0.45))
        return {"pred_unusable": 0, "score": round(score, 4), "reason": "severity_1_night_usable", "ssim": round(ssim_val, 4)}

    # 3. Check Blur (sharpness drop with unchanged mean brightness)
    if lap_ratio < th["blur_sev1_lap_ratio"] and 0.92 <= mean_ratio <= 1.08:
        if lap_ratio < th["blur_sev2_lap_ratio"] and std_ratio < th["blur_sev2_std_ratio"]:
            score = float(np.clip(0.85 + 0.14 * (1.0 - lap_ratio / th["blur_sev2_lap_ratio"]), 0.80, 0.99))
            return {"pred_unusable": 1, "score": round(score, 4), "reason": "severity_2_blur", "ssim": round(ssim_val, 4)}
        score = float(np.clip(0.30 + 0.15 * (1.0 - lap_ratio / th["blur_sev1_lap_ratio"]), 0.20, 0.48))
        return {"pred_unusable": 0, "score": round(score, 4), "reason": "severity_1_blur_usable", "ssim": round(ssim_val, 4)}

    # 4. Check Fog vs Glare (both increase mean brightness: mean_ratio > 1.08)
    # Fog reduces contrast (std_ratio < 0.82) and sharpness (lap_ratio < 0.50);
    # Glare increases local brightness and std_ratio >= 0.95 with high p99_diff.
    if mean_ratio > 1.12 and std_ratio < th["fog_sev1_std_ratio"] and lap_ratio < 0.50:
        if std_ratio < th["fog_sev2_std_ratio"] and lap_ratio < th["fog_sev2_lap_ratio"]:
            score = float(np.clip(0.82 + 0.16 * (1.0 - std_ratio / th["fog_sev2_std_ratio"]), 0.78, 0.98))
            return {"pred_unusable": 1, "score": round(score, 4), "reason": "severity_2_fog", "ssim": round(ssim_val, 4)}
        score = float(np.clip(0.25 + 0.20 * (1.0 - std_ratio / th["fog_sev1_std_ratio"]), 0.20, 0.45))
        return {"pred_unusable": 0, "score": round(score, 4), "reason": "severity_1_fog_usable", "ssim": round(ssim_val, 4)}

    # 5. Check Glare (localized bright Gaussian peak)
    if mean_ratio > 1.10 and p99_diff > 80.0:
        cx, cy, s_est = estimate_glare_peak(img, parent_img)
        in_bbox = False
        if len(boxes) > 0:
            for _, box in boxes.iterrows():
                if float(box.x) <= cx <= float(box.x + box.w) and float(box.y) <= cy <= float(box.y + box.h):
                    in_bbox = True
                    break
        if s_est >= th["glare_sev2_strength"]:
            reason = "severity_2_glare_in_bbox" if in_bbox else "severity_2_glare"
            score = float(np.clip(0.85 + 0.14 * min(1.0, (s_est - th["glare_sev2_strength"]) / 0.30), 0.82, 0.99))
            return {"pred_unusable": 1, "score": round(score, 4), "reason": reason, "ssim": round(ssim_val, 4)}
        if s_est >= th["glare_sev1_strength"]:
            if in_bbox:
                score = float(np.clip(0.80 + 0.12 * (s_est / th["glare_sev2_strength"]), 0.78, 0.92))
                return {"pred_unusable": 1, "score": round(score, 4), "reason": "glare_in_bbox", "ssim": round(ssim_val, 4)}
            score = float(np.clip(0.30 + 0.15 * (s_est / th["glare_sev2_strength"]), 0.20, 0.48))
            return {"pred_unusable": 0, "score": round(score, 4), "reason": "severity_1_glare_outside_bbox", "ssim": round(ssim_val, 4)}

    # 6. Clean / severity 0
    score = float(np.clip(1.0 - ssim_val, 0.0, 0.15))
    return {"pred_unusable": 0, "score": round(score, 4), "reason": "usable_clean", "ssim": round(ssim_val, 4)}


def predict(ctx: RunContext, data: dict) -> pd.DataFrame:
    hold = data["holdout"]
    if len(hold) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    orig_df = data["originals"].set_index("image_id")
    objs_df = data["objects"]
    thresholds = ctx.param("thresholds") or {}

    parent_cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for _, row in hold.iterrows():
        img_id = str(row["image_id"])
        parent_id = str(row["parent_image"])
        img_path = ctx.input_dir / str(row["path"])
        img = load_rgb(img_path)

        if parent_id not in parent_cache:
            parent_rel = str(orig_df.loc[parent_id, "path"])
            parent_cache[parent_id] = load_rgb(ctx.input_dir / parent_rel)
        parent_img = parent_cache[parent_id]

        boxes = objs_df[objs_df["image_id"] == parent_id]
        res = evaluate_single_image(img, parent_img, boxes, thresholds=thresholds)
        rows.append({
            "image_id": img_id,
            "pred_unusable": int(res["pred_unusable"]),
            "score": float(res["score"]),
            "reason": str(res["reason"]),
        })

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def check_output(scored: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Fail loudly (exit 2) instead of writing a qc.csv the evaluator would silently penalise."""
    missing_cols = [c for c in OUTPUT_COLUMNS if c not in scored.columns]
    if missing_cols:
        raise ValueError(f"qc.predict(): missing columns {missing_cols}")
    if scored.image_id.duplicated().any():
        raise ValueError("qc.predict(): duplicated image_id")
    missing_ids = set(holdout.image_id) - set(scored.image_id)
    if missing_ids:  # the evaluator would count these as 'usable'
        raise ValueError(f"qc.predict(): {len(missing_ids)} holdout images have no prediction")
    if not set(scored.pred_unusable.astype(int).unique()) <= {0, 1}:
        raise ValueError("qc.predict(): pred_unusable must be 0 or 1")
