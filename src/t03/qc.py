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

import pandas as pd

from ..common.context import RunContext

OUTPUT_COLUMNS = ["image_id", "pred_unusable", "score", "reason"]


def predict(ctx: RunContext, data: dict) -> pd.DataFrame:
    hold = data["holdout"]
    # TODO(Member 4): replace this placeholder (everything "usable", score 0) with the real gate:
    #   guess transform + severity vs parent, glare-centre-in-box check, erased-object check.
    return pd.DataFrame({
        "image_id": hold.image_id,
        "pred_unusable": 0,
        "score": 0.0,
        "reason": "placeholder",
    })[OUTPUT_COLUMNS]


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
