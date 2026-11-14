"""Member 6 acceptance tests for the T03 QC output contract and hard cases."""

import numpy as np
import pandas as pd
import pytest

from src.t03.qc import check_output, evaluate_single_image


def test_t03_qc_schema_accepts_one_binary_prediction_per_holdout_image():
    holdout = pd.DataFrame({"image_id": ["hold_000", "hold_001"]})
    scored = pd.DataFrame(
        {
            "image_id": ["hold_000", "hold_001"],
            "pred_unusable": [0, 1],
            "score": [0.1, 0.9],
            "reason": ["usable_clean", "severity_2_glare_in_bbox"],
        }
    )

    check_output(scored, holdout)


def test_t03_qc_schema_rejects_missing_required_column():
    holdout = pd.DataFrame({"image_id": ["hold_000"]})
    scored = pd.DataFrame({"image_id": ["hold_000"], "pred_unusable": [0], "score": [0.1]})

    with pytest.raises(ValueError, match="missing columns.*reason"):
        check_output(scored, holdout)


def test_t03_qc_schema_rejects_duplicate_or_missing_predictions():
    holdout = pd.DataFrame({"image_id": ["hold_000", "hold_001"]})
    duplicate = pd.DataFrame(
        {
            "image_id": ["hold_000", "hold_000"],
            "pred_unusable": [0, 1],
            "score": [0.1, 0.9],
            "reason": ["usable_clean", "severity_2_glare_in_bbox"],
        }
    )
    with pytest.raises(ValueError, match="duplicated image_id"):
        check_output(duplicate, holdout)

    missing = duplicate.iloc[:1].copy()
    with pytest.raises(ValueError, match="holdout images have no prediction"):
        check_output(missing, holdout)

    non_binary = duplicate.iloc[:1].copy()
    non_binary["pred_unusable"] = 2
    with pytest.raises(ValueError, match="pred_unusable must be 0 or 1"):
        check_output(non_binary, holdout.iloc[:1])


def test_t03_glare_peak_inside_labeled_bbox_is_flagged_unusable():
    height, width = 180, 320
    parent = np.full((height, width, 3), 70, dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    gaussian_glare = 220.0 * np.exp(-((xx - 160) ** 2 + (yy - 90) ** 2) / (2 * 22.0**2))
    transformed = np.clip(parent.astype(np.float32) + gaussian_glare[..., None], 0, 255).astype(np.uint8)
    boxes = pd.DataFrame([{"x": 142.0, "y": 72.0, "w": 36.0, "h": 36.0}])

    result = evaluate_single_image(transformed, parent, boxes)

    assert result["pred_unusable"] == 1
    assert result["reason"] == "severity_2_glare_in_bbox"
    assert result["score"] >= 0.82
