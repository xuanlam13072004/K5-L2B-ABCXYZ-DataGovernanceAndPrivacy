"""Quality gate v2 regressions (Lead, 15/11): the two T03 misses fixed on 1280x720 images.

Both tests build a synthetic 1280x720 scene, so they need no data files.
"""
import numpy as np
import pandas as pd
import yaml

from src.common import corruptions
from src.t03.generate import erase_object
from src.t03.qc import estimate_blur_radius, evaluate_single_image

THRESHOLDS = yaml.safe_load(open("configs/lab.yaml", encoding="utf-8"))["params"]["thresholds"]


def _scene(seed=0, H=720, W=1280):
    rng = np.random.default_rng(seed)
    img = (rng.random((H, W, 3)) * 60 + 90).astype(np.uint8)          # textured road
    img[300:420, 500:700] = [200, 30, 30]                             # a red "car"
    img[310:330, 520:680] = [240, 240, 240]                           # with a bright stripe
    boxes = pd.DataFrame([{"image_id": "p", "obj_id": "p_o0", "class": "car", "x": 500, "y": 300, "w": 200, "h": 120}])
    return img, boxes


def test_inpainted_erase_is_flagged_on_large_images():
    img, boxes = _scene()
    erased = erase_object(img, boxes.iloc[0])                         # textured in-painting, not a flat patch
    res = evaluate_single_image(erased, img, boxes, THRESHOLDS)
    assert res["pred_unusable"] == 1 and res["reason"] == "object_removed_label_kept"
    same = evaluate_single_image(img.copy(), img, boxes, THRESHOLDS)  # untouched image stays usable
    assert same["pred_unusable"] == 0


def test_blur_severity_is_read_from_radius_not_resolution():
    img, boxes = _scene(1)
    sev1 = corruptions.blur(img, severity=1)
    sev2 = corruptions.blur(img, severity=2)
    assert estimate_blur_radius(sev1, img) < THRESHOLDS["blur_sev2_radius_px"] <= estimate_blur_radius(sev2, img)
    assert evaluate_single_image(sev1, img, boxes, THRESHOLDS)["pred_unusable"] == 0
    assert evaluate_single_image(sev2, img, boxes, THRESHOLDS)["pred_unusable"] == 1
