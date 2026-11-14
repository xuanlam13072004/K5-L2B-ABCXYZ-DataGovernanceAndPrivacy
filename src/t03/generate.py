"""Synthetic variant generator — owner: Member 3.

Contract with the pipeline:
    generate_dev_variants(ctx, data) -> DataFrame, one row per generated image, with at least
        GENERATED_COLUMNS. Images are written under ctx.out("generated/").

Rules:
  * generate ONLY from originals with split == "dev" (never from holdout)
  * transforms night / glare / fog / blur at severity 0, 1, 2 (ctx.param("transforms"), ctx.param("severities"))
    + an "erase" variant that removes one labelled object but keeps its label (usability rule 3)
  * deterministic: derive each image's seed from ctx.seed (same run twice -> same sha256)
  * fill `params` (JSON string) with everything needed to re-create the image, and for glare / erase
    record where it happened (glare centre x,y / erased obj_id) so Member 2 can label the dev gold set
  * use src/common/corruptions.py if it fits, but do NOT edit it (shared infrastructure)
"""
from __future__ import annotations

import pandas as pd

from ..common.context import RunContext

GENERATED_COLUMNS = ["image_id", "path", "parent_image", "transform", "severity", "seed", "params"]


def generate_dev_variants(ctx: RunContext, data: dict) -> pd.DataFrame:
    dev = data["originals"][data["originals"].split == "dev"]
    ctx.stats["n_dev_originals"] = len(dev)
    # TODO(Member 3): loop over dev originals x transforms x severities (+ erase), write each image to
    #   ctx.out(f"generated/{image_id}.jpg") and append a row with GENERATED_COLUMNS.
    rows: list[dict] = []
    return pd.DataFrame(rows, columns=GENERATED_COLUMNS)
