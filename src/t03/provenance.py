"""Provenance for synthetic images — owner: Member 2.

Contract with the pipeline:
    write_transform_manifest(ctx, generated) -> Path of out/.../transform_manifest.csv

The evaluator scores "Provenance completeness" = share of manifest rows where ALL of
image_id, parent_image, transform, severity, seed are filled. Target: 100%.
Extra columns make each image reproducible: params, generator, version, parent_sha256, created_by.

Also owned here (TODO Member 2): build the dev gold set (dev_gold.csv) by applying
fixtures/t03_synthetic/README_usability_rule.md to the generated dev images — it is the only
labelled data the group has for tuning, because gt/ is not distributed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.context import RunContext

REQUIRED = ["image_id", "parent_image", "transform", "severity", "seed"]
EXTRA = ["params", "generator", "version", "parent_sha256", "created_by"]
GENERATOR_NAME = "src.t03.generate"
GENERATOR_VERSION = "0.1.0"   # TODO(Member 2): bump whenever generate.py changes behaviour


def write_transform_manifest(ctx: RunContext, generated: pd.DataFrame) -> Path:
    m = generated.copy()
    m["generator"] = GENERATOR_NAME
    m["version"] = GENERATOR_VERSION
    m["created_by"] = ctx.run_name
    # TODO(Member 2): parent_sha256 = sha256 of the parent original (see src/common/io.py), ...
    if "parent_sha256" not in m:
        m["parent_sha256"] = ""
    for c in REQUIRED + EXTRA:
        if c not in m:
            m[c] = ""
    missing = m[REQUIRED].isna().any(axis=1) | (m[REQUIRED].astype(str) == "").any(axis=1)
    if missing.any():
        raise ValueError(f"transform_manifest: {int(missing.sum())} rows miss one of {REQUIRED}")
    ctx.stats["n_generated"] = len(m)
    path = ctx.out("transform_manifest.csv")
    m[REQUIRED + EXTRA + [c for c in m.columns if c not in REQUIRED + EXTRA]].to_csv(path, index=False)
    return path
