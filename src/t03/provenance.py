"""Provenance for synthetic images — owner: Member 2.

Contract with the pipeline:
    write_transform_manifest(ctx, generated, data) -> Path of out/.../transform_manifest.csv
    (also writes out/.../dev_gold.csv, the labelled dev set used to tune the quality gate)

The evaluator scores "Provenance completeness" = share of manifest rows where ALL of
image_id, parent_image, transform, severity, seed are filled (it reads transform_manifest.csv
next to qc.csv). Target: 100%. Extra columns make each image reproducible and verifiable:
params, generator, version, parent_sha256, sha256, created_by.

dev_gold.csv applies the frozen usability rule (fixtures/t03_synthetic/README_usability_rule.md)
to the generated dev images. The label follows from how the image was made — no guessing:
  unusable = 1 if severity == 2 (night / fog / blur / glare)
               or glare severity >= 1 with its centre inside a labelled object box of the parent
               or erase removed a labelled object while its label is kept
Columns match what src.evaluate expects for T03: image_id, unusable, transform, severity, note.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..common.context import RunContext
from ..common.io import sha256_file
from .generate import inside_any_box

REQUIRED = ["image_id", "parent_image", "transform", "severity", "seed"]
EXTRA = ["params", "generator", "version", "parent_sha256", "sha256", "created_by"]
GENERATOR_NAME = "src.t03.generate"
GENERATOR_VERSION = "0.2.0"   # bump whenever generate.py changes what it produces


def usability_label(row, objects: pd.DataFrame) -> tuple[int, str]:
    if row.transform == "erase":
        return 1, "object_removed_label_kept"
    if int(row.severity) == 2:
        return 1, "severity_2"
    if row.transform == "glare" and int(row.severity) >= 1:
        cx, cy = json.loads(row.params)["center"]
        if inside_any_box(cx, cy, objects[objects.image_id == row.parent_image]):
            return 1, "glare_on_object"
    return 0, ""


def write_transform_manifest(ctx: RunContext, generated: pd.DataFrame, data: dict) -> Path:
    m = generated.copy()
    parents = data["originals"].set_index("image_id")
    parent_sha = {pid: sha256_file(ctx.input_dir / str(parents.loc[pid, "path"])) for pid in m.parent_image.unique()}
    m["parent_sha256"] = m.parent_image.map(parent_sha)
    m["sha256"] = [sha256_file(ctx.output_dir / p) for p in m.path] if len(m) else []
    m["generator"] = GENERATOR_NAME
    m["version"] = GENERATOR_VERSION
    m["created_by"] = f"{GENERATOR_NAME} (Member 3) via run '{ctx.run_name}'"
    for c in REQUIRED + EXTRA:
        if c not in m:
            m[c] = ""
    missing = m[REQUIRED].isna().any(axis=1) | (m[REQUIRED].astype(str) == "").any(axis=1)
    if missing.any():
        raise ValueError(f"transform_manifest: {int(missing.sum())} rows miss one of {REQUIRED}")
    ctx.stats["n_generated"] = len(m)
    path = ctx.out("transform_manifest.csv")
    m[REQUIRED + EXTRA + [c for c in m.columns if c not in REQUIRED + EXTRA]].to_csv(path, index=False)

    labels = [usability_label(r, data["objects"]) for r in m.itertuples(index=False)]
    gold = pd.DataFrame({"image_id": m.image_id, "unusable": [u for u, _ in labels], "transform": m["transform"],
                         "severity": m.severity, "note": [n for _, n in labels]})
    gold.to_csv(ctx.out("dev_gold.csv"), index=False)
    ctx.stats["dev_gold_unusable"] = int(gold.unusable.sum())
    return path
