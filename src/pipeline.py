"""T03 Synthetic-vs-Real — PoC pipeline (owner: Member 1, Lead / Integrator).

This file only WIRES the 5 lab steps together. The real work lives in src/t03/, one module per owner,
so two people never edit the same function:

    01 load      this file (Member 1)          input tables + provenance check (manifest.csv)
    02 process   src/t03/generate.py (M3)      synthetic variants of the dev originals
                 src/t03/provenance.py (M2)    out/transform_manifest.csv
                 src/t03/qc.py (M4)            quality gate -> image_id, pred_unusable, score
    03 rank      src/t03/review_flow.py (M5)   top-B review queue + before/after contact sheet
    04 review    src/t03/review_flow.py (M5)   reviewer decisions -> overrides + audit log
    05 decide    src/t03/review_flow.py (M5)   GO / HOLD / REWORK with reason_code + owner

Run:  python -m src.main --config configs/lab.yaml --output out/poc --run-name poc --offline --seed 42
Rules: never read fixtures/*/gt/; seed everything via ctx.rng / ctx.seed; no network calls.
"""
from __future__ import annotations

import pandas as pd

from .common.context import RunContext
from .common.decision import write_decision
from .common.manifest import verify_manifest
from .metrics.tracks import get_track
from .t03 import generate, provenance, qc, review_flow


# ---------------------------------------------------------------- 01 load  (Member 1)
def load(ctx: RunContext) -> dict:
    d = ctx.input_dir
    manifest = d / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    data = {
        "manifest": pd.read_csv(manifest),
        "originals": pd.read_csv(d / "originals.csv"),        # image_id, path, split (dev / holdout)
        "objects": pd.read_csv(d / "objects.csv"),            # image_id, obj_id, class, x, y, w, h (bbox of ORIGINAL)
    }
    hold = d / "holdout_transformed.csv"                      # image_id, path, parent_image (no transform info!)
    data["holdout"] = pd.read_csv(hold) if hold.exists() else pd.DataFrame(columns=["image_id", "path", "parent_image"])
    ctx.stats["n_originals"] = len(data["originals"])
    ctx.stats["n_holdout"] = len(data["holdout"])
    return data


# ---------------------------------------------------------------- 02 process  (M3 + M2 + M4)
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    generated = generate.generate_dev_variants(ctx, data)     # M3: images on disk + one row per variant
    provenance.write_transform_manifest(ctx, generated)       # M2: out/transform_manifest.csv
    data["generated"] = generated
    scored = qc.predict(ctx, data)                            # M4: one row per holdout image
    qc.check_output(scored, data["holdout"])
    return scored


# ---------------------------------------------------------------- 03-05  (M5)
def rank(ctx: RunContext, scored: pd.DataFrame, data: dict) -> pd.DataFrame:
    return review_flow.rank(ctx, scored, data)


def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    return review_flow.review(ctx, scored, queue)


def decide(ctx: RunContext, final: pd.DataFrame, data: dict):
    return review_flow.decide(ctx, final, data)


# ---------------------------------------------------------------- wiring  (Member 1)
def run(ctx: RunContext) -> None:
    if ctx.track != "t03":
        raise NotImplementedError(f"this pipeline implements t03 only (got {ctx.track})")
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored, data)
    final = review(ctx, scored, queue)
    cols = spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]
    final[cols].to_csv(ctx.out(spec.pred_file), index=False)                     # out/.../qc.csv
    write_decision(decide(ctx, final, data), ctx.out("decision.json"), ctx.out("decision_log.csv"))
