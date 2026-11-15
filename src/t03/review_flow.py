"""Human-in-the-loop + decision — owner: Member 5.

Contract with the pipeline:
    rank(ctx, scored, data)      -> review queue (top-B), writes review_queue.csv + contact_sheet.html
    review(ctx, scored, queue)   -> final predictions after reviewer overrides (audit-logged)
    decide(ctx, final, data)     -> Decision (GO / HOLD / REWORK + reason_code + owner)

Review loop: open out/.../contact_sheet.html, pick a decision per card, press "Export CSV",
save the file as out/.../review_queue_reviewed.csv and run the pipeline again.
`edit` / `reject` with override_value 0 or 1 replaces pred_unusable for that image.
"""
from __future__ import annotations

import pandas as pd

from ..common.contact_sheet import render_contact_sheet
from ..common.context import RunContext
from ..common.decision import Decision
from ..common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue


def rank(ctx: RunContext, scored: pd.DataFrame, data: dict) -> pd.DataFrame:
    budget = int(ctx.param("budget", 10))
    hold = data["holdout"][["image_id", "path", "parent_image"]]
    parents = data["originals"][["image_id", "path"]].rename(columns={"image_id": "parent_image", "path": "parent_path"})
    items = scored.merge(hold, on="image_id", how="left").merge(parents, on="parent_image", how="left")
    # Uncertainty first: spend the review budget where the quality gate is most likely wrong, i.e. scores
    # closest to the boundary between "usable" (qc.py scores <= 0.48) and "unusable" (>= 0.78).
    boundary = float(ctx.param("review_boundary", 0.63))
    items["review_priority"] = (1.0 - (items.score - boundary).abs()).round(4)
    items["suggested_action"] = items.pred_unusable.map({1: "confirm_unusable", 0: "confirm_usable"})
    q = make_review_queue(items, id_col="image_id", score_col="review_priority", budget=budget,
                          reason_col="reason", action_col="suggested_action", evidence_col="path")
    by_id = items.set_index("image_id")
    q["qc_score"] = q.item_id.map(by_id.score)
    q["pred_unusable"] = q.item_id.map(by_id.pred_unusable)
    q["parent_path"] = q.item_id.map(by_id.parent_path)
    write_review_queue(q, ctx.out("review_queue.csv"))

    objs = data["objects"]
    parent_of = items.set_index("image_id").parent_image
    boxes = {iid: [{"x": o.x, "y": o.y, "w": o.w, "h": o.h, "label": o["class"]}
                   for _, o in objs[objs.image_id == parent_of[iid]].iterrows()] for iid in q.item_id}
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_col="evidence_path", before_image_col="parent_path",
                         image_root=ctx.input_dir, boxes_by_item=boxes, title="T03 QC review: original (before) vs transformed (after)")
    return q


def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        ctx.stats["n_reviewed"] = 0
        return scored
    reviewed = load_review_decisions(reviewed_path)        # fails loudly if a decision has no reviewer
    ctx.stats["n_reviewed"] = len(reviewed)
    overrides = reviewed[reviewed.reviewer_decision.isin(["edit", "reject"]) & (reviewed.override_value.str.strip() != "")]
    bad = sorted(set(overrides.override_value.str.strip()) - {"0", "1"})
    if bad:
        raise ValueError(f"review_queue_reviewed.csv: override_value must be 0 or 1 for T03, got {bad}")
    # apply_overrides writes the CSV string ('1') into the column; pandas >= 3 refuses that on an int column,
    # so override on a string copy and cast back to 0/1 afterwards.
    final = apply_overrides(scored.astype({"pred_unusable": str}), reviewed, id_col="image_id",
                            value_col="pred_unusable", audit=ctx.audit)
    final["pred_unusable"] = final.pred_unusable.astype(int)
    return final


def decide(ctx: RunContext, final: pd.DataFrame, data: dict) -> Decision:
    """Release policy (Member 5, ported from commit 9bb095c). Thresholds live in configs/lab.yaml -> params.

    HOLD   if no holdout image was scored, or the provenance record is missing or empty
    REWORK if the predicted unusable rate exceeds max_unusable_rate
    GO     otherwise
    """
    owner = ctx.config.get("owner", "Member5-T03")
    n = len(final)
    rate = float(final.pred_unusable.mean()) if n else 0.0
    max_rate = float(ctx.param("max_unusable_rate", 0.30))
    manifest = ctx.output_dir / "transform_manifest.csv"
    metrics = {"n_holdout": n, "unusable_count": int(final.pred_unusable.sum()), "unusable_rate": round(rate, 4),
               "max_unusable_rate": max_rate, "n_reviewed": ctx.stats.get("n_reviewed", 0),
               "n_generated": ctx.stats.get("n_generated", 0)}
    reviewed = ctx.output_dir / "review_queue_reviewed.csv"
    evidence = [str(ctx.out("qc.csv")), str(manifest)] + ([str(reviewed)] if reviewed.exists() else [])

    if n == 0:
        return Decision(status="HOLD", reason_code="NO_HOLDOUT_IMAGES",
                        reason="No holdout image was scored: nothing supports a release decision.",
                        owner=owner, scope="t03 holdout_transformed", evidence=evidence, metrics=metrics)
    if not manifest.exists() or metrics["n_generated"] == 0:
        return Decision(status="HOLD", reason_code="MISSING_PROVENANCE_MANIFEST",
                        reason="Transform manifest is missing or empty: synthetic images cannot be traced to their parents.",
                        owner=owner, scope="t03 holdout_transformed", evidence=evidence, metrics=metrics)
    if rate > max_rate:
        return Decision(status="REWORK", reason_code="UNUSABLE_RATE_EXCEEDED",
                        reason=f"Unusable rate {rate:.2%} exceeds threshold {max_rate:.2%}; regenerate with milder transforms.",
                        owner=owner, scope="t03 holdout_transformed", evidence=evidence, metrics=metrics)
    return Decision(status="GO", reason_code="QC_AND_PROVENANCE_PASSED",
                    reason="Provenance recorded and unusable rate within threshold.",
                    owner=owner, scope="t03 holdout_transformed", evidence=evidence, metrics=metrics)
