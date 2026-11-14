"""Standard Heuristic Baseline for T03 (Synthetic-vs-Real Quality Gate).

Operates directly on custom input (data/image):
  01 load      Scan images from data/image/*.jpg
  02 process   Evaluate usability via blur (Laplacian variance) and darkness (luminance)
  03 rank      Prioritize suspicious images into review queue
  04 review    Log to audit trail
  05 decide    Emit HOLD decision (baseline rule) and export out/qc.csv
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.review import make_review_queue, write_review_queue
from .metrics.tracks import get_track


def _load_images(input_dir: Path) -> pd.DataFrame:
    """Load images from data/image or fallback to holdout_transformed.csv."""
    holdout_csv = input_dir / "holdout_transformed.csv"
    if holdout_csv.exists():
        df = pd.read_csv(holdout_csv)
        df["full_path"] = df["path"].apply(lambda p: str(input_dir / p))
        return df

    # Custom images mode (e.g. data/image/*.jpg)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return pd.DataFrame([{"image_id": p.stem, "path": p.name, "full_path": str(p)} for p in files])


def run(ctx: RunContext) -> None:
    spec = get_track("t03")
    df = _load_images(ctx.input_dir)                                      # 01 load

    # 02 process: Heuristic baseline (blur & dark)
    scores, unusable = [], []
    for p in df["full_path"]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            scores.append(1.0)
            unusable.append(1)
            continue

        lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
        is_dark = float(np.mean(img)) < 30.0
        is_unusable = 1 if (lap_var < 45.0 or is_dark) else 0

        # Higher score = more suspicious of being defective
        score = max(0.85, 1.0 - lap_var / 300.0) if is_dark else float(np.clip(1.0 - lap_var / 300.0, 0.0, 1.0))
        scores.append(round(score, 4))
        unusable.append(is_unusable)

    df["score"] = scores
    df["pred_unusable"] = unusable
    df["reason"] = "heuristic_laplacian_luma"
    ctx.stats["n_items"] = len(df)

    # 03 rank: review queue top-B
    budget = int(ctx.param("budget", min(10, len(df))))
    q = make_review_queue(df, "image_id", "score", budget, reason_col="reason")
    write_review_queue(q, ctx.out("review_queue.csv"))                    # 03 rank

    # 04 review: audit trail
    ctx.audit.log(actor="role3_baseline", action="evaluate", item_id=str(ctx.input_dir), after={"n": len(df)})

    # 05 decide: output qc.csv + decision HOLD
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    df[["image_id", "pred_unusable", "score"]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(
        Decision(status="HOLD", reason_code="BASELINE_ONLY",
                 reason=f"Baseline heuristic (blur/dark) evaluated on {ctx.input_dir.name}",
                 owner=ctx.config.get("owner", "Member 3 (Baseline)"), scope="t03"),
        ctx.out("decision.json"), ctx.out("decision_log.csv")
    )
    print(f"[Baseline] Evaluated {len(df)} images -> {ctx.out(spec.pred_file)}")
