"""Summarize the team's independently generated T03 benchmark; never used by the pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import sha256_file, write_json
from src.metrics.classification import prf
from src.common.image_ops import load_rgb, to_gray, laplacian_variance, bbox_erase_metrics


def summarize(root: Path, out: Path) -> dict:
    gold = root / "gold" / "team_holdout"
    gt = pd.read_csv(root / "gold" / "team_holdout_gt.csv").fillna("")
    parents = pd.read_csv(gold / "holdout_transformed.csv")
    sources = pd.read_csv(root / "source" / "source_map.csv")
    data = gt.merge(parents, on="image_id", validate="one_to_one")
    data = data.merge(sources[["image_id", "source_file"]].rename(columns={"image_id": "parent_image"}),
                      on="parent_image", validate="many_to_one")
    result = {
        "protocol": "Team holdout originals + controlled synthetic corruptions, not instructor hidden eval",
        "seed": 42, "n": len(data), "n_parents": data.parent_image.nunique(),
        "unusable": int(data.unusable.sum()),
        "archive_sha256": sha256_file("data/obj_train_data.zip"),
        "gt_sha256": sha256_file(root / "gold" / "team_holdout_gt.csv"),
        "config_sha256": sha256_file("configs/lab.yaml"), "runs": {},
    }
    lines = ["# T03 benchmark nội bộ — chi tiết do script sinh",
             "", result["protocol"], "",
             "| Run | N | TP | FP | FN | TN | F1 | F1 95% CI (parent bootstrap) |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    slices, cases = [], []
    for run in ("baseline", "poc"):
        pred = pd.read_csv(root / run / "qc.csv")
        if set(pred.image_id) != set(gt.image_id) or pred.image_id.duplicated().any():
            raise ValueError("prediction pool differs from GT")
        joined = data.merge(pred, on="image_id", validate="one_to_one")
        counts = prf(joined.unusable, joined.pred_unusable)
        metrics_path = out / f"metrics_{run}.json"
        if metrics_path.exists():
            official = json.loads(metrics_path.read_text(encoding="utf-8"))
            primary = next(m for m in official["metrics"] if m["primary"])
            if not np.isclose(primary["value"], counts["f1"]):
                raise ValueError("diagnostic F1 differs from eval_t03 output")
        groups = [g for _, g in joined.groupby("parent_image", sort=True)]
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(2000):
            sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            f1 = prf(sample.unusable, sample.pred_unusable)["f1"]
            if f1 is not None:
                boot.append(f1)
        ci = np.quantile(boot, [0.025, 0.975]).tolist() if boot else [None, None]
        errors = joined[joined.unusable != joined.pred_unusable]
        cols = ["image_id", "parent_image", "source_file", "transform", "severity", "note",
                "unusable", "pred_unusable"]
        if "reason" in errors:
            cols.append("reason")
        records = errors[cols].to_dict(orient="records")
        result["runs"][run] = {"counts": counts, "f1_ci95": ci, "bootstrap_valid": len(boot),
                                "bootstrap_replicates": 2000, "errors": records}
        lines.append(f"| {run} | {counts['n']} | {counts['tp']} | {counts['fp']} | {counts['fn']} | "
                     f"{counts['tn']} | {counts['f1']:.6f} | {ci[0]:.6f} – {ci[1]:.6f} |")
        for col in ("transform", "severity"):
            for key, g in joined.groupby(col):
                sl = {"run": run, "slice": f"{col}={key}", **prf(g.unusable, g.pred_unusable)}
                slices.append(sl)
        if run == "poc":
            # Prefer two distinct missed transform types for concrete failure analysis.
            for transform in errors["transform"].drop_duplicates():
                row = errors[errors["transform"] == transform].iloc[0]
                cases.append({c: row[c] for c in cols})
                if len(cases) == 2:
                    break
    result["slices"] = slices
    # Convert pandas/numpy scalar types through JSON serialization.
    result["failure_cases"] = json.loads(pd.DataFrame(cases).to_json(orient="records"))
    provenance = pd.read_csv(root / "gold" / "team_holdout_provenance.csv").set_index("image_id")
    objects = pd.read_csv(gold / "objects.csv").set_index("obj_id")
    originals = pd.read_csv(gold / "originals.csv").set_index("image_id")
    for case in result["failure_cases"]:
        img = load_rgb(gold / parents.set_index("image_id").loc[case["image_id"], "path"])
        parent = load_rgb(gold / originals.loc[case["parent_image"], "path"])
        g, pg = to_gray(img), to_gray(parent)
        features = {"mean_ratio": float(g.mean() / max(pg.mean(), 1e-5)),
                    "std_ratio": float(g.std() / max(pg.std(), 1e-5)),
                    "lap_ratio": laplacian_variance(img) / max(laplacian_variance(parent), 1e-5)}
        if case["transform"] == "erase":
            params = json.loads(provenance.loc[case["image_id"], "params"])
            box = objects.loc[params["erased_obj_id"]]
            features["bbox"] = bbox_erase_metrics(img, parent, box.x, box.y, box.w, box.h)
        case["features"] = features
    lines += ["", "## Slices", "",
              "| Run | Slice | N | TP | FP | FN | F1 | Precision | Recall |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in slices:
        fmt = lambda v: "n/a" if v is None else f"{v:.6f}"
        lines.append(f"| {s['run']} | {s['slice']} | {s['n']} | {s['tp']} | {s['fp']} | {s['fn']} | "
                     f"{fmt(s['f1'])} | {fmt(s['precision'])} | {fmt(s['recall'])} |")
    lines += ["", "## Failure cases (PoC)", "",
              "| Image | Parent / source | Transform | Severity | GT | Pred | Reason |",
              "|---|---|---|---:|---:|---:|---|"]
    for c in result["failure_cases"]:
        lines.append(f"| {c['image_id']} | {c['parent_image']} / {c['source_file']} | {c['transform']} | "
                     f"{c['severity']} | {c['unusable']} | {c['pred_unusable']} | {c.get('reason', '')} |")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "t03_benchmark_details.json", result)
    (out / "t03_benchmark_details.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("out/member6"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = summarize(args.root, args.out)
    print(json.dumps({k: v["counts"] for k, v in result["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()

