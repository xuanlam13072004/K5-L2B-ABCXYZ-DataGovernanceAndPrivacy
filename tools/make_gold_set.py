"""Build a self-labelled T03 test set ("gold set") from original images whose object boxes we know.

Every transformed image is created here, so its true label follows from the frozen usability rule
(fixtures/t03_synthetic/README_usability_rule.md) — no guessing:
  unusable = 1 if  severity == 2 (night / fog / blur / glare)
               or  glare severity >= 1 whose centre falls inside a labelled object box
               or  erase removed a labelled object while its label is kept
This does NOT reproduce the instructor's hidden labels for fixtures/t03_synthetic/holdout_transformed;
it is the group's own, independently generated test set.

    python -m tools.make_gold_set --source fixtures/t03_synthetic --split dev --name fixture_dev   # tune here
    python -m tools.make_gold_set --source data/t03_real --split all --name real                  # test here

Output (data/gold/ is gitignored; re-create with the same command, same seed -> same files):
  data/gold/<name>/                 pipeline INPUT (same layout as fixtures/t03_synthetic)
      originals/, originals.csv, objects.csv, holdout_transformed/, holdout_transformed.csv, manifest.csv
  data/gold/<name>_gt.csv           ANSWERS: image_id, unusable, transform, severity, note   (never give to the pipeline)
  data/gold/<name>_provenance.csv   how each image was made: parent, transform, severity, seed, params

Run + score:
  python -m src.main --config configs/lab.yaml --input data/gold/<name> --output out/gold_<name> --run-name poc --offline --seed 42
  python -m src.evaluate --track t03 --gt data/gold/<name>_gt.csv --pred out/gold_<name>/qc.csv --run-name poc --out reports/metrics_gold_<name>_poc.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import corruptions  # noqa: E402
from src.common.manifest import build_manifest, write_manifest  # noqa: E402

TRANSFORMS = ["night", "glare", "fog", "blur"]
SEVERITIES = [0, 1, 2]


def _inside(cx: float, cy: float, boxes: pd.DataFrame) -> bool:
    return bool(((boxes.x <= cx) & (cx <= boxes.x + boxes.w) & (boxes.y <= cy) & (cy <= boxes.y + boxes.h)).any())


def _glare_centre(rng: np.random.Generator, boxes: pd.DataFrame, W: int, H: int, on_object: bool) -> tuple[float, float]:
    if on_object and len(boxes):
        b = boxes.iloc[int(rng.integers(len(boxes)))]
        return float(b.x + b.w / 2), float(b.y + b.h / 2)
    for _ in range(200):  # a point outside every box
        cx, cy = float(rng.uniform(0.05, 0.95) * W), float(rng.uniform(0.05, 0.6) * H)
        if not _inside(cx, cy, boxes):
            return cx, cy
    return float(W * 0.5), float(H * 0.05)


def _erase(img: np.ndarray, box) -> np.ndarray:
    H, W = img.shape[:2]
    pad = 3
    x0, y0 = max(0, int(box.x) - pad), max(0, int(box.y) - pad)
    x1, y1 = min(W, int(box.x + box.w) + pad), min(H, int(box.y + box.h) + pad)
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def build(source: Path, split: str, name: str, out_root: Path, seed: int) -> None:
    originals = pd.read_csv(source / "originals.csv")
    objects = pd.read_csv(source / "objects.csv")
    if split != "all":
        originals = originals[originals.split == split]
    if originals.empty:
        sys.exit(f"[gold] no originals with split={split} in {source}")
    out = out_root / name
    if out.exists():
        shutil.rmtree(out)
    (out / "originals").mkdir(parents=True)
    (out / "holdout_transformed").mkdir()
    rng = np.random.default_rng(seed)

    hold, gt, prov, orig_rows = [], [], [], []
    for _, o in originals.iterrows():
        pid = o.image_id
        shutil.copyfile(source / o.path, out / "originals" / f"{pid}.jpg")
        orig_rows.append({"image_id": pid, "path": f"originals/{pid}.jpg", "split": "dev"})
        img = np.asarray(Image.open(source / o.path).convert("RGB"))
        H, W = img.shape[:2]
        boxes = objects[objects.image_id == pid]
        variants = []
        for t in TRANSFORMS:
            for s in SEVERITIES:
                params: dict = {}
                if t == "glare" and s > 0:
                    on = bool(rng.random() < 0.5)
                    cx, cy = _glare_centre(rng, boxes, W, H, on)
                    params = {"center": [round(cx, 1), round(cy, 1)]}
                    new = corruptions.glare(img, severity=s, center=(cx, cy))
                else:
                    new = corruptions.TRANSFORMS[t](img, severity=s, rng=np.random.default_rng(seed + len(hold)))
                variants.append((t, s, new, params))
        if len(boxes):
            big = boxes.assign(a=boxes.w * boxes.h).sort_values("a", ascending=False).head(3)
            victim = big.iloc[int(rng.integers(len(big)))]
            variants.append(("erase", 1, _erase(img, victim), {"erased_obj_id": victim.obj_id}))

        for t, s, new, params in variants:
            gid = f"g{len(hold):04d}"          # neutral id: the name must not leak the answer
            rel = f"holdout_transformed/{gid}.jpg"
            Image.fromarray(new).save(out / rel, quality=92)
            if t == "erase":
                unusable, note = 1, "object_removed_label_kept"
            elif s == 2:
                unusable, note = 1, "severity_2"
            elif t == "glare" and s >= 1 and _inside(*params["center"], boxes):
                unusable, note = 1, "glare_on_object"
            else:
                unusable, note = 0, ""
            hold.append({"image_id": gid, "path": rel, "parent_image": pid})
            gt.append({"image_id": gid, "unusable": unusable, "transform": t, "severity": s, "note": note})
            prov.append({"image_id": gid, "parent_image": pid, "transform": t, "severity": s, "seed": seed,
                         "params": json.dumps(params), "generator": "tools.make_gold_set"})

    pd.DataFrame(orig_rows).to_csv(out / "originals.csv", index=False)
    objects[objects.image_id.isin(originals.image_id)].to_csv(out / "objects.csv", index=False)
    pd.DataFrame(hold).to_csv(out / "holdout_transformed.csv", index=False)
    files = [r["path"] for r in orig_rows] + [h["path"] for h in hold] + ["originals.csv", "objects.csv", "holdout_transformed.csv"]
    write_manifest(build_manifest(out, files, f"gold set from {source} (split={split}) via tools/make_gold_set.py seed {seed}",
                                  "same as source images"), out / "manifest.csv")
    g = pd.DataFrame(gt)
    g.to_csv(out_root / f"{name}_gt.csv", index=False)
    pd.DataFrame(prov).to_csv(out_root / f"{name}_provenance.csv", index=False)
    print(f"[gold] {name}: {len(orig_rows)} originals -> {len(g)} images, unusable={int(g.unusable.sum())} "
          f"({g.unusable.mean():.0%}); notes={g.note.replace('', 'usable').value_counts().to_dict()}")
    print(f"       input: {out}   answers: {out_root / (name + '_gt.csv')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="fixtures/t03_synthetic")
    ap.add_argument("--split", default="dev", help="dev / holdout / all")
    ap.add_argument("--name", default="fixture_dev")
    ap.add_argument("--out-root", default=str(ROOT / "data" / "gold"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    build(Path(a.source), a.split, a.name, Path(a.out_root), a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
