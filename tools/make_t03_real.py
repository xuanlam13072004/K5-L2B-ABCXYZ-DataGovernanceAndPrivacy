"""Build a 30-image REAL road-traffic set for T03 from the BDD100K pack, in the same
layout as fixtures/t03_synthetic (originals/ + originals.csv + objects.csv + manifest.csv).

Step 1 (needs internet, run once BEFORE the lab):
    python -m tools.download_real_samples --packs bdd --bdd-n 600   # -> data/real/bdd/
    (the default 240 favours rare fog/tunnel/night scenes, so it may hold < 30 clean daytime images)
Step 2 (offline):
    python -m tools.make_t03_real                              # -> data/real/t03_bdd/

Selection rule (frozen, deterministic with --seed 42):
  * weather in {clear, partly cloudy, overcast} and timeofday == daytime
    -> clean daytime originals, so synthetic night/fog/glare/blur are real changes
  * >= --min-vehicles vehicle boxes (car/truck/bus/motorcycle) with area >= --min-area px^2
  * at most --max-per-scene images per BDD scene type, for some variety
  * first 20 -> split=dev, last 10 -> split=holdout (same 20/10 as the fixture)

WARNING: real people and real licence plates. BDD100K licence = non-commercial
research/education. Keep data/real/ local (gitignored); never push it to a public repo.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.manifest import build_manifest, write_manifest  # noqa: E402

VEHICLES = {"car", "truck", "bus", "motorcycle"}
KEEP_CLASSES = VEHICLES | {"pedestrian", "rider", "bicycle"}  # road users kept in objects.csv
GOOD_WEATHER = {"clear", "partly cloudy", "overcast"}
LICENSE = "BDD100K License - non-commercial research/education only (https://doc.bdd100k.com/license.html)"


def select(meta: pd.DataFrame, dets: pd.DataFrame, n: int, min_vehicles: int, min_area: float,
           max_per_scene: int, seed: int) -> pd.DataFrame:
    dets = dets.assign(area=dets.w * dets.h)
    big_veh = dets[dets["class"].isin(VEHICLES) & (dets.area >= min_area)]
    n_veh = big_veh.groupby("file_id").size().rename("n_big_vehicles")
    cand = meta.merge(n_veh, left_on="file_id", right_index=True, how="left").fillna({"n_big_vehicles": 0})
    cand = cand[cand.weather.isin(GOOD_WEATHER) & (cand.timeofday == "daytime") & (cand.n_big_vehicles >= min_vehicles)]
    if len(cand) < n:
        sys.exit(f"[t03-real] only {len(cand)} candidates match the rule (need {n}). "
                 f"Re-download with a larger --bdd-n or relax --min-vehicles / --min-area.")
    rng = np.random.default_rng(seed)
    cand = cand.iloc[rng.permutation(len(cand))]
    picked, per_scene = [], {}
    for _, r in cand.iterrows():
        if per_scene.get(r.scene, 0) < max_per_scene:
            picked.append(r)
            per_scene[r.scene] = per_scene.get(r.scene, 0) + 1
        if len(picked) == n:
            break
    for _, r in cand.iterrows():  # fill up if the scene cap was too strict
        if len(picked) == n:
            break
        if r.file_id not in {p.file_id for p in picked}:
            picked.append(r)
    return pd.DataFrame(picked).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bdd", default=str(ROOT / "data" / "real" / "bdd"))
    ap.add_argument("--out", default=str(ROOT / "data" / "real" / "t03_bdd"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-holdout", type=int, default=10)
    ap.add_argument("--min-vehicles", type=int, default=2)
    ap.add_argument("--min-area", type=float, default=400.0, help="min vehicle box area in px^2 (after resize)")
    ap.add_argument("--max-per-scene", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    bdd, out = Path(a.bdd), Path(a.out)
    if not (bdd / "metadata.csv").exists():
        print(f"[t03-real] {bdd}/metadata.csv not found. Run first:\n"
              f"    python -m tools.download_real_samples --packs bdd --bdd-n 600", file=sys.stderr)
        return 2
    meta = pd.read_csv(bdd / "metadata.csv")
    dets = pd.read_csv(bdd / "detections.csv")
    sel = select(meta, dets, a.n, a.min_vehicles, a.min_area, a.max_per_scene, a.seed)

    if out.exists():
        shutil.rmtree(out)
    (out / "originals").mkdir(parents=True)
    rows, objs, src = [], [], []
    for i, r in sel.iterrows():
        iid = f"orig_{i:02d}"
        split = "dev" if i < a.n - a.n_holdout else "holdout"
        shutil.copyfile(bdd / r.path, out / "originals" / f"{iid}.jpg")
        rows.append({"image_id": iid, "path": f"originals/{iid}.jpg", "split": split})
        src.append({"image_id": iid, "bdd_file_id": r.file_id, "weather": r.weather, "timeofday": r.timeofday,
                    "scene": r.scene, "width": r.width, "height": r.height, "n_big_vehicles": int(r.n_big_vehicles)})
        d = dets[(dets.file_id == r.file_id) & dets["class"].isin(KEEP_CLASSES)].reset_index(drop=True)
        for j, b in d.iterrows():
            objs.append({"image_id": iid, "obj_id": f"{iid}_o{j}", "class": b["class"],
                         "x": b.x, "y": b.y, "w": b.w, "h": b.h})

    originals = pd.DataFrame(rows)
    originals.to_csv(out / "originals.csv", index=False)
    pd.DataFrame(objs, columns=["image_id", "obj_id", "class", "x", "y", "w", "h"]).to_csv(out / "objects.csv", index=False)
    pd.DataFrame(src).to_csv(out / "source_map.csv", index=False)

    files = list(originals.path) + ["originals.csv", "objects.csv", "source_map.csv"]
    split_map = {p: s for p, s in zip(originals.path, originals.split)}
    write_manifest(build_manifest(out, files, "hf:dgural/bdd100k (BDD100K val) via tools/make_t03_real.py seed "
                                  f"{a.seed}", LICENSE, split=split_map), out / "manifest.csv")

    print(f"[t03-real] {len(originals)} images ({(originals.split == 'dev').sum()} dev / "
          f"{(originals.split == 'holdout').sum()} holdout), {len(objs)} boxes -> {out}")
    print("           scene:", pd.DataFrame(src).scene.value_counts().to_dict())
    print("           classes:", pd.DataFrame(objs)["class"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
