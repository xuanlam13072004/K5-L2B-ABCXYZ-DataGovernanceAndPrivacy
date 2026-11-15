"""Turn the team's 30 labelled road images into a T03 input folder (same layout as fixtures/t03_synthetic).

    python -m tools.import_team_labels                      # -> data/t03_real/
    python -m src.main --config configs/lab.yaml --input data/t03_real --output out/real --run-name real --offline --seed 42

Input : data/image/*.jpg  +  data/obj_train_data.zip (CVAT "YOLO 1.1" export: obj.names + obj_train_data/*.txt)
Output: data/t03_real/originals/orig_XX.jpg, originals.csv, objects.csv, label_fixes.csv, source_map.csv, manifest.csv

Label clean-up rules (frozen; every removed box is written to label_fixes.csv with its reason):
  * class mapping to the fixture taxonomy:  car/truck/bus -> car, person -> pedestrian,
    stop sign / traffic light -> sign  (original YOLO class kept in column `orig_class`)
  * classes outside the mapping are dropped (YOLO false positives, e.g. train, potted plant)
  * ego-vehicle hood: a box wider than 90% of the image that reaches the bottom 5% -> dropped
  * duplicates: same mapped class and IoU >= 0.8 in the same image -> keep the first
Split: 20 dev / 10 holdout, shuffled with --seed (deterministic).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.manifest import build_manifest, write_manifest  # noqa: E402
from src.common import corruptions  # noqa: E402
from src.common.image_ops import load_rgb  # noqa: E402

CLASS_MAP = {"car": "car", "truck": "car", "bus": "car", "person": "pedestrian",
             "stop sign": "sign", "traffic light": "sign"}
SOURCE = "30 BDD100K dashcam frames selected by team G01 (data/image) + CVAT YOLO 1.1 labels (data/obj_train_data.zip)"
LICENSE = "BDD100K License - non-commercial research/education only (https://doc.bdd100k.com/license.html)"


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def read_yolo_zip(zip_path: Path) -> tuple[list[str], dict[str, list[list[str]]]]:
    with zipfile.ZipFile(zip_path) as z:
        names = [l.strip() for l in z.read("obj.names").decode("utf-8").splitlines() if l.strip()]
        labels = {}
        for n in z.namelist():
            if n.startswith("obj_train_data/") and n.endswith(".txt"):
                rows = [l.split() for l in z.read(n).decode("utf-8").splitlines() if l.strip()]
                labels[Path(n).stem] = rows
    return names, labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(ROOT / "data" / "image"))
    ap.add_argument("--labels-zip", default=str(ROOT / "data" / "obj_train_data.zip"))
    ap.add_argument("--out", default=str(ROOT / "data" / "t03_real"))
    ap.add_argument("--n-holdout", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    img_dir, out = Path(a.images), Path(a.out)
    if not Path(a.labels_zip).exists():
        print(f"[labels] not found: {a.labels_zip}", file=sys.stderr)
        return 2
    names, labels = read_yolo_zip(Path(a.labels_zip))
    stems = sorted((p.stem for p in img_dir.glob("*.jpg")), key=lambda s: int(s))
    missing = [s for s in stems if s not in labels]
    if missing:
        print(f"[labels] images without a label file: {missing}", file=sys.stderr)
        return 2

    order = np.random.default_rng(a.seed).permutation(len(stems))
    holdout = {stems[i] for i in order[: a.n_holdout]}

    for sub in ("originals", "holdout_transformed"):   # clear only what this script generates
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / "originals").mkdir(parents=True)
    originals, objects, fixes, source = [], [], [], []
    for k, stem in enumerate(stems):
        iid = f"orig_{k:02d}"
        src = img_dir / f"{stem}.jpg"
        shutil.copyfile(src, out / "originals" / f"{iid}.jpg")
        W, H = Image.open(src).size
        split = "holdout" if stem in holdout else "dev"
        originals.append({"image_id": iid, "path": f"originals/{iid}.jpg", "split": split})
        source.append({"image_id": iid, "source_file": f"data/image/{stem}.jpg", "label_file": f"obj_train_data/{stem}.txt",
                       "width": W, "height": H})
        kept: list[dict] = []
        for j, r in enumerate(labels[stem]):
            cid = int(r[0])
            orig_cls = names[cid] if cid < len(names) else f"class_{cid}"
            xc, yc, w, h = (float(v) for v in r[1:5])
            box = [round((xc - w / 2) * W, 2), round((yc - h / 2) * H, 2), round(w * W, 2), round(h * H, 2)]
            row = {"image_id": iid, "source_file": f"{stem}.jpg", "line": j + 1, "orig_class": orig_cls,
                   "x": box[0], "y": box[1], "w": box[2], "h": box[3]}
            if orig_cls not in CLASS_MAP:
                fixes.append({**row, "reason": "class_not_in_taxonomy (YOLO false positive)"})
                continue
            if w > 0.9 and yc + h / 2 > 0.95:
                fixes.append({**row, "reason": "ego_vehicle_hood"})
                continue
            cls = CLASS_MAP[orig_cls]
            dup = next((k2 for k2 in kept if k2["class"] == cls and _iou(box, [k2["x"], k2["y"], k2["w"], k2["h"]]) >= 0.8), None)
            if dup is not None:
                fixes.append({**row, "reason": f"duplicate_of {dup['obj_id']} (IoU>=0.8)"})
                continue
            kept.append({"image_id": iid, "obj_id": f"{iid}_o{len(kept)}", "class": cls,
                         "x": box[0], "y": box[1], "w": box[2], "h": box[3], "orig_class": orig_cls})
        objects.extend(kept)

    cols = ["image_id", "obj_id", "class", "x", "y", "w", "h", "orig_class"]
    pd.DataFrame(originals).to_csv(out / "originals.csv", index=False)
    pd.DataFrame(objects, columns=cols).to_csv(out / "objects.csv", index=False)
    pd.DataFrame(fixes, columns=["image_id", "source_file", "line", "orig_class", "x", "y", "w", "h", "reason"]).to_csv(
        out / "label_fixes.csv", index=False)
    pd.DataFrame(source).to_csv(out / "source_map.csv", index=False)

    # Generate 45 holdout_transformed images from the 10 holdout originals (deterministic with --seed)
    hold_dir = out / "holdout_transformed"
    hold_dir.mkdir(parents=True, exist_ok=True)
    hold_origs = [o["image_id"] for o in originals if o["split"] == "holdout"]
    objs_df = pd.DataFrame(objects, columns=cols)
    rng_h = np.random.default_rng(a.seed + 100)
    hold_rows = []
    h_idx = 0
    for idx_p, pid in enumerate(hold_origs):
        p_img = load_rgb(out / "originals" / f"{pid}.jpg")
        H, W = p_img.shape[:2]
        p_boxes = objs_df[objs_df["image_id"] == pid].reset_index(drop=True)
        # 4 standard transforms per holdout original (night, glare, fog, blur) + 1 erase on every second holdout original (total 40 + 5 = 45)
        for t_idx, t_name in enumerate(["night", "glare", "fog", "blur"]):
            sev = int(rng_h.choice([0, 1, 2], p=[0.2, 0.5, 0.3]))
            sub_rng = np.random.default_rng(a.seed * 1000 + h_idx)
            if t_name == "glare" and sev >= 1 and len(p_boxes) > 0 and (idx_p % 2 == 0):
                b0 = p_boxes.iloc[0]
                cx = float(b0.x + b0.w * 0.5)
                cy = float(b0.y + b0.h * 0.5)
                t_img = corruptions.glare(p_img, severity=sev, rng=sub_rng, center=(cx, cy))
            else:
                t_img = corruptions.apply(t_name, p_img, severity=sev, seed=a.seed * 1000 + h_idx)
            hid = f"hold_{h_idx:03d}"
            rel_p = f"holdout_transformed/{hid}.jpg"
            Image.fromarray(t_img).save(out / rel_p, quality=95)
            hold_rows.append({"image_id": hid, "path": rel_p, "parent_image": pid})
            h_idx += 1
        if idx_p % 2 == 0 and len(p_boxes) > 0:
            t_img = p_img.copy()
            b0 = p_boxes.iloc[0]
            x0, y0 = max(0, int(b0.x)), max(0, int(b0.y))
            x1, y1 = min(W, int(b0.x + b0.w)), min(H, int(b0.y + b0.h))
            t_img[y0:y1, x0:x1] = 128
            hid = f"hold_{h_idx:03d}"
            rel_p = f"holdout_transformed/{hid}.jpg"
            Image.fromarray(t_img).save(out / rel_p, quality=95)
            hold_rows.append({"image_id": hid, "path": rel_p, "parent_image": pid})
            h_idx += 1

    pd.DataFrame(hold_rows).to_csv(out / "holdout_transformed.csv", index=False)
    files = ([o["path"] for o in originals]
             + [r["path"] for r in hold_rows]
             + ["originals.csv", "objects.csv", "label_fixes.csv", "source_map.csv", "holdout_transformed.csv"])
    split_map = {**{o["path"]: o["split"] for o in originals}, **{r["path"]: "holdout" for r in hold_rows}}
    write_manifest(build_manifest(out, files, SOURCE, LICENSE, split=split_map), out / "manifest.csv")

    obj = pd.DataFrame(objects)
    print(f"[labels] {len(originals)} images ({len(originals) - len(holdout)} dev / {len(holdout)} holdout) -> {out}")
    print(f"         kept {len(obj)} boxes: {obj['class'].value_counts().to_dict()}")
    print(f"         removed {len(fixes)} boxes -> label_fixes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
