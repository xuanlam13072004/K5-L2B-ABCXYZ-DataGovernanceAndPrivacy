"""Synthetic Data Generator with Full Provenance (Owned by Role 3).

Applies controlled, deterministic corruptions (night, glare, fog, blur) to dev images
and exports a compliant transform_manifest.csv (aligned with PROV-O / Croissant 1.1).

Usage:
    python tools/generate_synthetic_dev.py --seed 42 --output-manifest out/baseline/transform_manifest.csv
    python tools/generate_synthetic_dev.py --seed 42 --output-manifest out/poc/transform_manifest.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from src.common import corruptions


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_dev_synthetic(
    input_dir: Path,
    output_img_dir: Path,
    output_manifest_paths: list[Path],
    seed: int = 42,
) -> pd.DataFrame:
    input_dir = Path(input_dir).resolve()
    output_img_dir = Path(output_img_dir).resolve()
    output_img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1. Read input images (either from originals.csv or directly from custom folder)
    orig_csv = input_dir / "originals.csv"
    if orig_csv.exists():
        df_orig = pd.read_csv(orig_csv)
        dev_images = df_orig[df_orig["split"] == "dev"].reset_index(drop=True)
        img_items = [(str(row["image_id"]), input_dir / str(row["path"])) for _, row in dev_images.iterrows()]
    else:
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_exts])
        if not image_files:
            raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {input_dir}")
        img_items = [(p.stem, p) for p in image_files]

    print(f"[Role 3 · Generator] Tìm thấy {len(img_items)} ảnh gốc tại {input_dir.name} để áp dụng biến đổi.")

    manifest_rows = []

    # Transforms to apply: (name, function, severity_levels)
    transform_configs = [
        ("night", corruptions.night, [1, 2]),
        ("glare", corruptions.glare, [1, 2]),
        ("fog", corruptions.fog, [1, 2]),
        ("blur", corruptions.blur, [1, 2]),
    ]

    counter = 0

    for parent_id, src_path in img_items:
        if not src_path.exists():
            continue

        parent_sha = compute_sha256(src_path)

        # Load RGB image
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for transform_name, transform_fn, severities in transform_configs:
            for sev in severities:
                synth_id = f"synth_{parent_id}_{transform_name}_s{sev}_{counter:04d}"
                out_filename = f"{synth_id}.jpg"
                out_filepath = output_img_dir / out_filename

                # Apply deterministic transformation
                if transform_name in ("night", "glare", "noise", "rain"):
                    transformed_rgb = transform_fn(rgb, severity=sev, rng=rng)
                else:
                    transformed_rgb = transform_fn(rgb, severity=sev)

                # Save output image
                transformed_bgr = cv2.cvtColor(transformed_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_filepath), transformed_bgr)

                # Compute sha256 of synthetic image
                child_sha = compute_sha256(out_filepath)

                # Log provenance record
                manifest_rows.append({
                    "image_id": synth_id,
                    "parent_image": parent_id,
                    "transform": transform_name,
                    "severity": sev,
                    "seed": seed,
                    "params": f"severity={sev};seed={seed}",
                    "parent_sha256": parent_sha,
                    "sha256": child_sha,
                    "created_by": "Role 3 (Synthetic Generator)",
                })
                counter += 1

    df_manifest = pd.DataFrame(manifest_rows)

    # Save to all target manifest paths
    for p in output_manifest_paths:
        p = Path(p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        df_manifest.to_csv(p, index=False)
        print(f"[✓] Đã ghi Provenance Manifest ({len(df_manifest)} bản ghi) -> {p}")

    print(f"\n[Role 3 · Generator] Hoàn tất sinh {counter} ảnh synthetic tại: {output_img_dir}")
    return df_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Generator for T03")
    parser.add_argument("--input", default="data/image", help="Directory of source images (e.g. data/image)")
    parser.add_argument("--out-dir", default="data/custom_synthetic", help="Output directory for generated images")
    parser.add_argument("--out-manifest", nargs="+", default=["out/baseline/transform_manifest.csv", "out/poc/transform_manifest.csv"], help="Output path(s) for transform_manifest.csv")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")

    args = parser.parse_args()
    generate_dev_synthetic(
        input_dir=Path(args.input),
        output_img_dir=Path(args.out_dir),
        output_manifest_paths=[Path(p) for p in args.out_manifest],
        seed=args.seed,
    )
