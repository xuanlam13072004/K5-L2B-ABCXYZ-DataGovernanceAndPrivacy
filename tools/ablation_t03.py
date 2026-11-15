"""T03 ablation (Member 6): switch off one part of the quality gate at a time and re-score.

Every variant runs the normal pipeline (python -m src.main) and the official evaluator (src.evaluate)
on the group's self-labelled gold set; only `params.thresholds` in a temporary copy of configs/lab.yaml
changes. No threshold is tuned here — the "full" row uses configs/lab.yaml exactly as committed.

    python -m tools.ablation_t03          # -> reports/ablation_t03.md + reports/ablation_t03.json

Variants (how a check is switched off):
  full                 configs/lab.yaml as committed
  no_erase_check       erase_max_std = -1, erase_min_inside_mad = 1e9 -> "object removed, label kept" is never flagged
  no_glare_box_check   glare_sev1_strength = 99    -> severity-1 glare is never flagged, even on an object
  no_blur_check        blur_sev1_lap_ratio = -1    -> blur is never recognised
  no_night_check       night_sev1_mean_ratio = -1  -> darkening is never recognised
Gold set (re-created if missing, same seed -> same files; identical to Member 6's benchmark set):
  team_holdout  10 holdout originals of the team's 30 images (data/t03_real) x 13 variants = 130 images
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "full": {},
    "no_erase_check": {"erase_max_std": -1, "erase_min_inside_mad": 1e9},
    "no_glare_box_check": {"glare_sev1_strength": 99},
    "no_blur_check": {"blur_sev1_lap_ratio": -1},
    "no_night_check": {"night_sev1_mean_ratio": -1},
}


def _run(args: list[str]) -> None:
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[ablation] failed: {' '.join(args)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def ensure_gold(work: Path) -> dict[str, tuple[Path, Path]]:
    gold = work / "gold"
    source = ROOT / "data" / "t03_real"
    if not (source / "originals.csv").exists():
        _run(["-m", "tools.import_team_labels", "--out", str(source), "--seed", "42"])
    if not (gold / "team_holdout").exists():
        _run(["-m", "tools.make_gold_set", "--source", str(source), "--split", "holdout",
              "--name", "team_holdout", "--out-root", str(gold), "--seed", "42"])
    return {"team_holdout": (gold / "team_holdout", gold / "team_holdout_gt.csv")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(ROOT / "out" / "ablation"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "ablation_t03.md"))
    a = ap.parse_args(argv)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sets = ensure_gold(work)

    base_cfg = yaml.safe_load((ROOT / "configs" / "lab.yaml").read_text(encoding="utf-8"))
    base_cfg["track_config"] = str((ROOT / "configs" / "tracks" / "t03.yaml").resolve())
    rows = []
    for vname, override in VARIANTS.items():
        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("params", {}).setdefault("thresholds", {}).update(override)
        cfg_path = work / f"config_{vname}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        for sname, (inp, gt) in sets.items():
            out = work / vname / sname
            _run(["-m", "src.main", "--config", str(cfg_path), "--input", str(inp), "--output", str(out),
                  "--run-name", vname, "--offline", "--seed", "42"])
            mpath = out / "metrics.json"
            _run(["-m", "src.evaluate", "--track", "t03", "--config", str(cfg_path), "--gt", str(gt),
                  "--pred", str(out / "qc.csv"), "--run-name", vname, "--out", str(mpath)])
            met = {m["name"]: m for m in json.loads(mpath.read_text(encoding="utf-8"))["metrics"]}
            sl = {s["slice"]: s["f1"] for s in json.loads(mpath.read_text(encoding="utf-8"))["extra"].get("slices", [])}
            rows.append({"variant": vname, "gold_set": sname, "n": met["Transform QC F1"]["n"],
                         "f1": met["Transform QC F1"]["value"], "precision": met["Precision (unusable)"]["value"],
                         "recall": met["Recall (unusable)"]["value"],
                         "missed_erase": met["Missed label-invalidating edits"]["value"], "slice_f1": sl,
                         "thresholds_override": override})
            print(f"[ablation] {vname:20s} {sname:13s} F1={rows[-1]['f1']:.4f}")

    full = {r["gold_set"]: r["f1"] for r in rows if r["variant"] == "full"}
    lines = ["# T03 ablation — sinh bằng `python -m tools.ablation_t03`", "",
             "Mỗi dòng tắt **một** phần của bộ lọc QC (chỉ đổi `params.thresholds` trong bản sao tạm của "
             "`configs/lab.yaml`), chạy lại pipeline và chấm bằng `src.evaluate`. Không tune ngưỡng ở đây.", "",
             "| Biến thể | Gold set | N | F1 | ΔF1 so với full | Precision | Recall | Erase bị lọt |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    for r in rows:
        d = None if r["f1"] is None or full.get(r["gold_set"]) is None else r["f1"] - full[r["gold_set"]]
        lines.append(f"| {r['variant']} | {r['gold_set']} | {r['n']} | {fmt(r['f1'])} | "
                     f"{'—' if r['variant'] == 'full' else fmt(d)} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                     f"{int(r['missed_erase'])} |")
    lines += ["", "## F1 theo kiểu biến đổi", "", "| Biến thể | Gold set | blur | erase | fog | glare | night |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        s = r["slice_f1"]
        lines.append(f"| {r['variant']} | {r['gold_set']} | " + " | ".join(fmt(s.get(k)) for k in
                     ("blur", "erase", "fog", "glare", "night")) + " |")
    lines += ["", "Gold set: nhãn suy ra từ phép biến đổi theo `README_usability_rule.md` "
              "(`tools/make_gold_set.py`), không phải nhãn ẩn của giảng viên."]
    Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(a.report).with_suffix(".json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ablation] -> {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
