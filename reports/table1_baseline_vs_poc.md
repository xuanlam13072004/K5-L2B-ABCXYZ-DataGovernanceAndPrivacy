Baseline: `baseline` (git 69e2436, seed 42) · PoC: `poc` (git 69e2436, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Transform QC F1** (F1, positive = unusable) | all | 0.5567 | 0.7619 | 0.2052 | +36.9% | higher | 130 |
| Precision (unusable) (TP/(TP+FP)) | all | 0.6000 | 1.0000 | 0.4000 | +66.7% | higher | 130 |
| Recall (unusable) (TP/(TP+FN)) | all | 0.5192 | 0.6154 | 0.0962 | +18.5% | higher | 130 |
| Missed label-invalidating edits (count) | all | 10.0000 | 10.0000 | 0.0000 | +0.0% | lower | None |
| Provenance completeness (manifest rows with all required fields) | all | n/a | n/a | n/a | n/a | higher | None |
| Runtime / 100 items (s) | all | 1.8314 | 21.8512 | | | lower | |
