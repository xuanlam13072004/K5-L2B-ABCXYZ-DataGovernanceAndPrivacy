Baseline: `baseline` (git debf3cf, seed 42) · PoC: `poc` (git debf3cf, seed 42)

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Transform QC F1** (F1, positive = unusable) | all | 0.5567 | 1.0000 | 0.4433 | +79.6% | higher | 130 |
| Precision (unusable) (TP/(TP+FP)) | all | 0.6000 | 1.0000 | 0.4000 | +66.7% | higher | 130 |
| Recall (unusable) (TP/(TP+FN)) | all | 0.5192 | 1.0000 | 0.4808 | +92.6% | higher | 130 |
| Missed label-invalidating edits (count) | all | 10.0000 | 0.0000 | -10.0000 | -100.0% | lower | None |
| Provenance completeness (manifest rows with all required fields) | all | n/a | 1.0000 | n/a | n/a | higher | None |
| Runtime / 100 items (s) | all | 0.6258 | 17.4670 | | | lower | |
