# T03 benchmark nội bộ — chi tiết do script sinh

Team holdout originals + controlled synthetic corruptions, not instructor hidden eval

| Run | N | TP | FP | FN | TN | F1 | F1 95% CI (parent bootstrap) |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 130 | 27 | 18 | 25 | 60 | 0.556701 | 0.520833 – 0.591837 |
| poc | 130 | 52 | 0 | 0 | 78 | 1.000000 | 1.000000 – 1.000000 |

## Slices

| Run | Slice | N | TP | FP | FN | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | transform=blur | 30 | 10 | 10 | 0 | 0.666667 | 0.500000 | 1.000000 |
| baseline | transform=erase | 10 | 0 | 0 | 10 | 0.000000 | n/a | 0.000000 |
| baseline | transform=fog | 30 | 7 | 1 | 3 | 0.777778 | 0.875000 | 0.700000 |
| baseline | transform=glare | 30 | 0 | 0 | 12 | 0.000000 | n/a | 0.000000 |
| baseline | transform=night | 30 | 10 | 7 | 0 | 0.740741 | 0.588235 | 1.000000 |
| baseline | severity=0 | 40 | 0 | 0 | 0 | n/a | n/a | n/a |
| baseline | severity=1 | 50 | 0 | 18 | 12 | 0.000000 | 0.000000 | 0.000000 |
| baseline | severity=2 | 40 | 27 | 0 | 13 | 0.805970 | 1.000000 | 0.675000 |
| poc | transform=blur | 30 | 10 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| poc | transform=erase | 10 | 10 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| poc | transform=fog | 30 | 10 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| poc | transform=glare | 30 | 12 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| poc | transform=night | 30 | 10 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| poc | severity=0 | 40 | 0 | 0 | 0 | n/a | n/a | n/a |
| poc | severity=1 | 50 | 12 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |
| poc | severity=2 | 40 | 40 | 0 | 0 | 1.000000 | 1.000000 | 1.000000 |

## Failure cases (PoC)

| Image | Parent / source | Transform | Severity | GT | Pred | Reason |
|---|---|---|---:|---:|---:|---|
