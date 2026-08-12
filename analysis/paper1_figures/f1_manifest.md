# F1 — Variance decomposition

## Data source
`data/ea_ip.csv` (42966 rows). No model predictions.

## Metric
Sequential R² via `sklearn.metrics.r2_score` on group-transform means
(same as `audit_b_heldout_design.factor_variance`).

## Cumulative R² values
| Keys | EA | IP |
|---|---|---|
| smiles_A | 0.418156 | 0.500349 |
| +smiles_B | 0.929302 | 0.896551 |
| +fracA | 0.990206 | 0.985408 |
| +poly_type | 1.000000 | 1.000000 |

## Component shares
| Component | EA | IP |
|---|---|---|
| A | 0.418156 | 0.500349 |
| B\|A | 0.511146 | 0.396202 |
| comp\|A,B | 0.060905 | 0.088857 |
| arch\|A,B,comp | 0.009794 | 0.014592 |
| residual | 0.000000 | 0.000000 |

## Architecture annotations
- EA: 0.98% of total; 13.9% of post-AB residual
- IP: 1.46% of total; 14.1% of post-AB residual

## Cells: 1 whole-dataset computation per target (2 targets).