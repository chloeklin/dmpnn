# Further Experiments Required

**Date:** March 2026

18 of 88 model–split–target combinations have no results.

## Missing Experiments

| model_variant   | missing_splits   |
|:----------------|:-----------------|
| DMPNN +PT       | random           |
| HPG +desc +PT   | random           |
| Linear +PT      | monomer, random  |
| RF +PT          | monomer, random  |
| XGB +PT         | monomer, random  |
| wDMPNN          | random           |

---

## Prioritised Experiment List

### Priority 1 — Critical for paper comparison
- **wDMPNN, random split**: Required to compare against the published ~0.03 eV benchmark.

### Priority 2 — Required to complete the PT analysis
- **DMPNN +PT, random split**: Needed to check consistency with GAT/GIN +PT results.
- **HPG +desc +PT, random split**: Needed to complete the HPG variant grid.
- **All Tabular +PT variants** (both splits): Quantify PT benefit for classical ML.

### Priority 3 — Validation / ablation
- **Polymer-type-stratified random split for all +PT models**: The current random split may not stratify by polymer type. Adding a split that ensures each type is equally represented in train/test is needed to confirm whether +PT improvements are real or artefactual.
- **HPG hyperparameter sweep**: Current HPG RMSE > 0.5 eV. Recommended sweep:
  - hidden_dim: [128, 256, 512]
  - depth: [4, 6, 8]
  - max_lr: [1e-3, 3e-3]
  - dropout_ffn: [0.1, 0.2, 0.3]

### Priority 4 — Interpretability
- **Ablation: Identity baseline (PT only) vs GNN+PT**: The identity baseline already achieves RMSE ~0.07 eV. An ablation comparing GNN+PT against PT-only would isolate how much of the GNN+PT gain comes from molecular graph features versus polymer topology alone.

## Summary Table
| Experiment | Reason | Estimated runtime |
|------------|--------|-------------------|
| wDMPNN random split | Paper comparison | ~1.5h |
| DMPNN +PT random split | Complete PT grid | ~2.5h |
| Tabular +PT (all splits) | PT for classical ML | ~2h |
| HPG +desc +PT random | Complete HPG grid | ~1h |
| PT-stratified split ablation | Validate +PT gains | ~5h |
| HPG hyperparameter sweep | Fix poor HPG perf. | ~10h |