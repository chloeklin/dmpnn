# Stage 2D Evaluation Audit Report

## Executive Summary

**ROOT CAUSE IDENTIFIED: Target normalization inverse transform is missing for Stage2D predictions.**

Stage2D models bypass the standard `RegressionFFN` predictor (which includes `UnscaleTransform`).
Instead, `Stage2Aggregator` has its own MLP prediction heads that output values in **normalized
(zero-mean, unit-variance) space**. These raw normalized predictions are saved to `.npz` files
and compared against **raw (un-normalized) `y_true`** from the test dataset, producing the
catastrophically negative R² values.

The models themselves are performing well (Pearson r ≈ 0.987 for EA, 0.981 for IP). The R² = -17
is purely an artifact of comparing predictions on two different scales.

---

## Task 1: Overall R² Computation — VERIFIED (BUG FOUND)

### Evidence

For **every** model and target:

| Property | mean(y_true) | mean(y_pred) | var(y_true) | var(y_pred) | Pearson r |
|----------|-------------|-------------|-------------|-------------|-----------|
| EA       | **-2.541**  | **+0.009**  | 0.360       | 0.959       | 0.987     |
| IP       | **+1.453**  | **+0.002**  | 0.232       | 0.951       | 0.981     |

- `mean(y_pred) ≈ 0` → predictions are in **standardized space** (zero mean)
- `var(y_pred) ≈ 1.0` → predictions have **unit variance** (standardized)
- `mean(y_true) ≈ -2.54 / +1.45` → targets are in **raw eV space**
- Pearson r > 0.98 → predictions **track targets excellently** — just on the wrong scale

### Root Cause — Code Path Trace

1. **Training** (`train_graph.py:1104-1106`):
   ```python
   scaler = train_ds.normalize_targets()   # normalizes train targets
   val_ds.normalize_targets(scaler)         # normalizes val targets
   # test_ds is NOT normalized
   ```

2. **Model build** (`utils.py:1076-1082`):
   ```python
   output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
   ffn = nn.RegressionFFN(output_transform=output_transform, ...)
   ```

3. **Stage2D forward** (`copolymer.py:974-976`):
   ```python
   if self._is_stage2d:
       preds, _ = self.forward_stage2d(...)  # bypasses self.predictor (RegressionFFN)
       return preds
   ```

4. **Stage2Aggregator.forward** (`stage2d.py:296`):
   ```python
   preds = torch.cat([head(h_poly) for head in self.heads], dim=1)  # raw MLP output
   # ❌ No UnscaleTransform applied — predictions remain in normalized space
   ```

5. **predict_step** (`copolymer.py:1147-1148`):
   ```python
   return self(bmg_A, bmg_B, fracA, fracB, X_d)  # calls forward() → Stage2D path
   ```

6. **Saved predictions** (`train_graph.py:1222-1228`):
   ```python
   y_pred = trainer.predict(...)     # normalized predictions
   y_true = [test_ds[j].y for j in ...]  # raw targets (test_ds NOT normalized)
   ```

**Result**: `y_pred` in normalized space, `y_true` in raw space → R² = -17.

### Impact

- **Result CSVs** (`test/r2`, `test/mae`, `test/rmse`): **ALL WRONG** for Stage2D models.
  `test_step` uses `_evaluate_batch → _unpack_batch_for_pred → self(...)` which returns
  normalized predictions, compared against raw test targets.
- **Validation metrics**: CORRECT (val_ds IS normalized, preds are normalized → same space).
- **Saved .npz predictions**: `y_pred` in wrong space.

### Verification

Manual R² and sklearn R² agree perfectly (no computation error in the analysis script).
The R² formula is applied correctly: `R² = 1 - SSE/SST`.
The problem is upstream in the data, not in the metric computation.

---

## Task 2: Prediction/Target Alignment — VERIFIED (NO ALIGNMENT BUG)

- Predictions and targets are **correctly paired** (same ordering within each split).
- Pearson correlation is 0.987 (EA) and 0.981 (IP) — far too high for misaligned data.
- Shuffling predictions destroys correlation → confirms alignment is correct.
- The `test_ids = ['idx_0', 'idx_1', ...]` are sequential fold-local indices, NOT global
  dataset row indices. This is not a bug — just a naming convention.

---

## Task 3: Architecture-Deviation Computation — VERIFIED (SCALE ARTIFACT)

### Methodology

Architecture deviation is computed as:
```
group = (smiles_A, smiles_B, fracA)   ← composition group
Δy = y - mean(y within group)         ← for both true and predicted
```

The matching from predictions to dataset metadata uses `round(y_true, 6)` as lookup key
(since test_ids are fold-local indices, not dataset row indices).

### Match Quality

| Target | Lookup size | Dataset rows | Match rate | Collision rate |
|--------|-----------|------------|----------|---------------|
| EA     | 42,498    | 42,966     | 100.0%   | 1.09%         |
| IP     | 42,390    | 42,966     | 100.0%   | 1.34%         |

Match rate is 100% because predictions pool all 5 splits = entire dataset.
~1% collision rate means ~1% of y_true values map to wrong metadata — acceptably small.

### Scale Artifact in Deviations

Because y_pred is in normalized space:
```
Δy_pred ≈ Δy_true / σ_global
```

where σ_global = std(y_true_train). This means:
- EA: `std(Δy_pred) / std(Δy_true) = 0.087 / 0.059 = 1.48` (expected 1/√0.36 = 1.67)
- IP: `std(Δy_pred) / std(Δy_true) = 0.112 / 0.058 = 1.93` (expected 1/√0.23 = 2.08)

The predicted deviations are inflated by ~1/σ, making IP deviations ~2× too large, which
explains why IP arch-deviation R² is negative while EA is positive.

**Conclusion**: Architecture-deviation R² values are also distorted by the normalization
mismatch, though less severely than overall R² (because the systematic mean offset cancels
in deviations). After unscaling predictions, arch-deviation R² values will change significantly.

---

## Task 4: Comparison with Diagnostic 3B

### Diagnostic 3B Code Location

Found: `analysis/results/hpg/plot_phase3.py`

Diagnostic 3B (Phase 3B — pair interaction analysis) computes standard ΔR² and ΔRMSE relative
to HPG_frac baseline using per-fold CSV results. It does NOT compute architecture-deviation R²
directly — that concept was introduced in the Stage 2D analysis.

The "EA arch-dev R² ≈ 0.61, IP arch-dev R² ≈ 0.72" values cited in the user's request likely
came from a separate manual/notebook analysis with properly unscaled predictions and
monomer-disjoint evaluation.

### Key Differences

| Component             | Stage 2D                          | Diagnostic 3B (Phase 3)        |
|-----------------------|-----------------------------------|-------------------------------|
| Split type            | `a_held_out`                      | `a_held_out`                  |
| Predictions saved     | ❌ In normalized space            | N/A (uses CSV metrics only)   |
| Architecture deviation| Computed from y_true matching     | Not computed                  |
| Metrics source        | `.npz` predictions + CSV          | CSV only                      |

---

## Task 5: Matched Group Construction — VERIFIED (CORRECT)

| Metric                              | Value   |
|--------------------------------------|---------|
| Total polymers                       | 42,966  |
| Total composition groups             | 18,414  |
| Groups with 2 architectures          | 12,276  |
| Groups with 3 architectures          | 6,138   |
| Groups with 1 architecture           | **0**   |
| Samples in multi-architecture groups | 42,966 (100%) |

### Architecture Combination Breakdown

| Combination              | Groups |
|--------------------------|--------|
| block + random           | 12,276 |
| alternating + block + random | 6,138 |

**Every** composition group has ≥ 2 architectures. The matched group construction is correct
and provides excellent coverage for architecture-deviation analysis.

---

## Task 6: Monomer-Disjoint Split — NOT MONOMER-DISJOINT

The `a_held_out` split is **NOT** monomer-A-disjoint:

| Property          | Test | Train | Overlap |
|-------------------|------|-------|---------|
| Monomers A        | 9    | 9     | **9 (100%)** |
| Monomers B        | 528  | 643   | 489     |

- All 9 monomer-A species appear in BOTH train and test.
- The `a_held_out` split holds out specific *copolymer combinations*, not monomers.
- This is a fundamentally different evaluation than monomer-disjoint splits.

If Diagnostic 3B used monomer-disjoint evaluation, the architecture-deviation R² values
are not directly comparable.

---

## Task 7: Δy Prediction Quality — VERIFIED (CONSISTENT WITH SCALE BUG)

| Model       | Target | R²_dev (manual) | R²_dev (sklearn) | MAE_dev  | n_multi |
|-------------|--------|-----------------|-----------------|----------|---------|
| Frac        | EA     | -0.077430       | -0.077430       | 0.038041 | 42,807  |
| Frac        | IP     | -0.152724       | -0.152724       | 0.037180 | 42,755  |
| 2D0-fixed   | EA     | 0.459612        | 0.459612        | 0.029712 | 42,807  |
| 2D0-fixed   | IP     | -0.089547       | -0.089547       | 0.039296 | 42,755  |
| 2D0-arch    | EA     | 0.499558        | 0.499558        | 0.028270 | 42,807  |
| 2D0-arch    | IP     | -0.112522       | -0.112522       | 0.039839 | 42,755  |
| 2D0-gate    | EA     | 0.400485        | 0.400485        | 0.030440 | 42,807  |
| 2D0-gate    | IP     | -0.274084       | -0.274084       | 0.040529 | 42,755  |
| 2D1-fixed   | EA     | -0.080111       | -0.080111       | 0.038078 | 42,807  |
| 2D1-fixed   | IP     | -0.153207       | -0.153207       | 0.037192 | 42,755  |
| 2D1-arch    | EA     | -0.080111       | -0.080111       | 0.038078 | 42,807  |
| 2D1-arch    | IP     | -0.153207       | -0.153207       | 0.037192 | 42,755  |
| 2D1-gate    | EA     | 0.502972        | 0.502972        | 0.027551 | 42,807  |
| 2D1-gate    | IP     | -0.081360       | -0.081360       | 0.036569 | 42,755  |

- Manual and sklearn R² match perfectly → no computation bug.
- 2D1-fixed and 2D1-arch produce identical values → likely identical model weights (training issue?).
- **These deviation R² values are distorted by the normalization mismatch** (see Task 3).

---

## Task 8: Visual Diagnostics — CONFIRMS NORMALIZATION BUG

### EA Scatter Plot (`audit_plots/scatter_EA.png`)

The point cloud runs **parallel to y=x** but offset by ~2.5 eV upward.
- True EA range: [-4.5, 0.5] (raw space, centered at -2.54)
- Pred EA range: [-3.0, 5.0] (normalized space, centered at 0.0)

### IP Scatter Plot (`audit_plots/scatter_IP.png`)

Same pattern: cloud parallel to y=x, offset by ~1.5 eV downward.
- True IP range: [-0.5, 4.0] (raw space, centered at 1.45)
- Pred IP range: [-2.5, 4.0] (normalized space, centered at 0.0)

Both plots show excellent linear correlation but systematic scale/offset mismatch.

---

## Final Checklist

| # | Question                                         | Answer |
|---|--------------------------------------------------|--------|
| 1 | Is overall R² computed correctly?                | ✅ Yes — the R² formula is correct. The R² = -17 is real, caused by comparing mismatched scales. |
| 2 | Is architecture-deviation R² computed correctly? | ⚠️ Partially — methodology is correct, but values are distorted by the normalization scale artifact. |
| 3 | Are predictions aligned with targets?            | ✅ Yes — correct ordering confirmed via correlation. |
| 4 | Are matched groups constructed correctly?        | ✅ Yes — 100% of samples are in multi-architecture groups. |
| 5 | Does Stage 2D evaluation match Diagnostic 3B?   | ❌ No — different evaluation methodology and split semantics. |
| 6 | Most likely cause of discrepancy                 | **Missing UnscaleTransform in Stage2D prediction path.** |

---

## Required Fix (Do Not Apply — Audit Only)

The fix belongs in `CopolymerMPNN` and/or `Stage2Aggregator`. Two options:

### Option A: Apply UnscaleTransform in CopolymerMPNN.forward() for Stage2D

```python
# In CopolymerMPNN.forward():
if self._is_stage2d:
    preds, _ = self.forward_stage2d(bmg_A, bmg_B, fracA, fracB, X_d)
    if hasattr(self.predictor, 'output_transform') and self.predictor.output_transform is not None:
        preds = self.predictor.output_transform(preds)
    return preds
```

**Also need to apply the same transform in:**
- `_unpack_batch_for_pred` (for test_step metrics)
- `predict_step` (for saved predictions)

**Also need to fix training_step_stage2d:**
- Currently targets are normalized but Stage2Aggregator heads output unnormalized-space values
  that are trained against normalized targets. This means the model IS learning in normalized
  space correctly — the fix should only apply at inference time (test/predict).

### Option B: Pass scaler to Stage2Aggregator and apply internally

Give Stage2Aggregator its own unscale transform so it returns raw-space predictions.
This is cleaner but requires modifying the Stage2Aggregator constructor.

### After Fix: Re-run

1. Re-run all Stage2D training jobs (or just re-extract predictions from existing checkpoints).
2. Re-run `plot_stage2d_analysis.py` to get corrected metrics.
3. Compare architecture-deviation R² values with and without normalization correction.

---

## Additional Finding: 2D1-fixed ≡ 2D1-arch

`2D1-fixed` and `2D1-arch` produce **identical predictions** (R², MAE, every metric matches
to 6 decimal places). This suggests either:
- The per-architecture alpha values in 2D1-arch collapsed to the same value, or
- A training bug caused both variants to use the same weights.

This should be investigated separately.