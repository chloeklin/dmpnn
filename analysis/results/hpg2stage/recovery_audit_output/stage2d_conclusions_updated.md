# Stage 2D Updated Conclusions

## After Normalization Fix (Post-Hoc Inverse Transform)

### Best Models by Overall R²

- **EA**: stage2d_2d0_arch (R2=0.9812)
- **IP**: stage2d_2d1_gate (R2=0.9793)

### Best Models by Architecture-Deviation R²

- **EA**: stage2d_2d1_gate (R2_dev=0.8519)
- **IP**: stage2d_2d0_arch (R2_dev=0.8851)

### Does 2D1 outperform 2D0?

- **EA**: 2D1 best R²=0.9806 vs 2D0 best R²=0.9812 (Δ=-0.0006)
- **IP**: 2D1 best R²=0.9793 vs 2D0 best R²=0.9784 (Δ=+0.0009)

Architecture deviation:
- **EA**: 2D1 best R²(Δy)=0.8519 vs 2D0 best R²(Δy)=0.8240 (Δ=+0.0278)
- **IP**: 2D1 best R²(Δy)=0.8849 vs 2D0 best R²(Δy)=0.8851 (Δ=-0.0002)

### Statistical Significance

Cannot perform cross-validation significance test from pooled predictions.
Per-fold R² needed for paired t-test. Available from per-split predictions.

#### Per-Split R² Comparison (2D0-arch vs 2D1-gate):

- **EA**: 2D0-arch mean R²=0.9813, 2D1-gate mean R²=0.9812, paired t=-0.115, p=0.9141
- **IP**: 2D0-arch mean R²=0.9796, 2D1-gate mean R²=0.9801, paired t=0.340, p=0.7506

### Are 2D1-fixed and 2D1-arch genuinely different?

**NO.** They produce identical predictions and have identical training histories:
- Same epoch and step for every checkpoint
- Checkpoint file sizes differ by only 64 bytes (the extra per-arch alpha parameters)

Per-architecture alphas in 2D1-arch:
  - alternating: 0.000000 ± 0.000000
  - random: 0.000000 ± 0.000000
  - block: 0.000000 ± 0.000000
  - 2D1-fixed alpha: 0.000000 ± 0.000000

If per-architecture alphas ≈ fixed alpha, the models are functionally identical.

**Root Cause: Dead Initialization Symmetry**

In 2D1 variants, `h_poly = h_mix + alpha * r_arch` where:
- `alpha` initialized to 0.0
- `r_arch = mlp_2d1(z)` with **zero-initialized output layer** (weight=0, bias=0)

At initialization: `r_arch = 0` and `alpha = 0`. Gradients:
- `∂L/∂alpha = ∂L/∂h_poly · r_arch = 0` (since r_arch = 0)
- `∂L/∂mlp_out = ∂L/∂h_poly · alpha = 0` (since alpha = 0)

Both parameters have zero gradients forever → **stuck at initialization**.

This does NOT affect 2D0 because `e_arch = embedding(arch)` is randomly initialized (non-zero),
so `∂L/∂alpha = ∂L/∂h_poly · e_arch ≠ 0`, allowing alpha to escape zero.

**Fix**: Initialize `alpha_init = 0.1` (not 0.0) OR remove zero-init from mlp_2d1 output layer.

### Is Retraining Required?

**Assessment:**
- The normalization bug affects evaluation metrics ONLY, not the learned weights.
- Models trained correctly in normalized space.
- Post-hoc inverse transform recovers correct evaluation metrics.
- **Retraining is NOT required** for frac, 2d0_fixed, 2d0_arch, 2d0_gate, 2d1_gate.

**MUST be retrained:**
- `2d1_fixed` — alpha stuck at 0 (dead initialization), equivalent to Frac
- `2d1_arch` — all per-arch alphas stuck at 0 (dead initialization), equivalent to Frac

**Code bugs to fix before retraining:**
1. Missing `UnscaleTransform` in Stage2D prediction path (copolymer.py)
2. Dead initialization in 2D1 variants: either set `alpha_init ≠ 0` or don't zero-init mlp_2d1 output

### Experiments Needing Re-Running

| Experiment | Reason | Priority |
|-----------|--------|----------|
| 2d1_fixed (all reps, EA + IP) | Dead alpha=0, model is just Frac | HIGH |
| 2d1_arch (all reps, EA + IP) | Dead alpha=0, model is just Frac | HIGH |
| None of the others | Weights correct, metrics recovered post-hoc | - |