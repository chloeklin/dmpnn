# Stage 2D Rerun Readiness Report

## 1. Was the dead initialization bug fixed?

**YES.**

Change made in `chemprop/nn/stage2d.py` line 77:

```python
# BEFORE:
alpha_init: float = 0.0,

# AFTER:
alpha_init: float = 0.1,
```

This breaks the zero-gradient symmetry. With `alpha=0.1`, the gradient
`∂L/∂mlp_output = ∂L/∂h_poly · alpha = ∂L/∂h_poly · 0.1 ≠ 0` flows into
the residual MLP from the very first training step.

## 2. Do gradients flow into alpha?

**YES** (after 1 optimizer step).

Bootstrap mechanism:
- **Step 0**: `r_arch=0` (zero-init MLP), so `grad(alpha)=0`.
  But `grad(mlp) = ∂L/∂h_poly · alpha ≠ 0` — MLP learns immediately.
- **Step 1**: MLP outputs `r_arch ≠ 0`, so `grad(alpha) = ∂L/∂h_poly · r_arch ≠ 0`.

Measured values:
| Variant | Step 0: ||grad(mlp_w)|| | Step 1: ||grad(alpha)|| |
|---------|------------------------|------------------------|
| 2d1_fixed | 0.947211 | 0.085063 |
| 2d1_arch | 0.947211 | 0.053250 |

All non-zero. ✅

## 3. Do gradients flow into the residual MLP?

**YES** — from step 0 onward.

| Variant | ||grad(mlp_output_weight)|| | ||grad(mlp_output_bias)|| |
|---------|---------------------------|--------------------------|
| 2d1_fixed | 0.947211 | 0.196932 |
| 2d1_arch | 0.947211 | 0.196932 |

The non-zero `alpha=0.1` provides the gradient path immediately. ✅

## 4. Is 2d1_arch genuinely different from 2d1_fixed?

**YES.** Verified three ways:

1. **Parameter structure**: 2d1_fixed has 1 scalar alpha; 2d1_arch has 3 independent alphas (one per architecture).
2. **Gradient independence**: After 2 optimizer steps, per-arch alpha gradients are different for different architectures.
3. **Functional difference**: With non-zero MLP output, different architecture labels produce different predictions due to `alpha[arch] * r_arch` scaling.

After 2 training steps:
- alpha_alt, alpha_rand, alpha_block have diverged to different values. ✅

## 5. Which experiments remain enabled?

In `scripts/shell/batch_experiments.yaml`:

| Experiment | Status |
|-----------|--------|
| stage2d_frac | Commented out — already validated |
| stage2d_2d0_fixed | Commented out — already validated |
| stage2d_2d0_arch | Commented out — already validated |
| stage2d_2d0_gate | Commented out — already validated |
| **stage2d_2d1_fixed** | **ENABLED — rerun** |
| **stage2d_2d1_arch** | **ENABLED — rerun** |
| stage2d_2d1_gate | Commented out — already validated |

Only `2d1_fixed` and `2d1_arch` are active.

## 6. Is the project ready for rerunning?

**YES.** All prerequisites satisfied:

| Criterion | Status |
|-----------|--------|
| Dead initialization fixed | ✅ alpha_init = 0.1 |
| Gradients flow to MLP (step 0) | ✅ ||grad|| > 0 |
| Gradients flow to alpha (step 1) | ✅ ||grad|| > 0 |
| 2d1_arch ≠ 2d1_fixed | ✅ Independent per-arch alphas |
| Only target experiments enabled | ✅ 2d1_fixed + 2d1_arch only |
| No other code changes required | ✅ |

### To launch the rerun:

```bash
cd scripts/shell
bash batch_generate_scripts.sh
```

This will generate PBS scripts for only the two enabled experiments
(`stage2d_2d1_fixed` and `stage2d_2d1_arch`) and optionally submit them.

### Note on UnscaleTransform bug

The separate evaluation bug (Stage2D predictions in normalized space) still
exists in `chemprop/models/copolymer.py`. This does NOT affect training
(models learn correctly in normalized space). It only affects test-time
metrics and saved predictions. Fix this before evaluating the new checkpoints,
or continue using the post-hoc inverse transform from the recovery audit.
