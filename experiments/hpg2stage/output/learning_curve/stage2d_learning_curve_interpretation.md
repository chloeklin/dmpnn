# Stage 2D Learning Curve Interpretation

## Experiment Design

- **Models**: 2D0-arch, 2D1-arch
- **Training fractions**: 25%, 50%, 75%, 100% of training matched groups
- **Evaluation**: Full original test set (unchanged)
- **Split**: a_held_out (monomer A held out per fold)
- **Matched group**: (smiles_A, smiles_B, fracA, fracB)
- **Key constraint**: Entire groups selected together (no architecture leak)

## Summary Results

### Architecture-Deviation R²

| Model | Fraction | EA R²(Δ) | IP R²(Δ) |
|-------|----------|----------|----------|
| 2D0-arch | 25% | 0.3611 ± 0.3417 | 0.3173 ± 0.4079 |
| 2D0-arch | 50% | 0.4544 ± 0.2878 | 0.2524 ± 0.5671 |
| 2D0-arch | 75% | 0.4217 ± 0.2872 | 0.4837 ± 0.2560 |
| 2D0-arch | 100% | 0.4742 ± 0.3337 | 0.3984 ± 0.3010 |
| 2D1-arch | 25% | 0.3496 ± 0.4095 | 0.4360 ± 0.5561 |
| 2D1-arch | 50% | 0.5102 ± 0.2381 | 0.5166 ± 0.4526 |
| 2D1-arch | 75% | 0.4803 ± 0.3534 | 0.5408 ± 0.4029 |
| 2D1-arch | 100% | 0.3498 ± 0.6181 | 0.5318 ± 0.4691 |

### Overall R²

| Model | Fraction | EA R² | IP R² |
|-------|----------|-------|-------|
| 2D0-arch | 25% | 0.9450 ± 0.0386 | 0.9241 ± 0.0466 |
| 2D0-arch | 50% | 0.9531 ± 0.0326 | 0.9430 ± 0.0162 |
| 2D0-arch | 75% | 0.9528 ± 0.0291 | 0.9370 ± 0.0297 |
| 2D0-arch | 100% | 0.9436 ± 0.0509 | 0.9331 ± 0.0390 |
| 2D1-arch | 25% | 0.8055 ± 0.2708 | 0.8895 ± 0.0889 |
| 2D1-arch | 50% | 0.8169 ± 0.2666 | 0.9146 ± 0.0486 |
| 2D1-arch | 75% | 0.8556 ± 0.2031 | 0.9127 ± 0.0509 |
| 2D1-arch | 100% | 0.8740 ± 0.1671 | 0.9045 ± 0.0755 |

## Quantitative Analysis

### Improvement from 25% → 100%

| Model | Metric | 25% value | 100% value | Δ(100%-25%) |
|-------|--------|-----------|------------|-------------|
| 2D0-arch | EA R²(Δ) | 0.3611 | 0.4742 | +0.1132 |
| 2D0-arch | IP R²(Δ) | 0.3173 | 0.3984 | +0.0811 |
| 2D0-arch | EA R² | 0.9450 | 0.9436 | -0.0014 |
| 2D0-arch | IP R² | 0.9241 | 0.9331 | +0.0090 |
| 2D1-arch | EA R²(Δ) | 0.3496 | 0.3498 | +0.0002 |
| 2D1-arch | IP R²(Δ) | 0.4360 | 0.5318 | +0.0958 |
| 2D1-arch | EA R² | 0.8055 | 0.8740 | +0.0685 |
| 2D1-arch | IP R² | 0.8895 | 0.9045 | +0.0150 |

## Interpretation

### 1. Does R²(ΔEA) plateau?

- **2D0-arch**: NO, still improving. 75%→100% gain: +0.0525. Total 25%→100%: +0.1132
- **2D1-arch**: MARGINAL. 75%→100% gain: -0.1305. Total 25%→100%: +0.0002

### 2. Does R²(ΔIP) plateau?

- **2D0-arch**: MARGINAL. 75%→100% gain: -0.0853. Total 25%→100%: +0.0811
- **2D1-arch**: MARGINAL. 75%→100% gain: -0.0091. Total 25%→100%: +0.0958

### 3. Is 2D1 more data-hungry than 2D0?

- EA arch-dev improvement (25%→100%): 2D0=+0.1132, 2D1=+0.0002
- IP arch-dev improvement (25%→100%): 2D0=+0.0811, 2D1=+0.0958

**Mixed signal** — see per-target breakdown above.

### 4. Is performance still improving at 100% training data?

- 2D0-arch EA R²(Δ): **Still improving** (75%→100%: +0.0525)
- 2D0-arch IP R²(Δ): Saturated (75%→100%: -0.0853)
- 2D1-arch EA R²(Δ): Saturated (75%→100%: -0.1305)
- 2D1-arch IP R²(Δ): Saturated (75%→100%: -0.0091)

### 5. Final Verdict

| Criterion | Evidence |
|-----------|----------|
| Avg total improvement (25%→100%) | +0.0725 |
| Avg late improvement (75%→100%) | -0.0431 |
| **Verdict** | **Inconclusive / approaching saturation** |

Late-stage improvements are small but nonzero. The dataset may be approaching saturation, but marginal gains from more data cannot be ruled out.

## Decision Rule Application

```
Small but nonzero gains  →  approaching saturation (inconclusive)  ✓
```
