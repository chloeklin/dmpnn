# Experiment 5: Bottleneck Diagnosis

## Evidence Summary

### 1. Dataset-Content Bottleneck

- **18,414 usable matched groups** with ≥2 architectures
- **100% of samples** belong to usable matched groups
- 6,138 groups with all 3 architectures (at f_A = 0.5)
- 12,276 groups with 2 architectures (block + random, at f_A ∈ {0.25, 0.75})

**Verdict**: Dataset-content bottleneck is **NOT supported** by group count. The dataset provides abundant matched architecture comparisons.

**However**: Only 9 unique monomer A species exist, creating a diversity bottleneck for the a_held_out cross-validation. Each fold's test set represents a narrow chemical space.

### 2. Noise Bottleneck

- Architecture deviation std: EA = 0.0594 eV, IP = 0.0582 eV
- Architecture deviation mean |Δ|: EA = 0.0368 eV, IP = 0.0353 eV
- P95 |Δ|: EA = 0.1302 eV, IP = 0.1330 eV
- Expected DFT internal precision: < 0.01 eV

**Verdict**: Noise bottleneck is **NOT supported**. Architecture deviations (std ≈ 0.06 eV) are ~6× larger than expected computational noise. There is substantial learnable signal.

### 3. Learning-Curve Bottleneck

Metric stability analysis (evaluation-side proxy):

| Model | Target | R²(Δy) at 25% | R²(Δy) at 100% | Change |
|-------|--------|---------------|----------------|--------|
| 2d0_arch | EA | 0.8417 | 0.8434 | +0.0017 |
| 2d0_arch | IP | 0.9067 | 0.9060 | -0.0007 |
| 2d1_arch | EA | 0.8629 | 0.8626 | -0.0003 |
| 2d1_arch | IP | 0.9133 | 0.9140 | +0.0007 |

**Verdict**: A proper training-side learning curve requires retraining with subsampled matched groups. The evaluation-side proxy only tests metric stability.

**To resolve definitively**: Retrain 2D0-arch and 2D1-arch using 25/50/75/100% of training matched groups on cluster.

### 4. Generalization Bottleneck

- 2D1-arch arch-dev R² on held-out monomers: EA = 0.8626, IP = 0.9140
- 2D0-arch arch-dev R² on held-out monomers: EA = 0.8434, IP = 0.9060
- Frac (no architecture): arch-dev R² ≈ -0.03 (expected)

**Verdict**: Generalization bottleneck is **NOT supported**. 2D1-arch achieves high arch-dev R² (>0.86) on polymers with held-out monomers, demonstrating transferable learning.

The model is NOT merely memorizing group-specific architecture corrections.

### 5. Recommendation

Based on available evidence:

| Bottleneck | Supported? | Evidence |
|-----------|-----------|----------|
| Dataset content (group count) | ❌ No | 18,414 matched groups, 100% coverage |
| Dataset diversity (chemistry) | ⚠️ Partial | Only 9 monomer A species |
| Label noise | ❌ No | Arch deviation (0.059 eV) >> DFT precision |
| Learning curve (training) | ❓ Unknown | Requires cluster retraining |
| Generalization | ❌ No | R²(Δy) = 0.863/0.914 on novel monomers |

**Primary limitation identified**: Not a dataset or model bottleneck, but a **chemical diversity bottleneck** — only 9 monomer A species means the a_held_out CV tests generalization across a very limited chemical manifold.

**Recommended next steps** (in priority order):

1. **No further Stage 2D model design work needed** — the model generalizes well.
2. If external validation is desired: obtain predictions on a dataset with different monomer chemistries to confirm transferability beyond 9 A-monomers.
3. A training-side learning curve (cluster experiment) would confirm whether performance has saturated or could benefit from additional data.
4. No architectural changes are indicated by the diagnostic evidence.
