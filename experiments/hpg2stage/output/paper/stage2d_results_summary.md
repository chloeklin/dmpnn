# Stage 2D Results Summary

## 1. Composition Dominance

Composition explains **99.0%** of EA variance and **98.5%** of IP variance.
Architecture explains only **1.0%** (EA) and **1.5%** (IP).

Composition (monomer identity + fractions) overwhelmingly determines copolymer EA/IP. Architecture is a small but real residual effect.

## 2. Architecture Residual Contribution

- Frac baseline: R²(EA)=0.9741, capturing composition only
- 2D1-arch: R²(EA)=0.9820, adding architecture modeling
- Overall R² improvement from architecture: +0.79 percentage points

The small overall R² gain reflects the small variance fraction, but architecture-deviation R² reveals the model's ability to correctly rank architectures within matched groups.

## 3. Global vs Chemistry-Conditioned Architecture Effects

- **Diagnostic 3A** (global offset): Transfer R²(EA)=0.210, R²(IP)=0.249
- **Diagnostic 3B** (EA): arch_only R²=0.277 → arch_chem_frac R²=0.507
- **Diagnostic 3B** (IP): arch_only R²=0.340 → arch_chem_frac R²=0.621

Chemistry-conditioned models substantially outperform global offsets, confirming architecture effects are monomer-dependent and justifying graph-based Stage 2D models.

## 4. 2D0 vs 2D1 Findings

- EA: 2D0-arch R²(Δ)=0.9579, 2D1-arch R²(Δ)=0.9633
- IP: 2D0-arch R²(Δ)=0.9523, 2D1-arch R²(Δ)=0.9540

2D1 (learnable architecture embeddings) provides modest improvement over 2D0 (ordinal encoding). Both substantially outperform the Frac baseline for architecture ranking.

## 5. Generalization Findings

- **a_held_out**: 2D1-arch R²(Δ,EA)=0.848, R²(Δ,IP)=0.891
- **group_disjoint**: 2D1-arch R²(Δ,EA)=0.938, R²(Δ,IP)=0.965
- **pair_disjoint**: 2D1-arch R²(Δ,EA)=0.934, R²(Δ,IP)=0.964

Architecture-deviation R² is *maintained or improved* under stricter generalization splits (group-disjoint, pair-disjoint), demonstrating that architecture effects transfer to completely unseen monomer systems.

## 6. Learning-Curve Findings

- **2D0-arch EA R²(Δ)**: 25%: 0.746, 50%: 0.813, 75%: 0.833, 100%: 0.841
- **2D0-arch IP R²(Δ)**: 25%: 0.858, 50%: 0.889, 75%: 0.899, 100%: 0.905
- **2D1-arch EA R²(Δ)**: 25%: 0.812, 50%: 0.854, 75%: 0.854, 100%: 0.868
- **2D1-arch IP R²(Δ)**: 25%: 0.887, 50%: 0.897, 75%: 0.915, 100%: 0.917

Performance at 25% is already substantial, indicating the model learns architecture effects efficiently. Marginal gains from 75%→100% suggest near-saturation of available architecture signal.

## 7. wDMPNN Comparison

- wDMPNN EA: R²=0.9700, R²(Δ, a_held_out)=0.9276
- wDMPNN IP: R²=0.9523, R²(Δ, a_held_out)=0.8797

Under a_held_out, all models (including wDMPNN and Frac) achieve high R²(Δ) because composition groups are shared between train and test, allowing group-level memorization. The critical comparison is in the **generalization** splits: Frac drops to R²(Δ)≈0 under group-disjoint and pair-disjoint, while 2D0/2D1 maintain R²(Δ)≈0.89-0.96. This confirms that only the architecture-aware models genuinely capture architecture effects rather than memorizing group patterns.

**Pending**: wDMPNN group_disjoint and pair_disjoint results not yet available. These are expected to also collapse to near-zero R²(Δ) like Frac, since wDMPNN treats each input SMILES independently without architecture encoding.
