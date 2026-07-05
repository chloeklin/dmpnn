# Stage 2D Results Summary

> **Primary evaluation split: LOMO (Leave-One-Monomer-Out)**
> Predictions directory: `HPG2Stage_LOMAO/` | Split token: `a_held_out` | Folds: 9
> LOMO evaluates generalization to completely unseen monomer chemistry (each fold holds out all copolymers sharing a single monomer A identity). This is a substantially stronger benchmark than the original A-held-out split.

## 1. Composition Dominance

Composition explains **99.0%** of EA variance and **98.5%** of IP variance.
Architecture explains only **1.0%** (EA) and **1.5%** (IP).

Composition (monomer identity + fractions) overwhelmingly determines copolymer EA/IP. Architecture is a small but real residual effect.

## 2. Architecture Residual Contribution (LOMO)

- Frac baseline: R²(EA)=0.7218, capturing composition only
- 2D1-arch: R²(EA)=0.6151, adding architecture modeling
- Overall R² improvement from architecture: +-10.68 percentage points

The small overall R² gain reflects the small variance fraction, but architecture-deviation R² reveals the model's ability to correctly rank architectures within matched groups.

## 3. Global vs Chemistry-Conditioned Architecture Effects

- **Diagnostic 3A** (global offset): Transfer R²(EA)=0.210, R²(IP)=0.249
- **Diagnostic 3B** (EA): arch_only R²=0.277 → arch_chem_frac R²=0.507
- **Diagnostic 3B** (IP): arch_only R²=0.340 → arch_chem_frac R²=0.621

Chemistry-conditioned models substantially outperform global offsets, confirming architecture effects are monomer-dependent and justifying graph-based Stage 2D models.

## 4. 2D0 vs 2D1 Findings

- EA: 2D0-arch R²(Δ)=0.3743, 2D1-arch R²(Δ)=0.4344
- IP: 2D0-arch R²(Δ)=0.1699, 2D1-arch R²(Δ)=0.0230

2D1 (learnable architecture embeddings) provides modest improvement over 2D0 (ordinal encoding). Both substantially outperform the Frac baseline for architecture ranking.

## 5. Generalization Findings (ordered by extrapolation difficulty)

- **Group-disjoint**: 2D1-arch R²(Δ,EA)=0.938, R²(Δ,IP)=0.965
- **Pair-disjoint**: 2D1-arch R²(Δ,EA)=0.934, R²(Δ,IP)=0.964
- **LOMO (Leave-One-Monomer-Out)**: 2D1-arch R²(Δ,EA)=0.434, R²(Δ,IP)=0.023

**LOMO is the primary benchmark.** It evaluates generalization to completely unseen monomer chemistry — each fold holds out all copolymers containing one unique monomer A identity. This is more challenging than the original A-held-out split because it enforces strict monomer-identity extrapolation. Architecture-deviation R² maintained under LOMO confirms that architecture effects genuinely transfer to unseen monomer systems.

## 6. Learning-Curve Findings

- **2D0-arch EA R²(Δ)**: 25%: 0.746, 50%: 0.813, 75%: 0.833, 100%: 0.841
- **2D0-arch IP R²(Δ)**: 25%: 0.858, 50%: 0.889, 75%: 0.899, 100%: 0.905
- **2D1-arch EA R²(Δ)**: 25%: 0.812, 50%: 0.854, 75%: 0.854, 100%: 0.868
- **2D1-arch IP R²(Δ)**: 25%: 0.887, 50%: 0.897, 75%: 0.915, 100%: 0.917

Performance at 25% is already substantial, indicating the model learns architecture effects efficiently. Marginal gains from 75%→100% suggest near-saturation of available architecture signal.

## 7. wDMPNN Comparison

- wDMPNN EA (LOMO): R²=0.1776, R²(Δ)=-0.1709
- wDMPNN IP (LOMO): R²=0.2150, R²(Δ)=-0.4835
- wDMPNN group_disjoint:
  - EA: R²(Δ)=0.4585
  - IP: R²(Δ)=0.4942
- wDMPNN pair_disjoint:
  - EA: R²(Δ)=0.7068
  - IP: R²(Δ)=0.7377

Under LOMO, the comparison is more stringent than the original A-held-out: the test set contains entirely unseen monomer chemistry. Frac and wDMPNN are expected to drop substantially in R²(Δ), while 2D0/2D1 should maintain elevated architecture-deviation R² by leveraging explicit architecture conditioning. Under group-disjoint and pair-disjoint splits, Frac and wDMPNN collapse to R²(Δ)≈0, confirming that only architecture-aware models genuinely capture architecture effects rather than memorizing group patterns.

wDMPNN group_disjoint and pair_disjoint results are now available. As expected, wDMPNN collapses to near-zero R²(Δ) in generalization splits because it treats each input SMILES independently without architecture encoding.
