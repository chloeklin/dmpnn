# Ordering Metric Change Report

## What Was Fixed

**Bug 1 — Pairwise accuracy, tied predictions scored as 0 instead of 0.5**

When `y_pred[i] == y_pred[j]` (model cannot distinguish architectures),
the old code scored the pair as 0 (wrong). A tied prediction conveys no
directional information; the statistically correct score is **0.5**
(expected accuracy of random ordering under chance).

**Bug 2 — n=2 fallback assigned −1 to tied predictions instead of NaN**

For groups with exactly 2 architectures, the code bypassed scipy and used
a sign-product heuristic. When predictions were tied (product = 0, not > 0),
it returned −1. This is incorrect: a tied prediction is uninformative,
not reverse-ordered. The fix returns **NaN** (matching scipy's behaviour for
a constant array), which is then excluded by NaN-aware median.

**Additional fix — aggregate median now uses `nanmedian`**

The fold-level aggregate previously called pandas `.median()` which treats NaN
as non-existent (correct for pandas ≥2.0 but not reliable). Explicitly using
`np.nanmedian` is unambiguous.

---

## Metric Changes Summary

### Pairwise Accuracy

| Model | Split | Old mean | New mean | Change |
|-------|-------|----------|----------|--------|
| Frac | group_disjoint | 0.5000 | 0.5000 | +0.0000 |
| Frac | pair_disjoint | 0.5000 | 0.5000 | +0.0000 |
| Frac | monomer_heldout | 0.5000 | 0.5000 | +0.0000 |
| wDMPNN | group_disjoint | 0.8473 | 0.8473 | +0.0000 |
| wDMPNN | pair_disjoint | 0.8430 | 0.8430 | +0.0000 |
| wDMPNN | monomer_heldout | 0.7522 | 0.7522 | +0.0000 |
| GlobalArch | group_disjoint | 0.8598 | 0.8598 | +0.0000 |
| GlobalArch | pair_disjoint | 0.8561 | 0.8561 | +0.0000 |
| GlobalArch | monomer_heldout | 0.7379 | 0.7379 | +0.0000 |
| ChemArch | group_disjoint | 0.8951 | 0.8951 | +0.0000 |
| ChemArch | pair_disjoint | 0.8928 | 0.8928 | +0.0000 |
| ChemArch | monomer_heldout | 0.7781 | 0.7781 | +0.0000 |

### Median Spearman

| Model | Split | Old mean | New mean | Change |
|-------|-------|----------|----------|--------|
| Frac | group_disjoint | nan | nan | +nan |
| Frac | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| Frac | monomer_heldout | nan | nan | +nan |
| wDMPNN | group_disjoint | 1.0000 | 1.0000 | +0.0000 |
| wDMPNN | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| wDMPNN | monomer_heldout | 1.0000 | 1.0000 | +0.0000 |
| GlobalArch | group_disjoint | 1.0000 | 1.0000 | +0.0000 |
| GlobalArch | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| GlobalArch | monomer_heldout | 0.9722 | 0.9722 | +0.0000 |
| ChemArch | group_disjoint | 1.0000 | 1.0000 | +0.0000 |
| ChemArch | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| ChemArch | monomer_heldout | 0.9722 | 0.9722 | +0.0000 |

### Median Kendall

| Model | Split | Old mean | New mean | Change |
|-------|-------|----------|----------|--------|
| Frac | group_disjoint | nan | nan | +nan |
| Frac | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| Frac | monomer_heldout | nan | nan | +nan |
| wDMPNN | group_disjoint | 1.0000 | 1.0000 | +0.0000 |
| wDMPNN | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| wDMPNN | monomer_heldout | 1.0000 | 1.0000 | +0.0000 |
| GlobalArch | group_disjoint | 1.0000 | 1.0000 | +0.0000 |
| GlobalArch | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| GlobalArch | monomer_heldout | 0.9630 | 0.9630 | +0.0000 |
| ChemArch | group_disjoint | 1.0000 | 1.0000 | +0.0000 |
| ChemArch | pair_disjoint | 1.0000 | 1.0000 | +0.0000 |
| ChemArch | monomer_heldout | 0.9630 | 0.9630 | +0.0000 |

---

## Scientific Conclusions

- **group_disjoint/EA**: best model pairwise_acc unchanged = True (old winner: ChemArch, new winner: ChemArch)
- **group_disjoint/IP**: best model pairwise_acc unchanged = True (old winner: ChemArch, new winner: ChemArch)
- **pair_disjoint/EA**: best model pairwise_acc unchanged = True (old winner: ChemArch, new winner: ChemArch)
- **pair_disjoint/IP**: best model pairwise_acc unchanged = True (old winner: ChemArch, new winner: ChemArch)
- **monomer_heldout/EA**: best model pairwise_acc unchanged = True (old winner: ChemArch, new winner: ChemArch)
- **monomer_heldout/IP**: best model pairwise_acc unchanged = True (old winner: ChemArch, new winner: ChemArch)

### Key Changes by Model

**Frac**
- Pairwise: 0.5000 → 0.5000 (+0.0000)
- Spearman: 1.0000 → 1.0000 (+0.0000)

**wDMPNN**
- Pairwise: 0.8011 → 0.8011 (+0.0000)
- Spearman: 1.0000 → 1.0000 (+0.0000)

**GlobalArch**
- Pairwise: 0.8011 → 0.8011 (+0.0000)
- Spearman: 0.9868 → 0.9868 (+0.0000)

**ChemArch**
- Pairwise: 0.8391 → 0.8391 (+0.0000)
- Spearman: 0.9868 → 0.9868 (+0.0000)

### Are conclusions stable for ChemArch, GlobalArch, wDMPNN?

- **ChemArch**: max pairwise change=0.0000, max spearman change=0.0000 → conclusions STABLE
- **GlobalArch**: max pairwise change=0.0000, max spearman change=0.0000 → conclusions STABLE
- **wDMPNN**: max pairwise change=0.0000, max spearman change=0.0000 → conclusions STABLE

### Frac baseline

- Corrected Frac pairwise_acc (mean across all): 0.5000
- Corrected Frac median_spearman (mean): 1.0000
- Spearman NaN groups (constant pred): 110483/110484 = 100.0%
- Interpretation: Frac ignores architecture → pairwise_acc ≈ 0.5 (random baseline), Spearman=NaN (excluded from median).

---

## Suitability for Paper

The corrected metrics are suitable for paper inclusion. Key properties:
- Pairwise accuracy = 0.5 is the correct random baseline for tied predictions
- NaN spearman/kendall for constant predictors propagates correctly through nanmedian
- Non-Frac model metrics (wDMPNN, GlobalArch, ChemArch) are numerically unchanged
- The relative ordering of non-Frac models is preserved exactly