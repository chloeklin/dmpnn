# Diagnostic Summary Report

## Overview

This report synthesises results from the model diagnostics pipeline.

## Summary Tables

### group_disjoint

| Model | Overall R² | Group-mean R² | ΔR² | Cal. Slope | Disp. Ratio | Pairwise Ord. | Frac SSE_between | Frac SSE_within |
|-------|-----------|--------------|-----|-----------|------------|--------------|-----------------|----------------|
| Frac | — | 0.9980 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.144 | 0.856 |
| wDMPNN | — | 0.9984 | 0.8983 | 0.894 | 0.943 | 0.847 | 0.570 | 0.430 |
| GlobalArch | — | 0.9985 | 0.9146 | 0.864 | 0.901 | 0.859 | 0.599 | 0.401 |
| ChemArch | — | 0.9991 | 0.9515 | 0.945 | 0.969 | 0.894 | 0.607 | 0.393 |

### pair_disjoint

| Model | Overall R² | Group-mean R² | ΔR² | Cal. Slope | Disp. Ratio | Pairwise Ord. | Frac SSE_between | Frac SSE_within |
|-------|-----------|--------------|-----|-----------|------------|--------------|-----------------|----------------|
| Frac | — | 0.9977 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.165 | 0.835 |
| wDMPNN | — | 0.9979 | 0.8963 | 0.893 | 0.943 | 0.843 | 0.635 | 0.365 |
| GlobalArch | — | 0.9982 | 0.9134 | 0.866 | 0.904 | 0.855 | 0.645 | 0.355 |
| ChemArch | — | 0.9986 | 0.9492 | 0.945 | 0.969 | 0.892 | 0.702 | 0.298 |

### monomer_heldout

| Model | Overall R² | Group-mean R² | ΔR² | Cal. Slope | Disp. Ratio | Pairwise Ord. | Frac SSE_between | Frac SSE_within |
|-------|-----------|--------------|-----|-----------|------------|--------------|-----------------|----------------|
| Frac | — | -0.0184 | -0.0000 | 0.000 | 0.000 | 0.000 | 0.924 | 0.076 |
| wDMPNN | — | 0.9265 | 0.5209 | 0.565 | 0.740 | 0.751 | 0.681 | 0.319 |
| GlobalArch | — | 0.0239 | 0.4649 | 0.588 | 0.805 | 0.737 | 0.971 | 0.029 |
| ChemArch | — | -0.3658 | 0.5315 | 0.836 | 1.026 | 0.778 | 0.959 | 0.041 |

## Key Findings

### 1. Does wDMPNN win overall because it predicts group means better?

- **group_disjoint**: wDMPNN gm_R²=0.9984, ChemArch gm_R²=0.9991
- **pair_disjoint**: wDMPNN gm_R²=0.9979, ChemArch gm_R²=0.9986
- **monomer_heldout**: wDMPNN gm_R²=0.9265, ChemArch gm_R²=-0.3658

### 2. What fraction of each model's error is between-group vs within-group?

- **group_disjoint/Frac**: between=0.144, within=0.856
- **group_disjoint/wDMPNN**: between=0.570, within=0.430
- **group_disjoint/GlobalArch**: between=0.599, within=0.401
- **group_disjoint/ChemArch**: between=0.607, within=0.393
- **pair_disjoint/Frac**: between=0.165, within=0.835
- **pair_disjoint/wDMPNN**: between=0.635, within=0.365
- **pair_disjoint/GlobalArch**: between=0.645, within=0.355
- **pair_disjoint/ChemArch**: between=0.702, within=0.298
- **monomer_heldout/Frac**: between=0.924, within=0.076
- **monomer_heldout/wDMPNN**: between=0.681, within=0.319
- **monomer_heldout/GlobalArch**: between=0.971, within=0.029
- **monomer_heldout/ChemArch**: between=0.959, within=0.041

### 3. Does ChemArch preserve architecture-deviation magnitude better?

- **group_disjoint/wDMPNN**: slope=0.894, dispersion=0.943
- **group_disjoint/ChemArch**: slope=0.945, dispersion=0.969
- **pair_disjoint/wDMPNN**: slope=0.893, dispersion=0.943
- **pair_disjoint/ChemArch**: slope=0.945, dispersion=0.969
- **monomer_heldout/wDMPNN**: slope=0.565, dispersion=0.740
- **monomer_heldout/ChemArch**: slope=0.836, dispersion=1.026

### 4. Which model best ranks architectures within matched groups?

- **group_disjoint**: Best = ChemArch (pairwise=0.894)
- **pair_disjoint**: Best = ChemArch (pairwise=0.892)
- **monomer_heldout**: Best = ChemArch (pairwise=0.778)

### 5. Does wDMPNN shrink delta predictions toward zero?

- **group_disjoint**: slope=0.894, dispersion=0.943 → MILD
- **pair_disjoint**: slope=0.893, dispersion=0.943 → MILD
- **monomer_heldout**: slope=0.565, dispersion=0.740 → YES

### 6. Is wDMPNN's Monomer-heldout degradation correlated with chemical novelty?

- **EA**: Spearman(max_tanimoto, R²) = 0.373 (p=0.323, n=9)
- **IP**: Spearman(max_tanimoto, R²) = -0.424 (p=0.256, n=9)

### 7. Are hard folds driven by target-distribution shift or narrow variance?

- **EA**: Mean shift range [-0.513, 1.017] eV
  Std ratio range [0.457, 1.069]
- **IP**: Mean shift range [-0.696, 0.628] eV
  Std ratio range [0.414, 1.216]

### 8. Is the benzothiadiazole fold (fold 6) uniquely out-of-distribution?

- Fold 6 max Tanimoto = 0.308 (avg across folds = 0.407)
- **YES**: fold 6 is chemically more novel than average

### 9. Are the conclusions consistent across EA and IP?

- **overall_R2** wDMPNN-ChemArch: EA median diff=0.2936, IP median diff=0.2826, consistent=True
- **delta_R2** wDMPNN-ChemArch: EA median diff=-0.1208, IP median diff=-0.0634, consistent=True

### 10. Scientific Interpretation

See detailed metrics above. Key patterns:
- Compare group-mean R² (between-group prediction quality) across models
- Compare calibration slope and dispersion ratio (architecture-deviation magnitude preservation)
- The error decomposition reveals whether a model's advantage is from chemistry prediction or architecture recovery

## Statistical Comparisons (Monomer-Heldout, Paired by Fold)

| Metric | Model A | Model B | n | Median Diff | p-value | Wins A | Wins B |
|--------|---------|---------|---|------------|---------|--------|--------|
| EA_overall_R2 | wdmpnn | chemarch | 9 | 0.2936 | 0.0039 | 9 | 0 |
| EA_group_mean_R2 | wdmpnn | chemarch | 9 | 0.2713 | 0.0039 | 9 | 0 |
| EA_delta_R2 | wdmpnn | chemarch | 9 | -0.1208 | 0.1289 | 1 | 8 |
| EA_overall_R2 | wdmpnn | globalarch | 9 | 0.2638 | 0.0039 | 9 | 0 |
| EA_delta_R2 | wdmpnn | globalarch | 9 | -0.0214 | 0.7344 | 4 | 5 |
| IP_overall_R2 | wdmpnn | chemarch | 9 | 0.2826 | 0.0195 | 7 | 2 |
| IP_group_mean_R2 | wdmpnn | chemarch | 9 | 0.2517 | 0.0078 | 8 | 1 |
| IP_delta_R2 | wdmpnn | chemarch | 9 | -0.0634 | 0.8203 | 4 | 5 |
| IP_overall_R2 | wdmpnn | globalarch | 9 | 0.5335 | 0.0273 | 8 | 1 |
| IP_delta_R2 | wdmpnn | globalarch | 9 | 0.0131 | 0.3594 | 5 | 4 |
| EA_pairwise_acc | wdmpnn | chemarch | 9 | -0.0670 | 0.0273 | 2 | 7 |
| EA_median_kendall | wdmpnn | chemarch | 9 | 0.0000 | 1.0000 | 1 | 0 |
| IP_pairwise_acc | wdmpnn | chemarch | 9 | 0.0150 | 1.0000 | 5 | 4 |
| IP_median_kendall | wdmpnn | chemarch | 9 | 0.0000 | 1.0000 | 0 | 0 |
