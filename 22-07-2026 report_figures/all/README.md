# Seed-42 Diagnostics Figure Index

All figures use existing seed-42 prediction NPZs and diagnostics CSVs only; no models were trained. Bars are fold medians and dots are individual folds. LOMO bar annotations show fold means. Colors: HPG-hier `#E8A33D`, wDMPNN `#12314E`, ChemArch `#1C7293`.

## Figures

- `01_group_mean_r2.png` — Median bars and individual fold dots for Group-mean R² (chemistry baseline); LOMO text labels give fold mean.
- `02_architecture_delta_r2.png` — Median bars and individual fold dots for Architecture ΔR²; LOMO text labels give fold mean.
- `03_ordering_accuracy.png` — Median bars and individual fold dots for Architecture ordering accuracy; LOMO text labels give fold mean.
- `04_overall_r2.png` — Median bars and individual fold dots for Overall R²; LOMO text labels give fold mean.
- `05_overall_mae.png` — Median bars and individual fold dots for Overall MAE; LOMO text labels give fold mean.
- `06_overall_rmse.png` — Median bars and individual fold dots for Overall RMSE; LOMO text labels give fold mean.
- `07_calibration_slope.png` — Median bars and individual fold dots for Calibration slope; LOMO text labels give fold mean.
- `02a_overall_parity_EA.png` — 3×3 overall parity grid for EA; each panel pools matched test samples from all folds and annotates pooled R² and slope.
- `02b_group_mean_parity_EA.png` — 3×3 group_mean parity grid for EA; each panel pools matched test samples from all folds and annotates pooled R² and slope.
- `02c_architecture_deviation_parity_EA.png` — 3×3 architecture_deviation parity grid for EA; each panel pools matched test samples from all folds and annotates pooled R² and slope.
- `08_lomo_group_mean_r2_EA.png` — LOMO per-fold group-mean R² for EA; ChemArch EA fold 6 is clipped at -0.50 and annotated -12.87.
- `09_scorecard_EA.png` — Median fold scorecard for EA: x=group-mean R² and y=architecture ΔR²; values below -0.50 are clipped and annotated.
- `11_effect_size_error_EA.png` — Per-model median absolute architecture-deviation error in six equal-count |Δy| bins for EA.
- `12_error_decomposition_EA.png` — Stacked median fold fractions of squared error attributable to group-mean (between) and architecture (within) components for EA.
- `02a_overall_parity_IP.png` — 3×3 overall parity grid for IP; each panel pools matched test samples from all folds and annotates pooled R² and slope.
- `02b_group_mean_parity_IP.png` — 3×3 group_mean parity grid for IP; each panel pools matched test samples from all folds and annotates pooled R² and slope.
- `02c_architecture_deviation_parity_IP.png` — 3×3 architecture_deviation parity grid for IP; each panel pools matched test samples from all folds and annotates pooled R² and slope.
- `08_lomo_group_mean_r2_IP.png` — LOMO per-fold group-mean R² for IP; ChemArch EA fold 6 is clipped at -0.50 and annotated -12.87.
- `09_scorecard_IP.png` — Median fold scorecard for IP: x=group-mean R² and y=architecture ΔR²; values below -0.50 are clipped and annotated.
- `11_effect_size_error_IP.png` — Per-model median absolute architecture-deviation error in six equal-count |Δy| bins for IP.
- `12_error_decomposition_IP.png` — Stacked median fold fractions of squared error attributable to group-mean (between) and architecture (within) components for IP.
- `10_lomo_calibration_scatter.png` — LOMO architecture-deviation parity by target and model; panels pool all matched test samples across 9 folds.

## Anomalies

- All 114 prediction cells were present and valid according to `seed_42/01_validation/evaluation_inventory.csv`.
- ChemArch LOMO EA fold 6 has group-mean R² = -12.8726 and is clipped/annotated in robust-scale figures.
- ChemArch LOMO IP fold 5 has group-mean R² = -0.3493.
- ChemArch LOMO EA fold 4 has architecture ΔR² = -0.0163; fold 6 has architecture ΔR² = -2.4912.

### Numbers: group-mean R²

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.9969, 0.9968, 0.9978, 0.9956, 0.9966 | 0.9968 | 0.9967 |
| EA | HPG-hier | PD | 0.9972, 0.9974, 0.9976, 0.9973, 0.9944 | 0.9973 | 0.9968 |
| EA | HPG-hier | LOMO | 0.9253, 0.5751, 0.9216, 0.9746, 0.9506, 0.9690, 0.9166, 0.9019, 0.9631 | 0.9253 | 0.8998 |
| EA | wDMPNN | GD | 0.9983, 0.9982, 0.9982, 0.9982, 0.9985 | 0.9982 | 0.9983 |
| EA | wDMPNN | PD | 0.9978, 0.9977, 0.9981, 0.9978, 0.9976 | 0.9978 | 0.9978 |
| EA | wDMPNN | LOMO | 0.7601, 0.9446, 0.9803, 0.9903, 0.9456, 0.9687, 0.8936, 0.9648, 0.9941 | 0.9648 | 0.9380 |
| EA | ChemArch | GD | 0.9992, 0.9990, 0.9990, 0.9990, 0.9989 | 0.9990 | 0.9990 |
| EA | ChemArch | PD | 0.9988, 0.9986, 0.9985, 0.9985, 0.9986 | 0.9986 | 0.9986 |
| EA | ChemArch | LOMO | 0.9882, 0.5227, 0.8386, 0.9809, 0.8148, 0.9318, -12.8726, 0.7531, 0.7932 | 0.8148 | -0.6943 |
| IP | HPG-hier | GD | 0.9980, 0.9970, 0.9965, 0.9970, 0.9973 | 0.9970 | 0.9971 |
| IP | HPG-hier | PD | 0.9964, 0.9966, 0.9964, 0.9952, 0.9969 | 0.9964 | 0.9963 |
| IP | HPG-hier | LOMO | 0.9279, 0.9812, 0.7692, 0.9746, 0.9438, 0.7697, 0.9746, 0.9686, 0.9806 | 0.9686 | 0.9211 |
| IP | wDMPNN | GD | 0.9981, 0.9985, 0.9983, 0.9985, 0.9985 | 0.9985 | 0.9984 |
| IP | wDMPNN | PD | 0.9980, 0.9977, 0.9980, 0.9979, 0.9974 | 0.9979 | 0.9978 |
| IP | wDMPNN | LOMO | 0.2695, 0.9635, 0.9767, 0.9679, 0.7119, 0.9460, 0.9838, 0.9833, 0.9769 | 0.9679 | 0.8644 |
| IP | ChemArch | GD | 0.9993, 0.9992, 0.9992, 0.9992, 0.9994 | 0.9992 | 0.9993 |
| IP | ChemArch | PD | 0.9984, 0.9987, 0.9989, 0.9988, 0.9990 | 0.9988 | 0.9988 |
| IP | ChemArch | LOMO | 0.8788, 0.9209, 0.4621, 0.8534, 0.8189, -0.3493, 0.9223, 0.8979, 0.3791 | 0.8534 | 0.6427 |

### Numbers: architecture ΔR²

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.8976, 0.9025, 0.9097, 0.9097, 0.8764 | 0.9025 | 0.8992 |
| EA | HPG-hier | PD | 0.9080, 0.9076, 0.9142, 0.9183, 0.8757 | 0.9080 | 0.9048 |
| EA | HPG-hier | LOMO | 0.7336, 0.7903, 0.8455, 0.8372, 0.3470, 0.8820, 0.2429, 0.6496, 0.8423 | 0.7903 | 0.6856 |
| EA | wDMPNN | GD | 0.8706, 0.8498, 0.8587, 0.8748, 0.8664 | 0.8664 | 0.8641 |
| EA | wDMPNN | PD | 0.8659, 0.8602, 0.8619, 0.8741, 0.8520 | 0.8619 | 0.8628 |
| EA | wDMPNN | LOMO | 0.6836, 0.4364, 0.4851, 0.7257, 0.1453, 0.5802, 0.3389, 0.5803, 0.6275 | 0.5802 | 0.5115 |
| EA | ChemArch | GD | 0.9435, 0.9299, 0.9378, 0.9441, 0.9415 | 0.9415 | 0.9394 |
| EA | ChemArch | PD | 0.9340, 0.9316, 0.9276, 0.9497, 0.9391 | 0.9340 | 0.9364 |
| EA | ChemArch | LOMO | 0.7759, 0.8075, 0.7785, 0.6987, -0.0163, 0.9343, -2.4912, 0.6138, 0.7598 | 0.7598 | 0.3179 |
| IP | HPG-hier | GD | 0.9448, 0.9398, 0.9501, 0.9454, 0.9447 | 0.9448 | 0.9449 |
| IP | HPG-hier | PD | 0.9511, 0.9305, 0.9298, 0.9190, 0.9348 | 0.9305 | 0.9330 |
| IP | HPG-hier | LOMO | 0.8282, 0.7375, 0.2113, 0.8195, 0.5807, 0.8960, 0.8798, 0.7727, 0.8799 | 0.8195 | 0.7339 |
| IP | wDMPNN | GD | 0.9193, 0.9232, 0.9271, 0.9205, 0.9218 | 0.9218 | 0.9224 |
| IP | wDMPNN | PD | 0.9168, 0.9147, 0.9226, 0.9183, 0.9186 | 0.9183 | 0.9182 |
| IP | wDMPNN | LOMO | 0.7131, 0.3031, 0.4239, 0.8252, 0.4449, 0.4600, 0.3850, 0.6103, 0.7757 | 0.4600 | 0.5490 |
| IP | ChemArch | GD | 0.9641, 0.9663, 0.9668, 0.9634, 0.9673 | 0.9663 | 0.9656 |
| IP | ChemArch | PD | 0.9637, 0.9622, 0.9642, 0.9662, 0.9634 | 0.9637 | 0.9640 |
| IP | ChemArch | LOMO | 0.6507, 0.8168, 0.4294, 0.7662, 0.0794, 0.8679, 0.6642, 0.8300, 0.6317 | 0.6642 | 0.6374 |

### Numbers: pairwise ordering accuracy

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.8622, 0.8674, 0.8689, 0.8490, 0.8424 | 0.8622 | 0.8580 |
| EA | HPG-hier | PD | 0.8780, 0.8598, 0.8692, 0.8477, 0.8533 | 0.8598 | 0.8616 |
| EA | HPG-hier | LOMO | 0.7445, 0.8122, 0.8056, 0.7990, 0.5780, 0.8467, 0.7964, 0.7947, 0.8540 | 0.7990 | 0.7812 |
| EA | wDMPNN | GD | 0.8426, 0.8387, 0.8314, 0.8164, 0.8326 | 0.8326 | 0.8323 |
| EA | wDMPNN | PD | 0.8389, 0.8261, 0.8299, 0.8231, 0.8284 | 0.8284 | 0.8293 |
| EA | wDMPNN | LOMO | 0.7955, 0.7884, 0.7023, 0.7768, 0.5124, 0.7343, 0.6445, 0.7530, 0.7926 | 0.7530 | 0.7222 |
| EA | ChemArch | GD | 0.8950, 0.8954, 0.8972, 0.8848, 0.8969 | 0.8954 | 0.8939 |
| EA | ChemArch | PD | 0.9041, 0.8911, 0.8828, 0.8886, 0.8959 | 0.8911 | 0.8925 |
| EA | ChemArch | LOMO | 0.7678, 0.8354, 0.7902, 0.7964, 0.5228, 0.8732, 0.6523, 0.7417, 0.8412 | 0.7902 | 0.7579 |
| IP | HPG-hier | GD | 0.8661, 0.8675, 0.8818, 0.8730, 0.8745 | 0.8730 | 0.8726 |
| IP | HPG-hier | PD | 0.8778, 0.8613, 0.8602, 0.8524, 0.8547 | 0.8602 | 0.8613 |
| IP | HPG-hier | LOMO | 0.8268, 0.8257, 0.7569, 0.8219, 0.7639, 0.9167, 0.8462, 0.7305, 0.8728 | 0.8257 | 0.8179 |
| IP | wDMPNN | GD | 0.8471, 0.8508, 0.8598, 0.8513, 0.8537 | 0.8513 | 0.8525 |
| IP | wDMPNN | PD | 0.8458, 0.8435, 0.8508, 0.8401, 0.8525 | 0.8458 | 0.8465 |
| IP | wDMPNN | LOMO | 0.8162, 0.7434, 0.7173, 0.8700, 0.7577, 0.8286, 0.6026, 0.6605, 0.8281 | 0.7577 | 0.7583 |
| IP | ChemArch | GD | 0.8948, 0.9043, 0.8954, 0.8912, 0.9029 | 0.8954 | 0.8977 |
| IP | ChemArch | PD | 0.8979, 0.8991, 0.9037, 0.8948, 0.8930 | 0.8979 | 0.8977 |
| IP | ChemArch | LOMO | 0.7759, 0.8376, 0.6903, 0.8246, 0.7124, 0.9413, 0.7918, 0.7592, 0.8271 | 0.7918 | 0.7956 |

### Numbers: overall R²

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.9959, 0.9959, 0.9970, 0.9946, 0.9954 | 0.9959 | 0.9958 |
| EA | HPG-hier | PD | 0.9964, 0.9965, 0.9968, 0.9966, 0.9932 | 0.9965 | 0.9959 |
| EA | HPG-hier | LOMO | 0.9241, 0.5685, 0.9237, 0.9685, 0.9345, 0.9648, 0.8821, 0.9050, 0.9615 | 0.9241 | 0.8925 |
| EA | wDMPNN | GD | 0.9969, 0.9968, 0.9969, 0.9970, 0.9972 | 0.9969 | 0.9970 |
| EA | wDMPNN | PD | 0.9966, 0.9963, 0.9967, 0.9966, 0.9962 | 0.9966 | 0.9965 |
| EA | wDMPNN | LOMO | 0.7598, 0.9304, 0.9726, 0.9815, 0.9311, 0.9559, 0.8650, 0.9630, 0.9894 | 0.9559 | 0.9276 |
| EA | ChemArch | GD | 0.9986, 0.9984, 0.9984, 0.9985, 0.9984 | 0.9984 | 0.9985 |
| EA | ChemArch | PD | 0.9982, 0.9979, 0.9978, 0.9980, 0.9980 | 0.9980 | 0.9980 |
| EA | ChemArch | LOMO | 0.9856, 0.5153, 0.8369, 0.9717, 0.7907, 0.9288, -13.6518, 0.7553, 0.7898 | 0.7907 | -0.7864 |
| IP | HPG-hier | GD | 0.9972, 0.9961, 0.9958, 0.9962, 0.9965 | 0.9962 | 0.9964 |
| IP | HPG-hier | PD | 0.9958, 0.9956, 0.9954, 0.9941, 0.9960 | 0.9956 | 0.9954 |
| IP | HPG-hier | LOMO | 0.9232, 0.9790, 0.7743, 0.9631, 0.9321, 0.7680, 0.9716, 0.9663, 0.9762 | 0.9631 | 0.9171 |
| IP | wDMPNN | GD | 0.9969, 0.9975, 0.9973, 0.9974, 0.9973 | 0.9973 | 0.9973 |
| IP | wDMPNN | PD | 0.9969, 0.9965, 0.9969, 0.9968, 0.9963 | 0.9968 | 0.9967 |
| IP | wDMPNN | LOMO | 0.2648, 0.9536, 0.9730, 0.9569, 0.6896, 0.8682, 0.9685, 0.9778, 0.9675 | 0.9569 | 0.8467 |
| IP | ChemArch | GD | 0.9988, 0.9988, 0.9987, 0.9987, 0.9989 | 0.9988 | 0.9988 |
| IP | ChemArch | PD | 0.9979, 0.9982, 0.9983, 0.9983, 0.9984 | 0.9983 | 0.9982 |
| IP | ChemArch | LOMO | 0.8680, 0.9205, 0.4634, 0.8427, 0.7912, -0.2717, 0.9172, 0.8964, 0.3567 | 0.8427 | 0.6427 |

### Numbers: overall MAE (eV)

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.0283, 0.0279, 0.0234, 0.0333, 0.0300 | 0.0283 | 0.0286 |
| EA | HPG-hier | PD | 0.0252, 0.0244, 0.0250, 0.0249, 0.0379 | 0.0250 | 0.0275 |
| EA | HPG-hier | LOMO | 0.1068, 0.2140, 0.1182, 0.0547, 0.0733, 0.0583, 0.0580, 0.1555, 0.0939 | 0.0939 | 0.1036 |
| EA | wDMPNN | GD | 0.0234, 0.0228, 0.0237, 0.0244, 0.0226 | 0.0234 | 0.0234 |
| EA | wDMPNN | PD | 0.0241, 0.0257, 0.0239, 0.0248, 0.0252 | 0.0248 | 0.0248 |
| EA | wDMPNN | LOMO | 0.2269, 0.0800, 0.0630, 0.0410, 0.0760, 0.0645, 0.0685, 0.1008, 0.0441 | 0.0685 | 0.0850 |
| EA | ChemArch | GD | 0.0152, 0.0151, 0.0164, 0.0166, 0.0171 | 0.0164 | 0.0161 |
| EA | ChemArch | PD | 0.0163, 0.0174, 0.0193, 0.0183, 0.0181 | 0.0181 | 0.0179 |
| EA | ChemArch | LOMO | 0.0449, 0.2356, 0.2013, 0.0483, 0.1482, 0.0892, 0.9024, 0.2875, 0.2457 | 0.2013 | 0.2448 |
| IP | HPG-hier | GD | 0.0188, 0.0219, 0.0239, 0.0225, 0.0208 | 0.0219 | 0.0216 |
| IP | HPG-hier | PD | 0.0206, 0.0234, 0.0225, 0.0285, 0.0221 | 0.0225 | 0.0234 |
| IP | HPG-hier | LOMO | 0.0646, 0.0450, 0.1978, 0.0335, 0.0451, 0.1185, 0.0510, 0.0658, 0.0305 | 0.0510 | 0.0724 |
| IP | wDMPNN | GD | 0.0194, 0.0170, 0.0175, 0.0174, 0.0178 | 0.0175 | 0.0178 |
| IP | wDMPNN | PD | 0.0189, 0.0197, 0.0187, 0.0202, 0.0217 | 0.0197 | 0.0198 |
| IP | wDMPNN | LOMO | 0.2241, 0.0621, 0.0659, 0.0380, 0.0973, 0.0759, 0.0542, 0.0473, 0.0340 | 0.0621 | 0.0777 |
| IP | ChemArch | GD | 0.0113, 0.0117, 0.0119, 0.0120, 0.0108 | 0.0117 | 0.0116 |
| IP | ChemArch | PD | 0.0141, 0.0136, 0.0130, 0.0133, 0.0128 | 0.0133 | 0.0134 |
| IP | ChemArch | LOMO | 0.0851, 0.0971, 0.3480, 0.0812, 0.0821, 0.2896, 0.0898, 0.1301, 0.1788 | 0.0971 | 0.1535 |

### Numbers: overall RMSE (eV)

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.0390, 0.0381, 0.0329, 0.0443, 0.0405 | 0.0390 | 0.0389 |
| EA | HPG-hier | PD | 0.0361, 0.0355, 0.0340, 0.0352, 0.0490 | 0.0355 | 0.0380 |
| EA | HPG-hier | LOMO | 0.1370, 0.2347, 0.1452, 0.0706, 0.1028, 0.0688, 0.0834, 0.1902, 0.1147 | 0.1147 | 0.1275 |
| EA | wDMPNN | GD | 0.0337, 0.0335, 0.0334, 0.0332, 0.0316 | 0.0334 | 0.0331 |
| EA | wDMPNN | PD | 0.0352, 0.0366, 0.0344, 0.0351, 0.0367 | 0.0352 | 0.0356 |
| EA | wDMPNN | LOMO | 0.2436, 0.0943, 0.0869, 0.0541, 0.1054, 0.0770, 0.0892, 0.1187, 0.0601 | 0.0892 | 0.1033 |
| EA | ChemArch | GD | 0.0228, 0.0238, 0.0237, 0.0232, 0.0242 | 0.0237 | 0.0235 |
| EA | ChemArch | PD | 0.0257, 0.0273, 0.0284, 0.0268, 0.0264 | 0.0268 | 0.0269 |
| EA | ChemArch | LOMO | 0.0596, 0.2487, 0.2122, 0.0669, 0.1837, 0.0978, 0.9294, 0.3052, 0.2679 | 0.2122 | 0.2635 |
| IP | HPG-hier | GD | 0.0257, 0.0299, 0.0311, 0.0295, 0.0282 | 0.0295 | 0.0289 |
| IP | HPG-hier | PD | 0.0313, 0.0316, 0.0327, 0.0374, 0.0306 | 0.0316 | 0.0327 |
| IP | HPG-hier | LOMO | 0.0790, 0.0543, 0.2475, 0.0430, 0.0545, 0.1272, 0.0651, 0.0781, 0.0366 | 0.0651 | 0.0873 |
| IP | wDMPNN | GD | 0.0271, 0.0242, 0.0249, 0.0246, 0.0249 | 0.0249 | 0.0251 |
| IP | wDMPNN | PD | 0.0270, 0.0284, 0.0267, 0.0275, 0.0294 | 0.0275 | 0.0278 |
| IP | wDMPNN | LOMO | 0.2445, 0.0807, 0.0856, 0.0465, 0.1164, 0.0959, 0.0686, 0.0634, 0.0428 | 0.0807 | 0.0938 |
| IP | ChemArch | GD | 0.0170, 0.0168, 0.0173, 0.0174, 0.0161 | 0.0170 | 0.0169 |
| IP | ChemArch | PD | 0.0220, 0.0206, 0.0196, 0.0198, 0.0190 | 0.0198 | 0.0202 |
| IP | ChemArch | LOMO | 0.1036, 0.1057, 0.3816, 0.0888, 0.0955, 0.2978, 0.1112, 0.1369, 0.1906 | 0.1112 | 0.1680 |

### Numbers: calibration slope

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.8828, 0.8584, 0.9369, 0.9259, 0.8811 | 0.8828 | 0.8970 |
| EA | HPG-hier | PD | 0.8936, 0.8815, 0.9087, 0.8960, 0.7730 | 0.8936 | 0.8705 |
| EA | HPG-hier | LOMO | 0.6746, 0.7835, 0.8406, 0.7054, 0.5899, 1.0414, 1.0701, 1.0226, 0.8045 | 0.8045 | 0.8370 |
| EA | wDMPNN | GD | 0.8439, 0.8615, 0.8689, 0.8508, 0.8656 | 0.8615 | 0.8581 |
| EA | wDMPNN | PD | 0.8661, 0.8443, 0.8537, 0.8909, 0.8555 | 0.8555 | 0.8621 |
| EA | wDMPNN | LOMO | 0.5532, 0.4355, 0.4224, 0.5715, 0.4039, 0.4515, 0.3617, 0.7178, 0.5073 | 0.4515 | 0.4916 |
| EA | ChemArch | GD | 0.9202, 0.9176, 0.9223, 0.9337, 0.9206 | 0.9206 | 0.9229 |
| EA | ChemArch | PD | 0.9500, 0.9333, 0.9125, 0.9357, 0.9349 | 0.9349 | 0.9333 |
| EA | ChemArch | LOMO | 0.7380, 0.7502, 0.7865, 0.5361, 0.5446, 1.0324, 1.3714, 0.9204, 0.5820 | 0.7502 | 0.8068 |
| IP | HPG-hier | GD | 0.9338, 0.9373, 0.9784, 0.8992, 0.9473 | 0.9373 | 0.9392 |
| IP | HPG-hier | PD | 0.9464, 0.9217, 0.9487, 0.9726, 0.8960 | 0.9464 | 0.9371 |
| IP | HPG-hier | LOMO | 0.9158, 0.8361, 0.7565, 1.1385, 0.7452, 0.7567, 0.7483, 0.6513, 0.9806 | 0.7567 | 0.8366 |
| IP | wDMPNN | GD | 0.8925, 0.9232, 0.9222, 0.8962, 0.8975 | 0.8975 | 0.9063 |
| IP | wDMPNN | PD | 0.9032, 0.9219, 0.9181, 0.9233, 0.9453 | 0.9219 | 0.9224 |
| IP | wDMPNN | LOMO | 0.5542, 0.9138, 0.5328, 1.1356, 0.6826, 0.2867, 0.2968, 0.6090, 0.7476 | 0.6090 | 0.6399 |
| IP | ChemArch | GD | 0.9566, 0.9759, 0.9769, 0.9607, 0.9719 | 0.9719 | 0.9684 |
| IP | ChemArch | PD | 0.9367, 0.9565, 0.9642, 0.9544, 0.9605 | 0.9565 | 0.9545 |
| IP | ChemArch | LOMO | 0.5885, 0.8616, 0.5957, 1.2113, 1.0856, 0.6853, 0.5188, 0.8613, 0.8603 | 0.8603 | 0.8076 |

### Numbers: between-group squared-error fraction

| Target | Model | Split | Fold values | Median | Mean |
|---|---|---|---|---:|---:|
| EA | HPG-hier | GD | 0.7377, 0.7759, 0.7193, 0.8348, 0.7391 | 0.7391 | 0.7614 |
| EA | HPG-hier | PD | 0.7525, 0.7344, 0.7340, 0.7711, 0.8220 | 0.7525 | 0.7628 |
| EA | HPG-hier | LOMO | 0.9589, 0.9883, 0.9670, 0.8325, 0.8490, 0.8855, 0.7751, 0.9746, 0.9467 | 0.9467 | 0.9086 |
| EA | wDMPNN | GD | 0.5572, 0.5541, 0.5734, 0.5932, 0.5371 | 0.5572 | 0.5630 |
| EA | wDMPNN | PD | 0.6195, 0.6229, 0.5804, 0.6448, 0.6222 | 0.6222 | 0.6179 |
| EA | wDMPNN | LOMO | 0.9846, 0.8056, 0.6933, 0.5203, 0.8121, 0.6752, 0.8284, 0.9219, 0.5407 | 0.8056 | 0.7536 |
| EA | ChemArch | GD | 0.5775, 0.5871, 0.6273, 0.6264, 0.6549 | 0.6264 | 0.6147 |
| EA | ChemArch | PD | 0.6494, 0.6679, 0.6775, 0.7563, 0.7000 | 0.6775 | 0.6902 |
| EA | ChemArch | LOMO | 0.8176, 0.9905, 0.9779, 0.6553, 0.9264, 0.9685, 0.9917, 0.9891, 0.9851 | 0.9779 | 0.9225 |
| IP | HPG-hier | GD | 0.7137, 0.7761, 0.8238, 0.7955, 0.7554 | 0.7761 | 0.7729 |
| IP | HPG-hier | PD | 0.8288, 0.7648, 0.7733, 0.8059, 0.7705 | 0.7733 | 0.7887 |
| IP | HPG-hier | LOMO | 0.9143, 0.8525, 0.9787, 0.6583, 0.8596, 0.9311, 0.8903, 0.8978, 0.8002 | 0.8903 | 0.8648 |
| IP | wDMPNN | GD | 0.6239, 0.5655, 0.5987, 0.5713, 0.5550 | 0.5713 | 0.5829 |
| IP | wDMPNN | PD | 0.6094, 0.6421, 0.6246, 0.6378, 0.6887 | 0.6378 | 0.6405 |
| IP | wDMPNN | LOMO | 0.9850, 0.8226, 0.8699, 0.7165, 0.9594, 0.3708, 0.4955, 0.7342, 0.7269 | 0.7342 | 0.7423 |
| IP | ChemArch | GD | 0.5735, 0.6050, 0.6191, 0.6054, 0.5549 | 0.6050 | 0.5916 |
| IP | ChemArch | PD | 0.7420, 0.6986, 0.6772, 0.7104, 0.6664 | 0.6986 | 0.6989 |
| IP | ChemArch | LOMO | 0.8986, 0.9728, 0.9935, 0.8963, 0.8998, 0.9840, 0.8951, 0.9751, 0.9774 | 0.9728 | 0.9436 |
