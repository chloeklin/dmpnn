# Fusion Fold Win Summary

Comparing each fusion variant against Additive (2D1) on a fold-by-fold basis.

## Overall EA R²

### FiLM vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.7558 | 0.9702 | FiLM |
| 1 | 0.3284 | 0.6102 | FiLM |
| 2 | 0.7754 | 0.9407 | FiLM |
| 3 | 0.9734 | 0.8161 | Additive (2D1) |
| 4 | 0.6518 | 0.7013 | FiLM |
| 5 | 0.9559 | 0.9756 | FiLM |
| 6 | -17.3442 | -12.4113 | FiLM ← **fold 6 (OOD)** |
| 7 | 0.6918 | 0.7883 | FiLM |
| 8 | 0.6187 | 0.6178 | Additive (2D1) |

**FiLM wins: 7/9 folds**  | Additive wins: 2/9 folds  | Ties: 0

### NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.7558 | 0.9775 | NLMix |
| 1 | 0.3284 | 0.5400 | NLMix |
| 2 | 0.7754 | 0.9268 | NLMix |
| 3 | 0.9734 | 0.9626 | Additive (2D1) |
| 4 | 0.6518 | 0.5084 | Additive (2D1) |
| 5 | 0.9559 | 0.8820 | Additive (2D1) |
| 6 | -17.3442 | -17.0213 | NLMix ← **fold 6 (OOD)** |
| 7 | 0.6918 | 0.7220 | NLMix |
| 8 | 0.6187 | 0.7428 | NLMix |

**NLMix wins: 6/9 folds**  | Additive wins: 3/9 folds  | Ties: 0

### FiLM+NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.7558 | 0.9750 | FiLM+NLMix |
| 1 | 0.3284 | 0.7246 | FiLM+NLMix |
| 2 | 0.7754 | 0.9086 | FiLM+NLMix |
| 3 | 0.9734 | 0.9501 | Additive (2D1) |
| 4 | 0.6518 | 0.5642 | Additive (2D1) |
| 5 | 0.9559 | 0.9552 | Additive (2D1) |
| 6 | -17.3442 | -17.2123 | FiLM+NLMix ← **fold 6 (OOD)** |
| 7 | 0.6918 | 0.6731 | Additive (2D1) |
| 8 | 0.6187 | 0.5199 | Additive (2D1) |

**FiLM+NLMix wins: 4/9 folds**  | Additive wins: 5/9 folds  | Ties: 0

## Overall IP R²

### FiLM vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.2667 | 0.9151 | FiLM |
| 1 | 0.8134 | 0.8987 | FiLM |
| 2 | 0.4084 | 0.3503 | Additive (2D1) |
| 3 | 0.4973 | 0.7382 | FiLM |
| 4 | 0.6025 | 0.8708 | FiLM |
| 5 | -0.1710 | 0.1359 | FiLM |
| 6 | 0.9210 | 0.9826 | FiLM ← **fold 6 (OOD)** |
| 7 | 0.9863 | 0.9423 | Additive (2D1) |
| 8 | -0.1371 | 0.6361 | FiLM |

**FiLM wins: 7/9 folds**  | Additive wins: 2/9 folds  | Ties: 0

### NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.2667 | 0.7207 | NLMix |
| 1 | 0.8134 | 0.9246 | NLMix |
| 2 | 0.4084 | 0.3742 | Additive (2D1) |
| 3 | 0.4973 | 0.7703 | NLMix |
| 4 | 0.6025 | 0.7952 | NLMix |
| 5 | -0.1710 | -0.5140 | Additive (2D1) |
| 6 | 0.9210 | 0.9585 | NLMix ← **fold 6 (OOD)** |
| 7 | 0.9863 | 0.9092 | Additive (2D1) |
| 8 | -0.1371 | 0.2268 | NLMix |

**NLMix wins: 6/9 folds**  | Additive wins: 3/9 folds  | Ties: 0

### FiLM+NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.2667 | 0.8805 | FiLM+NLMix |
| 1 | 0.8134 | 0.9590 | FiLM+NLMix |
| 2 | 0.4084 | 0.4204 | FiLM+NLMix |
| 3 | 0.4973 | 0.6189 | FiLM+NLMix |
| 4 | 0.6025 | 0.7990 | FiLM+NLMix |
| 5 | -0.1710 | -0.0858 | FiLM+NLMix |
| 6 | 0.9210 | 0.9463 | FiLM+NLMix ← **fold 6 (OOD)** |
| 7 | 0.9863 | 0.9695 | Additive (2D1) |
| 8 | -0.1371 | 0.5676 | FiLM+NLMix |

**FiLM+NLMix wins: 8/9 folds**  | Additive wins: 1/9 folds  | Ties: 0

## EA ArchDev R²

### FiLM vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.6469 | 0.6301 | Additive (2D1) |
| 1 | 0.6778 | 0.7985 | FiLM |
| 2 | 0.9244 | 0.7906 | Additive (2D1) |
| 3 | 0.9736 | 0.6769 | Additive (2D1) |
| 4 | 0.8260 | -0.1630 | Additive (2D1) |
| 5 | 0.9823 | 0.8982 | Additive (2D1) |
| 6 | 0.5973 | -1.9215 | Additive (2D1) ← **fold 6 (OOD)** |
| 7 | 0.9426 | 0.5886 | Additive (2D1) |
| 8 | 0.9459 | 0.4628 | Additive (2D1) |

**FiLM wins: 1/9 folds**  | Additive wins: 8/9 folds  | Ties: 0

### NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.6469 | 0.6966 | NLMix |
| 1 | 0.6778 | 0.7981 | NLMix |
| 2 | 0.9244 | 0.7658 | Additive (2D1) |
| 3 | 0.9736 | 0.8404 | Additive (2D1) |
| 4 | 0.8260 | -0.0682 | Additive (2D1) |
| 5 | 0.9823 | 0.9369 | Additive (2D1) |
| 6 | 0.5973 | -0.3317 | Additive (2D1) ← **fold 6 (OOD)** |
| 7 | 0.9426 | 0.7324 | Additive (2D1) |
| 8 | 0.9459 | 0.7594 | Additive (2D1) |

**NLMix wins: 2/9 folds**  | Additive wins: 7/9 folds  | Ties: 0

### FiLM+NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.6469 | 0.6044 | Additive (2D1) |
| 1 | 0.6778 | 0.7722 | FiLM+NLMix |
| 2 | 0.9244 | 0.7118 | Additive (2D1) |
| 3 | 0.9736 | 0.8100 | Additive (2D1) |
| 4 | 0.8260 | -0.1501 | Additive (2D1) |
| 5 | 0.9823 | 0.9402 | Additive (2D1) |
| 6 | 0.5973 | -0.8843 | Additive (2D1) ← **fold 6 (OOD)** |
| 7 | 0.9426 | 0.6143 | Additive (2D1) |
| 8 | 0.9459 | 0.5895 | Additive (2D1) |

**FiLM+NLMix wins: 1/9 folds**  | Additive wins: 8/9 folds  | Ties: 0

## IP ArchDev R²

### FiLM vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.2384 | 0.8376 | FiLM |
| 1 | 0.8958 | 0.6322 | Additive (2D1) |
| 2 | 0.8975 | 0.2463 | Additive (2D1) |
| 3 | 0.8822 | 0.8053 | Additive (2D1) |
| 4 | 0.8568 | 0.2154 | Additive (2D1) |
| 5 | 0.8426 | 0.7773 | Additive (2D1) |
| 6 | 0.9604 | 0.8027 | Additive (2D1) ← **fold 6 (OOD)** |
| 7 | 0.9815 | 0.7793 | Additive (2D1) |
| 8 | 0.9109 | 0.8198 | Additive (2D1) |

**FiLM wins: 1/9 folds**  | Additive wins: 8/9 folds  | Ties: 0

### NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.2384 | 0.7935 | NLMix |
| 1 | 0.8958 | 0.8256 | Additive (2D1) |
| 2 | 0.8975 | 0.4027 | Additive (2D1) |
| 3 | 0.8822 | 0.7069 | Additive (2D1) |
| 4 | 0.8568 | 0.0520 | Additive (2D1) |
| 5 | 0.8426 | 0.8536 | NLMix |
| 6 | 0.9604 | 0.8200 | Additive (2D1) ← **fold 6 (OOD)** |
| 7 | 0.9815 | 0.7807 | Additive (2D1) |
| 8 | 0.9109 | 0.8955 | Additive (2D1) |

**NLMix wins: 2/9 folds**  | Additive wins: 7/9 folds  | Ties: 0

### FiLM+NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.2384 | 0.8813 | FiLM+NLMix |
| 1 | 0.8958 | 0.6688 | Additive (2D1) |
| 2 | 0.8975 | 0.4187 | Additive (2D1) |
| 3 | 0.8822 | 0.5604 | Additive (2D1) |
| 4 | 0.8568 | -0.0495 | Additive (2D1) |
| 5 | 0.8426 | 0.8182 | Additive (2D1) |
| 6 | 0.9604 | 0.8983 | Additive (2D1) ← **fold 6 (OOD)** |
| 7 | 0.9815 | 0.8204 | Additive (2D1) |
| 8 | 0.9109 | 0.8370 | Additive (2D1) |

**FiLM+NLMix wins: 1/9 folds**  | Additive wins: 8/9 folds  | Ties: 0

## EA MAE

### FiLM vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.1558 | 0.0702 | FiLM |
| 1 | 0.2385 | 0.2095 | FiLM |
| 2 | 0.2059 | 0.1156 | FiLM |
| 3 | 0.0495 | 0.1522 | Additive (2D1) |
| 4 | 0.1818 | 0.1767 | FiLM |
| 5 | 0.0672 | 0.0453 | FiLM |
| 6 | 1.0269 | 0.8683 | FiLM ← **fold 6 (OOD)** |
| 7 | 0.2968 | 0.2491 | FiLM |
| 8 | 0.3350 | 0.3345 | FiLM |

**FiLM wins: 8/9 folds**  | Additive wins: 1/9 folds  | Ties: 0

### NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.1558 | 0.0577 | NLMix |
| 1 | 0.2385 | 0.2283 | NLMix |
| 2 | 0.2059 | 0.1215 | NLMix |
| 3 | 0.0495 | 0.0633 | Additive (2D1) |
| 4 | 0.1818 | 0.2376 | Additive (2D1) |
| 5 | 0.0672 | 0.1149 | Additive (2D1) |
| 6 | 1.0269 | 1.0109 | NLMix ← **fold 6 (OOD)** |
| 7 | 0.2968 | 0.2994 | Additive (2D1) |
| 8 | 0.3350 | 0.2641 | NLMix |

**NLMix wins: 5/9 folds**  | Additive wins: 4/9 folds  | Ties: 0

### FiLM+NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.1558 | 0.0647 | FiLM+NLMix |
| 1 | 0.2385 | 0.1702 | FiLM+NLMix |
| 2 | 0.2059 | 0.1441 | FiLM+NLMix |
| 3 | 0.0495 | 0.0724 | Additive (2D1) |
| 4 | 0.1818 | 0.2272 | Additive (2D1) |
| 5 | 0.0672 | 0.0685 | Additive (2D1) |
| 6 | 1.0269 | 1.0117 | FiLM+NLMix ← **fold 6 (OOD)** |
| 7 | 0.2968 | 0.3257 | Additive (2D1) |
| 8 | 0.3350 | 0.3653 | Additive (2D1) |

**FiLM+NLMix wins: 4/9 folds**  | Additive wins: 5/9 folds  | Ties: 0

## IP MAE

### FiLM vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.1718 | 0.0721 | FiLM |
| 1 | 0.1377 | 0.1036 | FiLM |
| 2 | 0.3633 | 0.3776 | Additive (2D1) |
| 3 | 0.1470 | 0.1078 | FiLM |
| 4 | 0.1175 | 0.0629 | FiLM |
| 5 | 0.2745 | 0.2355 | FiLM |
| 6 | 0.0831 | 0.0415 | FiLM ← **fold 6 (OOD)** |
| 7 | 0.0392 | 0.0858 | Additive (2D1) |
| 8 | 0.2412 | 0.1307 | FiLM |

**FiLM wins: 7/9 folds**  | Additive wins: 2/9 folds  | Ties: 0

### NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.1718 | 0.1319 | NLMix |
| 1 | 0.1377 | 0.0889 | NLMix |
| 2 | 0.3633 | 0.3719 | Additive (2D1) |
| 3 | 0.1470 | 0.0989 | NLMix |
| 4 | 0.1175 | 0.0806 | NLMix |
| 5 | 0.2745 | 0.3187 | Additive (2D1) |
| 6 | 0.0831 | 0.0643 | NLMix ← **fold 6 (OOD)** |
| 7 | 0.0392 | 0.1107 | Additive (2D1) |
| 8 | 0.2412 | 0.2039 | NLMix |

**NLMix wins: 6/9 folds**  | Additive wins: 3/9 folds  | Ties: 0

### FiLM+NLMix vs Additive (2D1)

| Fold | Additive | Variant | Winner |
|---|---|---|---|
| 0 | 0.1718 | 0.0824 | FiLM+NLMix |
| 1 | 0.1377 | 0.0653 | FiLM+NLMix |
| 2 | 0.3633 | 0.3536 | FiLM+NLMix |
| 3 | 0.1470 | 0.1287 | FiLM+NLMix |
| 4 | 0.1175 | 0.0825 | FiLM+NLMix |
| 5 | 0.2745 | 0.2667 | FiLM+NLMix |
| 6 | 0.0831 | 0.0754 | FiLM+NLMix ← **fold 6 (OOD)** |
| 7 | 0.0392 | 0.0615 | Additive (2D1) |
| 8 | 0.2412 | 0.1499 | FiLM+NLMix |

**FiLM+NLMix wins: 8/9 folds**  | Additive wins: 1/9 folds  | Ties: 0

