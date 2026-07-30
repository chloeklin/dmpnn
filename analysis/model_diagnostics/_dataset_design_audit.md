# Dataset Design Audit

## 0.1 Design

Rows: `42966`. Unique A: `9`. Unique B: `682`. A and B sets disjoint: `True`.

The factorial claim `9 × 682 × 7 = 42,966` is confirmed. All `6138` A/B pairs are present with per-pair rows min/median/max = `7` / `7` / `7`.

Distinct `(fracA, fracB, poly_type)` cells across the full dataset:

| fracA | fracB | poly_type | count |
| --- | --- | --- | --- |
| 0.25000 | 0.75000 | block | 6138 |
| 0.25000 | 0.75000 | random | 6138 |
| 0.50000 | 0.50000 | alternating | 6138 |
| 0.50000 | 0.50000 | block | 6138 |
| 0.50000 | 0.50000 | random | 6138 |
| 0.75000 | 0.25000 | block | 6138 |
| 0.75000 | 0.25000 | random | 6138 |

Rows per A min/median/max: `4774` / `4774` / `4774`. Rows per B min/median/max: `63` / `63` / `63`.

## 0.2 Signal axis

| target | A_identity | B_identity | A_plus_B | fracA | poly_type | within_A_B_fracA |
| --- | --- | --- | --- | --- | --- | --- |
| EA | 0.41816 | 0.45710 | 0.92930 | 0.00117 | 0.00256 | 0.00979 |
| IP | 0.50035 | 0.32933 | 0.89655 | 0.01932 | 0.00464 | 0.01459 |

Mean EA and IP by A monomer:

| smiles_A | EA_mean | IP_mean |
| --- | --- | --- |
| CC1(C)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | -2.78894 | 1.43708 |
| O=S1(=O)c2cc(B(O)O)ccc2-c2ccc(B(O)O)cc21 | -2.37811 | 1.74583 |
| OB(O)c1cc(F)c(B(O)O)cc1F | -2.66739 | 2.01133 |
| OB(O)c1cc2cc3sc(B(O)O)cc3cc2s1 | -2.44097 | 1.10715 |
| OB(O)c1cc2ccc3cc(B(O)O)cc4ccc(c1)c2c34 | -2.63953 | 1.36860 |
| OB(O)c1ccc(-c2ccc(B(O)O)s2)s1 | -2.36282 | 0.83506 |
| OB(O)c1ccc(B(O)O)c2nsnc12 | -1.63749 | 1.62003 |
| OB(O)c1ccc(B(O)O)cc1 | -2.95955 | 1.71206 |
| OB(O)c1ccc2c(c1)[nH]c1cc(B(O)O)ccc12 | -2.99745 | 1.24352 |

Mean target ranges by monomer role:

| target | A_mean_min | A_mean_max | A_mean_spread | B_mean_min | B_mean_max | B_mean_spread |
| --- | --- | --- | --- | --- | --- | --- |
| EA | -2.99745 | -1.63749 | 1.35996 | -3.06914 | -0.14205 | 2.92709 |
| IP | 0.83506 | 2.01133 | 1.17627 | 0.47109 | 2.23057 | 1.75947 |

## 0.3 Frozen B-heldout splits

Random and Murcko-scaffold-balanced metadata are frozen at split seed 42. Fold k uses fold k+1 as validation, so validation and test B identities are disjoint and both fixed across model seeds. This leaves 530–532 B monomers for training and yields validation/test rows near the A-heldout 4,774-row size.

| assignment | n_train_min | n_val_min | n_test_min | n_test_max |
| --- | --- | --- | --- | --- |
| random | 33390 | 4725 | 4725 | 4788 |
| clustered | 33390 | 4725 | 4725 | 4788 |

The existing A-heldout generator uses one whole distinct A identity for validation: it excludes both the test A and a second A from training, producing 33,418 train / 4,774 validation / 4,774 test rows per fold.

## 0.4 Role-matched novelty

Morgan r=2, 2048-bit maximum Tanimoto to the actual role-matched training identities. This reuses the fingerprint convention in `analysis/diagnostics/novelty.py`; the computation is generalized here to frozen B folds.

A-heldout:

| fold | min | median | max |
| --- | --- | --- | --- |
| 0 | 0.41667 | 0.41667 | 0.41667 |
| 1 | 0.43750 | 0.43750 | 0.43750 |
| 2 | 0.30769 | 0.30769 | 0.30769 |
| 3 | 0.38462 | 0.38462 | 0.38462 |
| 4 | 0.47368 | 0.47368 | 0.47368 |
| 5 | 0.38462 | 0.38462 | 0.38462 |
| 6 | 0.30769 | 0.30769 | 0.30769 |
| 7 | 0.47368 | 0.47368 | 0.47368 |
| 8 | 0.45455 | 0.45455 | 0.45455 |

Random B-heldout:

| fold | min | median | max |
| --- | --- | --- | --- |
| 0 | 0.32000 | 0.51862 | 1.00000 |
| 1 | 0.30000 | 0.57895 | 1.00000 |
| 2 | 0.25000 | 0.56386 | 1.00000 |
| 3 | 0.31579 | 0.53846 | 0.75000 |
| 4 | 0.30769 | 0.54167 | 0.95000 |
| 5 | 0.27778 | 0.54545 | 0.82609 |
| 6 | 0.33333 | 0.57143 | 1.00000 |
| 7 | 0.31429 | 0.52174 | 0.82609 |
| 8 | 0.28571 | 0.54545 | 0.76190 |

### Near-duplicate random B monomers

Counts are held-out B monomers whose maximum Morgan similarity to the fold's B-training identities reaches each threshold. These identities remain in the frozen split and must be reported both in full-fold and filtered (`max Tanimoto < 0.95`) Step-1 metrics.

| fold | n_heldout_B | n_max_ge_0_99 | n_max_ge_0_95 | n_max_ge_0_90 |
| --- | --- | --- | --- | --- |
| 0 | 76 | 1 | 1 | 1 |
| 1 | 76 | 1 | 2 | 3 |
| 2 | 76 | 1 | 1 | 1 |
| 3 | 76 | 0 | 0 | 0 |
| 4 | 76 | 0 | 1 | 1 |
| 5 | 76 | 0 | 0 | 0 |
| 6 | 76 | 1 | 1 | 2 |
| 7 | 75 | 0 | 0 | 0 |
| 8 | 75 | 0 | 0 | 0 |

Morgan-identical-or-near-identical (`≥0.99`) held-out/training pairs:

| fold | held_out_smiles_B | training_smiles_B | max_tanimoto |
| --- | --- | --- | --- |
| 0 | Brc1ccc(-c2ccc(-c3ccc(Br)cc3)cc2)cc1 | Brc1ccc(-c2ccc(Br)cc2)cc1 | 1.00000 |
| 1 | CCCCCCCCc1cc(Br)sc1Br | CCCCCCc1cc(Br)sc1Br | 1.00000 |
| 2 | Brc1ccc(-c2ccc(Br)cc2)cc1 | Brc1ccc(-c2ccc(-c3ccc(Br)cc3)cc2)cc1 | 1.00000 |
| 6 | CCCCCCc1cc(Br)sc1Br | CCCCCCCCc1cc(Br)sc1Br | 1.00000 |

Clustered B-heldout:

| fold | min | median | max |
| --- | --- | --- | --- |
| 0 | 0.27660 | 0.48584 | 0.72727 |
| 1 | 0.30233 | 0.48268 | 0.66667 |
| 2 | 0.35484 | 0.50000 | 0.75000 |
| 3 | 0.36842 | 0.54167 | 0.66667 |
| 4 | 0.25000 | 0.43406 | 0.95000 |
| 5 | 0.36364 | 0.47368 | 0.70588 |
| 6 | 0.21429 | 0.46429 | 0.71429 |
| 7 | 0.26087 | 0.43243 | 0.95000 |
| 8 | 0.25000 | 0.44444 | 0.75000 |

## 0.5 Null floors

All nulls use train-only lookup means and the same matched group key `(smiles_A, smiles_B, fracA)` with at least two `poly_type` values. The role-blind null is A-blind for A-heldout and B-blind for B-heldout; global-mean is an absolute reference.

| split | target | fold | null | group_mean_r2 | overall_r2 | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-heldout | EA | 0 | A-blind | 0.69380 | 0.69158 | 0.26002 | 0.27609 | 0.25962 |
| A-heldout | EA | 0 | global-mean | -0.27273 | -0.27272 | 0.48270 | 0.56086 | 0.25962 |
| A-heldout | EA | 1 | A-blind | 0.48731 | 0.48470 | 0.21665 | 0.25645 | -0.21208 |
| A-heldout | EA | 1 | global-mean | -0.35015 | -0.35241 | 0.29632 | 0.41546 | -0.21208 |
| A-heldout | EA | 2 | A-blind | 0.96114 | 0.95597 | 0.08439 | 0.11023 | 0.01491 |
| A-heldout | EA | 2 | global-mean | -0.00070 | -0.00081 | 0.39441 | 0.52554 | 0.01491 |
| A-heldout | EA | 3 | A-blind | 0.95276 | 0.94747 | 0.07258 | 0.09121 | -0.04957 |
| A-heldout | EA | 3 | global-mean | -0.01540 | -0.01551 | 0.27761 | 0.40107 | -0.04957 |
| A-heldout | EA | 4 | A-blind | 0.88373 | 0.86128 | 0.12697 | 0.14955 | 0.08668 |
| A-heldout | EA | 4 | global-mean | -0.04848 | -0.04660 | 0.31912 | 0.41079 | 0.08668 |
| A-heldout | EA | 5 | A-blind | 0.67571 | 0.67461 | 0.17628 | 0.20909 | -0.16868 |
| A-heldout | EA | 5 | global-mean | -0.21216 | -0.21176 | 0.29708 | 0.40351 | -0.16868 |
| A-heldout | EA | 6 | A-blind | -19.06946 | -19.97913 | 1.05641 | 1.11216 | -1.05632 |
| A-heldout | EA | 6 | global-mean | -18.07671 | -18.92530 | 1.05632 | 1.08387 | -1.05632 |
| A-heldout | EA | 7 | A-blind | 0.09751 | 0.10387 | 0.51370 | 0.58404 | 0.51330 |
| A-heldout | EA | 7 | global-mean | -0.69798 | -0.69219 | 0.70466 | 0.80257 | 0.51330 |
| A-heldout | EA | 8 | A-blind | 0.42772 | 0.42902 | 0.39270 | 0.44157 | 0.39212 |
| A-heldout | EA | 8 | global-mean | -0.45055 | -0.45026 | 0.61922 | 0.70373 | 0.39212 |
| A-heldout | IP | 0 | A-blind | 0.96902 | 0.96650 | 0.04433 | 0.05219 | -0.02312 |
| A-heldout | IP | 0 | global-mean | -0.00683 | -0.00657 | 0.21760 | 0.28610 | -0.02312 |
| A-heldout | IP | 1 | A-blind | 0.50927 | 0.50602 | 0.24599 | 0.26343 | -0.24587 |
| A-heldout | IP | 1 | global-mean | -0.43104 | -0.43032 | 0.38588 | 0.44825 | -0.24587 |
| A-heldout | IP | 2 | A-blind | -1.01943 | -1.00861 | 0.66143 | 0.73829 | -0.66143 |
| A-heldout | IP | 2 | global-mean | -1.62134 | -1.61217 | 0.73079 | 0.84194 | -0.66143 |
| A-heldout | IP | 3 | A-blind | -3.20636 | -3.20740 | 0.42575 | 0.45940 | 0.42571 |
| A-heldout | IP | 3 | global-mean | -3.60120 | -3.61294 | 0.43305 | 0.48103 | 0.42571 |
| A-heldout | IP | 4 | A-blind | -0.25097 | -0.34241 | 0.19383 | 0.24213 | 0.18526 |
| A-heldout | IP | 4 | global-mean | -0.74342 | -0.78588 | 0.22255 | 0.27927 | 0.18526 |
| A-heldout | IP | 5 | A-blind | -7.52773 | -6.93823 | 0.70901 | 0.74407 | 0.70901 |
| A-heldout | IP | 5 | global-mean | -7.75557 | -7.20793 | 0.71093 | 0.75660 | 0.70901 |
| A-heldout | IP | 6 | A-blind | 0.56868 | 0.56208 | 0.23231 | 0.25583 | -0.23220 |
| A-heldout | IP | 6 | global-mean | -0.36024 | -0.36077 | 0.36438 | 0.45097 | -0.23220 |
| A-heldout | IP | 7 | A-blind | 0.40980 | 0.40641 | 0.29355 | 0.32787 | -0.29327 |
| A-heldout | IP | 7 | global-mean | -0.47843 | -0.47492 | 0.42981 | 0.51682 | -0.29327 |
| A-heldout | IP | 8 | A-blind | -0.03401 | -0.06530 | 0.21632 | 0.24527 | 0.21607 |
| A-heldout | IP | 8 | global-mean | -0.79694 | -0.82676 | 0.26886 | 0.32118 | 0.21607 |
| B-heldout random | EA | 0 | B-blind | 0.58850 | 0.58831 | 0.23945 | 0.33791 | 0.02778 |
| B-heldout random | EA | 0 | global-mean | -0.00302 | -0.00278 | 0.41518 | 0.52737 | 0.02778 |
| B-heldout random | EA | 1 | B-blind | 0.47565 | 0.47562 | 0.29832 | 0.41897 | 0.01566 |
| B-heldout random | EA | 1 | global-mean | -0.00078 | -0.00073 | 0.46200 | 0.57878 | 0.01566 |
| B-heldout random | EA | 2 | B-blind | 0.34006 | 0.34234 | 0.35849 | 0.54081 | -0.04133 |
| B-heldout random | EA | 2 | global-mean | -0.00376 | -0.00384 | 0.51024 | 0.66815 | -0.04133 |
| B-heldout random | EA | 3 | B-blind | 0.39426 | 0.39583 | 0.33946 | 0.49302 | -0.01631 |
| B-heldout random | EA | 3 | global-mean | -0.00062 | -0.00066 | 0.50945 | 0.63450 | -0.01631 |
| B-heldout random | EA | 4 | B-blind | 0.41966 | 0.42079 | 0.33179 | 0.45334 | -0.02234 |
| B-heldout random | EA | 4 | global-mean | -0.00133 | -0.00141 | 0.48140 | 0.59609 | -0.02234 |
| B-heldout random | EA | 5 | B-blind | 0.49126 | 0.49267 | 0.30845 | 0.42195 | 0.06543 |
| B-heldout random | EA | 5 | global-mean | -0.01238 | -0.01220 | 0.48200 | 0.59600 | 0.06543 |
| B-heldout random | EA | 6 | B-blind | 0.46035 | 0.46085 | 0.31264 | 0.42374 | -0.01206 |
| B-heldout random | EA | 6 | global-mean | -0.00040 | -0.00044 | 0.46396 | 0.57722 | -0.01206 |
| B-heldout random | EA | 7 | B-blind | 0.40982 | 0.41009 | 0.31657 | 0.44527 | -0.04835 |
| B-heldout random | EA | 7 | global-mean | -0.00689 | -0.00696 | 0.46798 | 0.58175 | -0.04835 |
| B-heldout random | EA | 8 | B-blind | 0.41769 | 0.41970 | 0.34540 | 0.48085 | 0.03126 |
| B-heldout random | EA | 8 | global-mean | -0.00255 | -0.00245 | 0.50816 | 0.63200 | 0.03126 |
| B-heldout random | IP | 0 | B-blind | 0.60967 | 0.61091 | 0.22410 | 0.30160 | -0.04666 |
| B-heldout random | IP | 0 | global-mean | -0.00957 | -0.00931 | 0.38845 | 0.48576 | -0.04666 |
| B-heldout random | IP | 1 | B-blind | 0.51899 | 0.51800 | 0.23346 | 0.32790 | 0.02911 |
| B-heldout random | IP | 1 | global-mean | -0.00359 | -0.00380 | 0.38119 | 0.47319 | 0.02911 |
| B-heldout random | IP | 2 | B-blind | 0.56042 | 0.55810 | 0.21788 | 0.30931 | 0.00319 |
| B-heldout random | IP | 2 | global-mean | -0.00003 | -0.00005 | 0.36573 | 0.46531 | 0.00319 |
| B-heldout random | IP | 3 | B-blind | 0.56286 | 0.56236 | 0.22508 | 0.31767 | -0.00722 |
| B-heldout random | IP | 3 | global-mean | -0.00029 | -0.00023 | 0.38279 | 0.48025 | -0.00722 |
| B-heldout random | IP | 4 | B-blind | 0.50968 | 0.50908 | 0.24434 | 0.34135 | 0.01968 |
| B-heldout random | IP | 4 | global-mean | -0.00150 | -0.00163 | 0.38795 | 0.48757 | 0.01968 |
| B-heldout random | IP | 5 | B-blind | 0.57851 | 0.57862 | 0.22928 | 0.31454 | -0.02490 |
| B-heldout random | IP | 5 | global-mean | -0.00273 | -0.00264 | 0.38408 | 0.48519 | -0.02490 |
| B-heldout random | IP | 6 | B-blind | 0.50954 | 0.50936 | 0.24106 | 0.33876 | 0.02596 |
| B-heldout random | IP | 6 | global-mean | -0.00270 | -0.00288 | 0.38997 | 0.48433 | 0.02596 |
| B-heldout random | IP | 7 | B-blind | 0.56709 | 0.56766 | 0.22674 | 0.31902 | -0.01713 |
| B-heldout random | IP | 7 | global-mean | -0.00134 | -0.00125 | 0.38677 | 0.48548 | -0.01713 |
| B-heldout random | IP | 8 | B-blind | 0.50502 | 0.50551 | 0.24204 | 0.34355 | 0.01803 |
| B-heldout random | IP | 8 | global-mean | -0.00129 | -0.00136 | 0.38570 | 0.48888 | 0.01803 |
| B-heldout clustered | EA | 0 | B-blind | 0.54191 | 0.54156 | 0.28001 | 0.36514 | 0.16699 |
| B-heldout clustered | EA | 0 | global-mean | -0.09630 | -0.09588 | 0.45001 | 0.56455 | 0.16699 |
| B-heldout clustered | EA | 1 | B-blind | 0.38352 | 0.38518 | 0.41264 | 0.49026 | 0.23715 |
| B-heldout clustered | EA | 1 | global-mean | -0.14454 | -0.14386 | 0.56562 | 0.66871 | 0.23715 |
| B-heldout clustered | EA | 2 | B-blind | 0.74719 | 0.74623 | 0.18824 | 0.24794 | 0.14657 |
| B-heldout clustered | EA | 2 | global-mean | -0.09036 | -0.08868 | 0.41193 | 0.51354 | 0.14657 |
| B-heldout clustered | EA | 3 | B-blind | 0.37390 | 0.37750 | 0.37628 | 0.51046 | -0.04224 |
| B-heldout clustered | EA | 3 | global-mean | -0.00423 | -0.00426 | 0.53224 | 0.64835 | -0.04224 |
| B-heldout clustered | EA | 4 | B-blind | 0.31687 | 0.31610 | 0.30547 | 0.51057 | -0.08631 |
| B-heldout clustered | EA | 4 | global-mean | -0.01898 | -0.01954 | 0.45552 | 0.62339 | -0.08631 |
| B-heldout clustered | EA | 5 | B-blind | 0.62946 | 0.62934 | 0.24619 | 0.32255 | 0.06268 |
| B-heldout clustered | EA | 5 | global-mean | -0.01435 | -0.01400 | 0.43045 | 0.53349 | 0.06268 |
| B-heldout clustered | EA | 6 | B-blind | 0.01524 | 0.01539 | 0.40134 | 0.58476 | -0.28881 |
| B-heldout clustered | EA | 6 | global-mean | -0.23820 | -0.24017 | 0.52240 | 0.65627 | -0.28881 |
| B-heldout clustered | EA | 7 | B-blind | 0.18923 | 0.18946 | 0.39888 | 0.57380 | -0.18940 |
| B-heldout clustered | EA | 7 | global-mean | -0.08750 | -0.08831 | 0.52083 | 0.66489 | -0.18940 |
| B-heldout clustered | EA | 8 | B-blind | 0.44717 | 0.44674 | 0.31250 | 0.41988 | -0.00890 |
| B-heldout clustered | EA | 8 | global-mean | -0.00018 | -0.00025 | 0.45614 | 0.56456 | -0.00890 |
| B-heldout clustered | IP | 0 | B-blind | 0.66315 | 0.66157 | 0.18821 | 0.26484 | -0.04825 |
| B-heldout clustered | IP | 0 | global-mean | -0.01160 | -0.01124 | 0.36805 | 0.45779 | -0.04825 |
| B-heldout clustered | IP | 1 | B-blind | 0.50476 | 0.50217 | 0.23861 | 0.32670 | 0.04497 |
| B-heldout clustered | IP | 1 | global-mean | -0.00892 | -0.00943 | 0.37561 | 0.46521 | 0.04497 |
| B-heldout clustered | IP | 2 | B-blind | 0.58485 | 0.58612 | 0.24821 | 0.32243 | -0.17085 |
| B-heldout clustered | IP | 2 | global-mean | -0.11758 | -0.11621 | 0.41620 | 0.52950 | -0.17085 |
| B-heldout clustered | IP | 3 | B-blind | 0.52889 | 0.52808 | 0.27037 | 0.35930 | -0.08805 |
| B-heldout clustered | IP | 3 | global-mean | -0.02894 | -0.02834 | 0.41401 | 0.53039 | -0.08805 |
| B-heldout clustered | IP | 4 | B-blind | 0.65442 | 0.64976 | 0.16198 | 0.24629 | 0.01156 |
| B-heldout clustered | IP | 4 | global-mean | -0.00067 | -0.00077 | 0.32181 | 0.41632 | 0.01156 |
| B-heldout clustered | IP | 5 | B-blind | 0.55571 | 0.55874 | 0.24630 | 0.33944 | -0.04143 |
| B-heldout clustered | IP | 5 | global-mean | -0.00662 | -0.00657 | 0.41385 | 0.51268 | -0.04143 |
| B-heldout clustered | IP | 6 | B-blind | 0.56935 | 0.56847 | 0.23456 | 0.32116 | -0.03556 |
| B-heldout clustered | IP | 6 | global-mean | -0.00544 | -0.00529 | 0.38331 | 0.49019 | -0.03556 |
| B-heldout clustered | IP | 7 | B-blind | 0.34716 | 0.34751 | 0.24351 | 0.35199 | 0.14188 |
| B-heldout clustered | IP | 7 | global-mean | -0.10488 | -0.10601 | 0.37423 | 0.45827 | 0.14188 |
| B-heldout clustered | IP | 8 | B-blind | 0.16978 | 0.17279 | 0.29334 | 0.41116 | 0.18917 |
| B-heldout clustered | IP | 8 | global-mean | -0.17367 | -0.17509 | 0.40456 | 0.49005 | 0.18917 |

Median and mean across folds:

| split | target | null | group_mean_r2_median | group_mean_r2_mean | overall_r2_median | overall_r2_mean | mae_median | mae_mean | bias_median | bias_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-heldout | EA | A-blind | 0.67571 | -1.54331 | 0.67461 | -1.64785 | 0.21665 | 0.32219 | 0.01491 | -0.02445 |
| A-heldout | EA | global-mean | -0.27273 | -2.23609 | -0.27272 | -2.32973 | 0.39441 | 0.49416 | 0.01491 | -0.02445 |
| A-heldout | IP | A-blind | -0.03401 | -1.06464 | -0.06530 | -1.01344 | 0.24599 | 0.33584 | -0.02312 | 0.00891 |
| A-heldout | IP | global-mean | -0.74342 | -1.75500 | -0.78588 | -1.70203 | 0.38588 | 0.41821 | -0.02312 | 0.00891 |
| B-heldout clustered | EA | B-blind | 0.38352 | 0.40494 | 0.38518 | 0.40528 | 0.31250 | 0.32462 | -0.00890 | -0.00025 |
| B-heldout clustered | EA | global-mean | -0.08750 | -0.07718 | -0.08831 | -0.07722 | 0.45614 | 0.48279 | -0.00890 | -0.00025 |
| B-heldout clustered | IP | B-blind | 0.55571 | 0.50868 | 0.55874 | 0.50836 | 0.24351 | 0.23612 | -0.03556 | 0.00038 |
| B-heldout clustered | IP | global-mean | -0.01160 | -0.05093 | -0.01124 | -0.05100 | 0.38331 | 0.38574 | -0.03556 | 0.00038 |
| B-heldout random | EA | B-blind | 0.41966 | 0.44414 | 0.42079 | 0.44513 | 0.31657 | 0.31673 | -0.01206 | -0.00003 |
| B-heldout random | EA | global-mean | -0.00255 | -0.00353 | -0.00245 | -0.00350 | 0.48140 | 0.47782 | -0.01206 | -0.00003 |
| B-heldout random | IP | B-blind | 0.56042 | 0.54686 | 0.55810 | 0.54662 | 0.22928 | 0.23155 | 0.00319 | 0.00001 |
| B-heldout random | IP | global-mean | -0.00150 | -0.00256 | -0.00163 | -0.00257 | 0.38570 | 0.38363 | 0.00319 | 0.00001 |

## 0.6 Verdict

Headroom is `1 - median null R²`; values near zero are degenerate for that metric.

| split | target | null | metric | null_floor_median | headroom_to_1 |
| --- | --- | --- | --- | --- | --- |
| A-heldout | EA | A-blind | group_mean_r2 | 0.67571 | 0.32429 |
| A-heldout | EA | A-blind | overall_r2 | 0.67461 | 0.32539 |
| A-heldout | EA | global-mean | group_mean_r2 | -0.27273 | 1.27273 |
| A-heldout | EA | global-mean | overall_r2 | -0.27272 | 1.27272 |
| A-heldout | IP | A-blind | group_mean_r2 | -0.03401 | 1.03401 |
| A-heldout | IP | A-blind | overall_r2 | -0.06530 | 1.06530 |
| A-heldout | IP | global-mean | group_mean_r2 | -0.74342 | 1.74342 |
| A-heldout | IP | global-mean | overall_r2 | -0.78588 | 1.78588 |
| B-heldout clustered | EA | B-blind | group_mean_r2 | 0.38352 | 0.61648 |
| B-heldout clustered | EA | B-blind | overall_r2 | 0.38518 | 0.61482 |
| B-heldout clustered | EA | global-mean | group_mean_r2 | -0.08750 | 1.08750 |
| B-heldout clustered | EA | global-mean | overall_r2 | -0.08831 | 1.08831 |
| B-heldout clustered | IP | B-blind | group_mean_r2 | 0.55571 | 0.44429 |
| B-heldout clustered | IP | B-blind | overall_r2 | 0.55874 | 0.44126 |
| B-heldout clustered | IP | global-mean | group_mean_r2 | -0.01160 | 1.01160 |
| B-heldout clustered | IP | global-mean | overall_r2 | -0.01124 | 1.01124 |
| B-heldout random | EA | B-blind | group_mean_r2 | 0.41966 | 0.58034 |
| B-heldout random | EA | B-blind | overall_r2 | 0.42079 | 0.57921 |
| B-heldout random | EA | global-mean | group_mean_r2 | -0.00255 | 1.00255 |
| B-heldout random | EA | global-mean | overall_r2 | -0.00245 | 1.00245 |
| B-heldout random | IP | B-blind | group_mean_r2 | 0.56042 | 0.43958 |
| B-heldout random | IP | B-blind | overall_r2 | 0.55810 | 0.44190 |
| B-heldout random | IP | global-mean | group_mean_r2 | -0.00150 | 1.00150 |
| B-heldout random | IP | global-mean | overall_r2 | -0.00163 | 1.00163 |

The A-heldout EA chemistry metric is demonstrably constrained: its A-blind median floor is 0.67571 and individual folds reach 0.96114. This directly qualifies the EA chemistry headline in `variant_results_report.md` (the sections titled `What changed / what flipped` and `Headline`): current A-heldout group-mean R² rankings cannot by themselves establish useful unseen-chemistry learning.

Random B-heldout is worth GPU for both EA and IP. Its B-blind group-mean floors are 0.41966 (EA) and 0.56042 (IP), leaving 0.58034 and 0.43958 R² headroom. The B-blind floor does not collapse to the global mean because it averages roughly 530 seen B identities conditional on A, and A identity itself explains substantial variance. Clustered B-heldout is also viable and is chemically harder by its lower median nearest-neighbor similarities, but it should remain a follow-up until random B-heldout establishes the baseline comparison.

Architecture variance within `(A, B, fracA)` is 0.979% for EA and 1.459% for IP, confirming the known 1–4% scale. Architecture-recovery metrics remain meaningful but require the Step 1 paired comparison; no existing model ranking is changed by Step 0 alone.
