# Experiment C: Pair-Held-Out Architecture Transfer Analysis

```
======================================================================
EXPERIMENT C: PAIR-HELD-OUT ARCHITECTURE TRANSFER ANALYSIS
======================================================================
Dataset: 42966 rows
Unique smiles_A: 9
Unique (A,B) pairs: 6138
Unique (A,B,fA,fB) groups: 18414
Architectures: ['alternating', 'block', 'random']
======================================================================
PART 1: VERIFY PAIR-DISJOINT SPLIT CORRECTNESS
======================================================================

--- Pair-Disjoint Overlap Audit ---
 Fold  n_train  n_test  tr_pairs  te_pairs  overlap
--------------------------------------------------
    0    30933    8596      4419      1228        0
    1    30933    8596      4419      1228        0
    2    30933    8596      4419      1228        0
    3    30940    8589      4420      1227        0
    4    30940    8589      4420      1227        0

  ✓ All folds: zero pair overlap confirmed.

--- Group-Disjoint Overlap Audit ---
 Fold  n_train  n_test  tr_groups  te_groups  overlap
-------------------------------------------------------
    0    30916    8594      13258       3683        0
    1    30984    8594      13258       3683        0
    2    30931    8592      13259       3682        0
    3    30941    8593      13258       3683        0
    4    30947    8593      13258       3683        0

  ✓ All folds: zero group overlap confirmed.

--- A-held-out smiles_A Overlap Audit ---
 Fold  n_test  te_A  tr_A  A_overlap
----------------------------------------
    0    9548     2     7          0
    1    9548     2     7          0
    2    9548     2     7          0
    3    9548     2     7          0
    4    4774     1     8          0

  ✓ A-held-out: smiles_A completely disjoint across all folds.

--- Split Hierarchy Summary ---
  A-held-out:     Entire monomer A chemistries unseen in test
  Pair-disjoint:  Entire (A,B) pairs unseen, but A and B seen separately
  Group-disjoint: Entire (A,B,fA,fB) groups unseen, but pairs/monomers seen

======================================================================
PART 2: DISTRIBUTION MATCHING ACROSS SPLITS
======================================================================

--- EA Overall Distribution (per-fold mean ± std) ---
Split                 fold0    fold1    fold2    fold3    fold4       mean      std
-------------------------------------------------------------------------------------
a_held_out          -2.7192  -2.8135  -2.0078  -2.5759  -2.6395    -2.5414   0.6001
group_disjoint      -2.5507  -2.5277  -2.5425  -2.5393  -2.5465    -2.5414   0.6001
pair_disjoint       -2.5361  -2.5131  -2.5479  -2.5516  -2.5580    -2.5414   0.6001

--- EA Architecture Deviation |Δy| Statistics ---
Split                mean|Δ|    median       std       p90       p95       max
---------------------------------------------------------------------------
a_held_out            0.0368    0.0211    0.0594    0.0883    0.1302    0.6102
group_disjoint        0.0368    0.0211    0.0594    0.0883    0.1302    0.6102
pair_disjoint         0.0368    0.0211    0.0594    0.0883    0.1302    0.6102

--- IP Overall Distribution (per-fold mean ± std) ---
Split                 fold0    fold1    fold2    fold3    fold4       mean      std
-------------------------------------------------------------------------------------
a_held_out           1.1753   1.8617   1.6829   1.1361   1.3686     1.4534   0.4818
group_disjoint       1.4689   1.4473   1.4615   1.4601   1.4291     1.4534   0.4818
pair_disjoint        1.4517   1.4601   1.4518   1.4544   1.4490     1.4534   0.4818

--- IP Architecture Deviation |Δy| Statistics ---
Split                mean|Δ|    median       std       p90       p95       max
---------------------------------------------------------------------------
a_held_out            0.0353    0.0185    0.0582    0.0890    0.1330    0.5755
group_disjoint        0.0353    0.0185    0.0582    0.0890    0.1330    0.5755
pair_disjoint         0.0353    0.0185    0.0582    0.0890    0.1330    0.5755

NOTE: Concatenated test sets = full dataset (5-fold CV), so overall
distributions are identical. Per-fold variation reveals split difficulty.
Saved: EA_distribution_comparison.png, IP_distribution_comparison.png
Saved: EA_arch_distribution_comparison.png, IP_arch_distribution_comparison.png

Saved: pair_disjoint_metrics.csv

======================================================================
PART 3: COMPUTE METRICS FROM PREDICTIONS
======================================================================

Split              Model       Tgt       R²      ±     MAE      ±   ArchR²      ±  ArchMAE      ±
-----------------------------------------------------------------------------------------------
a_held_out         frac         EA   0.9741 0.0043  0.0642 0.0065  -0.0035 0.0024   0.0369 0.0013
a_held_out         frac         IP   0.9642 0.0094  0.0584 0.0072  -0.0028 0.0014   0.0353 0.0020
a_held_out         2d0_arch     EA   0.9813 0.0053  0.0481 0.0081   0.8438 0.0195   0.0156 0.0011
a_held_out         2d0_arch     IP   0.9796 0.0084  0.0374 0.0074   0.9064 0.0240   0.0117 0.0010
a_held_out         2d1_arch     EA   0.9831 0.0031  0.0470 0.0059   0.8626 0.0137   0.0143 0.0008
a_held_out         2d1_arch     IP   0.9801 0.0096  0.0374 0.0067   0.9135 0.0207   0.0111 0.0011
group_disjoint     frac         EA   0.9879 0.0004  0.0436 0.0008   0.0000 0.0000   0.0368 0.0006
group_disjoint     frac         IP   0.9838 0.0004  0.0392 0.0005   0.0000 0.0000   0.0353 0.0004
group_disjoint     2d0_arch     EA   0.9971 0.0004  0.0213 0.0007   0.8868 0.0143   0.0134 0.0004
group_disjoint     2d0_arch     IP   0.9980 0.0001  0.0149 0.0003   0.9423 0.0032   0.0095 0.0001
group_disjoint     2d1_arch     EA   0.9984 0.0002  0.0163 0.0013   0.9381 0.0070   0.0098 0.0002
group_disjoint     2d1_arch     IP   0.9987 0.0001  0.0118 0.0004   0.9649 0.0017   0.0074 0.0001
pair_disjoint      frac         EA   0.9878 0.0002  0.0441 0.0004   0.0000 0.0000   0.0368 0.0002
pair_disjoint      frac         IP   0.9831 0.0007  0.0407 0.0016   0.0000 0.0000   0.0353 0.0002
pair_disjoint      2d0_arch     EA   0.9968 0.0002  0.0226 0.0007   0.8862 0.0045   0.0136 0.0002
pair_disjoint      2d0_arch     IP   0.9976 0.0002  0.0163 0.0008   0.9406 0.0023   0.0096 0.0001
pair_disjoint      2d1_arch     EA   0.9979 0.0002  0.0178 0.0006   0.9346 0.0067   0.0098 0.0003
pair_disjoint      2d1_arch     IP   0.9981 0.0000  0.0139 0.0003   0.9637 0.0026   0.0075 0.0002

Saved: pair_disjoint_metrics.csv

======================================================================
PART 4: TRANSFERABILITY ANALYSIS
======================================================================

--- Summary Table ---
Split              Model         EA R²    IP R²  EA ArchR²  IP ArchR²
------------------------------------------------------------
a_held_out         frac         0.9741   0.9642    -0.0035    -0.0028
a_held_out         2d0_arch     0.9813   0.9796     0.8438     0.9064
a_held_out         2d1_arch     0.9831   0.9801     0.8626     0.9135
group_disjoint     frac         0.9879   0.9838     0.0000     0.0000
group_disjoint     2d0_arch     0.9971   0.9980     0.8868     0.9423
group_disjoint     2d1_arch     0.9984   0.9987     0.9381     0.9649
pair_disjoint      frac         0.9878   0.9831     0.0000     0.0000
pair_disjoint      2d0_arch     0.9968   0.9976     0.8862     0.9406
pair_disjoint      2d1_arch     0.9979   0.9981     0.9346     0.9637

Saved: pair_disjoint_summary_table.csv

--- Transferability Deltas: Pair-Disjoint − Group-Disjoint ---
Model         ΔEA_R²    ΔIP_R²   ΔEA_ArchR²   ΔIP_ArchR²
-------------------------------------------------------
frac         -0.0001   -0.0007      +0.0000      +0.0000
2d0_arch     -0.0003   -0.0004      -0.0006      -0.0016
2d1_arch     -0.0005   -0.0006      -0.0035      -0.0011

--- Transferability Deltas: Pair-Disjoint − A-held-out ---
Model         ΔEA_R²    ΔIP_R²   ΔEA_ArchR²   ΔIP_ArchR²
-------------------------------------------------------
frac         +0.0137   +0.0189      +0.0035      +0.0028
2d0_arch     +0.0155   +0.0179      +0.0424      +0.0342
2d1_arch     +0.0148   +0.0180      +0.0719      +0.0502
Saved: overall_R2_comparison.png
Saved: arch_R2_comparison.png

======================================================================
PART 5: INTERPRETATION
======================================================================

--- Case A: Does architecture effect generalise across unseen chemistry? ---
  EA 2D1-arch: GroupDisjoint ArchR² = 0.9381±0.0070, PairDisjoint = 0.9346±0.0067, Δ = -0.0035 → ≈ COMPARABLE (within 1 std)
  IP 2D1-arch: GroupDisjoint ArchR² = 0.9649±0.0017, PairDisjoint = 0.9637±0.0026, Δ = -0.0011 → ≈ COMPARABLE (within 1 std)

--- Case B: Chemistry-specific corrections? ---
  EA 2D1: Overall R² GD=0.9984 PD=0.9979 (Δ=-0.0005)
  EA 2D1: ArchR²    GD=0.9381 PD=0.9346 (Δ=-0.0035)
    → No strong evidence of chemistry-specific memorisation
  IP 2D1: Overall R² GD=0.9987 PD=0.9981 (Δ=-0.0006)
  IP 2D1: ArchR²    GD=0.9649 PD=0.9637 (Δ=-0.0011)
    → No strong evidence of chemistry-specific memorisation

--- Case C: 2D1 vs 2D0 under pair-disjoint (transferable interaction?) ---
  EA: 2D0 ArchR² = 0.8862±0.0045, 2D1 ArchR² = 0.9346±0.0067, Δ = +0.0483 → 2D1 > 2D0: interaction provides transferable signal
  IP: 2D0 ArchR² = 0.9406±0.0023, 2D1 ArchR² = 0.9637±0.0026, Δ = +0.0231 → 2D1 > 2D0: interaction provides transferable signal

--- Case D: A-held-out performance (hardest split) ---
  EA: Frac=-0.0035, 2D0=0.8438, 2D1=0.8626
    2D1 lift over Frac: +0.8661
  IP: Frac=-0.0028, 2D0=0.9064, 2D1=0.9135
    2D1 lift over Frac: +0.9163

======================================================================
FINAL CONCLUSION
======================================================================

Evidence Assessment:

  1. DATASET BOTTLENECK:
     Architecture effects = ~1% of total variance.
     This is a fundamental signal-to-noise challenge, not a data quantity issue.
     → Partial evidence for dataset bottleneck (small signal).

  2. GENERALISATION BOTTLENECK:
     Group-disjoint mean ArchR² (2D1): 0.9515
     Pair-disjoint mean ArchR²  (2D1): 0.9492
     Drop: +0.0023
     → ArchR² is stable (0.0023) — architecture learning transfers.
     → Evidence AGAINST generalisation bottleneck.

  3. TRANSFERABLE ARCHITECTURE LEARNING:
     2D0 pair-disjoint mean ArchR²: 0.9134
     2D1 pair-disjoint mean ArchR²: 0.9492
     2D1 − 2D0: +0.0357
     → 2D1 outperforms 2D0 on unseen pairs: interaction MLP provides
       genuinely transferable architecture-chemistry signal.

  4. SPLIT COMPARISON:
     A-held-out mean ArchR²  (2D1): 0.8881  ← unseen monomers
     Group-disjoint mean ArchR² (2D1): 0.9515  ← unseen compositions
     Pair-disjoint mean ArchR²  (2D1): 0.9492  ← unseen pairs

     The hierarchy should be:  A-held-out ≤ Pair-disjoint ≤ Group-disjoint
     because A-held-out is the hardest (unseen monomers) and Group-disjoint
     is the easiest (only unseen compositions of known pairs).

```
