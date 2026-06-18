# Dataset Bottleneck Assessment

Combines architecture signal analysis with matched-group saturation learning curves.

```
======================================================================
STAGE 2D DATASET BOTTLENECK ASSESSMENT
======================================================================
Dataset: 42966 rows
Unique groups (A,B,fA,fB): 18414
Unique (A,B) pairs: 6138
======================================================================
EXPERIMENT A: ARCHITECTURE SIGNAL MAGNITUDE ANALYSIS
======================================================================

--- A1/A2: Variance Decomposition ---

  EA:
    Var_total       = 0.360127
    Var_comp        = 0.356600  (99.02%)
    Var_arch        = 0.003527  (0.98%)
    Var_comp + Var_arch = 0.360127  (ratio to total: 1.000000)

  IP:
    Var_total       = 0.232108
    Var_comp        = 0.228721  (98.54%)
    Var_arch        = 0.003387  (1.46%)
    Var_comp + Var_arch = 0.232108  (ratio to total: 1.000000)

  Saved: architecture_variance_table.csv

--- A3: Architecture Signal Statistics ---
          mean|Δ|     med|Δ|     std(Δ)     p90|Δ|     p95|Δ|     max|Δ|
  ------------------------------------------------------------------
  EA       0.0368     0.0211     0.0594     0.0883     0.1302     0.6102
  IP       0.0353     0.0185     0.0582     0.0890     0.1330     0.5755
  Saved: deltaEA_distribution.png, deltaIP_distribution.png

--- A4: Architecture Variance By Group ---

  EA:
    Total groups: 18414
    Multi-arch groups (≥2 members): 18414
    Median group variance: 0.000613
    Mean group variance:   0.003042
    Max group variance:    0.230890
    Top 10 highest-variance groups:
      var=0.230890  n=3  (A=OB(O)c1ccc2c(c1)[nH], B=O=[N+]([O-])c1c(Br)s, fA=0.5)
      var=0.228909  n=3  (A=OB(O)c1cc2ccc3cc(B(O, B=O=[N+]([O-])c1c(Br)s, fA=0.5)
      var=0.223195  n=2  (A=OB(O)c1ccc2c(c1)[nH], B=[C-]#[N+]c1c(Br)cc(F, fA=0.25)
      var=0.209380  n=3  (A=CC1(C)c2cc(B(O)O)ccc, B=O=[N+]([O-])c1c(Br)s, fA=0.5)
      var=0.138590  n=2  (A=OB(O)c1ccc2c(c1)[nH], B=O=[N+]([O-])c1c(Br)s, fA=0.75)
      var=0.138274  n=3  (A=OB(O)c1ccc2c(c1)[nH], B=O=C1OC(=O)c2c(Br)sc(, fA=0.5)
      var=0.128810  n=2  (A=CC1(C)c2cc(B(O)O)ccc, B=O=[N+]([O-])c1c(Br)s, fA=0.75)
      var=0.126023  n=3  (A=OB(O)c1ccc2c(c1)[nH], B=O=[N+]([O-])c1c(Br)c, fA=0.5)
      var=0.111237  n=3  (A=OB(O)c1cc2ccc3cc(B(O, B=O=C1OC(=O)c2c(Br)sc(, fA=0.5)
      var=0.109883  n=2  (A=OB(O)c1cc2ccc3cc(B(O, B=O=[N+]([O-])c1c(Br)s, fA=0.75)

  IP:
    Total groups: 18414
    Multi-arch groups (≥2 members): 18414
    Median group variance: 0.000483
    Mean group variance:   0.002933
    Max group variance:    0.190402
    Top 10 highest-variance groups:
      var=0.190402  n=3  (A=OB(O)c1ccc(B(O)O)cc1, B=O=c1oc(=O)c2c(Br)c3c, fA=0.5)
      var=0.174827  n=3  (A=OB(O)c1ccc(B(O)O)c2n, B=O=c1oc(=O)c2c(Br)c3c, fA=0.5)
      var=0.115339  n=2  (A=OB(O)c1ccc(B(O)O)c2n, B=O=c1oc(=O)c2c(Br)c3c, fA=0.25)
      var=0.094129  n=3  (A=OB(O)c1ccc(-c2ccc(B(, B=O=[N+]([O-])c1c(Br)c, fA=0.5)
      var=0.092128  n=3  (A=OB(O)c1ccc(-c2ccc(B(, B=CC(C)(C)N=[N+]([O-]), fA=0.5)
      var=0.091250  n=3  (A=OB(O)c1ccc(-c2ccc(B(, B=O=[N+]([O-])c1c(Br)c, fA=0.5)
      var=0.089251  n=2  (A=OB(O)c1ccc(B(O)O)cc1, B=O=c1oc(=O)c2c(Br)c3c, fA=0.25)
      var=0.088727  n=3  (A=OB(O)c1ccc(-c2ccc(B(, B=O=[N+]([O-])c1c(Br)c, fA=0.5)
      var=0.086158  n=3  (A=OB(O)c1ccc(-c2ccc(B(, B=O=[N+]([O-])c1ncc(Br, fA=0.5)
      var=0.086157  n=3  (A=OB(O)c1ccc(-c2ccc(B(, B=O=[N+]([O-])c1cnc(Br, fA=0.5)
  Saved: architecture_group_variance.csv
  Saved: architecture_variance_distribution_EA.png, architecture_variance_distribution_IP.png

--- A5: Architecture Signal vs Frac Residual ---

  EA:
    Frac residual variance:     0.009261
    Architecture variance:       0.003527
    Var_arch / Var_residual:      0.3808
    → Architecture effects explain 38.1% of remaining Frac error

  IP:
    Frac residual variance:     0.008385
    Architecture variance:       0.003387
    Var_arch / Var_residual:      0.4039
    → Architecture effects explain 40.4% of remaining Frac error

======================================================================
EXPERIMENT B: MATCHED-GROUP SATURATION LEARNING CURVE
======================================================================

--- B1–B3: Load Learning Curve Predictions ---
  Loaded 16 data points
  Saved: learning_curve_metrics.csv

target      model  frac  R2_mean  R2_std  R2_arch R2a_std
-------------------------------------------------------
  EA   2d0_arch    25   0.9450  0.0345   0.3611  0.3056
  EA   2d0_arch    50   0.9531  0.0292   0.4544  0.2575
  EA   2d0_arch    75   0.9528  0.0260   0.4217  0.2569
  EA   2d0_arch   100   0.9436  0.0455   0.4742  0.2984
  EA   2d1_arch    25   0.8055  0.2422   0.3496  0.3662
  EA   2d1_arch    50   0.8169  0.2384   0.5102  0.2130
  EA   2d1_arch    75   0.8556  0.1817   0.4803  0.3161
  EA   2d1_arch   100   0.8740  0.1495   0.3498  0.5528
  IP   2d0_arch    25   0.9241  0.0417   0.3173  0.3649
  IP   2d0_arch    50   0.9430  0.0145   0.2524  0.5073
  IP   2d0_arch    75   0.9370  0.0265   0.4837  0.2289
  IP   2d0_arch   100   0.9331  0.0349   0.3984  0.2692
  IP   2d1_arch    25   0.8895  0.0796   0.4360  0.4974
  IP   2d1_arch    50   0.9146  0.0434   0.5166  0.4048
  IP   2d1_arch    75   0.9127  0.0455   0.5408  0.3603
  IP   2d1_arch   100   0.9045  0.0676   0.5318  0.4196

--- B4: Saturation Analysis ---
  EA 2d0_arch R2_mean:
    Asymptote a = 0.9498 ± 0.0032
    Rate b = 21.13 ± 5.22
    Current (100%) = 0.9436
    Headroom = 0.0063
    Saturated = 99.3%
  EA 2d1_arch R2_mean:
    Asymptote a = 0.8510 ± 0.0167
    Rate b = 11.42 ± 2.67
    Current (100%) = 0.8740
    Headroom = -0.0230
    Saturated = 102.7%
  IP 2d0_arch R2_mean:
    Asymptote a = 0.9377 ± 0.0030
    Rate b = 16.98 ± 1.77
    Current (100%) = 0.9331
    Headroom = 0.0046
    Saturated = 99.5%
  IP 2d1_arch R2_mean:
    Asymptote a = 0.9107 ± 0.0033
    Rate b = 15.09 ± 1.23
    Current (100%) = 0.9045
    Headroom = 0.0062
    Saturated = 99.3%
  EA 2d0_arch R2_arch_mean:
    Asymptote a = 0.4568 ± 0.0198
    Rate b = 6.39 ± 1.49
    Current (100%) = 0.4742
    Headroom = -0.0174
    Saturated = 103.8%
  EA 2d1_arch R2_arch_mean:
    Asymptote a = 0.4431 ± 0.0607
    Rate b = 7.49 ± 6.50
    Current (100%) = 0.3498
    Headroom = 0.0933
    Saturated = 78.9%
  IP 2d0_arch R2_arch_mean:
    Asymptote a = 0.4218 ± 0.1033
    Rate b = 3.98 ± 3.47
    Current (100%) = 0.3984
    Headroom = 0.0233
    Saturated = 94.5%
  IP 2d1_arch R2_arch_mean:
    Asymptote a = 0.5376 ± 0.0043
    Rate b = 6.66 ± 0.30
    Current (100%) = 0.5318
    Headroom = 0.0059
    Saturated = 98.9%
  Saved: *_learning_curve.png (4 figures)

  Saturation Summary:
  target      model          metric   asympt  current  headroom   %sat
  ------------------------------------------------------------
    EA   2d0_arch         R2_mean   0.9498   0.9436   +0.0063  99.3%
    EA   2d1_arch         R2_mean   0.8510   0.8740   -0.0230 102.7%
    IP   2d0_arch         R2_mean   0.9377   0.9331   +0.0046  99.5%
    IP   2d1_arch         R2_mean   0.9107   0.9045   +0.0062  99.3%
    EA   2d0_arch    R2_arch_mean   0.4568   0.4742   -0.0174 103.8%
    EA   2d1_arch    R2_arch_mean   0.4431   0.3498   +0.0933  78.9%
    IP   2d0_arch    R2_arch_mean   0.4218   0.3984   +0.0233  94.5%
    IP   2d1_arch    R2_arch_mean   0.5376   0.5318   +0.0059  98.9%

--- B5: Interpretation ---

======================================================================
INTERPRETATION
======================================================================

  EA 2d0_arch: R²_arch(75→100%) = +0.0525, std at 100% = 0.2984
    → Marginal improvement (Δ < std): APPROACHING SATURATION

  IP 2d0_arch: R²_arch(75→100%) = -0.0853, std at 100% = 0.2692
    → No improvement or decreasing: LIKELY SATURATED

  EA 2d1_arch: R²_arch(75→100%) = -0.1305, std at 100% = 0.5528
    → No improvement or decreasing: LIKELY SATURATED

  IP 2d1_arch: R²_arch(75→100%) = -0.0091, std at 100% = 0.4196
    → No improvement or decreasing: LIKELY SATURATED

======================================================================
DATASET BOTTLENECK ASSESSMENT
======================================================================

QUESTION: Are we hitting a dataset bottleneck (insufficient architecture
information), or would more matched architecture examples continue to
improve performance?

EVIDENCE SUMMARY:

  EA: Architecture variance = 0.98% of total variance
  IP: Architecture variance = 1.46% of total variance

  EA 2d0_arch: R²_arch(25→100%) = +0.1132
  IP 2d0_arch: R²_arch(25→100%) = +0.0811
  EA 2d1_arch: R²_arch(25→100%) = +0.0002
  IP 2d1_arch: R²_arch(25→100%) = +0.0958

ASSESSMENT:

  CONCLUSION: INCONCLUSIVE due to high fold-to-fold variance.
  The large standard deviations across folds (driven by the a_held_out
  split's extreme monomer-level hold-out) prevent definitive determination
  of saturation vs data limitation. The signal-to-noise ratio in the
  learning curves is insufficient for a clear verdict.

QUANTITATIVE SUMMARY:

  EA:
    Architecture effects = 0.98% of total variance
    |Δy| mean = 0.0368 eV
    |Δy| p95  = 0.1302 eV
  IP:
    Architecture effects = 1.46% of total variance
    |Δy| mean = 0.0353 eV
    |Δy| p95  = 0.1330 eV

  2D1-arch EA R²_arch: 0.3496 (25%) → 0.3498 (100%)   Δ = +0.0002

  2D1-arch IP R²_arch: 0.4360 (25%) → 0.5318 (100%)   Δ = +0.0958
```
