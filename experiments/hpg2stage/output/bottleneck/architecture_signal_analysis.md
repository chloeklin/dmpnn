# Architecture Signal Magnitude Analysis

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
======================================================================

======================================================================
======================================================================

======================================================================
======================================================================
```
