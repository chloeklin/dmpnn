# EA/IP Copolymer Model Comparison — Analysis Report

**Generated:** March 19, 2026  
**Source:** `plots/ea_ip_random_vs_monomer/ea_ip_random_vs_monomer_consolidated.csv`  
**Script:** `scripts/python/analyze_ea_ip_report.py`

---

## Folder Structure
```
analysis/ea_ip_report/
├── tables/
│   ├── 01_schema_summary.txt          schema, coverage, missing entries
│   ├── 02_best_by_family.csv          best model per family × split × target
│   ├── 03_pt_effect.csv               RMSE delta from adding polymer type
│   ├── 04_paper_comparison.csv        our results vs published D-MPNN / wD-MPNN
│   └── 05_generalization_gap.csv      random vs monomer RMSE gap per model
├── plots/
│   ├── 01_performance_by_family.png   all base models, all conditions
│   ├── 02_pt_gain.png                 RMSE reduction from adding PT
│   ├── 03_generalization_gap.png      paired random vs monomer bars
│   ├── 04_hpg_comparison.png          HPG variants + GNN reference lines
│   ├── 05_tabular_comparison.png      tabular model comparison
│   └── 06_paper_comparison.png        parity plot vs published values
├── reports/
│   ├── executive_summary.md           key findings, 1-page summary
│   ├── supervisor_summary.md          formal report for supervisor
│   └── further_experiments.md         prioritised list of pending work
└── README.md                          this file
```

## Dataset Coverage
- **Total model–split–target combinations:** 88
- **With results:** 70
- **Pending (no results):** 18

## Best Model per Family (RMSE, eV)

| family                 | split   | target   | best_model     |   RMSE |      ± |    MAE |     R² |
|:-----------------------|:--------|:---------|:---------------|-------:|-------:|-------:|-------:|
| Graph (standard GNN)   | monomer | EA       | DMPNN +PT      | 0.0888 | 0.0192 | 0.053  | 0.9769 |
| Graph (topology-aware) | monomer | EA       | wDMPNN         | 0.0895 | 0.0432 | 0.0528 | 0.9736 |
| Tabular                | monomer | EA       | RF             | 0.1811 | 0.0162 | 0.1159 | 0.9082 |
| Identity               | monomer | EA       | Identity (mix) | 0.3514 | 0.1131 | 0.2169 | 0.6195 |
| Graph (standard GNN)   | monomer | IP       | DMPNN +PT      | 0.0704 | 0.0183 | 0.0399 | 0.9774 |
| Graph (topology-aware) | monomer | IP       | wDMPNN         | 0.0859 | 0.047  | 0.0498 | 0.963  |
| Tabular                | monomer | IP       | XGB            | 0.1488 | 0.0153 | 0.099  | 0.903  |
| Identity               | monomer | IP       | Identity +PT   | 0.2481 | 0.0973 | 0.1506 | 0.624  |
| Graph (standard GNN)   | random  | EA       | GAT +PT        | 0.0227 | 0.001  | 0.0159 | 0.9985 |
| Identity               | random  | EA       | Identity +PT   | 0.0438 | 0.0015 | 0.0301 | 0.9914 |
| Tabular                | random  | EA       | XGB            | 0.1171 | 0.0089 | 0.0798 | 0.9625 |
| Graph (topology-aware) | random  | EA       | HPG +desc      | 0.5729 | 0.2878 | 0.3575 | 0.6507 |
| Graph (standard GNN)   | random  | IP       | GAT +PT        | 0.0171 | 0.0005 | 0.0116 | 0.9987 |
| Identity               | random  | IP       | Identity +PT   | 0.0337 | 0.0016 | 0.0243 | 0.9918 |
| Tabular                | random  | IP       | XGB            | 0.1096 | 0.0044 | 0.0744 | 0.9487 |
| Graph (topology-aware) | random  | IP       | HPG +desc      | 0.4012 | 0.187  | 0.2464 | 0.8132 |

## Paper Reference Values
| Model | Split | EA RMSE | IP RMSE |
|-------|-------|---------|---------|
| D-MPNN (paper)  | random  | ~0.17 | ~0.16 |
| D-MPNN (paper)  | monomer | ~0.20 | ~0.20 |
| wD-MPNN (paper) | random  | ~0.03 | ~0.03 |
| wD-MPNN (paper) | monomer | ~0.10 | ~0.09 |

## Notes
- All metrics: mean ± std over 5-fold CV; RMSE in eV; lower is better.
- +PT = polymer type (alternating/block/random) added as one-hot encoding.
- Missing entries indicate experiments not yet run.
- See `reports/further_experiments.md` for prioritised list of pending work.