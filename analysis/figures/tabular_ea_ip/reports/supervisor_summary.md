# Supervisor-Facing Findings Summary

**Prepared:** March 2026  
**Project context:** Benchmarking graph neural networks and classical ML for copolymer electrochemical property prediction (EA, IP vs SHE).  
**Evaluation protocol:** 5-fold CV; random and monomer-held-out splits; RMSE (eV).

---

## Model Families
| Family | Models | Notes |
|--------|--------|-------|
| Identity baseline | IdentityBaseline (mix, interact) | Polymer type one-hot only |
| Standard GNNs | DMPNN, GAT, GIN | mix/interact modes; ±polymer type |
| Topology-aware GNNs | wDMPNN, HPG | Designed for polymer structure |
| Tabular ML | Linear, RF, XGBoost | Molecular descriptors (AB, RDKit) |

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

## Polymer Type (PT) Effect — Selected Pairs

| base_model          | pt_model      | split   | target   |   RMSE_base |   RMSE_+PT |   delta_RMSE |   improvement_% |
|:--------------------|:--------------|:--------|:---------|------------:|-----------:|-------------:|----------------:|
| GAT (interact)      | GAT +PT       | random  | IP       |      0.0821 |     0.0171 |       0.0649 |            79.1 |
| GAT (mix)           | GAT +PT       | random  | IP       |      0.06   |     0.0171 |       0.0428 |            71.4 |
| GIN (interact)      | GIN +PT       | random  | IP       |      0.0604 |     0.018  |       0.0424 |            70.2 |
| GIN (mix)           | GIN +PT       | random  | IP       |      0.0605 |     0.018  |       0.0424 |            70.2 |
| HPG                 | HPG +desc +PT | monomer | EA       |      0.6538 |     0.6123 |       0.0414 |             6.3 |
| GAT (interact)      | GAT +PT       | random  | EA       |      0.0634 |     0.0227 |       0.0407 |            64.1 |
| GIN (mix)           | GIN +PT       | random  | EA       |      0.0638 |     0.0231 |       0.0407 |            63.7 |
| GAT (mix)           | GAT +PT       | random  | EA       |      0.0627 |     0.0227 |       0.0399 |            63.7 |
| GIN (interact)      | GIN +PT       | random  | EA       |      0.0626 |     0.0231 |       0.0395 |            63.1 |
| Identity (interact) | Identity +PT  | random  | EA       |      0.0826 |     0.0438 |       0.0388 |            47   |
| Identity (interact) | Identity +PT  | random  | IP       |      0.0715 |     0.0337 |       0.0378 |            52.9 |
| Identity (interact) | Identity +PT  | monomer | IP       |      0.284  |     0.2481 |       0.0359 |            12.6 |

## Comparison with Published Values

| comparison   | split   | target   | our_variant   |   our_RMSE |   paper_RMSE |     diff |   rel_% |
|:-------------|:--------|:---------|:--------------|-----------:|-------------:|---------:|--------:|
| D-MPNN       | random  | EA       | DMPNN (mix)   |     0.0622 |         0.17 |  -0.1078 |   -63.4 |
| D-MPNN       | random  | IP       | DMPNN (mix)   |     0.0599 |         0.16 |  -0.1001 |   -62.6 |
| D-MPNN       | monomer | EA       | DMPNN (mix)   |     0.0991 |         0.2  |  -0.1009 |   -50.4 |
| D-MPNN       | monomer | IP       | DMPNN (mix)   |     0.0904 |         0.2  |  -0.1096 |   -54.8 |
| wD-MPNN      | random  | EA       | —             |   nan      |         0.03 | nan      |   nan   |
| wD-MPNN      | random  | IP       | —             |   nan      |         0.03 | nan      |   nan   |
| wD-MPNN      | monomer | EA       | wDMPNN        |     0.0895 |         0.1  |  -0.0105 |   -10.5 |
| wD-MPNN      | monomer | IP       | wDMPNN        |     0.0859 |         0.09 |  -0.0041 |    -4.6 |

## Generalisation Gap (random → monomer, base variants)

| model               | target   | family                 |   RMSE_random |   RMSE_monomer |    gap |   gap_% |
|:--------------------|:---------|:-----------------------|--------------:|---------------:|-------:|--------:|
| Identity (mix)      | EA       | Identity               |        0.073  |         0.3514 | 0.2784 |   381.5 |
| Identity (interact) | EA       | Identity               |        0.0826 |         0.3963 | 0.3137 |   379.9 |
| GAT (interact)      | EA       | Graph (standard GNN)   |        0.0634 |         0.1231 | 0.0597 |    94.1 |
| GIN (interact)      | EA       | Graph (standard GNN)   |        0.0626 |         0.1125 | 0.0499 |    79.7 |
| DMPNN (interact)    | EA       | Graph (standard GNN)   |        0.0627 |         0.1069 | 0.0442 |    70.5 |
| GAT (mix)           | EA       | Graph (standard GNN)   |        0.0627 |         0.1055 | 0.0428 |    68.3 |
| GIN (mix)           | EA       | Graph (standard GNN)   |        0.0638 |         0.1031 | 0.0393 |    61.6 |
| DMPNN (mix)         | EA       | Graph (standard GNN)   |        0.0622 |         0.0991 | 0.037  |    59.5 |
| XGB                 | EA       | Tabular                |        0.1171 |         0.1816 | 0.0645 |    55.1 |
| RF                  | EA       | Tabular                |        0.1249 |         0.1811 | 0.0562 |    45   |
| Linear              | EA       | Tabular                |        0.2107 |         0.3    | 0.0893 |    42.4 |
| HPG                 | EA       | Graph (topology-aware) |        0.6083 |         0.6538 | 0.0454 |     7.5 |
| HPG +desc           | EA       | Graph (topology-aware) |        0.5729 |         0.5936 | 0.0207 |     3.6 |
| Identity (mix)      | IP       | Identity               |        0.0666 |         0.2659 | 0.1993 |   299.2 |
| Identity (interact) | IP       | Identity               |        0.0715 |         0.284  | 0.2125 |   297.3 |
| GAT (mix)           | IP       | Graph (standard GNN)   |        0.06   |         0.0967 | 0.0367 |    61.3 |
| GIN (interact)      | IP       | Graph (standard GNN)   |        0.0604 |         0.096  | 0.0356 |    59   |
| DMPNN (interact)    | IP       | Graph (standard GNN)   |        0.0602 |         0.0944 | 0.0342 |    56.9 |
| GIN (mix)           | IP       | Graph (standard GNN)   |        0.0605 |         0.0948 | 0.0344 |    56.9 |
| DMPNN (mix)         | IP       | Graph (standard GNN)   |        0.0599 |         0.0904 | 0.0305 |    50.9 |
| XGB                 | IP       | Tabular                |        0.1096 |         0.1488 | 0.0392 |    35.8 |
| Linear              | IP       | Tabular                |        0.1767 |         0.2391 | 0.0624 |    35.3 |
| GAT (interact)      | IP       | Graph (standard GNN)   |        0.0821 |         0.1011 | 0.0191 |    23.2 |
| RF                  | IP       | Tabular                |        0.1242 |         0.1509 | 0.0266 |    21.4 |
| HPG +desc           | IP       | Graph (topology-aware) |        0.4012 |         0.4752 | 0.074  |    18.4 |
| HPG                 | IP       | Graph (topology-aware) |        0.4689 |         0.5073 | 0.0384 |     8.2 |

---

## Discussion Points for Supervisor Meeting

**1. PT leakage risk (high priority):**  
Standard GNNs with polymer type (+PT) on random splits show RMSE ~0.022 eV — an order of magnitude better than without PT, and better than the wD-MPNN benchmark. This is likely inflated because random splits may place all three polymer types in both train and test. A polymer-type-stratified split is needed before reporting these numbers.

**2. wDMPNN gap from benchmark:**  
Our wDMPNN monomer-split results (EA: 0.089 eV, IP: 0.086 eV) are approximately 3× worse than the published benchmark. Likely causes: fewer training epochs, different LR schedule, or a dataset/preprocessing discrepancy. Recommend checking against the original paper's training configuration.

**3. HPG poor performance:**  
HPG RMSE > 0.5 eV on all conditions suggests the model is not training effectively on this dataset. Recommend: (a) verify graph construction for copolymers, (b) sweep hidden dim (256–512) and depth (4–8), (c) increase training budget.

**4. Tabular models are surprisingly competitive:**  
XGBoost and RF without graph features match the D-MPNN paper baseline (~0.17 eV). This is useful context: graph-based approaches need to clearly exceed this threshold to justify their added complexity.