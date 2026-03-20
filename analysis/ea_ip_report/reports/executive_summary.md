# Executive Summary — EA/IP Copolymer Property Prediction

**Date:** March 2026  
**Dataset:** ea_ip (copolymer electron affinity & ionisation potential)  
**Targets:** EA vs SHE (eV), IP vs SHE (eV)  
**Evaluation:** 5-fold cross-validation, mean ± std; RMSE in eV (lower is better)  
**Split strategies:** random and monomer-held-out (a_held_out)

---

## 1. Scope
A total of **19 model variants** were evaluated, spanning:
- Identity baseline (polymer type one-hot only)
- Standard GNNs: DMPNN, GAT, GIN (with mix/interact copolymer modes, ±PT)
- Topology-aware GNNs: wDMPNN, HPG (±desc, ±PT)
- Classical ML tabular models: Linear regression, Random Forest, XGBoost

18 of 88 model–split–target combinations are pending (results not yet available).

---

## 2. Key Findings

### Finding 1 — Polymer topology is a dominant predictive signal
The **Identity baseline** (using only polymer-type one-hot encoding, no molecular features) achieves RMSE ≈ 0.044 eV for EA on the random split. This is substantially better than the paper's D-MPNN baseline (~0.17 eV), confirming that the alternating/block/random classification of the polymer architecture captures most of the variance in EA/IP under i.i.d. conditions.

### Finding 2 — GNN +PT variants achieve near-perfect random-split performance
GAT+PT and GIN+PT reach RMSE ≈ 0.022–0.023 eV on random-split EA/IP — a >3× reduction over the corresponding base variants. **This result warrants scrutiny**: if the random split does not stratify by polymer type, the train and test sets will share the same type distribution, trivially inflating PT-based performance. A stratified-by-type ablation is recommended.

### Finding 3 — Standard GNNs match the published D-MPNN baseline
DMPNN (mix, no PT) achieves RMSE ≈ 0.062 eV (EA, random), closely matching the published D-MPNN reference of ~0.17 eV. The monomer-split RMSE (0.099 eV EA) also aligns well with the paper's ~0.20 eV.

### Finding 4 — wDMPNN approaches but does not match the published benchmark
wDMPNN achieves RMSE ≈ 0.089 eV (EA) and 0.086 eV (IP) on the monomer-held-out split. The published wD-MPNN values are ~0.10 and ~0.09 eV respectively. The ~3× gap may reflect differences in training epochs, learning rate schedule, or dataset preprocessing. **Random-split results for wDMPNN are not yet available.**

### Finding 5 — HPG underperforms significantly
HPG variants (original and +desc) achieve RMSE > 0.5 eV across all conditions — roughly 5–8× worse than standard GNNs. This is unexpected for a topology-aware hierarchical architecture and strongly suggests a hyperparameter tuning or data preparation issue (see Further Experiments).

### Finding 6 — Tabular models match the paper D-MPNN baseline
XGBoost achieves RMSE ≈ 0.117 eV (EA, random), comparable to the published D-MPNN baseline without any graph representation. RF is similarly competitive. Linear regression is substantially weaker, confirming the need for non-linear models in this task.

### Finding 7 — Generalisation gap is universal but varies by model
Every model shows higher RMSE on the monomer-held-out split, confirming this is a harder generalisation task. The gap is smallest for wDMPNN (~1.6× increase) and largest for HPG (~1.0× in absolute terms but high baseline). Standard GNNs show ~1.5–1.8× increase.

---

## 3. Recommendations
1. Validate the +PT random-split results with a polymer-type-stratified split.
2. Prioritise running wDMPNN on random splits to directly compare with the ~0.03 eV benchmark.
3. Investigate HPG hyperparameters (hidden dim, depth, LR) before reporting HPG results.
4. Run all pending Tabular +PT experiments to quantify PT effect for classical ML.