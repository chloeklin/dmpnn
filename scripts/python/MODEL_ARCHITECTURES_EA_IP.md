# Model Architecture Summary for EA/IP Prediction

This document describes the architectures and configurations of all models displayed in `plot_ea_ip_random_vs_monomer.py` for electron affinity (EA) and ionization potential (IP) prediction on copolymer systems.

## Dataset: EA/IP Copolymers
- **Targets**: EA vs SHE (eV), IP vs SHE (eV)
- **Polymer Type**: Copolymers (binary mixtures with A/B monomers and fractions)
- **Split Strategies**:
  - **Random**: Standard random train/val/test splits
  - **Monomer (a_held_out)**: Group-based splits where entire monomer A groups are held out in test set

---

## 1. Identity Baseline Models

**Architecture**: Simple tabular baseline using polymer type as the only feature.

### Variants:
1. **Identity (mix)**: Uses `mix` copolymer mode (simple averaging of monomer features)
2. **Identity (interact)**: Uses `interact` copolymer mode (interaction-aware feature combination)
3. **Identity +PT**: Automatically selects best performing mode (mix or interact) with polymer type one-hot encoding

**Features**:
- Polymer type categorical variable (alternating, block, random) encoded as one-hot
- Copolymer mode determines how A/B monomer information is combined

**Purpose**: Establishes baseline performance to assess if polymer topology alone provides predictive signal.

---

## 2. Graph Neural Network Models

All graph models use message-passing neural networks (MPNNs) that operate on molecular graph representations.

### 2.1 DMPNN (Directed Message Passing Neural Network)

**Architecture**: Chemprop's directed MPNN with edge-centric message passing.

**Variants**:
1. **DMPNN (mix)**: Simple averaging of A/B monomer embeddings weighted by fractions
2. **DMPNN (interact)**: Interaction-aware combination with cross-monomer message passing
3. **DMPNN +PT**: Best of mix_meta/interact_meta modes with polymer type metadata

**Copolymer Handling**:
- `mix`: Encodes A and B separately, combines embeddings: `fA * emb(A) + fB * emb(B)`
- `interact`: Allows message passing between A and B fragments
- `_meta` suffix: Includes additional metadata features (polymer type, fractions)

### 2.2 GAT (Graph Attention Network)

**Architecture**: Graph attention mechanism with multi-head attention for edge weighting.

**Variants**: Same as DMPNN (mix, interact, +PT)

**Key Feature**: Learns attention weights to focus on important atoms/bonds during aggregation.

### 2.3 GIN (Graph Isomorphism Network)

**Architecture**: Based on Weisfeiler-Lehman graph isomorphism test, uses sum aggregation.

**Variants**: Same as DMPNN (mix, interact, +PT)

**Key Feature**: Theoretically more expressive than standard MPNNs for distinguishing graph structures.

### 2.4 wDMPNN (Weighted DMPNN)

**Architecture**: Modified DMPNN that reads pre-computed weighted molecular representations.

**Variants**: 
1. **wDMPNN**: Single variant (no copolymer modes)

**Copolymer Handling**:
- Uses pre-computed `WDMPNN_Input` column from dataset
- Weighted combination of A/B fragments is done during preprocessing
- Does not use `mix` or `interact` modes

**Key Feature**: Optimized for copolymer systems with explicit fragment weighting.

### 2.5 HPG (Hierarchical Polymer Graph)

**Architecture**: Two-level hierarchical graph with fragment nodes and atom nodes.

**Variants**:
1. **HPG**: Original architecture without additional descriptors
2. **HPG +desc**: Includes dataset-specific descriptors as node/edge features
3. **HPG +desc +PT**: Adds polymer type one-hot encoding to descriptors

**Copolymer Handling**:
- Fragment-level nodes represent A and B monomers
- Atom-level nodes within each fragment
- Hierarchical message passing: atom → fragment → graph
- Naturally handles copolymer topology without explicit modes

**Key Features**:
- Explicit representation of polymer topology (alternating, block, random)
- Can incorporate domain-specific descriptors
- No separate copolymer modes needed (topology is in graph structure)

---

## 3. Tabular Models

Classical machine learning models using molecular descriptors.

### 3.1 Linear Regression

**Architecture**: Ridge regression with L2 regularization.

**Variants**:
1. **Linear**: Uses AB features (atom/bond counts) and/or RDKit descriptors
2. **Linear +PT**: Adds polymer type one-hot encoding

**Features**:
- **AB features**: Pooled atom and bond type counts from molecular graphs
- **RDKit descriptors**: 200+ molecular descriptors (MW, logP, TPSA, etc.)
- Copolymer handling: Weighted average of A/B descriptor vectors

### 3.2 Random Forest (RF)

**Architecture**: Ensemble of decision trees with bootstrap aggregation.

**Variants**: Same as Linear (RF, RF +PT)

**Hyperparameters**: Typically 100-500 trees, max depth tuned via validation.

### 3.3 XGBoost (XGB)

**Architecture**: Gradient boosted decision trees with regularization.

**Variants**: Same as Linear (XGB, XGB +PT)

**Key Features**: 
- L1/L2 regularization on leaf weights
- Column subsampling for robustness
- Early stopping based on validation performance

---

## Feature Engineering Summary

### Copolymer Modes (DMPNN, GAT, GIN):
- **mix**: `emb = fA * φ(A) + fB * φ(B)` - Simple weighted average
- **interact**: Cross-fragment message passing before pooling
- **mix_meta / interact_meta**: Includes polymer type and fraction metadata

### Descriptor Types:
- **AB features**: Atom/bond type counts pooled from molecular graphs
- **RDKit descriptors**: Physicochemical properties (200+ features)
- **Dataset descriptors**: Domain-specific features from input CSV
- **Polymer type**: One-hot encoding of topology (alternating, block, random)

### Preprocessing:
- **Imputation**: Median imputation for missing descriptor values
- **Scaling**: StandardScaler for RDKit descriptors (fit on train only)
- **Feature selection**: Remove constant and highly correlated features (train-only)
- **Fraction normalization**: Ensure fA + fB = 1.0

---

## Model Selection Strategy

### Poly_type Variants (+PT):
- For models with `+PT` suffix and `mode=None`, the script automatically selects the best performing copolymer mode
- Selection criterion: Lowest RMSE on the target metric
- Applies to: Identity, DMPNN, GAT, GIN (not HPG, wDMPNN, or Tabular which have fixed configurations)

### Missing Data:
- Random split results missing for: wDMPNN, HPG +desc +PT, all Tabular +PT variants
- These models only have monomer (a_held_out) split results available

---

## Performance Metrics

All models are evaluated on:
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **R²** (Coefficient of Determination): Proportion of variance explained

Results are averaged across 5 cross-validation folds with standard deviation reported.

---

## Key Architectural Differences

| Model | Graph-based | Hierarchical | Copolymer Modes | Descriptors | Poly Type |
|-------|-------------|--------------|-----------------|-------------|-----------|
| Identity | ✗ | ✗ | mix/interact | ✗ | ✓ |
| DMPNN | ✓ | ✗ | mix/interact/meta | Optional | Optional |
| GAT | ✓ | ✗ | mix/interact/meta | Optional | Optional |
| GIN | ✓ | ✗ | mix/interact/meta | Optional | Optional |
| wDMPNN | ✓ | ✗ | Pre-computed | ✗ | ✗ |
| HPG | ✓ | ✓ | Implicit in graph | Optional | Optional |
| Linear/RF/XGB | ✗ | ✗ | Weighted avg | AB + RDKit | Optional |

---

## References

- **Chemprop**: Yang et al., "Analyzing Learned Molecular Representations for Property Prediction" (2019)
- **GAT**: Veličković et al., "Graph Attention Networks" (2018)
- **GIN**: Xu et al., "How Powerful are Graph Neural Networks?" (2019)
- **HPG**: Hierarchical polymer graph architecture (custom implementation)

---

*Generated for: `plot_ea_ip_random_vs_monomer.py`*  
*Dataset: EA/IP copolymer prediction*  
*Date: March 2026*
