# Model Guide: Polymer Graph Neural Networks

This guide covers all graph-based models available in this codebase, including their
input formats, featurization schemes, and architecture details.

All models are trained via:

```bash
python scripts/python/train_graph.py \
  --dataset_name <name> \
  --model_name <MODEL> \
  --task_type reg
```

---

## Table of Contents

1. [DMPNN](#1-dmpnn)
2. [GIN / GIN0 / GINE](#2-gin--gin0--gine)
3. [GAT / GATv2](#3-gat--gatv2)
4. [PPG](#4-ppg-periodic-polymer-graph)
5. [wDMPNN](#5-wdmpnn-weighted-dmpnn)
6. [HPG](#6-hpg-hierarchical-polymer-graph)
7. [Shared Options](#7-shared-options)
8. [Node and Edge Feature Summary](#8-node-and-edge-feature-summary)

---

## 1. DMPNN

**Directed Message Passing Neural Network** — the standard baseline.

### Architecture

- **Message passing**: directed on bonds (`BondMessagePassing`)
- **Pooling**: mean aggregation over atom hidden states
- **Prediction head**: 2-layer FFN
- **Depth**: 3 layers (default)
- **Hidden dim**: 300 (default)

### Input Format

Single SMILES string per sample in the dataset CSV:

```csv
smiles,target
CCO,1.23
c1ccccc1,4.56
```

For **homopolymers with wildcards**, the wildcards are treated as unknown atoms
(atomic number 0) and get "unknown" feature bits:

```csv
smiles,target
[*]CC[*],1.23
```

### Node & Edge Features

| Feature type | Dim | Details |
|---|---|---|
| Atom (node) | **72** | See [Section 8](#8-node-and-edge-feature-summary) |
| Bond (edge) | **14** | See [Section 8](#8-node-and-edge-feature-summary) |

### Usage

```bash
# Basic homopolymer regression
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name DMPNN \
  --task_type reg

# With RDKit descriptors (late concat)
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name DMPNN \
  --task_type reg \
  --incl_rdkit

# With FiLM conditioning instead of late concat
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name DMPNN \
  --task_type reg \
  --incl_rdkit \
  --fusion_mode film

# Copolymer (mix mode)
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --model_name DMPNN \
  --polymer_type copolymer \
  --copolymer_mode mix
```

### Copolymer Modes

DMPNN supports two copolymer integration strategies. Both encode each monomer
independently with a **shared** encoder then combine the embeddings:

| Mode | Embedding formula | FFN input |
|---|---|---|
| `mix` | `z = fA * z_A + fB * z_B` | `d_mp` |
| `interact` | `[z_A \|\| z_B \|\| \|z_A−z_B\| \|\| z_A⊙z_B \|\| fA \|\| fB]` | `4*d_mp + 2` |

---

## 2. GIN / GIN0 / GINE

**Graph Isomorphism Network** variants.

### Architecture

| Variant | ε | Edge features | Notes |
|---|---|---|---|
| `GIN` | Learnable | Yes | Standard GIN + edge features |
| `GIN0` | Fixed = 0 | Yes | Simpler, no ε parameter |
| `GINE` | Learnable | No | Pure node aggregation |

- **Pooling**: mean aggregation
- **MLP layers per step**: 2 (default, `--gin_mlp_layers`)
- **Depth**: 3 (default)
- **Hidden dim**: 300 (default)

### Input Format

Same as DMPNN — single SMILES per sample.

### Node & Edge Features

Same as DMPNN: **72-dim** atom features, **14-dim** bond features.

### Usage

```bash
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name GIN \
  --task_type reg

# Copolymer
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --model_name GIN \
  --polymer_type copolymer \
  --copolymer_mode interact
```

---

## 3. GAT / GATv2

**Graph Attention Network** variants.

### Architecture

| Variant | Attention | Notes |
|---|---|---|
| `GAT` | `a(Wh_i, Wh_j, e_ij)` | Standard GAT with edge features |
| `GATv2` | `a(W[h_i \|\| h_j \|\| e_ij])` | Dynamic attention (GATv2 paper) |

- **Attention heads**: 4 (default, `--gat_num_heads`)
- **Head output**: concatenated (default, `--gat_concat_heads`)
- **Attention dropout**: 0.0 (default, `--gat_attention_dropout`)
- **Pooling**: mean aggregation
- **Depth**: 3 (default)
- **Hidden dim**: 300 (default)

### Input Format

Same as DMPNN — single SMILES per sample.

### Node & Edge Features

Same as DMPNN: **72-dim** atom features, **14-dim** bond features.

### Usage

```bash
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name GAT \
  --task_type reg

python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name GATv2 \
  --task_type reg
```

---

## 4. PPG (Periodic Polymer Graph)

Based on [Gurnani et al.](https://github.com/rishigurnani/ppg). Adds **periodic bonds**
between atoms at wildcard connection points to simulate infinite polymer chains.

### Architecture

- **Message passing**: `BondMessagePassing` (same as DMPNN)
- **Pooling**: mean aggregation
- **Key difference**: featurizer adds periodic bonds between atoms adjacent to `[*]` atoms
- **Explicit H**: added before graph construction (required for periodic bond detection)
- **Prediction head**: 2-layer FFN

### Input Format

Single SMILES with wildcard connection points `[*]`:

```csv
smiles,target
[*]CC[*],1.23
[*]c1ccc([*])cc1,4.56
```

Wildcard atoms `[*]` mark the polymer chain ends. The featurizer:
1. Identifies atoms bonded to `[*]` (the "nearest neighbor" atoms)
2. Adds a new **periodic bond** between those atoms with bond-length-binned features
3. Removes the `[*]` atoms from the final graph

### Node & Edge Features

| Feature type | Dim | Details |
|---|---|---|
| Atom (node) | **72** | Same as DMPNN (MultiHotAtomFeaturizer.v2) |
| Bond (edge) | **24** | 14-dim standard + **10 bond-length bins** for periodic bonds |

The 10 extra bond-length bins are only non-zero for periodic bonds; regular bonds
have zeros in those positions.

### Usage

```bash
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name PPG \
  --task_type reg
```

> **Note**: PPG does not support copolymer mode.

---

## 5. wDMPNN (Weighted DMPNN)

Polymer-aware DMPNN that encodes **fragment weights** (composition ratios) directly
into the message passing via bond-level weight scaling.

### Architecture

- **Message passing**: `WeightedBondMessagePassing` — scales messages by fragment weight
- **Pooling**: `WeightedMeanAggregation` — weight-aware mean over fragment atoms
- **Prediction head**: 2-layer FFN
- **Depth**: 3 (default)
- **Hidden dim**: 300 (default)

### Input Format

A pipe-delimited SMILES string encoding fragments, weights, and connectivity:

```
SMILES|weight1|weight2<edge1<edge2...
```

Example for a 50/50 A-B copolymer:

```
[*:1]CC[*:2]|0.5|0.5<0:1:single<1:2:single
```

The format encodes:
- **SMILES**: combined fragment SMILES with numbered wildcards
- **Weights**: mole fractions per fragment (pipe-separated)
- **Edges**: connectivity between fragments (angle-bracket-separated)

### Node & Edge Features

| Feature type | Dim | Details |
|---|---|---|
| Atom (node) | **72** | Same as DMPNN (MultiHotAtomFeaturizer.v2) |
| Bond (edge) | **14** | Same as DMPNN (MultiHotBondFeaturizer) |
| Fragment weight | scalar | Embedded via message scaling (not a feature vector) |

### Usage

```bash
python scripts/python/train_graph.py \
  --dataset_name htpmd \
  --model_name wDMPNN \
  --task_type reg
```

> **Note**: wDMPNN requires the wDMPNN-format SMILES column in the dataset. It does not
> support the standard copolymer mode (`--polymer_type copolymer`).

---

## 6. HPG (Hierarchical Polymer Graph)

Based on [Park et al., HPG-GAT](https://github.com/park-sungook/HPG). Encodes polymers as
**hierarchical graphs** with two node types (fragment nodes + atom nodes) and three
directed edge types.

### Architecture

- **Message passing**: edge-aware multi-head GAT (`HPGGATLayer`)
- **Pooling**: sum aggregation over all nodes
- **Prediction head**: linear FFN
- **Framework**: PyTorch Lightning (`HPGMPNN`)
- **Depth**: 6 GAT layers (default, `--hpg_depth`)
- **Hidden dim**: 128 (default, `--hpg_hidden_dim`)
- **Attention heads**: 8 (default, `--hpg_num_heads`)
- **FFN dim**: 64 (default, `--hpg_ffn_dim`)

### Graph Structure

For a polymer with *F* fragments and *A* total (real) atoms, the graph has:

```
Nodes: [frag_0, ..., frag_{F-1}, atom_0, ..., atom_{A-1}]
       ───────────────────────── ────────────────────────
         F fragment nodes          A atom nodes (wildcards excluded)
```

Three **directed** edge types:

| Edge type | Direction | Feature | Notes |
|---|---|---|---|
| Fragment–Fragment | Directed (one-way) | Degree of polymerization (float) | Self-loop for homopolymers |
| Atom–Atom | Bidirectional | Bond order (1.0/1.5/2.0/3.0) | Within each fragment |
| Atom→Fragment | Directed | 1.0 | Each atom points to its owning fragment |

### Input Format

Fragment SMILES with wildcard connection points (`[*]`, `[R]`, `[Q]`, `[T]`, `[U]`).
Wildcards are **removed** from the atom graph — they only define topology.

Dataset CSV format (same column as other models):

```csv
smiles,target
[*]CC[*],1.23
```

For **copolymers**, pass multiple fragment SMILES and connections via the data pipeline.

### Node & Edge Features

| Feature | Dim | Details |
|---|---|---|
| Fragment node | **49** | All-ones vector: `ones(49)` |
| Atom node | **49** | Original HPG-GAT encoding (see below) |
| Edge | **1** | Scalar: bond order or DP |

**Atom node features (49-dim)**:

| Sub-feature | Dim | Values |
|---|---|---|
| Symbol (one-hot) | 20 | C, N, O, S, H, F, Cl, Br, I, Se, Te, Si, P, B, Ca, Mg, Al, Sb, Ge, As |
| Num H (one-hot) | 5 | 0–4 |
| Degree (one-hot) | 7 | 0–6 |
| Aromatic | 1 | bool |
| Hybridization | 6 | S, SP, SP2, SP3, SP3D, SP3D2 |
| In ring | 1 | bool |
| Formal charge | 9 | −4 to +4 |

> Elements not in the vocabulary of 20 will raise a `ValueError`. Verify your dataset
> does not contain unsupported elements before running HPG.

### Wildcard Handling

Non-standard wildcard notation is normalized before parsing:

| Input | Normalized to |
|---|---|
| `[R]`, `[Q]`, `[T]`, `[U]` | `[*]` |
| `[*:1]`, `[*:2]`, ... | `[*]` (already atomic num 0) |

After RDKit parsing, all atoms with atomic number 0 (`[*]`) are **removed** from the
atom graph. Bonds touching a wildcard atom are also removed.

### Homopolymer Handling

For a single fragment with no explicit connections, a **self-loop** is added on the
fragment node with degree = 1.0 (matching the original HPG code).

Unknown degree `"?"` is treated as 1.0.

### Usage

```bash
# HPG regression
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name HPG \
  --task_type reg

# With scalar descriptors
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name HPG \
  --task_type reg \
  --incl_rdkit

# Tuning architecture
python scripts/python/train_graph.py \
  --dataset_name insulator \
  --model_name HPG \
  --task_type reg \
  --hpg_depth 4 \
  --hpg_hidden_dim 256 \
  --hpg_num_heads 4
```

---

## 7. Shared Options

These flags apply to all models:

### Training

| Flag | Default | Description |
|---|---|---|
| `--task_type` | `reg` | `reg`, `binary`, `multi` |
| `--batch_size` | 64 | Mini-batch size |
| `--train_size` | full | Subsample training set (`500`, `2000`, `full`) |
| `--batch_norm` | off | Enable batch normalization |
| `--save_checkpoint` | off | Save model checkpoints |
| `--export_embeddings` | off | Save GNN embeddings post-training |

### Descriptors

| Flag | Description |
|---|---|
| `--incl_desc` | Include dataset-specific tabular descriptors |
| `--incl_rdkit` | Include RDKit 2D descriptors (200 features) |

### Descriptor Fusion Modes

| Mode | Behaviour |
|---|---|
| `late_concat` (default) | Concatenate descriptors to GNN embedding before FFN |
| `film` | FiLM conditioning inside message passing layers |
| `none` | No descriptors |

```bash
--fusion_mode film \
--film_layers all \      # or 'last'
--film_hidden_dim 128
```

### Data Splitting

| Flag | Description |
|---|---|
| `--split_type random` (default) | 80/10/10 random splits |
| `--split_type a_held_out` | For copolymers: group by monomer A |

---

## 8. Node and Edge Feature Summary

### Standard Models (DMPNN, GIN, GAT, PPG, wDMPNN)

**Atom features — 72-dim** (`MultiHotAtomFeaturizer.v2`):

| Sub-feature | Dim | Notes |
|---|---|---|
| Atomic number (1–36, 53=I) | 38 | 37 elements + 1 unknown padding |
| Degree (0–5) | 7 | + 1 unknown |
| Formal charge (−2,−1,0,1,2) | 6 | + 1 unknown |
| Chiral tag (0–3) | 5 | + 1 unknown |
| Num H (0–4) | 6 | + 1 unknown |
| Hybridization (S,SP,SP2,SP2D,SP3,SP3D,SP3D2) | 8 | + 1 unknown |
| Aromatic | 1 | bool |
| Mass | 1 | scaled by 0.01 |
| **Total** | **72** | |

**Bond features — 14-dim** (`MultiHotBondFeaturizer`):

| Sub-feature | Dim | Notes |
|---|---|---|
| Null bond | 1 | is bond None? |
| Bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC) | 4 | one-hot |
| Conjugated | 1 | bool |
| In ring | 1 | bool |
| Stereochemistry (0–5) | 7 | + 1 unknown |
| **Total** | **14** | |

> **PPG only**: bond features are **24-dim** (14 standard + 10 bond-length bins).
> The extra 10 dims are only non-zero for the added periodic bonds.

---

### HPG Model

**Node features — 49-dim** (all node types share the same dimension):

| Node type | Value | Notes |
|---|---|---|
| Fragment nodes | `ones(49)` | Fixed all-ones vector |
| Atom nodes | 49-dim one-hot | See atom feature breakdown below |

**Atom features breakdown (49-dim)**:

| Sub-feature | Dim | Notes |
|---|---|---|
| Symbol (20 elements) | 20 | Strict one-hot — raises if element not in list |
| Num H (0–4) | 5 | Strict one-hot |
| Degree (0–6) | 7 | Strict one-hot |
| Aromatic | 1 | bool |
| Hybridization (S,SP,SP2,SP3,SP3D,SP3D2) | 6 | Strict one-hot |
| In ring | 1 | bool |
| Formal charge (−4 to +4) | 9 | Strict one-hot |
| **Total** | **49** | |

**Edge features — 1-dim scalar**:

| Edge type | Value |
|---|---|
| Fragment–Fragment | Degree of polymerization (float, unknown → 1.0) |
| Atom–Atom | Bond order (1.0=single, 1.5=aromatic, 2.0=double, 3.0=triple) |
| Atom→Fragment | 1.0 |

---

## 9. Model Comparison

| Model | Node dim | Edge dim | Polymer topology | Copolymer | Notes |
|---|---|---|---|---|---|
| DMPNN | 72 | 14 | Implicit (single mol) | ✅ mix/interact | Standard baseline |
| GIN / GIN0 / GINE | 72 | 14 | Implicit | ✅ mix/interact | Sum aggregation inside |
| GAT / GATv2 | 72 | 14 | Implicit | ✅ mix/interact | Attention weights |
| PPG | 72 | 24 | Periodic bonds | ❌ | Explicit periodicity |
| wDMPNN | 72 | 14 | Weighted bonds | ❌ | Fragment weight scaling |
| HPG | 49 | 1 | Explicit hierarchy | ⚠️ planned | Fragment + atom nodes |
