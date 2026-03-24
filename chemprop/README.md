# Chemprop Package — Developer Reference

This document covers the package layout, every available model with usage
examples, and a step-by-step tutorial for adding a new model.

**See also:**

- [`MODEL_GUIDE.md`](MODEL_GUIDE.md) — command-line flags and featurization details for each model
- [`IMPLEMENTING_CUSTOM_MPNN.md`](IMPLEMENTING_CUSTOM_MPNN.md) — deep-dive on the training workflow and custom MPNN skeleton

---

## Table of Contents

1. [Package Layout](#1-package-layout)
2. [Submodule Reference](#2-submodule-reference)
3. [Model Usage](#3-model-usage)
   - [DMPNN](#31-dmpnn)
   - [GIN / GIN0 / GINE](#32-gin--gin0--gine)
   - [GAT / GATv2](#33-gat--gatv2)
   - [wDMPNN](#34-wdmpnn)
   - [CopolymerMPNN](#35-copolymermpnn)
   - [HPGMPNN](#36-hpgmpnn)
   - [MulticomponentMPNN](#37-multicomponentmpnn)
   - [MolAtomBondMPNN](#38-molatombondmpnn)
4. [Tutorial: Adding a New Model](#4-tutorial-adding-a-new-model)

---

## 1. Package Layout

```text
chemprop/
├── conf.py                  # Global defaults (atom/bond dims, hidden dim)
├── exceptions.py            # Custom exception types
├── schedulers.py            # Noam-style LR scheduler
├── types.py                 # Shared type aliases
│
├── data/                    # Data containers and loading
│   ├── datapoints.py        # MoleculeDatapoint, ReactionDatapoint
│   ├── datasets.py          # MoleculeDataset, ReactionDataset
│   ├── copolymer.py         # CopolymerDatum, CopolymerDataset
│   ├── hpg.py               # HPG-specific data (BatchHPGMolGraph)
│   ├── collate.py           # BatchMolGraph and collate functions
│   ├── molgraph.py          # MolGraph (graph representation of one molecule)
│   ├── splitting.py         # Train/val/test split utilities
│   ├── dataloader.py        # build_dataloader helper
│   └── samplers.py          # Weighted samplers
│
├── featurizers/             # Molecule → graph featurization
│   ├── atom.py              # Atom feature set
│   ├── bond.py              # Bond feature set
│   ├── molecule.py          # RDKit 2D descriptor featurizer
│   └── molgraph/            # MolGraph featurizer classes
│       ├── molecule.py      # SimpleMoleculeMolGraphFeaturizer
│       │                    # PolymerMolGraphFeaturizer
│       └── reaction.py      # ReactionMolGraphFeaturizer
│
├── nn/                      # Neural network building blocks
│   ├── message_passing/     # MP encoder implementations
│   │   ├── base.py          # _MessagePassingBase (DMPNN core)
│   │   ├── gat.py           # GATMessagePassing, GATv2MessagePassing
│   │   ├── gin.py           # GINMessagePassing, GIN0/GINEMessagePassing
│   │   ├── mixins.py        # Atom/Bond/WeightedBond/DiffPool mixins
│   │   └── multi.py         # MulticomponentMessagePassing
│   ├── agg.py               # Aggregation classes (Mean, Sum, Norm, Attentive…)
│   ├── predictors.py        # Predictor FFN heads (Regression, Classification…)
│   ├── metrics.py           # Loss functions and evaluation metrics
│   ├── ffn.py               # MLP / ConstrainerFFN
│   ├── film.py              # FiLM conditioning layer
│   ├── hpg.py               # HPGGATLayer, HPGMessagePassing
│   └── transforms.py        # ScaleTransform, UnscaleTransform, GraphTransform
│
├── models/                  # Full Lightning models
│   ├── model.py             # MPNN  (base model)
│   ├── multi.py             # MulticomponentMPNN
│   ├── copolymer.py         # CopolymerMPNN
│   ├── hpg.py               # HPGMPNN
│   ├── mol_atom_bond.py     # MolAtomBondMPNN
│   └── utils.py             # load_model / save_model helpers
│
├── cli/                     # Command-line interface (chemprop train/predict)
├── uncertainty/             # Uncertainty quantification heads
└── utils/                   # Registry, Factory, and misc helpers
```

---

## 2. Submodule Reference

### `chemprop.data`

Handles the full data pipeline from raw SMILES to batched tensors.

| Class / function | Purpose |
| --- | --- |
| `MoleculeDatapoint` | Single molecule with target `y`, extra atom descriptors `V_d`, and extra molecule descriptors `x_d` |
| `MoleculeDataset` | Dataset wrapping a list of `MoleculeDatapoint`s and a featurizer |
| `CopolymerDatum` | One copolymer sample: two mol-graphs + monomer fractions + optional meta |
| `CopolymerDataset` | Dataset for copolymer samples used with `CopolymerMPNN` |
| `BatchMolGraph` | Collated batch of molecular graphs (adjacency, atom/bond features, batch index) |
| `build_dataloader` | Convenience wrapper around `DataLoader` with sensible defaults |
| `make_split_indices` | Random / scaffold / k-fold split indices |

### `chemprop.featurizers`

Converts RDKit `Mol` objects to `MolGraph` instances.

| Class | Purpose |
| --- | --- |
| `SimpleMoleculeMolGraphFeaturizer` | Standard atom + bond features for small molecules and homopolymers |
| `PolymerMolGraphFeaturizer` | Extended feature set including polymer-specific bits |
| `AtomFeaturizer` | Configurable atom feature list (element, degree, charge, …) |
| `BondFeaturizer` | Configurable bond feature list (bond type, ring, stereo, …) |

Default dimensions (from `conf.py`):

```python
DEFAULT_ATOM_FDIM  = 72   # SimpleMoleculeMolGraphFeaturizer
DEFAULT_BOND_FDIM  = 14
DEFAULT_POLY_ATOM_FDIM  = 75   # PolymerMolGraphFeaturizer
DEFAULT_POLY_BOND_FDIM  = 14
DEFAULT_HIDDEN_DIM = 300
```

### `chemprop.nn`

Reusable neural network modules.

**Message passing** (`chemprop.nn.message_passing`):

| Class | Algorithm |
| --- | --- |
| `BondMessagePassing` | DMPNN — directed messages on bonds |
| `AtomMessagePassing` | Messages on atoms (undirected) |
| `WeightedBondMessagePassing` | DMPNN with degree-of-polymerisation weighting (wDMPNN) |
| `GATMessagePassing` | Graph Attention Network (Veličković et al. 2018) |
| `GATv2MessagePassing` | GATv2 (Brody et al. 2022) |
| `GINMessagePassing` | Graph Isomorphism Network (Xu et al. 2019) |
| `GIN0MessagePassing` | GIN with ε fixed at 0 |
| `GINEMessagePassing` | GIN with edge features |
| `MulticomponentMessagePassing` | Runs a shared MP block over multiple graph inputs |
| `BondMessagePassingWithDiffPool` | BondMP with differentiable graph pooling |
| `HPGMessagePassing` | Edge-aware GAT for hierarchical polymer graphs |

**Aggregation** (`chemprop.nn.agg`):

| Class | Behaviour |
| --- | --- |
| `MeanAggregation` | Mean over atom representations |
| `SumAggregation` | Sum over atom representations |
| `NormAggregation` | Mean divided by √N |
| `AttentiveAggregation` | Learnable attention-weighted sum |
| `WeightedMeanAggregation` | Mean weighted by per-atom scalars |

**Predictor heads** (`chemprop.nn.predictors`):

| Class | Task |
| --- | --- |
| `RegressionFFN` | Regression (MSE loss) |
| `MveFFN` | Mean–variance estimation |
| `EvidentialFFN` | Evidential uncertainty |
| `BinaryClassificationFFN` | Binary classification (BCE) |
| `MulticlassClassificationFFN` | Multiclass (cross-entropy) |
| `SpectralFFN` | Spectral / SID targets |
| `MixedRegMultiFFN` | Multi-target with mixed regression tasks |

### `chemprop.models`

Complete PyTorch Lightning modules ready for `pl.Trainer`.

| Class | Use case |
| --- | --- |
| `MPNN` | Standard single-molecule regression / classification |
| `MulticomponentMPNN` | Multiple input graphs concatenated (reactions, mixtures) |
| `CopolymerMPNN` | Two-monomer copolymers with composition-aware integration |
| `HPGMPNN` | Hierarchical polymer graph with edge-aware GAT |
| `MolAtomBondMPNN` | Simultaneous molecule-, atom-, and bond-level predictions |

---

## 3. Model Usage

All examples assume the standard training entry point:

```bash
python scripts/python/train_graph.py --dataset_name <ds> --model_name <MODEL> --task_type reg
```

Python snippets show the equivalent programmatic construction used inside
`train_graph.py`.

---

### 3.1 DMPNN

**Directed Message Passing Neural Network.**  The default baseline — directed
messages flow along bonds, then atom representations are mean-pooled.

```bash
python scripts/python/train_graph.py \
  --dataset_name htpmd \
  --model_name DMPNN \
  --task_type reg \
  --depth 3 \
  --hidden_size 300 \
  --ffn_num_layers 2
```

```python
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import MPNN

mp  = BondMessagePassing(d_v=72, d_e=14, d_h=300, depth=3)
agg = MeanAggregation()
ffn = RegressionFFN(input_dim=300, hidden_dim=300, n_layers=2, n_tasks=1)

model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
```

Key options:

- `--batch_norm` — applies batch normalisation on graph embeddings
- `--extra_desc` — concatenate RDKit descriptors to the graph embedding
  (`fusion_mode=late_concat`)
- `--film` — condition MP layers on descriptors via FiLM (`fusion_mode=film`)

---

### 3.2 GIN / GIN0 / GINE

**Graph Isomorphism Network.**  Node-centric update:
`h_v' = MLP((1+ε)·h_v + Σ_{u∈N(v)} h_u)`

```bash
python scripts/python/train_graph.py \
  --dataset_name htpmd \
  --model_name GIN \
  --task_type reg \
  --depth 3 \
  --hidden_size 300
```

```python
from chemprop.nn import GINMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import MPNN

mp  = GINMessagePassing(d_v=72, d_e=14, d_h=300, depth=3, eps_learnable=True)
agg = MeanAggregation()
ffn = RegressionFFN(input_dim=300, hidden_dim=300, n_layers=2, n_tasks=1)

model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
```

Variants:
- `GIN0MessagePassing` — ε fixed at 0
- `GINEMessagePassing` — incorporates bond features in the aggregation

---

### 3.3 GAT / GATv2

**Graph Attention Network.**  Attention-weighted neighbourhood aggregation.

```bash
python scripts/python/train_graph.py \
  --dataset_name htpmd \
  --model_name GAT \
  --task_type reg \
  --depth 3 \
  --hidden_size 300 \
  --num_heads 4
```

```python
from chemprop.nn import GATMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import MPNN

mp  = GATMessagePassing(d_v=72, d_e=14, d_h=300, depth=3,
                        num_heads=4, concat_heads=True)
agg = MeanAggregation()
ffn = RegressionFFN(input_dim=300 * 4, hidden_dim=300, n_layers=2, n_tasks=1)

model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
```

> **Note:** when `concat_heads=True` the MP output dim is `d_h × num_heads`; pass
> this as `input_dim` to the FFN.

Use `GATv2MessagePassing` for the dynamic attention variant (Brody et al. 2022).

---

### 3.4 wDMPNN

**Weighted DMPNN.**  Identical to DMPNN but messages are scaled by the
degree-of-polymerisation so the model is sensitive to chain length.
Designed for homopolymers with BigSMILES / PSMILES repeat-unit notation.

```bash
python scripts/python/train_graph.py \
  --dataset_name htpmd \
  --model_name wDMPNN \
  --task_type reg
```

```python
from chemprop.nn import WeightedBondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import MPNN
from chemprop.featurizers.molgraph.molecule import PolymerMolGraphFeaturizer

mp  = WeightedBondMessagePassing(
          d_v=DEFAULT_POLY_ATOM_FDIM,
          d_e=DEFAULT_POLY_BOND_FDIM,
          d_h=300, depth=3)
agg = MeanAggregation()
ffn = RegressionFFN(input_dim=300, hidden_dim=300, n_layers=2, n_tasks=1)

model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
```

> Uses `PolymerMolGraphFeaturizer` (larger atom/bond dims) and
> `BatchPolymerMolGraph` which carries `degree_of_polym` as a tensor.

---

### 3.5 CopolymerMPNN

**Two-monomer copolymer model** with a shared encoder and composition-aware
embedding integration.

```bash
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --model_name DMPNN \
  --task_type reg \
  --copolymer_mode mix_meta \
  --incl_poly_type
```

Available `--copolymer_mode` values:

| Mode | Embedding formula | Extra head input |
| --- | --- | --- |
| `mean` | `(z_A + z_B)/2` | — |
| `mean_meta` | `(z_A + z_B)/2` | `meta` scalars |
| `mix` | `fA·z_A + fB·z_B` | — |
| `mix_meta` | `fA·z_A + fB·z_B` | `meta` scalars |
| `mix_frac` | `fA·z_A + fB·z_B` | `fA, fB` |
| `mix_frac_meta` | `fA·z_A + fB·z_B` | `fA, fB, meta` |
| `interact` | concat of z_A, z_B, abs(z_A-z_B), z_A⊙z_B, fA, fB | — |
| `interact_meta` | same as interact + meta scalars | — |

```python
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import CopolymerMPNN

mp  = BondMessagePassing(d_v=72, d_e=14, d_h=300, depth=3)
agg = MeanAggregation()
ffn = RegressionFFN(input_dim=300, hidden_dim=300, n_layers=2, n_tasks=1)
# For interact modes, input_dim = 4 * d_h + 2 (+ meta_dim for _meta modes)

model = CopolymerMPNN(
    message_passing=mp,
    agg=agg,
    predictor=ffn,
    copolymer_mode="mix_meta",
)
```

Data must use `CopolymerDataset` which yields `CopolymerDatum` objects
(monomer A graph, monomer B graph, `fracA`, `fracB`, optional meta).

---

### 3.6 HPGMPNN

**Hierarchical Polymer Graph.**  Uses an edge-aware GAT on a hierarchical
graph where nodes represent monomer units and edges encode connectivity.

```bash
python scripts/python/train_graph.py \
  --dataset_name ea_ip \
  --model_name HPG \
  --task_type reg \
  --depth 3 \
  --hidden_size 256 \
  --num_heads 4
```

```python
from chemprop.models import HPGMPNN

model = HPGMPNN(
    d_v=75,          # node feature dim (PolymerMolGraphFeaturizer)
    d_h=256,         # hidden dim
    d_ffn=256,       # FFN dim after pooling
    depth=3,         # number of GAT layers
    num_heads=4,
    dropout_mp=0.0,
    dropout_ffn=0.1,
    n_tasks=1,
    d_xd=0,          # set > 0 to concatenate extra molecule descriptors
    task_type="regression",
)
```

`HPGMPNN` does not use the standard `MPNN(mp, agg, ffn)` constructor —
it owns its own internal `HPGMessagePassing` block.
Data must use `HPGDataset` which builds `BatchHPGMolGraph` batches.

---

### 3.7 MulticomponentMPNN

Runs a **shared message-passing block** over several input graphs and
concatenates the resulting embeddings before the prediction head.  Used for
reactions (reactant + product graphs) or solvent–solute pairs.

```python
from chemprop.nn import (BondMessagePassing, MeanAggregation,
                          MulticomponentMessagePassing, RegressionFFN)
from chemprop.models import MulticomponentMPNN

mp_block = BondMessagePassing(d_v=72, d_e=14, d_h=300, depth=3)
mp       = MulticomponentMessagePassing(blocks=[mp_block, mp_block],
                                         n_components=2)
agg      = MeanAggregation()
ffn      = RegressionFFN(input_dim=300 * 2, hidden_dim=300,
                          n_layers=2, n_tasks=1)

model = MulticomponentMPNN(message_passing=mp, agg=agg, predictor=ffn)
```

---

### 3.8 MolAtomBondMPNN

Simultaneously predicts at **molecule, atom, and bond** levels from a single
forward pass.  Uses `MABMessagePassing` which retains per-node and per-edge
representations after message passing.

```python
from chemprop.nn import (MABBondMessagePassing, MeanAggregation, RegressionFFN)
from chemprop.models import MolAtomBondMPNN

mp  = MABBondMessagePassing(d_v=72, d_e=14, d_h=300, depth=3)
agg = MeanAggregation()

mol_pred  = RegressionFFN(input_dim=300, n_tasks=1)   # molecule-level
atom_pred = RegressionFFN(input_dim=300, n_tasks=1)   # per-atom
bond_pred = RegressionFFN(input_dim=300, n_tasks=1)   # per-bond

model = MolAtomBondMPNN(
    message_passing=mp,
    agg=agg,
    mol_predictor=mol_pred,
    atom_predictor=atom_pred,
    bond_predictor=bond_pred,
)
```

---

## 4. Tutorial: Adding a New Model

This section walks through adding a new GNN from scratch — a **GraphSAGE**
encoder — as a complete worked example.

### Step 1 — Implement the message passing block

All message passing modules must subclass `chemprop.nn.message_passing.proto.MessagePassing`
and `HyperparametersMixin`.  The only required method is `forward`.

```python
# chemprop/nn/message_passing/sage.py

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
import torch.nn as nn
from torch import Tensor

from chemprop.data import BatchMolGraph
from chemprop.nn.message_passing.proto import MessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM


class SAGEMessagePassing(MessagePassing, HyperparametersMixin):
    """GraphSAGE-style message passing.

    h_v' = ReLU( W · concat(h_v, mean_{u∈N(v)} h_u) )
    """

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,    # unused here, kept for API compat
        d_h: int = DEFAULT_HIDDEN_DIM,
        depth: int = 3,
        dropout: float = 0.0,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["V_d_transform", "graph_transform"])
        self.hparams["V_d_transform"] = V_d_transform
        self.hparams["graph_transform"] = graph_transform
        self.hparams["cls"] = self.__class__

        self.d_h = d_h
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        # Initial projection from atom features
        self.input_proj = nn.Linear(d_v, d_h)
        # GraphSAGE aggregation: concat self + mean-neighbour → d_h
        self.layers = nn.ModuleList([
            nn.Linear(2 * d_h, d_h) for _ in range(depth)
        ])
        self.act = nn.ReLU()

        # Required by MPNN for descriptor concatenation
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self.d_h

    def forward(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> Tensor:
        self.graph_transform(bmg)

        # bmg.V  : (total_atoms, d_v)
        # bmg.E  : (total_bonds, d_e)  — unused in SAGE
        # bmg.a2b: edge index mapping atom → incoming bond indices
        # bmg.b2a: bond → source atom index
        # bmg.batch: (total_atoms,) — which graph each atom belongs to

        H = self.act(self.input_proj(bmg.V))    # (N_atoms, d_h)

        for layer in self.layers:
            # Scatter-mean: for each atom, average its neighbours' hidden states
            src = bmg.b2a[bmg.a2b]              # (N_atoms, max_bonds) source atoms
            # Simple mean via index_add
            neigh = H[src].mean(dim=1)           # (N_atoms, d_h)  ← approximation
            H_new = self.act(layer(torch.cat([H, neigh], dim=-1)))
            H = self.dropout(H_new)

        return H   # (N_atoms, d_h) — MPNN.fingerprint aggregates this
```

> **Tip:** the `hparams["cls"]` line is required for checkpoint round-trips —
> `MPNN._load` uses it to reconstruct the module.

### Step 2 — Register the block (optional but recommended)

If you want the block to be available via `AggregationRegistry` style lookup,
add it to `chemprop/nn/message_passing/__init__.py`:

```python
# chemprop/nn/message_passing/__init__.py  — add to existing imports
from .sage import SAGEMessagePassing
```

And expose it from `chemprop/nn/__init__.py`:

```python
from .message_passing import (
    ...
    SAGEMessagePassing,   # ← add
)
```

### Step 3 — Wire up the full model in the training script

No new model class is needed if your encoder produces per-atom embeddings
that the existing `MPNN` wrapper can aggregate.

```python
from chemprop.nn import MeanAggregation, RegressionFFN
from chemprop.nn.message_passing.sage import SAGEMessagePassing
from chemprop.models import MPNN

mp  = SAGEMessagePassing(d_v=72, d_e=14, d_h=300, depth=3, dropout=0.1)
agg = MeanAggregation()
ffn = RegressionFFN(input_dim=300, hidden_dim=300, n_layers=2, n_tasks=1)

model = MPNN(message_passing=mp, agg=agg, predictor=ffn)
```

### Step 4 — (If needed) Write a new top-level model class

Only do this when the standard `MPNN(mp, agg, ffn)` contract is not enough
(e.g. two input graphs, custom loss logic, new data batch type).

```python
# chemprop/models/my_model.py

import lightning.pytorch as pl
import torch
from torch import Tensor, nn

from chemprop.data import BatchMolGraph, TrainingBatch
from chemprop.nn import Aggregation, ChempropMetric, Predictor
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.nn.message_passing.sage import SAGEMessagePassing


class MySAGEModel(pl.LightningModule):
    """Custom model wrapping SAGEMessagePassing with a bespoke loss."""

    def __init__(
        self,
        message_passing: SAGEMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        warmup_epochs: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["message_passing", "agg", "predictor"])
        # Store hparams for checkpoint compatibility
        self.hparams.update({
            "message_passing": message_passing.hparams,
            "agg": agg.hparams,
            "predictor": predictor.hparams,
        })
        self.message_passing = message_passing
        self.agg = agg
        self.predictor = predictor
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs

    def forward(self, bmg: BatchMolGraph, V_d=None, X_d=None) -> Tensor:
        H_v = self.message_passing(bmg, V_d)
        H   = self.agg(H_v, bmg.batch)
        return self.predictor(H)

    def training_step(self, batch: TrainingBatch, batch_idx: int):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds  = self(bmg, V_d, X_d)
        mask   = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        loss = self.predictor.criterion(preds, targets, mask, weights)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0):
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds  = self(bmg, V_d, X_d)
        mask   = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        self.predictor.criterion(preds, targets, mask, weights)
        self.log("val_loss", self.predictor.criterion,
                 prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.init_lr)
        steps_per_epoch = self.trainer.num_training_batches
        warmup  = self.warmup_epochs * steps_per_epoch
        cooldown = (self.trainer.max_epochs - self.warmup_epochs) * steps_per_epoch
        sched = build_NoamLike_LRSched(
            opt, warmup, cooldown, self.init_lr, self.max_lr, self.final_lr
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}
```

Then register in `chemprop/models/__init__.py`:

```python
from .my_model import MySAGEModel
__all__ = [..., "MySAGEModel"]
```

### Step 5 — Add to `train_graph.py`

In `scripts/python/train_graph.py`, find the model-selection block (the
`if model_name == "DMPNN"` chain) and add a branch:

```python
elif model_name == "SAGE":
    from chemprop.nn.message_passing.sage import SAGEMessagePassing
    mp = SAGEMessagePassing(
        d_v=atom_fdim, d_e=bond_fdim,
        d_h=args.hidden_size,
        depth=args.depth,
        dropout=args.dropout,
    )
    model = MPNN(message_passing=mp, agg=agg, predictor=predictor)
```

Then run:

```bash
python scripts/python/train_graph.py \
  --dataset_name htpmd \
  --model_name SAGE \
  --task_type reg
```

### Checklist

- [ ] `output_dim` property returns the embedding dimensionality
- [ ] `hparams["cls"] = self.__class__` set in `__init__` for checkpoint support
- [ ] `V_d_transform` and `graph_transform` attributes exist (can be `nn.Identity()`)
- [ ] Added to `chemprop/nn/message_passing/__init__.py`
- [ ] Exposed from `chemprop/nn/__init__.py`
- [ ] Branch added in `train_graph.py`
- [ ] (If custom model class) registered in `chemprop/models/__init__.py`
