# Implementing Custom MPNN Models in Chemprop

This guide explains the key Chemprop modules and workflow for implementing custom Message Passing Neural Network (MPNN) models. It follows the architecture used in `scripts/python/train_graph.py`.

---

## Table of Contents

1. [Overview: Training Workflow](#overview-training-workflow)
2. [Key Modules](#key-modules)
   - [Data Representation](#1-data-representation)
   - [Graph Featurization](#2-graph-featurization)
   - [Message Passing](#3-message-passing)
   - [Aggregation](#4-aggregation)
3. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
4. [Example: Adding a Custom MPNN](#example-adding-a-custom-mpnn)

---

## Overview: Training Workflow

The typical training workflow in `train_graph.py` follows these steps:

```python
# 1. Load data and create datapoints
smis = ["CCO", "c1ccccc1", ...]  # SMILES strings
ys = np.array([[0.5], [1.2], ...])  # Target values
x_d = np.array([[...], [...], ...])  # Optional: additional descriptors

# 2. Create datapoints
datapoints = [
    MoleculeDatapoint.from_smi(smi, y=y, x_d=xd) 
    for smi, y, xd in zip(smis, ys, x_d)
]

# 3. Featurize molecules into graphs
featurizer = SimpleMoleculeMolGraphFeaturizer()
dataset = MoleculeDataset(datapoints, featurizer)

# 4. Create dataloaders
train_loader = build_dataloader(train_dataset, num_workers=4)

# 5. Build MPNN model
mpnn = BondMessagePassing(d_h=300, depth=3)  # Message passing encoder
model = MPNN(mpnn, agg=MeanAggregation(), ...)  # Full model with aggregation

# 6. Train
trainer = pl.Trainer(...)
trainer.fit(model, train_loader, val_loader)
```

---

## Key Modules

### 1. Data Representation

#### **`chemprop.data.datapoints`**

**Purpose:** Container classes for molecular data with targets and features.

**Key Classes:**

- **`MoleculeDatapoint`**: Single molecule with targets and features
  ```python
  @dataclass
  class MoleculeDatapoint:
      mol: Chem.Mol              # RDKit molecule
      y: np.ndarray | None       # Target values (can have NaNs for missing)
      x_d: np.ndarray | None     # Global descriptors (concatenated after aggregation)
      V_f: np.ndarray | None     # Atom-level features (before message passing)
      E_f: np.ndarray | None     # Bond-level features (before message passing)
      weight: float = 1.0        # Sample weight for loss calculation
      name: str | None = None    # Identifier (usually SMILES)
  ```

- **`PolymerDatapoint`**: For polymer structures (uses BigSMILES)
  ```python
  @dataclass
  class PolymerDatapoint:
      polymer_mol: Chem.Mol      # Polymer structure
      y: np.ndarray | None
      x_d: np.ndarray | None
      # ... similar to MoleculeDatapoint
  ```

**Usage in train_graph.py:**
```python
# Create datapoints from SMILES, targets, and descriptors
all_data = [
    MoleculeDatapoint.from_smi(
        smi=smiles,
        y=y_val,
        x_d=desc_val if combined_descriptor_data else None
    )
    for smiles, y_val, desc_val in zip(smis, ys, combined_descriptor_data)
]
```

**Key Methods:**
- `from_smi()`: Create datapoint from SMILES string
- `mol` property: Access RDKit molecule (lazy loading for `LazyMoleculeDatapoint`)

---

#### **`chemprop.data.molgraph`**

**Purpose:** Graph representation of molecules for neural network input.

**Key Classes:**

- **`MolGraph`**: Stores graph structure and features
  ```python
  @dataclass
  class MolGraph:
      V: np.ndarray              # Atom features [n_atoms, atom_fdim]
      E: np.ndarray              # Bond features [n_bonds*2, bond_fdim]
      edge_index: np.ndarray     # Edge connectivity [2, n_bonds*2]
  ```

- **`BatchMolGraph`**: Batched version for efficient GPU processing
  ```python
  @dataclass
  class BatchMolGraph:
      V: Tensor                  # Batched atom features
      E: Tensor                  # Batched bond features
      edge_index: Tensor         # Batched edge indices
      batch: Tensor              # Atom-to-graph mapping
      # ... additional batching metadata
  ```

**Key Properties:**
- Directed edges: Each bond creates 2 directed edges (i→j and j→i)
- Batching: Multiple graphs concatenated with `batch` tensor tracking which atoms belong to which molecule

---

#### **`chemprop.data.datasets`**

**Purpose:** PyTorch Dataset wrapper that applies featurization.

**Key Class:**

- **`MoleculeDataset`**: Applies featurizer to datapoints on-the-fly
  ```python
  class MoleculeDataset(Dataset):
      def __init__(self, data: Sequence[MoleculeDatapoint], featurizer: GraphFeaturizer):
          self.data = data
          self.featurizer = featurizer
      
      def __getitem__(self, idx):
          dp = self.data[idx]
          # Featurize molecule into MolGraph
          mg = self.featurizer(dp.mol, dp.V_f, dp.E_f)
          return mg, dp.y, dp.x_d, dp.weight, ...
  ```

**Usage:**
```python
featurizer = SimpleMoleculeMolGraphFeaturizer()
train_dataset = MoleculeDataset(train_datapoints, featurizer)
```

---

#### **`chemprop.data.collate`**

**Purpose:** Collate individual graphs into batches for DataLoader.

**Key Function:**

- **`collate_batch()`**: Combines multiple `MolGraph` objects into `BatchMolGraph`
  ```python
  def collate_batch(batch):
      # Unpack batch items
      mgs, ys, x_ds, weights, ... = zip(*batch)
      
      # Create batched graph
      bmg = BatchMolGraph.from_molgraphs(mgs)
      
      # Stack targets and features
      Y = torch.tensor(ys)
      X_d = torch.tensor(x_ds) if x_ds[0] is not None else None
      
      return bmg, Y, X_d, weights, ...
  ```

**Usage:**
```python
from chemprop.data import build_dataloader

# Automatically uses collate_batch
train_loader = build_dataloader(train_dataset, batch_size=32, num_workers=4)
```

---

#### **`chemprop.data.dataloader`**

**Purpose:** Convenience function to create DataLoader with proper collation.

**Key Function:**

- **`build_dataloader()`**: Creates PyTorch DataLoader with custom collate function
  ```python
  def build_dataloader(
      dataset: MoleculeDataset,
      batch_size: int = 64,
      num_workers: int = 0,
      shuffle: bool = True,
      **kwargs
  ) -> DataLoader:
      return DataLoader(
          dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          num_workers=num_workers,
          collate_fn=collate_batch,  # Custom collation
          **kwargs
      )
  ```

---

### 2. Graph Featurization

#### **`chemprop.featurizers.molgraph.molecule`**

**Purpose:** Convert RDKit molecules into `MolGraph` representations.

**Key Classes:**

- **`SimpleMoleculeMolGraphFeaturizer`**: Default featurizer for small molecules
  ```python
  @dataclass
  class SimpleMoleculeMolGraphFeaturizer(GraphFeaturizer):
      atom_featurizer: AtomFeaturizer = MultiHotAtomFeaturizer()
      bond_featurizer: BondFeaturizer = MultiHotBondFeaturizer()
      extra_atom_fdim: int = 0
      extra_bond_fdim: int = 0
      
      def __call__(self, mol: Chem.Mol, 
                   atom_features_extra: np.ndarray | None = None,
                   bond_features_extra: np.ndarray | None = None) -> MolGraph:
          # 1. Featurize atoms
          V = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()])
          
          # 2. Featurize bonds (creates 2 directed edges per bond)
          E = []
          edge_index = [[], []]
          for bond in mol.GetBonds():
              i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
              e_ij = self.bond_featurizer(bond)
              
              # Add both directions
              E.append(e_ij)
              edge_index[0].append(i)
              edge_index[1].append(j)
              
              E.append(e_ij)  # Same features for reverse edge
              edge_index[0].append(j)
              edge_index[1].append(i)
          
          # 3. Concatenate extra features if provided
          if atom_features_extra is not None:
              V = np.hstack((V, atom_features_extra))
          if bond_features_extra is not None:
              E = np.hstack((E, bond_features_extra))
          
          return MolGraph(V=V, E=np.array(E), edge_index=np.array(edge_index))
  ```

- **`PolymerMolGraphFeaturizer`**: For polymer structures (BigSMILES)
  - Similar to `SimpleMoleculeMolGraphFeaturizer` but handles polymer-specific features
  - Returns `PolymerMolGraph` with additional polymer metadata

**Default Featurizers:**
- **`MultiHotAtomFeaturizer`**: Encodes atom type, degree, formal charge, chirality, hybridization, aromaticity, mass
- **`MultiHotBondFeaturizer`**: Encodes bond type, conjugation, ring membership, stereochemistry

**Usage:**
```python
# For small molecules (DMPNN)
featurizer = SimpleMoleculeMolGraphFeaturizer()

# For polymers (wDMPNN)
featurizer = PolymerMolGraphFeaturizer()

# Featurize a molecule
mol = Chem.MolFromSmiles("CCO")
molgraph = featurizer(mol)
```

**Key Properties:**
- `atom_fdim`: Dimension of atom features (default: 133)
- `bond_fdim`: Dimension of bond features (default: 14)

---

### 3. Message Passing

#### **`chemprop.nn.message_passing.base`**

**Purpose:** Core message passing implementation for MPNN encoders.

**Key Classes:**

- **`_MessagePassingBase`**: Abstract base class for all message passing schemes
  ```python
  class _MessagePassingBase(MessagePassing, HyperparametersMixin):
      def __init__(
          self,
          d_v: int = DEFAULT_ATOM_FDIM,      # Atom feature dim (133)
          d_e: int = DEFAULT_BOND_FDIM,      # Bond feature dim (14)
          d_h: int = DEFAULT_HIDDEN_DIM,     # Hidden dim (300)
          bias: bool = False,
          depth: int = 3,                     # Number of message passing steps
          dropout: float = 0.0,
          activation: str = "relu",
          undirected: bool = False,
          d_vd: int | None = None,           # Extra atom descriptor dim
          V_d_transform: ScaleTransform | None = None,
          graph_transform: GraphTransform | None = None,
      ):
          # Initialize weight matrices
          self.W_i, self.W_h, self.W_o, self.W_d = self.setup(d_v, d_e, d_h, d_vd, bias)
          self.depth = depth
          self.dropout = nn.Dropout(dropout)
          self.tau = get_activation_function(activation)
      
      @abstractmethod
      def setup(self, d_v, d_e, d_h, d_vd, bias):
          """Initialize weight matrices (implemented by subclasses)"""
          pass
      
      @abstractmethod
      def aggregate_messages(self, H, bmg, V_d):
          """Aggregate messages from neighbors (implemented by subclasses)"""
          pass
      
      def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
          """
          Forward pass: performs message passing and returns atom/bond encodings
          
          Returns:
              H: Hidden representations [n_atoms or n_bonds, d_h]
          """
          bmg = self.graph_transform(bmg)
          V_d = self.V_d_transform(V_d) if V_d is not None else None
          
          # Initialize hidden states
          H = self.W_i(bmg)  # [n_atoms/bonds, d_h]
          
          # Message passing iterations
          for _ in range(self.depth - 1):
              M = self.aggregate_messages(H, bmg, V_d)  # Aggregate from neighbors
              H = self.tau(self.W_h(M))                  # Update hidden states
              H = self.dropout(H)
          
          # Final output projection
          M = self.aggregate_messages(H, bmg, V_d)
          H = self.W_o(M)
          
          # Concatenate extra descriptors if provided
          if self.W_d is not None and V_d is not None:
              H = torch.cat([H, V_d], dim=1)
              H = self.W_d(H)
          
          return H
  ```

**Key Subclasses:**

- **`AtomMessagePassing`**: Messages passed on atoms (D-MPNN style)
  ```python
  class AtomMessagePassing(_AtomMessagePassingMixin, _MessagePassingBase):
      """
      Message passing where hidden states are on atoms.
      Messages are aggregated from neighboring atoms via bonds.
      """
      def aggregate_messages(self, H, bmg, V_d):
          # H: [n_atoms, d_h]
          # Aggregate messages from neighbors through edges
          src, dst = bmg.edge_index
          E = bmg.E  # Bond features
          
          # Message from neighbor j to atom i through bond (j→i)
          M = scatter_sum(H[src] * E, dst, dim=0, dim_size=H.size(0))
          return M
  ```

- **`BondMessagePassing`**: Messages passed on bonds (original DMPNN)
  ```python
  class BondMessagePassing(_BondMessagePassingMixin, _MessagePassingBase):
      """
      Message passing where hidden states are on directed bonds.
      Messages are aggregated from neighboring bonds.
      """
      def aggregate_messages(self, H, bmg, V_d):
          # H: [n_bonds*2, d_h]
          # For each bond (i→j), aggregate from bonds (k→i) where k≠j
          # This is the key innovation of D-MPNN
          ...
  ```

- **`WeightedBondMessagePassing`**: For polymers with weighted edges
  ```python
  class WeightedBondMessagePassing(_WeightedBondMessagePassingMixin, _MessagePassingBase):
      """
      Similar to BondMessagePassing but uses edge weights from polymer graphs.
      Used in wDMPNN for polymer property prediction.
      """
  ```

**Usage in train_graph.py:**
```python
# Create message passing encoder
if args.model_name == "DMPNN":
    mpnn = BondMessagePassing(
        d_h=300,           # Hidden dimension
        depth=3,           # Number of message passing steps
        dropout=0.0,
        activation="relu",
        d_vd=len(descriptor_columns) if descriptor_columns else None
    )
elif args.model_name == "wDMPNN":
    mpnn = WeightedBondMessagePassing(...)
```

**Key Methods:**
- `forward(bmg, V_d)`: Performs message passing, returns hidden states
- `encode(bmg, V_d)`: Alias for forward (used for embedding extraction)
- `output_dim`: Property returning the output dimension

---

### 4. Aggregation

#### **`chemprop.nn.agg`**

**Purpose:** Aggregate atom/bond-level representations into graph-level representations.

**Key Classes:**

- **`Aggregation`**: Base class for aggregation functions
  ```python
  class Aggregation(nn.Module):
      @abstractmethod
      def forward(self, H: Tensor, batch: Tensor, **kwargs) -> Tensor:
          """
          Aggregate node/edge features into graph-level features
          
          Args:
              H: Node/edge features [n_nodes, d_h]
              batch: Batch assignment [n_nodes] indicating which graph each node belongs to
          
          Returns:
              H_g: Graph-level features [batch_size, d_h]
          """
          pass
  ```

**Concrete Implementations:**

- **`MeanAggregation`**: Average pooling
  ```python
  class MeanAggregation(Aggregation):
      def forward(self, H, batch, **kwargs):
          return scatter_mean(H, batch, dim=0)
  ```

- **`SumAggregation`**: Sum pooling
  ```python
  class SumAggregation(Aggregation):
      def forward(self, H, batch, **kwargs):
          return scatter_sum(H, batch, dim=0)
  ```

- **`NormAggregation`**: Normalized sum (sum / sqrt(n_atoms))
  ```python
  class NormAggregation(Aggregation):
      def forward(self, H, batch, **kwargs):
          counts = scatter_sum(torch.ones_like(batch), batch)
          return scatter_sum(H, batch, dim=0) / counts.sqrt().unsqueeze(-1)
  ```

**Usage:**
```python
# In model definition
from chemprop.nn.agg import MeanAggregation

agg = MeanAggregation()
H_atoms = mpnn(bmg)  # [n_atoms, d_h]
H_graph = agg(H_atoms, bmg.batch)  # [batch_size, d_h]
```

---

## Step-by-Step Implementation Guide

### Step 1: Prepare Your Data

```python
import numpy as np
from chemprop.data import MoleculeDatapoint

# Load your data
smiles_list = ["CCO", "c1ccccc1", "CC(C)O"]
targets = np.array([[0.5], [1.2], [0.8]])
descriptors = np.random.randn(3, 200)  # Optional

# Create datapoints
datapoints = [
    MoleculeDatapoint.from_smi(smi, y=y, x_d=xd)
    for smi, y, xd in zip(smiles_list, targets, descriptors)
]
```

### Step 2: Create Datasets and DataLoaders

```python
from chemprop.data import MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

# Choose featurizer based on model type
featurizer = SimpleMoleculeMolGraphFeaturizer()  # For DMPNN
# featurizer = PolymerMolGraphFeaturizer()       # For wDMPNN

# Create datasets
train_dataset = MoleculeDataset(train_datapoints, featurizer)
val_dataset = MoleculeDataset(val_datapoints, featurizer)

# Create dataloaders
train_loader = build_dataloader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = build_dataloader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

### Step 3: Build Your MPNN Model

```python
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import MeanAggregation
from chemprop.models import MPNN

# Create message passing encoder
mpnn_encoder = BondMessagePassing(
    d_v=133,           # Atom feature dim (from featurizer)
    d_e=14,            # Bond feature dim (from featurizer)
    d_h=300,           # Hidden dimension
    depth=3,           # Number of message passing steps
    dropout=0.0,
    activation="relu",
    d_vd=200 if use_descriptors else None  # Extra descriptor dim
)

# Create full model with aggregation and prediction head
model = MPNN(
    message_passing=mpnn_encoder,
    agg=MeanAggregation(),
    n_tasks=1,              # Number of prediction tasks
    output_transform=None,  # Optional: UnscaleTransform for denormalization
)
```

### Step 4: Train the Model

```python
import lightning.pytorch as pl
from chemprop.nn.loss import MSELoss

# Setup trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    logger=True,
)

# Train
trainer.fit(model, train_loader, val_loader)

# Test
test_results = trainer.test(model, test_loader)
```

### Step 5: Extract Embeddings (Optional)

```python
import torch

model.eval()
embeddings = []

with torch.no_grad():
    for batch in test_loader:
        bmg, Y, X_d, weights, *_ = batch
        
        # Get atom-level encodings
        H_atoms = model.message_passing(bmg, X_d)
        
        # Get graph-level encodings
        H_graph = model.agg(H_atoms, bmg.batch)
        
        embeddings.append(H_graph.cpu().numpy())

embeddings = np.vstack(embeddings)
```

---

## Example: Adding a Custom MPNN

Here's how to implement a custom message passing scheme:

### 1. Create Custom Message Passing Class

```python
# chemprop/nn/message_passing/custom.py

from chemprop.nn.message_passing.base import _MessagePassingBase
from chemprop.nn.message_passing.mixins import _BondMessagePassingMixin
import torch
import torch.nn as nn

class CustomMessagePassing(_BondMessagePassingMixin, _MessagePassingBase):
    """
    Custom message passing with attention mechanism
    """
    
    def setup(self, d_v, d_e, d_h, d_vd, bias):
        """Initialize weight matrices"""
        # Input projection: bond features → hidden
        W_i = nn.Linear(d_e, d_h, bias=bias)
        
        # Message passing: hidden → hidden
        W_h = nn.Linear(d_h, d_h, bias=bias)
        
        # Output projection
        W_o = nn.Linear(d_h, d_h, bias=bias)
        
        # Attention weights (custom addition)
        self.W_attn = nn.Linear(d_h, 1, bias=False)
        
        # Descriptor projection (if using extra features)
        W_d = nn.Linear(d_h + d_vd, d_h, bias=bias) if d_vd else None
        
        return W_i, W_h, W_o, W_d
    
    def aggregate_messages(self, H, bmg, V_d):
        """
        Custom aggregation with attention
        
        Args:
            H: Hidden states [n_bonds*2, d_h]
            bmg: BatchMolGraph
            V_d: Extra atom descriptors [n_atoms, d_vd]
        
        Returns:
            M: Aggregated messages [n_bonds*2, d_h]
        """
        # Standard D-MPNN aggregation
        a2b = bmg.a2b  # Atom to incoming bonds mapping
        b2a = bmg.b2a  # Bond to target atom mapping
        b2revb = bmg.b2revb  # Bond to reverse bond mapping
        
        # For each bond (i→j), aggregate from bonds (k→i) where k≠j
        nei_messages = []
        for bond_idx in range(H.size(0)):
            target_atom = b2a[bond_idx]
            reverse_bond = b2revb[bond_idx]
            
            # Get incoming bonds to target atom (excluding reverse)
            incoming_bonds = a2b[target_atom]
            incoming_bonds = [b for b in incoming_bonds if b != reverse_bond]
            
            if len(incoming_bonds) > 0:
                # Get messages from neighbors
                nei_h = H[incoming_bonds]  # [n_neighbors, d_h]
                
                # Compute attention weights (custom addition)
                attn_scores = self.W_attn(nei_h)  # [n_neighbors, 1]
                attn_weights = torch.softmax(attn_scores, dim=0)
                
                # Weighted aggregation
                message = (nei_h * attn_weights).sum(dim=0)
            else:
                message = torch.zeros(H.size(1), device=H.device)
            
            nei_messages.append(message)
        
        M = torch.stack(nei_messages)  # [n_bonds*2, d_h]
        
        # Add atom features if provided
        if V_d is not None:
            atom_features = V_d[b2a]  # Map bond → atom → features
            M = torch.cat([M, atom_features], dim=1)
        
        return M
```

### 2. Register Your Model

```python
# chemprop/nn/message_passing/__init__.py

from chemprop.nn.message_passing.custom import CustomMessagePassing

__all__ = [
    "BondMessagePassing",
    "AtomMessagePassing",
    "WeightedBondMessagePassing",
    "CustomMessagePassing",  # Add your model
]
```

### 3. Add to Model Configuration

```python
# scripts/python/train_config.yaml

MODELS:
  CustomMPNN:
    smiles_column: smiles
    ignore_columns: []
```

### 4. Use in Training Script

```python
# scripts/python/train_graph.py

parser.add_argument('--model_name', type=str, 
                    choices=["DMPNN", "wDMPNN", "CustomMPNN"],  # Add your model
                    default="DMPNN")

# In model building section
if args.model_name == "CustomMPNN":
    from chemprop.nn.message_passing import CustomMessagePassing
    mpnn = CustomMessagePassing(d_h=300, depth=3, ...)
```

### 5. Train Your Model

```bash
python scripts/python/train_graph.py \
    --dataset_name my_dataset \
    --model_name CustomMPNN \
    --incl_rdkit
```

---

## Advanced Topics

### Adding DiffPool Aggregation

If you want hierarchical graph pooling (like `DMPNN_DiffPool`):

```python
from chemprop.nn.message_passing.mixins import _DiffPoolMixin

class DMPNNWithDiffPool(_DiffPoolMixin, BondMessagePassing):
    """DMPNN with differentiable pooling"""
    
    def __init__(self, *args, diffpool_depth=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffpool_depth = diffpool_depth
        self.setup_diffpool(self.output_dim, diffpool_depth)
    
    def forward(self, bmg, V_d=None):
        # Standard message passing
        H = super().forward(bmg, V_d)
        
        # Apply DiffPool
        H_pooled, _ = self.diffpool(H, bmg.batch)
        
        return H_pooled
```

### Custom Aggregation

```python
from chemprop.nn.agg import Aggregation

class AttentionAggregation(Aggregation):
    """Graph-level aggregation with attention"""
    
    def __init__(self, d_h):
        super().__init__()
        self.attn = nn.Linear(d_h, 1)
    
    def forward(self, H, batch, **kwargs):
        # H: [n_atoms, d_h]
        # batch: [n_atoms]
        
        # Compute attention scores
        scores = self.attn(H)  # [n_atoms, 1]
        
        # Softmax per graph
        scores = scatter_softmax(scores, batch, dim=0)
        
        # Weighted sum
        H_weighted = H * scores
        H_graph = scatter_sum(H_weighted, batch, dim=0)
        
        return H_graph
```

---

## Summary

**Key Workflow:**
1. **Datapoints** (`MoleculeDatapoint`) → Store SMILES, targets, descriptors
2. **Featurizer** (`SimpleMoleculeMolGraphFeaturizer`) → Convert molecules to graphs (`MolGraph`)
3. **Dataset** (`MoleculeDataset`) → Apply featurization on-the-fly
4. **DataLoader** (`build_dataloader`) → Batch graphs into `BatchMolGraph`
5. **Message Passing** (`BondMessagePassing`) → Encode graph structure
6. **Aggregation** (`MeanAggregation`) → Pool to graph-level representation
7. **Prediction Head** → Output final predictions

**To implement a custom MPNN:**
1. Subclass `_MessagePassingBase` (or use mixins)
2. Implement `setup()` and `aggregate_messages()`
3. Register in `__init__.py`
4. Add to training script

**Key Design Principles:**
- **Modularity**: Separate featurization, message passing, and aggregation
- **Flexibility**: Easy to swap components (e.g., different aggregations)
- **Efficiency**: Batched operations on GPU
- **Extensibility**: Clear interfaces for custom implementations

---

## References

- **Chemprop Paper**: [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237)
- **D-MPNN**: Directed Message Passing Neural Network
- **PyTorch Geometric**: Inspiration for batching and scatter operations
- **Lightning**: Training framework used in Chemprop v2

For more examples, see:
- `chemprop/models/model.py`: Full MPNN model implementation
- `scripts/python/train_graph.py`: Complete training workflow
- `chemprop/nn/message_passing/`: All message passing implementations
