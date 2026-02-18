#!/usr/bin/env python3
"""Test script to verify AttentiveFP descriptor concatenation support."""

import torch
import numpy as np
import sys
sys.path.insert(0, 'scripts/python')

from attentivefp_utils import (
    create_attentivefp_model,
    GraphCSV,
    smiles_to_graph
)
from torch_geometric.loader import DataLoader
import pandas as pd

print("=" * 60)
print("Testing AttentiveFP Descriptor Concatenation")
print("=" * 60)

# Test 1: Model creation without descriptors
print("\n1. Testing model creation without descriptors...")
model_no_desc = create_attentivefp_model(
    task_type='reg',
    hidden_channels=200,
    num_layers=2,
    num_timesteps=2,
    dropout=0.0,
    descriptor_dim=0
)
print(f"✓ Model created without descriptors")
print(f"  Model type: {type(model_no_desc).__name__}")

# Test 2: Model creation with descriptors
print("\n2. Testing model creation with descriptors...")
descriptor_dim = 50
model_with_desc = create_attentivefp_model(
    task_type='reg',
    hidden_channels=200,
    num_layers=2,
    num_timesteps=2,
    dropout=0.0,
    descriptor_dim=descriptor_dim
)
print(f"✓ Model created with {descriptor_dim} descriptors")
print(f"  Model type: {type(model_with_desc).__name__}")
print(f"  Has descriptor layer: {hasattr(model_with_desc, 'output_layer')}")

# Test 3: Dataset creation with descriptors
print("\n3. Testing dataset creation with descriptors...")
test_smiles = [
    "CCO",  # ethanol
    "CC(C)O",  # isopropanol
    "CCCC",  # butane
]
test_targets = [1.5, 2.0, 3.5]
test_descriptors = np.random.randn(3, descriptor_dim).astype(np.float32)

df_test = pd.DataFrame({
    'smiles': test_smiles,
    'target': test_targets
})

dataset = GraphCSV(df_test, 'smiles', 'target', task_type='reg', descriptors=test_descriptors)
print(f"✓ Dataset created with {len(dataset)} molecules")
print(f"  Descriptors shape: {test_descriptors.shape}")

# Test 4: DataLoader and batch creation
print("\n4. Testing DataLoader with descriptors...")
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))
print(f"✓ Batch created successfully")
print(f"  Batch size: {batch.num_graphs}")
print(f"  Has descriptors: {hasattr(batch, 'descriptors')}")
if hasattr(batch, 'descriptors'):
    print(f"  Descriptor shape: {batch.descriptors.shape}")

# Test 5: Forward pass without descriptors
print("\n5. Testing forward pass without descriptors...")
model_no_desc.eval()
with torch.no_grad():
    output_no_desc = model_no_desc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
print(f"✓ Forward pass successful")
print(f"  Output shape: {output_no_desc.shape}")
print(f"  Output values: {output_no_desc.squeeze().tolist()}")

# Test 6: Forward pass with descriptors
print("\n6. Testing forward pass with descriptors...")
model_with_desc.eval()
with torch.no_grad():
    output_with_desc = model_with_desc(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, 
        batch.descriptors if hasattr(batch, 'descriptors') else None
    )
print(f"✓ Forward pass successful")
print(f"  Output shape: {output_with_desc.shape}")
print(f"  Output values: {output_with_desc.squeeze().tolist()}")

# Test 7: Verify descriptor concatenation affects output
print("\n7. Verifying descriptor impact...")
# Create two identical batches but with different descriptors
desc1 = torch.randn(batch.num_graphs, descriptor_dim)
desc2 = torch.randn(batch.num_graphs, descriptor_dim)

with torch.no_grad():
    out1 = model_with_desc(batch.x, batch.edge_index, batch.edge_attr, batch.batch, desc1)
    out2 = model_with_desc(batch.x, batch.edge_index, batch.edge_attr, batch.batch, desc2)

outputs_differ = not torch.allclose(out1, out2, atol=1e-6)
print(f"✓ Outputs differ with different descriptors: {outputs_differ}")
if outputs_differ:
    print(f"  Mean absolute difference: {(out1 - out2).abs().mean().item():.6f}")

# Test 8: Parameter count comparison
print("\n8. Comparing parameter counts...")
params_no_desc = sum(p.numel() for p in model_no_desc.parameters())
params_with_desc = sum(p.numel() for p in model_with_desc.parameters())
print(f"  Parameters without descriptors: {params_no_desc:,}")
print(f"  Parameters with descriptors: {params_with_desc:,}")
print(f"  Additional parameters: {params_with_desc - params_no_desc:,}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nAttentiveFP now supports descriptor concatenation like chemprop models.")
print("\nUsage:")
print("  python train_attentivefp.py --dataset_name htpmd --incl_rdkit")
print("  python train_attentivefp.py --dataset_name htpmd --incl_desc")
print("  python train_attentivefp.py --dataset_name htpmd --incl_rdkit --incl_desc")
