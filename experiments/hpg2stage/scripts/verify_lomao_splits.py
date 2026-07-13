#!/usr/bin/env python3
"""
Verify Leave-One-Monomer-A-Out (LOMAO) Split Implementation
===========================================================

This script verifies that:
1. LOMAO creates one fold per unique monomer A
2. No data leakage occurs (held-out monomer A absent from train/val)
3. All monomers A appear exactly once as held-out
4. Dataset-driven implementation works for any dataset
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
sys.path.append('../../scripts/python')

from utils import generate_a_held_out_splits, canonicalize_smiles

def verify_lomao_implementation():
    """Verify LOMAO implementation on EA/IP dataset."""
    print("=" * 70)
    print("Verifying LOMAO Implementation on EA/IP Dataset")
    print("=" * 70)
    
    # Load dataset
    ROOT = Path(__file__).resolve().parents[3]
    data_path = ROOT / 'data' / 'ea_ip.csv'
    df = pd.read_csv(data_path)
    
    # Extract monomer A identities
    smiles_A = df['smiles_A'].astype(str).tolist()
    n_datapoints = len(df)
    
    print(f"Dataset: {n_datapoints} polymers")
    print(f"Unique monomer A species: {df['smiles_A'].nunique()}")
    print()
    
    # Test LOMAO protocol
    print("Testing LOMAO protocol...")
    train_idx, val_idx, test_idx, held_out_monomers = generate_a_held_out_splits(
        smiles_A, n_datapoints, seed=42, protocol='leave_one_A_out'
    )
    
    n_folds = len(train_idx)
    unique_monomers = set(df['smiles_A'].unique())
    held_out_set = set(held_out_monomers)
    
    print(f"✓ Created {n_folds} folds")
    print(f"✓ Expected {len(unique_monomers)} folds (one per monomer A)")
    assert n_folds == len(unique_monomers), f"Fold count mismatch: {n_folds} vs {len(unique_monomers)}"
    
    # Verify each monomer appears exactly once as held-out
    print("\nVerifying held-out monomer coverage...")
    for monomer in unique_monomers:
        count = held_out_monomers.count(monomer)
        if count != 1:
            print(f"✗ Monomer {monomer} appears {count} times (should be 1)")
            return False
    print("✓ Each monomer A appears exactly once as held-out")
    
    # Verify no data leakage
    print("\nVerifying no data leakage...")
    for fold_i in range(n_folds):
        held_out = held_out_monomers[fold_i]
        
        # Get monomers in each split
        train_monomers = set(df.iloc[train_idx[fold_i]]['smiles_A'].unique())
        val_monomers = set(df.iloc[val_idx[fold_i]]['smiles_A'].unique())
        test_monomers = set(df.iloc[test_idx[fold_i]]['smiles_A'].unique())
        
        # Check held-out monomer only appears in test set
        if held_out in train_monomers:
            print(f"✗ Fold {fold_i}: Held-out {held_out} found in training set")
            return False
        if held_out in val_monomers:
            print(f"✗ Fold {fold_i}: Held-out {held_out} found in validation set")
            return False
        if held_out not in test_monomers:
            print(f"✗ Fold {fold_i}: Held-out {held_out} not found in test set")
            return False
        
        # Verify test set contains ONLY the held-out monomer
        if len(test_monomers) != 1:
            print(f"✗ Fold {fold_i}: Test set has {len(test_monomers)} monomers (should be 1)")
            return False
    
    print("✓ No data leakage detected")
    print("✓ Each test set contains exactly one held-out monomer A")
    
    # Print fold statistics
    print("\nFold Statistics:")
    print("-" * 70)
    print(f"{'Fold':<5} {'Held-out Monomer':<30} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 70)
    
    for fold_i in range(n_folds):
        held_out = held_out_monomers[fold_i]
        n_train = len(train_idx[fold_i])
        n_val = len(val_idx[fold_i])
        n_test = len(test_idx[fold_i])
        
        # Truncate long SMILES for display
        display_monomer = held_out[:27] + "..." if len(held_out) > 30 else held_out
        print(f"{fold_i:<5} {display_monomer:<30} {n_train:<8} {n_val:<8} {n_test:<8}")
    
    print("-" * 70)
    print(f"Total: {n_folds} folds")
    
    return True

def verify_groupkfold_compatibility():
    """Verify that groupkfold protocol still works."""
    print("\n" + "=" * 70)
    print("Verifying GroupKFold Protocol Compatibility")
    print("=" * 70)
    
    # Load dataset
    ROOT = Path(__file__).resolve().parents[3]
    data_path = ROOT / 'data' / 'ea_ip.csv'
    df = pd.read_csv(data_path)
    
    smiles_A = df['smiles_A'].astype(str).tolist()
    n_datapoints = len(df)
    
    # Test groupkfold protocol
    print("Testing GroupKFold protocol (5-fold)...")
    train_idx, val_idx, test_idx, held_out_monomers = generate_a_held_out_splits(
        smiles_A, n_datapoints, seed=42, n_splits=5, protocol='groupkfold'
    )
    
    n_folds = len(train_idx)
    print(f"✓ Created {n_folds} folds")
    assert n_folds == 5, f"Expected 5 folds, got {n_folds}"
    assert held_out_monomers is None, f"GroupKFold should return None for held_out_monomers"
    
    # Verify no data leakage
    print("Verifying no data leakage...")
    for fold_i in range(n_folds):
        train_monomers = set(df.iloc[train_idx[fold_i]]['smiles_A'].unique())
        val_monomers = set(df.iloc[val_idx[fold_i]]['smiles_A'].unique())
        test_monomers = set(df.iloc[test_idx[fold_i]]['smiles_A'].unique())
        
        # Check for overlaps
        if train_monomers & val_monomers:
            print(f"✗ Fold {fold_i}: Train/val monomer overlap")
            return False
        if train_monomers & test_monomers:
            print(f"✗ Fold {fold_i}: Train/test monomer overlap")
            return False
        if val_monomers & test_monomers:
            print(f"✗ Fold {fold_i}: Val/test monomer overlap")
            return False
    
    print("✓ No data leakage detected")
    print("✓ GroupKFold protocol works correctly")
    
    return True

def test_generic_implementation():
    """Test that implementation works generically for any dataset."""
    print("\n" + "=" * 70)
    print("Testing Generic Implementation (Synthetic Dataset)")
    print("=" * 70)
    
    # Create synthetic dataset with varying number of monomers A
    np.random.seed(42)
    n_monomers_A = 4
    samples_per_monomer = 10
    
    smiles_A_list = [f"monomer_{i}" for i in range(n_monomers_A)]
    smiles_A = []
    for monomer in smiles_A_list:
        smiles_A.extend([monomer] * samples_per_monomer)
    
    n_datapoints = len(smiles_A)
    
    print(f"Synthetic dataset: {n_datapoints} samples, {n_monomers_A} unique monomers A")
    
    # Test LOMAO
    train_idx, val_idx, test_idx, held_out_monomers = generate_a_held_out_splits(
        smiles_A, n_datapoints, seed=42, protocol='leave_one_A_out'
    )
    
    n_folds = len(train_idx)
    print(f"✓ Created {n_folds} folds for {n_monomers_A} monomers")
    assert n_folds == n_monomers_A, f"Expected {n_monomers_A} folds, got {n_folds}"
    
    # Verify each monomer appears once
    for i, monomer in enumerate(smiles_A_list):
        assert held_out_monomers[i] == monomer, f"Fold {i} should hold out {monomer}"
    
    print("✓ Generic implementation works correctly")
    return True

if __name__ == "__main__":
    success = True
    
    try:
        success &= verify_lomao_implementation()
        success &= verify_groupkfold_compatibility()
        success &= test_generic_implementation()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ ALL VERIFICATIONS PASSED")
            print("LOMAO implementation is correct and ready for use.")
        else:
            print("❌ VERIFICATION FAILED")
            print("Please fix the issues before proceeding.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)
