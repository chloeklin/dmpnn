#!/usr/bin/env python3
"""
Diagnostic script to understand AB block features and why many have zero variance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add path to utilities
sys.path.append('/Users/u6788552/Desktop/experiments/dmpnn/scripts/python')
from tabular_utils import build_features

def analyze_ab_features():
    """Analyze AB block features to understand zero variance patterns."""
    
    # Load TC dataset (homopolymer)
    data_path = Path("/Users/u6788552/Desktop/experiments/dmpnn/data/tc.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Generate only AB features (no RDKit, no descriptors)
    print("Generating AB block features only...")
    ab_block, descriptor_block, feat_names = build_features(
        df, list(range(len(df))), [], "homo",  # Use all samples, no descriptors
        use_rdkit=False, use_ab=True, smiles_column="smiles"
    )
    
    print(f"AB block shape: {ab_block.shape}")
    print(f"Total AB features: {len([name for name in feat_names if name.startswith('AB_')])}")
    
    # Calculate variances for each AB feature
    ab_variances = np.var(ab_block, axis=0)
    zero_var_mask = ab_variances < 1e-10
    zero_var_count = np.sum(zero_var_mask)
    non_zero_var_count = len(ab_variances) - zero_var_count
    
    print(f"\nAB Feature Variance Analysis:")
    print(f"Zero variance features: {zero_var_count}/{len(ab_variances)} ({zero_var_count/len(ab_variances)*100:.1f}%)")
    print(f"Non-zero variance features: {non_zero_var_count}/{len(ab_variances)} ({non_zero_var_count/len(ab_variances)*100:.1f}%)")
    
    # Show which features have zero vs non-zero variance
    zero_var_indices = np.where(zero_var_mask)[0]
    non_zero_var_indices = np.where(~zero_var_mask)[0]
    
    print(f"\nFeatures with NON-ZERO variance:")
    for idx in non_zero_var_indices[:10]:  # Show first 10
        feature_name = f"AB_{idx}"
        variance = ab_variances[idx]
        print(f"  {feature_name}: variance = {variance:.6f}")
    
    if len(non_zero_var_indices) > 10:
        print(f"  ... and {len(non_zero_var_indices) - 10} more")
    
    print(f"\nFeatures with ZERO variance:")
    for idx in zero_var_indices[:10]:  # Show first 10
        feature_name = f"AB_{idx}"
        print(f"  {feature_name}: variance = {ab_variances[idx]:.2e}")
    
    if len(zero_var_indices) > 10:
        print(f"  ... and {len(zero_var_indices) - 10} more")
    
    # Let's understand what these features represent
    print(f"\n" + "="*60)
    print("UNDERSTANDING AB FEATURES")
    print("="*60)
    
    print("AB features are pooled atom and bond features:")
    print("- First part (features 0-126): Atom features (127 features)")
    print("- Second part (features 127-138): Bond features (12 features)")
    print()
    
    # Look at the actual values to understand the pattern
    print("Sample values for first few AB features:")
    print("Feature\tMin\tMax\tMean\tUnique Values")
    print("-" * 50)
    
    for i in range(min(20, len(ab_variances))):
        feature_values = ab_block[:, i]
        unique_count = len(np.unique(feature_values))
        print(f"AB_{i}\t{feature_values.min():.3f}\t{feature_values.max():.3f}\t{feature_values.mean():.3f}\t{unique_count}")
    
    # Check if this is a homopolymer-specific issue
    print(f"\n" + "="*60)
    print("WHY SO MANY ZERO VARIANCE FEATURES?")
    print("="*60)
    
    print("For HOMOPOLYMERS (like TC dataset):")
    print("1. All molecules are single monomer types")
    print("2. No monomer B present → fraction_B = 0 for all samples")
    print("3. Many atom/bond features are identical across similar monomers")
    print("4. Bond features especially: similar bonding patterns in homopolymers")
    print()
    
    print("For COPOLYMERS (would have different patterns):")
    print("1. Two different monomer types A and B")
    print("2. Variable fractions of A and B")
    print("3. More diverse atom/bond environments")
    print("4. Higher variance in AB features")
    
    # Let's also check the feature structure
    print(f"\n" + "="*60)
    print("FEATURE STRUCTURE ANALYSIS")
    print("="*60)
    
    # Check if there are patterns in which features have variance
    print("Analyzing which feature groups have variance...")
    
    atom_features_end = 127  # Based on ATOM_FEAT_LEN
    bond_features_start = 127
    bond_features_end = 139  # 127 + 12 = 139
    
    atom_variances = ab_variances[:atom_features_end]
    bond_variances = ab_variances[bond_features_start:bond_features_end]
    
    atom_zero_var = np.sum(atom_variances < 1e-10)
    atom_non_zero_var = len(atom_variances) - atom_zero_var
    
    bond_zero_var = np.sum(bond_variances < 1e-10)
    bond_non_zero_var = len(bond_variances) - bond_zero_var
    
    print(f"Atom features (0-126):")
    print(f"  Zero variance: {atom_zero_var}/{len(atom_variances)} ({atom_zero_var/len(atom_variances)*100:.1f}%)")
    print(f"  Non-zero variance: {atom_non_zero_var}/{len(atom_variances)} ({atom_non_zero_var/len(atom_variances)*100:.1f}%)")
    
    print(f"Bond features (127-138):")
    print(f"  Zero variance: {bond_zero_var}/{len(bond_variances)} ({bond_zero_var/len(bond_variances)*100:.1f}%)")
    print(f"  Non-zero variance: {bond_non_zero_var}/{len(bond_variances)} ({bond_non_zero_var/len(bond_variances)*100:.1f}%)")
    
    # Show which specific bond features have variance
    if bond_non_zero_var > 0:
        print(f"\nBond features with variance:")
        for i in range(bond_features_start, bond_features_end):
            if ab_variances[i] >= 1e-10:
                print(f"  AB_{i}: variance = {ab_variances[i]:.6f}")

if __name__ == "__main__":
    analyze_ab_features()
