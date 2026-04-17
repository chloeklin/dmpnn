#!/usr/bin/env python3
"""
Diagnostic script to check zero-variance features after preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging
import sys

# Add path to utilities
sys.path.append('/Users/u6788552/Desktop/experiments/dmpnn/scripts/python')
from tabular_utils import build_features, preprocess_descriptor_data

def check_zero_variance_after_preprocessing():
    """Check if zero-variance features are created after imputation."""
    
    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load a dataset for testing (use TC as example)
    data_path = Path("/Users/u6788552/Desktop/experiments/dmpnn/data/tc.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # Create train/val/test split
    np.random.seed(42)
    n_samples = len(df)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Generate features
    print("Generating features...")
    ab_block, descriptor_block, feat_names = build_features(
        df, train_idx, [], "homo",  # No descriptor columns for TC
        use_rdkit=True, use_ab=True, smiles_column="smiles"
    )
    
    print(f"Total features generated: {len(feat_names)}")
    print(f"AB block shape: {ab_block.shape if ab_block is not None else 'None'}")
    print(f"Descriptor block shape: {descriptor_block.shape if descriptor_block is not None else 'None'}")
    
    if descriptor_block is not None:
        # Get non-AB feature names
        non_ab_names = [name for name in feat_names if not name.startswith('AB_')]
        
        print(f"\nBefore preprocessing:")
        print(f"Descriptor features: {len(non_ab_names)}")
        
        # Check variance before preprocessing
        desc_X = np.asarray(descriptor_block, dtype=np.float64)
        inf_mask = np.isinf(desc_X)
        if np.any(inf_mask):
            desc_X[inf_mask] = np.nan
        
        # Check variance before constant removal
        variances_before = np.nanvar(desc_X, axis=0)
        zero_var_before = np.sum(variances_before < 1e-10)
        print(f"Zero variance features before constant removal: {zero_var_before}")
        
        # Apply preprocessing
        print(f"\nApplying preprocessing...")
        (desc_tr_selected, desc_val_selected, desc_te_selected, selected_desc_names, 
         preprocessing_metadata, imputer, constant_mask, corr_mask) = preprocess_descriptor_data(
             descriptor_block, train_idx, val_idx, test_idx, 
             non_ab_names,
             logger
         )
        
        print(f"\nAfter preprocessing:")
        print(f"Features before any selection: {preprocessing_metadata['n_desc_before_any_selection']}")
        print(f"Features after constant removal: {preprocessing_metadata['n_desc_after_constant_removal']}")
        print(f"Features after correlation removal: {preprocessing_metadata['n_desc_after_corr_removal']}")
        print(f"Constant features removed: {len(preprocessing_metadata['constant_features_removed'])}")
        print(f"Correlated features removed: {len(preprocessing_metadata['correlated_features_removed'])}")
        
        # Check variance after preprocessing
        variances_after = np.var(desc_tr_selected, axis=0)
        zero_var_after = np.sum(variances_after < 1e-10)
        print(f"Zero variance features after preprocessing: {zero_var_after}")
        
        if zero_var_after > 0:
            print(f"\n⚠️  WARNING: Found {zero_var_after} zero-variance features after preprocessing!")
            
            # Find which features have zero variance
            zero_var_indices = np.where(variances_after < 1e-10)[0]
            print("Zero variance features:")
            for idx in zero_var_indices:
                print(f"  - {selected_desc_names[idx]} (variance: {variances_after[idx]:.2e})")
            
            # Let's check what happened during imputation
            print(f"\nAnalyzing imputation impact...")
            
            # Get the data before imputation but after constant removal
            desc_X = np.asarray(descriptor_block, dtype=np.float64)
            inf_mask = np.isinf(desc_X)
            if np.any(inf_mask):
                desc_X[inf_mask] = np.nan
            
            # Apply constant removal
            const_keep_idx = np.where(constant_mask)[0]
            desc_X_no_const = desc_X[:, const_keep_idx]
            
            # Split before imputation
            desc_tr_before = desc_X_no_const[train_idx]
            
            # Check variance before imputation for kept features
            const_kept_names = [name for i, name in enumerate(non_ab_names) if constant_mask[i]]
            variances_before_impute = np.nanvar(desc_tr_before, axis=0)
            
            # Check which features became zero variance after imputation
            for i, (name, var_before, var_after) in enumerate(zip(const_kept_names, variances_before_impute, variances_after)):
                if var_after < 1e-10 and var_before >= 1e-10:
                    print(f"  Feature '{name}' became zero variance after imputation")
                    print(f"    Variance before: {var_before:.2e}")
                    print(f"    Variance after: {var_after:.2e}")
                    
                    # Show some statistics
                    feature_data = desc_tr_before[:, i]
                    non_nan_data = feature_data[~np.isnan(feature_data)]
                    if len(non_nan_data) > 0:
                        print(f"    Non-NaN values: {len(non_nan_data)}")
                        print(f"    Unique non-NaN values: {len(np.unique(non_nan_data))}")
                        print(f"    Median (used for imputation): {np.median(non_nan_data):.6f}")
                        print(f"    All non-NaN values same: {len(np.unique(non_nan_data)) == 1}")
        else:
            print(f"\n✅ No zero-variance features found after preprocessing!")
    
    # Also check AB block features
    if ab_block is not None:
        print(f"\nChecking AB block features...")
        ab_variances = np.var(ab_block[train_idx], axis=0)
        zero_var_ab = np.sum(ab_variances < 1e-10)
        print(f"AB block features: {ab_block.shape[1]}")
        print(f"Zero variance AB features: {zero_var_ab}")
        
        if zero_var_ab > 0:
            print("⚠️  AB block has zero-variance features!")

if __name__ == "__main__":
    check_zero_variance_after_preprocessing()
