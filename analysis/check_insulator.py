#!/usr/bin/env python3
"""
Check insulator dataset for zero variance features after preprocessing.
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

def check_insulator_zero_variance():
    """Check if zero-variance features are properly removed for insulator dataset."""
    
    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load insulator dataset
    data_path = Path("/Users/u6788552/Desktop/experiments/dmpnn/data/insulator.csv")
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
    
    print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Generate features
    print("Generating features...")
    ab_block, descriptor_block, feat_names = build_features(
        df, train_idx, [], "homo",  # No descriptor columns for insulator
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
        
        variances_before = np.nanvar(desc_X, axis=0)
        zero_var_before = np.sum(variances_before < 1e-10)
        print(f"Zero variance descriptors before constant removal: {zero_var_before}")
        
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
        print(f"Features after final zero-var removal: {preprocessing_metadata['n_desc_after_final_zero_var_removal']}")
        print(f"Constant features removed: {len(preprocessing_metadata['constant_features_removed'])}")
        print(f"Correlated features removed: {len(preprocessing_metadata['correlated_features_removed'])}")
        print(f"Zero-var features removed after imputation: {len(preprocessing_metadata['zero_var_after_impute_removed'])}")
        
        if preprocessing_metadata['zero_var_after_impute_removed']:
            print(f"Zero-var features: {preprocessing_metadata['zero_var_after_impute_removed']}")
        
        # Check variance after preprocessing
        variances_after = np.var(desc_tr_selected, axis=0)
        zero_var_after = np.sum(variances_after < 1e-10)
        print(f"Zero variance descriptors after preprocessing: {zero_var_after}")
        
        # Combine with AB block
        if ab_block is not None:
            features_combined = np.concatenate([ab_block[train_idx], desc_tr_selected], axis=1)
            final_feat_names = ([name for name in feat_names if name.startswith('AB_')] + 
                               selected_desc_names)
        else:
            features_combined = desc_tr_selected
            final_feat_names = selected_desc_names
        
        print(f"\nFinal feature matrix shape: {features_combined.shape}")
        
        # Check variance of ALL final features (including AB)
        final_variances = np.var(features_combined, axis=0)
        zero_var_final = np.sum(final_variances < 1e-10)
        non_zero_var_final = len(final_variances) - zero_var_final
        
        print(f"Final feature variance analysis:")
        print(f"Total features: {len(final_variances)}")
        print(f"Zero variance features: {zero_var_final}")
        print(f"Non-zero variance features: {non_zero_var_final}")
        print(f"Percentage zero variance: {zero_var_final/len(final_variances)*100:.1f}%")
        
        # Break down by feature type
        ab_names = [name for name in final_feat_names if name.startswith('AB_')]
        desc_names = [name for name in final_feat_names if not name.startswith('AB_')]
        
        ab_indices = [i for i, name in enumerate(final_feat_names) if name.startswith('AB_')]
        desc_indices = [i for i, name in enumerate(final_feat_names) if not name.startswith('AB_')]
        
        ab_variances = final_variances[ab_indices] if ab_indices else np.array([])
        desc_variances = final_variances[desc_indices] if desc_indices else np.array([])
        
        ab_zero_var = np.sum(ab_variances < 1e-10) if len(ab_variances) > 0 else 0
        desc_zero_var = np.sum(desc_variances < 1e-10) if len(desc_variances) > 0 else 0
        
        print(f"\nBreakdown by feature type:")
        print(f"AB features: {len(ab_variances)} total, {ab_zero_var} zero variance ({ab_zero_var/len(ab_variances)*100:.1f}%)" if len(ab_variances) > 0 else "AB features: 0")
        print(f"Descriptor features: {len(desc_variances)} total, {desc_zero_var} zero variance ({desc_zero_var/len(desc_variances)*100:.1f}%)" if len(desc_variances) > 0 else "Descriptor features: 0")
        
        if zero_var_after > 0:
            print(f"\n⚠️  WARNING: Found {zero_var_after} zero-variance features after preprocessing!")
            
            # Find which features have zero variance
            zero_var_indices = np.where(variances_after < 1e-10)[0]
            print("Zero variance descriptor features:")
            for idx in zero_var_indices:
                print(f"  - {selected_desc_names[idx]} (variance: {variances_after[idx]:.2e})")
        else:
            print(f"\n✅ No zero-variance DESCRIPTOR features found after preprocessing!")
        
        if ab_zero_var > 0:
            print(f"\n⚠️  Found {ab_zero_var} zero-variance AB features (this is expected for homopolymers)")
        else:
            print(f"\n✅ No zero-variance AB features found")
            
        return zero_var_final, len(final_variances)
    else:
        print("No descriptor features found!")
        return None, None

if __name__ == "__main__":
    zero_var_count, total_count = check_insulator_zero_variance()
    
    if zero_var_count is not None:
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Insulator dataset: {zero_var_count}/{total_count} zero variance features")
        
        if zero_var_count == 104:  # Matching what you observed
            print("✅ This matches your observation from the histogram!")
            print("The issue is likely that AB features (which are expected to have zero variance)")
            print("are being included in the final feature matrix before the variance check.")
        else:
            print(f"⚠️  Expected 104 zero variance features but found {zero_var_count}")
            print("There might be an inconsistency in the preprocessing.")
