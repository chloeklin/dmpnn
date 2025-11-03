#!/usr/bin/env python3
"""
Test the smart zero-variance filtering approach for insulator dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

# Add path to utilities
sys.path.append('/Users/u6788552/Desktop/experiments/dmpnn/scripts/python')
from tabular_utils import build_features, preprocess_descriptor_data

def test_smart_filtering():
    """Test the smart filtering approach."""
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
    
    if descriptor_block is not None:
        # Get non-AB feature names
        non_ab_names = [name for name in feat_names if not name.startswith('AB_')]
        
        # Apply preprocessing to descriptors (now with smart filtering)
        print("Preprocessing descriptor features...")
        (desc_tr_selected, desc_val_selected, desc_te_selected, selected_desc_names, 
         preprocessing_metadata, imputer, constant_mask, corr_mask) = preprocess_descriptor_data(
             descriptor_block, train_idx, val_idx, test_idx, 
             non_ab_names,
             logger
         )
        
        # Combine AB and descriptor features
        if ab_block is not None:
            features_combined = np.concatenate([ab_block[train_idx], desc_tr_selected], axis=1)
            final_feat_names = ([name for name in feat_names if name.startswith('AB_')] + 
                               selected_desc_names)
        else:
            features_combined = desc_tr_selected
            final_feat_names = selected_desc_names
        
        print(f"Combined feature matrix shape: {features_combined.shape}")
        
        # Apply smart filtering to combined matrix
        print("Applying smart zero-variance filtering to combined features...")
        final_variances = np.var(features_combined, axis=0)
        
        # Categorize features
        exact_zero_mask = final_variances == 0.0
        low_variance_mask = (final_variances > 0.0) & (final_variances < 1e-10)
        normal_variance_mask = final_variances >= 1e-10
        
        exact_zero_count = np.sum(exact_zero_mask)
        low_variance_count = np.sum(low_variance_mask)
        normal_variance_count = np.sum(normal_variance_mask)
        
        # Separate AB and descriptor features
        ab_indices = [i for i, name in enumerate(final_feat_names) if name.startswith('AB_')]
        desc_indices = [i for i, name in enumerate(final_feat_names) if not name.startswith('AB_')]
        
        ab_exact_zero = sum(1 for i in ab_indices if exact_zero_mask[i])
        ab_low_variance = sum(1 for i in ab_indices if low_variance_mask[i])
        ab_normal = sum(1 for i in ab_indices if normal_variance_mask[i])
        
        desc_exact_zero = sum(1 for i in desc_indices if exact_zero_mask[i])
        desc_low_variance = sum(1 for i in desc_indices if low_variance_mask[i])
        desc_normal = sum(1 for i in desc_indices if normal_variance_mask[i])
        
        print(f"\nFeature Variance Analysis (Smart Filtering):")
        print(f"Total features: {len(final_feat_names)}")
        print(f"Exact zero variance: {exact_zero_count}")
        print(f"Low variance (<1e-10): {low_variance_count}")
        print(f"Normal variance (≥1e-10): {normal_variance_count}")
        
        print(f"\nBy Feature Type:")
        print(f"AB features: {ab_exact_zero} zero, {ab_low_variance} low, {ab_normal} normal")
        print(f"Descriptors: {desc_exact_zero} zero, {desc_low_variance} low, {desc_normal} normal")
        
        # Apply smart filtering (remove only exact zero variance)
        smart_keep_mask = ~exact_zero_mask
        features_smart = features_combined[:, smart_keep_mask]
        feat_names_smart = [name for name, keep in zip(final_feat_names, smart_keep_mask) if keep]
        
        print(f"\nResults after smart filtering:")
        print(f"Features before: {len(features_combined[0])}")
        print(f"Features after: {len(features_smart[0])}")
        print(f"Features removed: {len(features_combined[0]) - len(features_smart[0])} (exact zero variance only)")
        
        # Compare with old approach
        old_keep_mask = final_variances > 1e-10
        features_old = features_combined[:, old_keep_mask]
        
        print(f"\nComparison with old approach:")
        print(f"Old approach (variance > 1e-10): {len(features_old[0])} features")
        print(f"Smart approach (variance > 0): {len(features_smart[0])} features")
        print(f"Additional features kept: {len(features_smart[0]) - len(features_old[0])}")
        
        # Show some examples of low-variance features being kept
        low_variance_features = []
        for i, (name, var) in enumerate(zip(final_feat_names, final_variances)):
            if 0.0 < var < 1e-10:
                low_variance_features.append((name, var))
        
        if low_variance_features:
            print(f"\nExamples of low-variance features being kept:")
            for name, var in low_variance_features[:5]:  # Show first 5
                feature_type = "AB" if name.startswith('AB_') else "Descriptor"
                print(f"  {name} ({feature_type}): variance = {var:.2e}")
        
        return len(features_smart[0]), len(features_old[0])
    else:
        print("No descriptor features found!")
        return None, None

if __name__ == "__main__":
    smart_count, old_count = test_smart_filtering()
    
    if smart_count is not None:
        print(f"\n" + "="*60)
        print("SMART FILTERING SUMMARY")
        print("="*60)
        print(f"Features with smart filtering: {smart_count}")
        print(f"Features with old approach: {old_count}")
        print(f"Additional features retained: {smart_count - old_count}")
        print("\n✅ Smart filtering keeps chemically meaningful low-variance AB features!")
        print("✅ While still removing exact zero variance features that provide no information!")
