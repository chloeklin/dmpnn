#!/usr/bin/env python3
"""
Test smart filtering with TC dataset to see if there are low-variance features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add path to utilities
sys.path.append('/Users/u6788552/Desktop/experiments/dmpnn/scripts/python')
from tabular_utils import build_features, preprocess_descriptor_data

def test_tc_smart_filtering():
    """Test smart filtering with TC dataset."""
    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load TC dataset
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
    
    if descriptor_block is not None:
        # Get non-AB feature names
        non_ab_names = [name for name in feat_names if not name.startswith('AB_')]
        
        # Apply preprocessing to descriptors
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
        
        # Analyze variances
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
        
        print(f"\nTC Dataset Feature Variance Analysis:")
        print(f"Total features: {len(final_feat_names)}")
        print(f"Exact zero variance: {exact_zero_count}")
        print(f"Low variance (<1e-10): {low_variance_count}")
        print(f"Normal variance (≥1e-10): {normal_variance_count}")
        
        print(f"\nBy Feature Type:")
        print(f"AB features: {ab_exact_zero} zero, {ab_low_variance} low, {ab_normal} normal")
        print(f"Descriptors: {desc_exact_zero} zero, {desc_low_variance} low, {desc_normal} normal")
        
        # Show examples of low-variance features if any exist
        if low_variance_count > 0:
            print(f"\nLow-variance features being kept:")
            for i, (name, var) in enumerate(zip(final_feat_names, final_variances)):
                if 0.0 < var < 1e-10:
                    feature_type = "AB" if name.startswith('AB_') else "Descriptor"
                    print(f"  {name} ({feature_type}): variance = {var:.2e}")
        else:
            print(f"\nNo low-variance features found in TC dataset")
        
        return True
    else:
        print("No descriptor features found!")
        return False

if __name__ == "__main__":
    test_tc_smart_filtering()
