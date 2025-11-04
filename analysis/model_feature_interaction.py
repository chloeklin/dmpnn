#!/usr/bin/env python3
"""
Model-Feature Interaction Analysis Script

This script analyzes feature importance from trained tabular models by:
1. Loading trained models and preprocessing objects from train_tabular.py results
2. Extracting feature importance (coefficients for linear, feature_importances_ for RF/XGB)
3. Creating bar plots of top 5 features for each model and target
4. Comparing feature importance between AB block, RDKit, and descriptor features

Usage:
    python3 analysis/model_feature_interaction.py --dataset_name insulator --task_type reg
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import yaml
from typing import Dict, List, Tuple, Optional
import sys

# Add path to utilities
sys.path.append('/Users/u6788552/Desktop/experiments/dmpnn/scripts/python')
from utils import setup_training_environment
from tabular_utils import load_preprocessing_objects

def load_feature_names(checkpoint_dir: Path, split_idx: int, model_name: str) -> Optional[List[str]]:
    """Load feature names from preprocessing metadata."""
    try:
        # Load preprocessing objects
        preprocessing_data = load_preprocessing_objects(checkpoint_dir, split_idx)
        
        if preprocessing_data is None:
            print(f"Warning: No preprocessing metadata found for split {split_idx}")
            return None
        
        # Get feature names from metadata
        metadata = preprocessing_data['preprocessing_metadata']
        feat_names = metadata.get('feat_names', [])
        
        if not feat_names:
            # Fallback: reconstruct feature names from metadata
            feat_names = []
            
            # Add AB feature names if AB was used
            if metadata.get('use_ab', False):
                ab_count = metadata.get('ab_feature_count', 0)
                feat_names.extend([f"AB_{i}" for i in range(ab_count)])
            
            # Add descriptor names
            feat_names.extend(preprocessing_data['selected_desc_names'])
        
        return feat_names
        
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return None

def load_model_and_data(checkpoint_dir: Path, split_idx: int, model_name: str) -> Tuple[Optional[object], Optional[List[str]]]:
    """Load trained model and feature names."""
    try:
        # Load trained model
        model_file = checkpoint_dir / f"{model_name}_split_{split_idx}.pkl"
        if not model_file.exists():
            print(f"Warning: Model file not found: {model_file}")
            return None, None
        
        model = joblib.load(model_file)
        
        # Load feature names
        feat_names = load_feature_names(checkpoint_dir, split_idx, model_name)
        
        return model, feat_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def extract_feature_importance(model, model_name: str, feat_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract feature importance from trained model."""
    try:
        if model is None or feat_names is None:
            return None
        
        n_features = len(feat_names)
        
        if model_name == "Linear":
            # For linear models, use absolute coefficients
            if hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                if len(importance.shape) > 1:  # Multi-class case
                    importance = np.mean(np.abs(importance), axis=0)
            else:
                print(f"Warning: Linear model has no coefficients")
                return None
                
        elif model_name in ["RF", "XGB"]:
            # For tree-based models, use feature_importances_
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                print(f"Warning: {model_name} model has no feature_importances_")
                return None
        else:
            print(f"Warning: Unknown model type: {model_name}")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature_name': feat_names,
            'importance': importance,
            'model': model_name
        })
        
        # Add feature family
        importance_df['feature_family'] = importance_df['feature_name'].apply(
            lambda x: 'AB_Block' if x.startswith('AB_') else 
                     ('RDKit' if x.startswith('RD_') else 'Descriptors')
        )
        
        # Normalize importance to sum to 1 for better comparison
        importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
        
        return importance_df
        
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return None

def plot_top_features(importance_df: pd.DataFrame, dataset_name: str, target_name: str, 
                     model_name: str, output_dir: Path, top_n: int = 5) -> None:
    """Plot bar chart of top N features by importance."""
    if importance_df is None or importance_df.empty:
        print(f"No importance data to plot for {model_name}")
        return
    
    # Get top N features
    top_features = importance_df.nlargest(top_n, 'importance_normalized').sort_values('importance_normalized', ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Color by feature family
    colors = {'AB_Block': '#2E86AB', 'RDKit': '#A23B72', 'Descriptors': '#F18F01'}
    bar_colors = [colors.get(family, '#808080') for family in top_features['feature_family']]
    
    bars = plt.barh(range(len(top_features)), top_features['importance_normalized'], color=bar_colors, alpha=0.8, edgecolor='black')
    
    # Customize plot
    plt.xlabel('Normalized Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'{dataset_name.upper()} - {target_name}\nTop {top_n} Features - {model_name} Model', 
             fontsize=14, fontweight='bold')
    
    # Set y-axis labels
    plt.yticks(range(len(top_features)), 
               [f"{name}\n({family})" for name, family in zip(top_features['feature_name'], top_features['feature_family'])])
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance_normalized'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=10)
    
    # Add legend for feature families
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[fam], alpha=0.8, edgecolor='black', 
                                    label=fam.replace('_', ' ')) for fam in colors.keys() 
                      if fam in top_features['feature_family'].values]
    if legend_elements:
        plt.legend(handles=legend_elements, loc='lower right')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{dataset_name}_{target_name}_{model_name}_top_features.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot: {output_file}")
    plt.close()

def compare_feature_families(importance_data: Dict[str, pd.DataFrame], dataset_name: str, 
                            target_name: str, output_dir: Path) -> None:
    """Compare feature importance between families across all models."""
    if not importance_data:
        print("No importance data for family comparison")
        return
    
    # Combine data from all models
    all_importance = pd.concat(importance_data.values(), ignore_index=True)
    
    # Group by feature family and model
    family_importance = all_importance.groupby(['feature_family', 'model'])['importance_normalized'].sum().reset_index()
    
    # Create pivot table for plotting
    pivot_data = family_importance.pivot(index='feature_family', columns='model', values='importance_normalized').fillna(0)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot grouped bar chart
    x = np.arange(len(pivot_data.index))
    width = 0.25
    
    models = pivot_data.columns
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, model in enumerate(models):
        plt.bar(x + i*width, pivot_data[model], width, label=model, color=colors[i], alpha=0.8, edgecolor='black')
    
    plt.xlabel('Feature Family', fontsize=12)
    plt.ylabel('Total Normalized Importance', fontsize=12)
    plt.title(f'{dataset_name.upper()} - {target_name}\nFeature Family Importance Comparison', 
             fontsize=14, fontweight='bold')
    
    plt.xticks(x + width, pivot_data.index, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{dataset_name}_{target_name}_family_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved family comparison plot: {output_file}")
    plt.close()
    
    # Also save summary table
    summary_file = output_dir / f"{dataset_name}_{target_name}_family_importance_summary.csv"
    pivot_data.to_csv(summary_file)
    print(f"Saved family importance summary: {summary_file}")

def analyze_dataset_feature_importance(dataset_name: str, task_type: str, models_to_analyze: List[str] = None) -> None:
    """Analyze feature importance for a specific dataset."""
    
    if models_to_analyze is None:
        models_to_analyze = ["Linear", "RF", "XGB"]
    
    # Setup environment
    args = argparse.Namespace(
        dataset_name=dataset_name,
        task_type=task_type,
        polymer_type="homo"  # Default, will be updated from setup
    )
    
    setup_info = setup_training_environment(args, model_type="tabular")
    results_dir = setup_info['results_dir']
    
    # Find result directories for this dataset
    dataset_pattern = f"*{dataset_name}*"
    result_dirs = list(results_dir.glob(dataset_pattern))
    
    if not result_dirs:
        print(f"No results found for dataset: {dataset_name}")
        return
    
    print(f"Found {len(result_dirs)} result directories for {dataset_name}")
    
    # Create output directory
    output_dir = Path("/Users/u6788552/Desktop/experiments/dmpnn") / "plots" / "feature_importance" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each result directory
    for result_dir in result_dirs:
        print(f"\nAnalyzing: {result_dir.name}")
        
        # Extract configuration from directory name
        config_parts = result_dir.name.split('__')
        feature_config = config_parts[1] if len(config_parts) > 1 else "unknown"
        
        # Find targets (from CSV files)
        csv_files = list(result_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {result_dir}")
            continue
        
        # Load results to get targets
        results_df = pd.read_csv(csv_files[0])
        targets = results_df['target'].unique()
        
        print(f"Found targets: {targets}")
        
        # Analyze each target
        for target in targets:
            print(f"\nAnalyzing target: {target}")
            
            importance_data = {}
            
            # Analyze each model
            for model_name in models_to_analyze:
                print(f"  Processing model: {model_name}")
                
                # Try to load data from split 0 (representative)
                model, feat_names = load_model_and_data(result_dir, 0, model_name)
                
                if model is not None and feat_names is not None:
                    importance_df = extract_feature_importance(model, model_name, feat_names)
                    
                    if importance_df is not None:
                        importance_data[model_name] = importance_df
                        
                        # Plot top features for this model
                        plot_top_features(importance_df, dataset_name, target, model_name, 
                                        output_dir / feature_config, top_n=5)
                        
                        print(f"    Extracted importance for {len(importance_df)} features")
                    else:
                        print(f"    Failed to extract importance")
                else:
                    print(f"    Failed to load model or feature names")
            
            # Compare feature families across models
            if importance_data:
                compare_feature_families(importance_data, dataset_name, target, 
                                        output_dir / feature_config)

def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance from trained tabular models')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset to analyze')
    parser.add_argument('--task_type', type=str, choices=['reg', 'binary', 'multi'], default="reg",
                        help='Task type: "reg" (regression), "binary", or "multi" (multi-class)')
    parser.add_argument('--models', nargs='+', default=["Linear", "RF", "XGB"],
                        help='Models to analyze (default: Linear RF XGB)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("MODEL-FEATURE INTERACTION ANALYSIS")
    print("="*60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Task type: {args.task_type}")
    print(f"Models: {args.models}")
    print("="*60)
    
    # Run analysis
    analyze_dataset_feature_importance(args.dataset_name, args.task_type, args.models)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: plots/feature_importance/{args.dataset_name}/")

if __name__ == "__main__":
    main()
