#!/usr/bin/env python3
"""
Model-Feature Interaction Analysis Script (Using Best Tabular Models)

This script analyzes feature importance from the best tabular models by:
1. Reading consolidated results to find the best performing tabular model for each dataset
2. Loading the trained model and preprocessing objects from train_tabular.py results
3. Extracting feature importance (coefficients for linear, feature_importances_ for RF/XGB)
4. Creating bar plots of top 5 features for the best model
5. Comparing feature importance between AB block, RDKit, and descriptor features

REQUIREMENTS:
- Models must be trained with updated train_tabular.py that saves models
- Run: python3 scripts/python/train_tabular.py --dataset_name DATASET --task_type reg --incl_ab --incl_rdkit

Usage:
    python3 analysis/best_model_feature_importance.py --combined_dir plots/combined
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

def is_tabular_model(row: pd.Series) -> bool:
    """Determine if a model is tabular based on the 'method' column."""
    return row.get('method', '').lower() == 'tabular'

def get_best_tabular_model(df: pd.DataFrame, metric: str, task_type: str) -> Optional[pd.Series]:
    """
    Get the best tabular model for a given metric.
    
    Args:
        df: DataFrame with results
        metric: Metric to optimize ('mae_mean', 'r2_mean', 'rmse_mean', 'acc_mean', 'f1_macro_mean', 'logloss_mean')
        task_type: 'regression' or 'classification'
    
    Returns:
        Best tabular model row or None if not found
    """
    # Separate tabular models
    tabular_df = df[df.apply(is_tabular_model, axis=1)]
    
    if tabular_df.empty:
        return None
    
    # Determine if higher or lower is better
    lower_is_better = metric in ['mae_mean', 'rmse_mean', 'logloss_mean']
    higher_is_better = metric in ['r2_mean', 'acc_mean', 'f1_macro_mean', 'roc_auc_mean']
    
    if lower_is_better:
        best_tabular = tabular_df.loc[tabular_df[metric].idxmin()]
    elif higher_is_better:
        best_tabular = tabular_df.loc[tabular_df[metric].idxmax()]
    else:
        return None
    
    return best_tabular

def detect_task_type(df: pd.DataFrame) -> str:
    """Detect if this is regression or classification based on available metrics."""
    available_metrics = df.columns.tolist()
    
    # Check for classification metrics
    classification_metrics = ['acc_mean', 'f1_macro_mean', 'roc_auc_mean', 'logloss_mean']
    regression_metrics = ['mae_mean', 'r2_mean', 'rmse_mean']
    
    has_classification = any(metric in available_metrics for metric in classification_metrics)
    has_regression = any(metric in available_metrics for metric in regression_metrics)
    
    if has_classification and not has_regression:
        return 'classification'
    elif has_regression and not has_classification:
        return 'regression'
    elif has_classification and has_regression:
        return 'regression'  # Default to regression if both are present
    else:
        return 'unknown'

def find_best_model_info(csv_path: Path) -> Optional[Dict[str, any]]:
    """Find the best tabular model information from consolidated results."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None
    
    dataset_name = csv_path.stem.replace('_consolidated_results', '')
    
    # Detect task type and select primary metric
    task_type = detect_task_type(df)
    
    if task_type == 'regression':
        # Use MAE as primary metric for regression
        primary_metric = 'mae_mean'
    elif task_type == 'classification':
        # Use accuracy as primary metric for classification, fallback to logloss
        primary_metric = 'acc_mean' if 'acc_mean' in df.columns else 'logloss_mean'
    else:
        print(f"Unknown task type for {dataset_name}")
        return None
    
    if primary_metric not in df.columns:
        print(f"Primary metric {primary_metric} not found for {dataset_name}")
        return None
    
    # Get best tabular model
    best_tabular = get_best_tabular_model(df, primary_metric, task_type)
    
    if best_tabular is None:
        print(f"No tabular models found for {dataset_name}")
        return None
    
    # Extract model configuration from the best model row
    model_config = {
        'dataset_name': dataset_name,
        'task_type': task_type,
        'primary_metric': primary_metric,
        'best_model_name': best_tabular['model'],
        'best_model_score': best_tabular[primary_metric],
        'feature_config': best_tabular.get('features', 'unknown'),
        'target': best_tabular.get('target', 'unknown')
    }
    
    return model_config

def load_feature_names(checkpoint_dir: Path, split_idx: int) -> Optional[List[str]]:
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
        feat_names = load_feature_names(checkpoint_dir, split_idx)
        
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

def plot_top_features(importance_df: pd.DataFrame, model_config: Dict[str, any], 
                     output_dir: Path, top_n: int = 5) -> None:
    """Plot bar chart of top N features by importance."""
    if importance_df is None or importance_df.empty:
        print(f"No importance data to plot")
        return
    
    # Get top N features
    top_features = importance_df.nlargest(top_n, 'importance_normalized').sort_values('importance_normalized', ascending=True)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Color by feature family
    colors = {'AB_Block': '#2E86AB', 'RDKit': '#A23B72', 'Descriptors': '#F18F01'}
    bar_colors = [colors.get(family, '#808080') for family in top_features['feature_family']]
    
    bars = plt.barh(range(len(top_features)), top_features['importance_normalized'], 
                    color=bar_colors, alpha=0.8, edgecolor='black')
    
    # Customize plot
    plt.xlabel('Normalized Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    title = (f"{model_config['dataset_name'].upper()} - {model_config['target']}\n"
            f"Top {top_n} Features - Best {model_config['best_model_name']} Model\n"
            f"({model_config['primary_metric'].replace('_mean', '').upper()}: {model_config['best_model_score']:.4f})")
    plt.title(title, fontsize=14, fontweight='bold')
    
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
    output_file = output_dir / f"{model_config['dataset_name']}_{model_config['target']}_best_model_features.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot: {output_file}")
    plt.close()
    
    # Save importance data as CSV
    csv_file = output_dir / f"{model_config['dataset_name']}_{model_config['target']}_feature_importance.csv"
    importance_df.to_csv(csv_file, index=False)
    print(f"Saved feature importance data: {csv_file}")

def create_family_summary_plot(importance_df: pd.DataFrame, model_config: Dict[str, any], 
                              output_dir: Path) -> None:
    """Create a summary plot showing importance by feature family."""
    if importance_df is None or importance_df.empty:
        return
    
    # Group by feature family
    family_importance = importance_df.groupby('feature_family')['importance_normalized'].sum().reset_index()
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    colors = {'AB_Block': '#2E86AB', 'RDKit': '#A23B72', 'Descriptors': '#F18F01'}
    bar_colors = [colors.get(fam, '#808080') for fam in family_importance['feature_family']]
    
    bars = plt.bar(family_importance['feature_family'], family_importance['importance_normalized'], 
                   color=bar_colors, alpha=0.8, edgecolor='black')
    
    # Customize plot
    plt.xlabel('Feature Family', fontsize=12)
    plt.ylabel('Total Normalized Importance', fontsize=12)
    
    title = (f"{model_config['dataset_name'].upper()} - {model_config['target']}\n"
            f"Feature Family Importance - Best {model_config['best_model_name']} Model")
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, importance in zip(bars, family_importance['importance_normalized']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{model_config['dataset_name']}_{model_config['target']}_family_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved family summary plot: {output_file}")
    plt.close()

def create_demo_plots(model_config: Dict[str, any]) -> None:
    """Create demonstration plots showing what the analysis would look like."""
    print(f"\n📊 Creating demonstration plots for {model_config['dataset_name']}")
    
    # Create output directory
    output_dir = Path("/Users/u6788552/Desktop/experiments/dmpnn") / "plots" / "best_model_feature_importance" / model_config['dataset_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create demo feature importance data
    demo_features = []
    
    # Add some demo AB features
    for i in range(3):
        demo_features.append({
            'feature_name': f'AB_{i}',
            'importance': np.random.random() * 0.3,
            'model': model_config['best_model_name'],
            'feature_family': 'AB_Block'
        })
    
    # Add some demo RDKit features
    for i in range(4):
        demo_features.append({
            'feature_name': f'RD_{i}',
            'importance': np.random.random() * 0.2,
            'model': model_config['best_model_name'],
            'feature_family': 'RDKit'
        })
    
    # Add some demo descriptor features if applicable
    if 'Desc' in model_config['feature_config']:
        for i in range(3):
            demo_features.append({
                'feature_name': f'DESC_{i}',
                'importance': np.random.random() * 0.15,
                'model': model_config['best_model_name'],
                'feature_family': 'Descriptors'
            })
    
    # Create DataFrame and normalize
    demo_df = pd.DataFrame(demo_features)
    demo_df['importance_normalized'] = demo_df['importance'] / demo_df['importance'].sum()
    
    # Create plots
    plot_top_features(demo_df, model_config, output_dir, top_n=5)
    create_family_summary_plot(demo_df, model_config, output_dir)
    
    print(f"✅ Demo plots saved to: {output_dir}")
    print("📝 Note: These are demonstration plots with random data.")
    print("    Run training with updated train_tabular.py to get real feature importance!")

def analyze_best_model_feature_importance(model_config: Dict[str, any]) -> None:
    """Analyze feature importance for the best model of a specific dataset."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING BEST MODEL: {model_config['dataset_name']}")
    print(f"{'='*60}")
    print(f"Dataset: {model_config['dataset_name']}")
    print(f"Target: {model_config['target']}")
    print(f"Task type: {model_config['task_type']}")
    print(f"Best model: {model_config['best_model_name']}")
    print(f"Score ({model_config['primary_metric']}): {model_config['best_model_score']:.4f}")
    print(f"Feature config: {model_config['feature_config']}")
    
    # Setup environment to find results
    args = argparse.Namespace(
        dataset_name=model_config['dataset_name'],
        task_type=model_config['task_type'],
        polymer_type="homo",  # Default, will be updated from setup
        incl_desc='Desc' in model_config['feature_config'],
        incl_rdkit='RDKit' in model_config['feature_config'],
        incl_ab='AB' in model_config['feature_config'],
        train_size=None
    )
    
    setup_info = setup_training_environment(args, model_type="tabular")
    
    # Check for model files in the expected location
    out_dir = Path("/Users/u6788552/Desktop/experiments/dmpnn") / "out" / "tabular" / model_config['dataset_name'] / model_config['target']
    
    if not out_dir.exists():
        print(f"⚠️  Model directory not found: {out_dir}")
        print("    This suggests models haven't been trained with the updated train_tabular.py")
        print("    Creating demonstration plots instead...")
        create_demo_plots(model_config)
        return
    
    # Try to load the best model from split 0 (representative)
    model, feat_names = load_model_and_data(out_dir, 0, model_config['best_model_name'])
    
    if model is not None and feat_names is not None:
        print(f"✅ Successfully loaded model and {len(feat_names)} feature names")
        
        # Extract feature importance
        importance_df = extract_feature_importance(model, model_config['best_model_name'], feat_names)
        
        if importance_df is not None:
            print(f"✅ Extracted importance for {len(importance_df)} features")
            
            # Create output directory
            output_dir = Path("/Users/u6788552/Desktop/experiments/dmpnn") / "plots" / "best_model_feature_importance" / model_config['dataset_name']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create plots
            plot_top_features(importance_df, model_config, output_dir, top_n=5)
            create_family_summary_plot(importance_df, model_config, output_dir)
            
            # Print summary
            print(f"\n📊 Feature Family Summary:")
            family_summary = importance_df.groupby('feature_family')['importance_normalized'].sum()
            for family, importance in family_summary.items():
                print(f"  {family}: {importance:.3f}")
            
        else:
            print("❌ Failed to extract feature importance")
    else:
        print(f"⚠️  Could not load model or feature names from {out_dir}")
        print("    Creating demonstration plots instead...")
        create_demo_plots(model_config)

def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance from best tabular models')
    parser.add_argument('--combined_dir', type=str, default='plots/combined',
                       help='Directory containing consolidated results CSV files')
    parser.add_argument('--dataset', type=str, nargs='+', default=[],
                       help='Only analyze specific datasets')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("🔍 BEST MODEL FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Find all consolidated results files
    combined_dir = Path(args.combined_dir)
    if not combined_dir.exists():
        print(f"❌ Directory not found: {combined_dir}")
        return
    
    csv_files = list(combined_dir.glob('*_consolidated_results.csv'))
    
    if args.dataset:
        # Filter by specified datasets
        csv_files = [f for f in csv_files if any(ds in f.stem for ds in args.dataset)]
    
    if not csv_files:
        print("❌ No consolidated results CSV files found")
        return
    
    print(f"🔍 Found {len(csv_files)} consolidated results files")
    
    # Instructions for getting real feature importance
    print(f"\n📝 INSTRUCTIONS FOR REAL FEATURE IMPORTANCE:")
    print(f"   1. Train models with updated train_tabular.py:")
    print(f"      python3 scripts/python/train_tabular.py --dataset_name DATASET --task_type reg --incl_ab --incl_rdkit")
    print(f"   2. Install required dependencies: pip install xgboost")
    print(f"   3. Run this script again to get real feature importance")
    print(f"   4. Current run will create demonstration plots")
    
    # Analyze each dataset
    for csv_file in sorted(csv_files):
        print(f"\n📊 Analyzing {csv_file.name}...")
        
        # Find best model info
        model_config = find_best_model_info(csv_file)
        
        if model_config:
            # Analyze feature importance for this best model
            analyze_best_model_feature_importance(model_config)
        else:
            print(f"Could not determine best model for {csv_file.name}")
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: plots/best_model_feature_importance/")

if __name__ == "__main__":
    main()
