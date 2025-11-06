#!/usr/bin/env python3
"""
Compare Tabular vs Graph Model Performance

This script reads all consolidated_results.csv files in plots/combined,
separates results into tabular and graph categories, finds the best model
in each category, and prints a summary of performance differences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patheffects
import seaborn as sns

def is_tabular_model(row: pd.Series) -> bool:
    """Determine if a model is tabular based on the 'method' column."""
    return row.get('method', '').lower() == 'tabular'

def get_best_models(df: pd.DataFrame, metric: str, task_type: str) -> Tuple[pd.Series, pd.Series]:
    """
    Get the best tabular and graph models for a given metric.
    
    Args:
        df: DataFrame with results
        metric: Metric to optimize ('mae_mean', 'r2_mean', 'rmse_mean', 'acc_mean', 'f1_macro_mean', 'logloss_mean')
        task_type: 'regression' or 'classification'
    
    Returns:
        Tuple of (best_tabular_row, best_graph_row)
    """
    # Separate tabular and graph models
    tabular_df = df[df.apply(is_tabular_model, axis=1)]
    graph_df = df[~df.apply(is_tabular_model, axis=1)]
    
    if tabular_df.empty or graph_df.empty:
        return None, None
    
    # Determine if higher or lower is better
    lower_is_better = metric in ['mae_mean', 'rmse_mean', 'logloss_mean']
    higher_is_better = metric in ['r2_mean', 'acc_mean', 'f1_macro_mean', 'roc_auc_mean']
    
    best_tabular = None
    best_graph = None
    
    if lower_is_better:
        best_tabular = tabular_df.loc[tabular_df[metric].idxmin()]
        best_graph = graph_df.loc[graph_df[metric].idxmin()]
    elif higher_is_better:
        best_tabular = tabular_df.loc[tabular_df[metric].idxmax()]
        best_graph = graph_df.loc[graph_df[metric].idxmax()]
    
    return best_tabular, best_graph

def calculate_improvement(best_tabular: pd.Series, best_graph: pd.Series, metric: str) -> Dict[str, Any]:
    """Calculate performance improvement between best models."""
    if best_tabular is None or best_graph is None:
        return {}
    
    tabular_val = best_tabular[metric]
    graph_val = best_graph[metric]
    
    # Calculate percentage improvement
    lower_is_better = metric in ['mae_mean', 'rmse_mean', 'logloss_mean']
    
    if lower_is_better:
        # For metrics where lower is better, improvement is (tabular - graph) / tabular
        improvement_pct = ((tabular_val - graph_val) / tabular_val) * 100
        better_model = 'graph' if graph_val < tabular_val else 'tabular'
    else:
        # For metrics where higher is better, improvement is (graph - tabular) / tabular
        improvement_pct = ((graph_val - tabular_val) / tabular_val) * 100
        better_model = 'graph' if graph_val > tabular_val else 'tabular'
    
    return {
        'tabular_value': tabular_val,
        'graph_value': graph_val,
        'improvement_pct': improvement_pct,
        'better_model': better_model,
        'tabular_model': best_tabular['model'],
        'graph_model': best_graph['model']
    }

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
        # Default to regression if both are present
        return 'regression'
    else:
        return 'unknown'

def analyze_dataset(csv_path: Path) -> Dict[str, Any]:
    """Analyze a single dataset's consolidated results."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return {}
    
    dataset_name = csv_path.stem.replace('_consolidated_results', '')
    
    # Detect task type and relevant metrics
    task_type = detect_task_type(df)
    
    if task_type == 'regression':
        metrics = ['mae_mean', 'r2_mean', 'rmse_mean']
    elif task_type == 'classification':
        # Use both logloss and ROC-AUC where available
        metrics = ['acc_mean', 'f1_macro_mean']
        if 'logloss_mean' in df.columns:
            metrics.append('logloss_mean')
        if 'roc_auc_mean' in df.columns:
            metrics.append('roc_auc_mean')
    else:
        print(f"Unknown task type for {dataset_name}")
        return {}
    
    results = {
        'dataset': dataset_name,
        'task_type': task_type,
        'total_models': len(df),
        'tabular_models': len(df[df.apply(is_tabular_model, axis=1)]),
        'graph_models': len(df[~df.apply(is_tabular_model, axis=1)]),
        'comparisons': {}
    }
    
    # Compare for each metric
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Check if we have valid (non-NaN) values for both categories
        tabular_valid = df[df.apply(is_tabular_model, axis=1)][metric].notna().any()
        graph_valid = df[~df.apply(is_tabular_model, axis=1)][metric].notna().any()
        
        if not tabular_valid or not graph_valid:
            print(f"  ⚠️  Skipping {metric}: missing data for one category")
            continue
        
        # Use clean metric name for display (remove _mean suffix)
        clean_metric = metric.replace('_mean', '')
        
        best_tabular, best_graph = get_best_models(df, metric, task_type)
        if best_tabular is not None and best_graph is not None:
            improvement = calculate_improvement(best_tabular, best_graph, metric)
            results['comparisons'][clean_metric] = improvement
    
    return results

def print_summary(all_results: List[Dict[str, Any]]):
    """Print a comprehensive summary of all comparisons."""
    print("\n" + "="*80)
    print("📊 TABULAR vs GRAPH MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    for result in all_results:
        if not result:
            continue
            
        dataset = result['dataset']
        task_type = result['task_type']
        
        print(f"\n🎯 Dataset: {dataset} ({task_type})")
        print(f"   Total models: {result['total_models']} "
              f"(Tabular: {result['tabular_models']}, Graph: {result['graph_models']})")
        
        if not result['comparisons']:
            print("   ⚠️  No valid comparisons found")
            continue
        
        print("   📈 Performance Comparison:")
        
        for metric, comp in result['comparisons'].items():
            better = comp['better_model']
            improvement = comp['improvement_pct']
            
            if better == 'graph':
                print(f"      {metric.upper()}: ✅ Graph models perform {improvement:+.2f}% better")
                print(f"         Best Graph: {comp['graph_model']} ({comp['graph_value']:.4f})")
                print(f"         Best Tabular: {comp['tabular_model']} ({comp['tabular_value']:.4f})")
            else:
                print(f"      {metric.upper()}: ⚠️  Tabular models perform {abs(improvement):+.2f}% better")
                print(f"         Best Tabular: {comp['tabular_model']} ({comp['tabular_value']:.4f})")
                print(f"         Best Graph: {comp['graph_model']} ({comp['graph_value']:.4f})")
    
    # Overall summary
    print(f"\n📋 OVERALL SUMMARY")
    print("-" * 40)
    
    total_comparisons = sum(len(r['comparisons']) for r in all_results if r)
    
    if total_comparisons == 0:
        print("No valid comparisons found")
        return
    
    graph_wins = 0
    tabular_wins = 0
    
    for result in all_results:
        if not result or not result['comparisons']:
            continue
        for comp in result['comparisons'].values():
            if comp['better_model'] == 'graph':
                graph_wins += 1
            else:
                tabular_wins += 1
    
    print(f"Total comparisons: {total_comparisons}")
    print(f"Graph models win: {graph_wins} ({graph_wins/total_comparisons*100:.1f}%)")
    print(f"Tabular models win: {tabular_wins} ({tabular_wins/total_comparisons*100:.1f}%)")
    
    if graph_wins > tabular_wins:
        print("🏆 Overall winner: GRAPH models")
    elif tabular_wins > graph_wins:
        print("🏆 Overall winner: TABULAR models")
    else:
        print("🏆 Overall winner: TIE")

def create_improvement_bar_plot(all_results: List[Dict[str, Any]], output_path: str = 'graph_vs_tabular_improvement.png'):
    """Create a bar plot showing % improvement of graph over tabular models."""
    
    # Define the desired dataset order
    dataset_order = ['tc', 'insulator', 'htpmd', 'polyinfo', 'camb3lyp']
    
    # Map opv_camb3lyp to camb3lyp for display
    dataset_mapping = {'opv_camb3lyp': 'camb3lyp'}
    
    # Collect improvement data
    plot_data = []
    
    for result in all_results:
        if not result or not result['comparisons']:
            continue
            
        dataset = result['dataset']
        task_type = result['task_type']
        
        # Apply dataset mapping for display
        display_dataset = dataset_mapping.get(dataset, dataset)
        
        # Determine which metric to use based on task type
        if task_type == 'classification' and dataset == 'polyinfo':
            # Use accuracy for polyinfo (classification)
            metric_key = 'acc'
        else:
            # Use RMSE for regression tasks
            metric_key = 'rmse'
        
        # Find the appropriate comparison
        if metric_key in result['comparisons']:
            comp = result['comparisons'][metric_key]
            improvement_pct = comp['improvement_pct']
            
            # For graph improvement, positive values mean graph is better
            if comp['better_model'] == 'tabular':
                improvement_pct = -abs(improvement_pct)  # Make negative if tabular is better
            
            plot_data.append({
                'dataset': display_dataset,
                'improvement_pct': improvement_pct,
                'task_type': task_type,
                'metric': metric_key.upper()
            })
    
    if not plot_data:
        print("❌ No data available for plotting")
        return
    
    # Convert to DataFrame and sort by desired order
    df_plot = pd.DataFrame(plot_data)
    
    # Filter to only include datasets in our desired order that we have data for
    available_datasets = df_plot['dataset'].unique()
    ordered_datasets = [ds for ds in dataset_order if ds in available_datasets]
    
    # Filter and reorder
    df_plot = df_plot[df_plot['dataset'].isin(ordered_datasets)]
    df_plot['dataset'] = pd.Categorical(df_plot['dataset'], categories=ordered_datasets, ordered=True)
    df_plot = df_plot.sort_values('dataset')
    
    # Create the plot with modern styling
    plt.style.use('default')  # Use default style to preserve colors
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Add custom grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#E0E0E0')
    ax.set_axisbelow(True)
    
    # Define a beautiful color palette
    colors = []
    for x in df_plot['improvement_pct']:
        if x >= 30:
            colors.append('#2E8B57')  # Sea Green for high improvement
        elif x >= 15:
            colors.append('#32CD32')  # Lime Green for good improvement
        elif x >= 5:
            colors.append('#90EE90')  # Light Green for moderate improvement
        elif x >= 0:
            colors.append('#98FB98')  # Pale Green for small improvement
        else:
            colors.append('#FF6B6B')  # Coral Red for tabular better
    
    # Create gradient bars with rounded edges
    bars = ax.bar(df_plot['dataset'], df_plot['improvement_pct'], 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=2,
                  capsize=5, width=0.6)
    
    # Add subtle shadow effect
    for bar in bars:
        bar.set_path_effects([plt.matplotlib.patheffects.SimplePatchShadow(offset=(1, -1), 
                                                                           shadow_rgbFace='gray', alpha=0.3)])
    
    # Customize the plot with modern styling
    ax.axhline(y=0, color='#2C3E50', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold', color='#2C3E50', labelpad=15)
    ax.set_ylabel('% Improvement of Graph over Tabular Models', fontsize=14, fontweight='bold', color='#2C3E50', labelpad=15)
    ax.set_title('🚀 Graph vs Tabular Model Performance Comparison', 
                fontsize=18, fontweight='bold', color='#2C3E50', pad=25)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    ax.tick_params(colors='#2C3E50', labelsize=12)
    
    # Add value labels on bars with better styling
    for bar, value in zip(bars, df_plot['improvement_pct']):
        height = bar.get_height()
        label_y = height + (2 if height >= 0 else -4)
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=12, color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add metric information with better styling
    for i, (dataset, metric) in enumerate(zip(df_plot['dataset'], df_plot['metric'])):
        ax.text(i, min(df_plot['improvement_pct']) - 8, f'({metric})', 
                ha='center', va='top', fontsize=11, style='italic', color='#7F8C8D',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ECF0F1', alpha=0.7, edgecolor='none'))
    
    # Add a subtle background gradient
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Bar plot saved to: {output_path}")
    
    # Show summary
    print(f"\n📊 IMPROVEMENT SUMMARY:")
    for _, row in df_plot.iterrows():
        status = "✅ Graph better" if row['improvement_pct'] >= 0 else "⚠️ Tabular better"
        print(f"   {row['dataset']}: {row['improvement_pct']:+.1f}% ({row['metric']}) - {status}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare tabular vs graph model performance')
    parser.add_argument('--combined_dir', type=str, default='plots/combined',
                       help='Directory containing consolidated results CSV files')
    parser.add_argument('--dataset', type=str, nargs='+', default=[],
                       help='Only analyze specific datasets')
    parser.add_argument('--plot', action='store_true',
                       help='Create bar plot showing improvement percentages')
    parser.add_argument('--output', type=str, default='graph_vs_tabular_improvement.png',
                       help='Output path for the bar plot')
    args = parser.parse_args()
    
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
    
    # Analyze each dataset
    all_results = []
    for csv_file in sorted(csv_files):
        print(f"📊 Analyzing {csv_file.name}...")
        result = analyze_dataset(csv_file)
        if result:
            all_results.append(result)
    
    # Print summary
    print_summary(all_results)
    
    # Create bar plot if requested
    if args.plot:
        create_improvement_bar_plot(all_results, args.output)

if __name__ == '__main__':
    main()
