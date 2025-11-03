# Dataset Analysis Configuration

This directory contains configuration files and scripts for analyzing target value distributions across datasets.

## Files

### `dataset_config.yaml`
Configuration file that specifies:
- **Datasets**: List of datasets to analyze with their file paths
- **Target columns**: Specific target columns for each dataset
- **Settings**: 
  - `plot`: Plot configuration options (bins, figure size, DPI, etc.)
  - `filter`: Data filtering criteria (min unique values, CV threshold)
  - `summary`: Summary table configuration (format, creation flag)

### `plot_target_histograms.py`
Script that reads the YAML configuration and generates:
- Combined histogram plots for each dataset
- Individual histogram plots for each target
- Summary statistics table

### `plot_correlation_heatmaps.py`
Script that reads the YAML configuration and generates:
- Pearson correlation heatmaps for datasets with multiple targets
- P-value heatmaps showing statistical significance
- Summary table of all significant correlations across datasets

### `feature_space_analysis.py`
Script that performs comprehensive feature space analysis for tabular models:
- Feature family summary tables (AB block, RDKit, descriptors)
- Feature variance histograms after preprocessing
- PCA 2D scatter plots colored by target values
- UMAP 2D embedding plots colored by target values
- Top 5 feature-target correlation bar charts

## Usage

### 1. Configure datasets
Edit `dataset_config.yaml` to specify which datasets and targets to analyze:

```yaml
datasets:
  your_dataset:
    file_path: "data/your_data.csv"
    targets:
      - "target_column_1"
      - "target_column_2"
    description: "Your dataset description"
```

### 2. Run analysis

**Target Histograms:**
```bash
cd /Users/u6788552/Desktop/experiments/dmpnn
python3 analysis/plot_target_histograms.py
```

**Correlation Heatmaps:**
```bash
cd /Users/u6788552/Desktop/experiments/dmpnn
python3 analysis/plot_correlation_heatmaps.py
```

**Feature Space Analysis:**
```bash
cd /Users/u6788552/Desktop/experiments/dmpnn
python3 analysis/feature_space_analysis.py
```

### 3. View results

**Histogram results** are saved in `plots/target_histograms/`:
- `{dataset}_target_histograms.png` - Combined plots
- `{dataset}_{target}_histogram.png` - Individual plots
- `target_summary_table.csv` - Statistics summary

**Correlation results** are saved in `plots/correlation_heatmaps/`:
- `{dataset}_correlation_heatmap.png` - Pearson correlation heatmaps
- `{dataset}_pvalues_heatmap.png` - P-value significance heatmaps
- `correlation_summary.csv` - All significant correlations

**Feature space analysis results** are saved in `plots/feature_space_analysis/`:
- `{dataset}_feature_family_summary.csv` - Feature count breakdown
- `{dataset}_{target}_feature_variance_histogram.png` - Variance distributions
- `{dataset}_{target}_pca_scatter.png` - PCA 2D projections
- `{dataset}_{target}_umap_embedding.png` - UMAP embeddings
- `{dataset}_{target}_top_correlations.png` - Top 5 feature correlations
- `top_feature_correlations_summary.csv` - Overall correlation summary

## Configuration Options

### Dataset Configuration
- `file_path`: Relative path to dataset CSV file
- `targets`: List of target column names
- `description`: Optional description

### Plot Settings
- `max_cols_per_row`: Maximum subplots per row in combined plots
- `default_bins`: Default number of histogram bins
- `subplot_size`: Figure size for each subplot [width, height]
- `dpi`: Resolution for saved figures
- `create_individual_plots`: Whether to generate individual plots
- `show_statistics`: Whether to show mean/median lines

### Filter Settings
- `min_unique_values`: Minimum unique values required for target columns
- `max_cv_threshold`: Maximum coefficient of variation before flagging

### Summary Settings
- `create_summary`: Whether to generate summary table
- `format`: File format for summary table (csv, tsv, etc.)

**Note**: Each script manages its own output directory. The `plot_target_histograms.py` script uses `plots/target_histograms/` by default.

## Example Output

### Target Histograms
The histogram script generates:
1. **Combined plots**: All targets for a dataset in one figure
2. **Individual plots**: Detailed plots for each target with statistics
3. **Summary table**: Comprehensive statistics across all datasets

Each plot includes:
- Histogram with appropriate binning
- Statistical information (mean, std, range)
- Scientific notation for large/small values
- Professional styling and formatting

### Correlation Heatmaps
The correlation script generates:
1. **Correlation heatmaps**: Pearson correlation coefficients between target pairs
2. **P-value heatmaps**: Statistical significance of correlations
3. **Summary table**: All significant correlations with effect sizes

Key features:
- Only processes datasets with 2+ targets
- Shows correlation coefficients with significance stars
- Identifies strong correlations (|r| > 0.5)
- Provides comprehensive statistical summary
- Professional color schemes (coolwarm for correlations, RdYlBu_r for p-values)

**Note**: Each script manages its own output directory. The `plot_target_histograms.py` script uses `plots/target_histograms/` by default, while `plot_correlation_heatmaps.py` uses `plots/correlation_heatmaps/`.
