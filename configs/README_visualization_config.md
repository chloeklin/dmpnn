# Visualization Configuration Guide

This guide explains how to use the YAML configuration system for the visualization script, making it easy to add new models without editing code.

## Overview

The `visualization_config.yaml` file controls:
- **Model discovery**: Which directories to scan and what file patterns to look for
- **Feature ordering**: How features appear in plots
- **Color schemes**: Colors for different model types
- **Plot settings**: Figure sizes, fonts, and metric configurations

## Quick Start: Adding a New Model

### Example: Adding a new "Graphormer" model

1. **Train your model** and save results to `results/Graphormer/`
2. **No code changes needed!** The visualization script will automatically:
   - Discover the `Graphormer` directory
   - Load any `*_results.csv` files
   - Include them in plots

3. **(Optional) Customize display order** by editing `visualization_config.yaml`:

```yaml
feature_order:
  - group: "Graph Models"
    features:
      - Graph
      - Graph+RDKit
      # ... existing features ...
      - Graphormer              # Add here for specific position
      - Graphormer+RDKit
      - Graphormer+Desc+RDKit
```

4. **(Optional) Set custom colors** in `visualization_config.yaml`:

```yaml
color_scheme:
  Graph:
    Graphormer: "#FF6B6B"  # Custom red color
```

That's it! Run the visualization and your new model will appear.

## Configuration File Structure

### 1. Model Discovery (`model_discovery`)

Controls how the script finds result files:

```yaml
model_discovery:
  exclude_dirs:
    - tabular        # Directories to skip globally
    - embeddings
  
  methods:
    Graph:
      suffix: "_results.csv"      # File pattern to match
      auto_discover: true          # Scan all directories
      exclude_dirs: ["tabular", "IdentityBaseline"]
    
    IdentityBaseline:
      suffix: "_results.csv"
      auto_discover: false         # Use specific directories only
      directories: ["IdentityBaseline"]
```

**Key fields:**
- `suffix`: File pattern to match (e.g., `_results.csv`, `_baseline.csv`)
- `auto_discover`: If `true`, scans all directories; if `false`, uses `directories` list
- `exclude_dirs`: Directories to skip when auto-discovering
- `directories`: Specific directories to use (when `auto_discover: false`)

### 2. Feature Order (`feature_order`)

Controls the order features appear in plots:

```yaml
feature_order:
  - group: "Tabular"
    features:
      - AB
      - RDKit
      - Desc
  
  - group: "Graph Models"
    pattern: "Graph \\((mix|interact)\\)"  # Regex pattern for dynamic features
```

**Two ways to specify features:**

1. **Explicit list** (`features`): List exact feature names
2. **Pattern matching** (`pattern`): Use regex to match dynamic features (e.g., copolymer modes)

**Important:** Features not listed in the config will be automatically appended at the end in alphabetical order.

### 3. Color Scheme (`color_scheme`)

Define colors for different model types:

```yaml
color_scheme:
  Graph:
    DMPNN: "#004488"
    GAT: "#994455"
    GIN: "#997700"
    # Add your model here:
    YourModel: "#FF6B6B"
```

Uses Paul Tol's colorblind-friendly palette by default.

### 4. Marker Styles (`marker_styles`)

Set marker shapes for different models:

```yaml
marker_styles:
  Graph_DMPNN: "o"      # Circle
  Graph_GAT: "p"        # Pentagon
  Graph_GIN: "h"        # Hexagon
  YourModel: "D"        # Diamond
```

### 5. Plot Settings (`plot_settings`)

Configure plot appearance and metrics:

```yaml
plot_settings:
  figure_size: [14, 10]
  dpi: 300
  font_size: 10
  
  metrics:
    regression: [mae, rmse, r2]
    classification: [acc, f1_macro, logloss, roc_auc]
  
  higher_is_better: [r2, acc, f1_macro, roc_auc]
  lower_is_better: [mae, rmse, mse, logloss]
```

## Common Use Cases

### Adding a New Graph Model

**Scenario:** You've implemented a new GNN architecture called "SuperGNN"

**Steps:**
1. Save results to `results/SuperGNN/dataset_results.csv`
2. (Optional) Add to config for custom ordering:

```yaml
feature_order:
  - group: "Graph Models"
    features:
      - Graph
      - SuperGNN        # Add here
      - SuperGNN+RDKit
```

3. (Optional) Set custom color:

```yaml
color_scheme:
  Graph:
    SuperGNN: "#9B59B6"  # Purple
```

### Adding a New Baseline Type

**Scenario:** You want to add "Transformer" baseline results

**Steps:**
1. Save results to `results/TransformerBaseline/dataset_baseline.csv`
2. Add to config:

```yaml
model_discovery:
  methods:
    TransformerBaseline:
      suffix: "_baseline.csv"
      auto_discover: false
      directories: ["TransformerBaseline"]

feature_order:
  - group: "Transformer Baseline"
    features:
      - Baseline_Transformer
      - Baseline_Transformer+RDKit
```

### Adding Copolymer Mode Variants

**Scenario:** You have new copolymer modes like "block" and "random"

**No changes needed!** The pattern matching will automatically include them:

```yaml
feature_order:
  - group: "Copolymer Variants"
    pattern: "Graph \\((mix|interact|block|random)\\)"
```

Or add to the pattern:
```yaml
pattern: "Graph \\((mix|interact|mix_meta|interact_meta|block|random)\\)"
```

### Excluding a Model from Plots

**Scenario:** You want to temporarily hide a model from visualizations

**Option 1:** Add to exclude list:
```yaml
model_discovery:
  exclude_dirs:
    - tabular
    - OldModel        # Add here
```

**Option 2:** Remove from feature order (it will still be discovered but appear at the end)

## File Naming Conventions

The script expects specific file naming patterns:

### Graph Models
- Format: `dataset_results.csv` or `dataset__copoly_mode_results.csv`
- Examples:
  - `htpmd_results.csv`
  - `block__copoly_mix_results.csv`
  - `ea_ip__copoly_interact_meta_results.csv`

### Baseline Models
- Format: `dataset_baseline.csv`
- Examples:
  - `htpmd_baseline.csv`
  - `opv_camb3lyp__desc__rdkit_baseline.csv`

### IdentityBaseline
- Format: `dataset__identity_mode_results.csv`
- Examples:
  - `block__identity_mix_results.csv`
  - `block__identity_interact_results.csv`

### Tabular Models
- Format: `dataset_descriptors_features.csv`
- Examples:
  - `htpmd_descriptors_ab.csv`
  - `opv_descriptors_rdkit_ab.csv`

## Troubleshooting

### Model not appearing in plots

**Check:**
1. File is in the correct directory (`results/ModelName/`)
2. File matches the suffix pattern (e.g., `_results.csv`)
3. Directory is not in `exclude_dirs`
4. Run with verbose output to see what's being discovered

### Features in wrong order

**Solution:**
Add them explicitly to `feature_order` in the desired position.

### Colors not applying

**Check:**
1. Model name matches exactly (case-sensitive)
2. Color is in the correct section (`Graph`, `Baseline`, etc.)
3. Color format is valid hex code (e.g., `"#FF6B6B"`)

### Pattern not matching

**Debug:**
- Test your regex pattern at https://regex101.com/
- Remember to escape backslashes in YAML: `\\(` not `\(`
- Check that feature names in CSV match the pattern

## Advanced: Dynamic Feature Discovery

For completely dynamic models (e.g., user can add any model without touching config):

The script automatically:
1. Scans all directories in `results/`
2. Finds files matching method suffixes
3. Adds discovered features not in config to the end (alphabetically)

This means **you can add any model and it will work** - the config just controls ordering and styling.

## Example Workflow

```bash
# 1. Train a new model
python train_my_new_model.py --dataset htpmd

# 2. Results automatically saved to results/MyNewModel/htpmd_results.csv

# 3. Run visualization (no code changes needed!)
python scripts/python/visualize_combined_results.py --dataset htpmd

# 4. (Optional) Customize appearance by editing configs/visualization_config.yaml

# 5. Re-run visualization to see customizations
python scripts/python/visualize_combined_results.py --dataset htpmd
```

## Config File Location

Default: `configs/visualization_config.yaml`

To use a different config file:
```python
# In visualize_combined_results.py, modify:
CONFIG = load_config(Path("path/to/your/config.yaml"))
```

## Summary

**Key Benefits:**
- ✅ Add new models without editing Python code
- ✅ Automatic discovery of model directories
- ✅ Flexible feature ordering with patterns
- ✅ Centralized styling configuration
- ✅ Backward compatible with existing models

**When to edit config:**
- Adding a new model type with different file patterns
- Customizing feature display order
- Setting custom colors or markers
- Filtering OPV dataset targets

**When NOT to edit config:**
- Adding a standard graph model (auto-discovered)
- Adding new copolymer modes (pattern-matched)
- Temporary experiments (will auto-appear at end)
