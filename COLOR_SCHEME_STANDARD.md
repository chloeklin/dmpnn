# Standard Color Scheme for All Plots

All plotting scripts in this repository now use **Paul Tol's colorblind-friendly palette**, which is the standard used in Nature, Science, and other top-tier scientific journals.

## Benefits

✅ **Colorblind-friendly**: Distinguishable by people with deuteranopia, protanopia, and tritanopia  
✅ **Grayscale-compatible**: Print well in black & white  
✅ **Journal-standard**: Used in Nature, Science, Cell, and other high-impact journals  
✅ **Professional**: Avoid garish or overly bright colors  
✅ **Consistent**: Follow scientifically-designed color schemes  

## Reference

Paul Tol's color schemes: https://personal.sron.nl/~pault/

## Color Definitions

### Model Colors

**Graph Models:**
- DMPNN: `#EE7733` (Orange)
- wDMPNN: `#0077BB` (Blue)
- PPG: `#33BBEE` (Cyan)
- AttentiveFP: `#EE3377` (Magenta)
- DMPNN_DiffPool: `#009988` (Teal)
- GAT: `#CC3311` (Red)
- GIN: `#AA3377` (Purple)
- Graphormer: `#CCBB44` (Yellow)

**Tabular Models:**
- Linear: `#4477AA` (Blue)
- RF: `#EE6677` (Red)
- XGB: `#228833` (Green)
- LogReg: `#4477AA` (Blue)

### Dataset Colors

- tc: `#4477AA` (Blue)
- insulator: `#EE7733` (Orange)
- htpmd: `#228833` (Green)
- polyinfo: `#EE6677` (Red)
- opv_camb3lyp: `#AA3377` (Purple)
- ea_ip: `#CCBB44` (Yellow)
- pae_tg_mono211: `#EE3377` (Magenta)
- pae_tg_paper211: `#BBBBBB` (Grey)

### Feature Family Colors

- AB_Block: `#4477AA` (Blue)
- RDKit: `#EE6677` (Red)
- Descriptors: `#EE7733` (Orange)
- Graph: `#228833` (Green)

### Split Colors (train/val/test)

- train: `#4477AA` (Blue)
- val: `#EE7733` (Orange)
- test: `#EE6677` (Red)

### Statistical Line Colors

- mean: `#EE6677` (Red)
- median: `#EE7733` (Orange)
- std: `#4477AA` (Blue)

### Correlation Colors

- positive: `#4477AA` (Blue)
- negative: `#EE6677` (Red)

### Standard Greys

- Edge colors, error bars: `#333333` (Dark grey)
- Light elements: `#BBBBBB` (Light grey)

### Background Colors for Text Boxes

- Info boxes: `#E8F4F8` (Light blue)
- Warning boxes: `#FFF4E6` (Light orange)
- Neutral boxes: `#F5F5F5` (Light grey)

## Updated Scripts

### Scripts Directory
1. ✅ `visualize_combined_results.py` - Main comparison plots
2. ✅ `plot_multi_model_learning_curves.py` - Learning curve plots
3. ✅ `plot_target_histograms.py` - Target distribution histograms
4. ✅ `plot_colors.py` - Centralized color configuration (NEW)

### Analysis Directory
1. ✅ `feature_space_analysis.py` - PCA, UMAP, t-SNE visualizations
2. ✅ `model_feature_interaction.py` - Feature importance plots
3. ✅ `graph_feature_space_analysis.py` - Graph embedding visualizations (partial)

## Usage

For new plotting scripts, import colors from the centralized configuration:

```python
import sys
sys.path.append('/path/to/scripts/python')
from plot_colors import MODEL_COLORS, DATASET_COLORS, TOL_VIBRANT

# Use in plots
plt.plot(x, y, color=MODEL_COLORS['DMPNN'])
plt.scatter(x, y, c=DATASET_COLORS['htpmd'])
```

## Notes

- All histogram bars now use `#4477AA` (Tol's blue) with `#333333` edges
- Error bars use `#333333` (dark grey) for consistency
- Mean/median lines use `#EE6677` (red) and `#EE7733` (orange) respectively
- Positive/negative correlations use `#4477AA` (blue) and `#EE6677` (red)
- All text box backgrounds use light, muted colors from the palette
