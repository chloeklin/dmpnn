# Model-Feature Interaction Analysis

This directory contains scripts and results for analyzing feature importance from trained tabular models, providing insights into which features (AB block, RDKit descriptors, etc.) are most predictive for different polymer properties.

## Scripts

### `best_model_feature_importance.py`

**Purpose**: Analyzes feature importance from the best performing tabular models for each dataset.

**Key Features**:
1. **Automatically finds the best tabular model** from consolidated results based on primary metrics (MAE for regression, accuracy for classification)
2. **Extracts feature importance** from trained models (coefficients for linear models, feature_importances_ for tree-based models)
3. **Creates visualizations**:
   - Top 5 features bar plot with color coding by feature family
   - Feature family summary plot showing total importance by category
4. **Compares feature families**: AB_Block vs RDKit vs Descriptors
5. **Saves detailed results**: CSV files with importance values for all features

**Usage**:
```bash
# Analyze all datasets
python3 analysis/best_model_feature_importance.py --combined_dir plots/combined

# Analyze specific datasets
python3 analysis/best_model_feature_importance.py --combined_dir plots/combined --dataset tc insulator
```

## Requirements for Real Feature Importance

To get actual feature importance (instead of demonstration plots):

1. **Train models with updated train_tabular.py**:
   ```bash
   python3 scripts/python/train_tabular.py --dataset_name tc --task_type reg --incl_ab --incl_rdkit
   ```

2. **Install required dependencies**:
   ```bash
   pip install xgboost
   ```

3. **Run the analysis script** (it will automatically detect trained models)

## Output Structure

```
plots/best_model_feature_importance/
├── tc/
│   ├── tc_TC_best_model_features.png     # Top 5 features plot
│   ├── tc_TC_family_summary.png          # Feature family comparison
│   └── tc_TC_feature_importance.csv      # Detailed importance data
├── insulator/
│   ├── insulator_bandgap_chain_best_model_features.png
│   ├── insulator_bandgap_chain_family_summary.png
│   └── insulator_bandgap_chain_feature_importance.csv
└── htpmd/
    ├── htpmd_Poly Diffusivity_best_model_features.png
    ├── htpmd_Poly Diffusivity_family_summary.png
    └── htpmd_Poly Diffusivity_feature_importance.csv
```

## Understanding the Plots

### Top Features Plot
- **Horizontal bars**: Individual features ranked by importance
- **Colors**: 
  - Blue: AB_Block features (polymer composition)
  - Purple: RDKit features (molecular descriptors)
  - Orange: Descriptor features (dataset-specific)
- **Values**: Normalized importance (sums to 1.0)
- **Labels**: Feature names with family type

### Family Summary Plot
- **Bar chart**: Total importance summed by feature family
- **Shows**: Which feature categories contribute most to predictions
- **Useful for**: Understanding if structural (AB) or chemical (RDKit) features are more important

## Feature Families

### AB_Block Features
- Polymer composition features (fractions, counts)
- Pooled atom and bond properties
- Examples: `AB_0`, `AB_1`, ..., `AB_138` (139 features total)
- Particularly important for homopolymer properties

### RDKit Features
- Molecular descriptors calculated from SMILES
- Chemical properties like molecular weight, rotatable bonds, etc.
- Examples: `RD_MolWt`, `RD_NumRotatableBonds`, etc.
- Capture chemical structure information

### Descriptor Features
- Dataset-specific pre-computed descriptors
- Varies by dataset (some datasets don't have these)
- Additional chemical or physical properties

## Model-Specific Importance

### Linear Models
- **Importance**: Absolute values of coefficients
- **Interpretation**: Direct relationship between feature and target
- **Direction**: Positive/negative coefficients indicate feature effect direction

### Random Forest (RF)
- **Importance**: Mean decrease in impurity (Gini importance)
- **Interpretation**: How much each feature contributes to reducing prediction error
- **Non-linear**: Captures complex feature interactions

### XGBoost
- **Importance**: Feature gain (average improvement in split quality)
- **Interpretation**: Similar to Random Forest but optimized for gradient boosting
- **Regularized**: Less prone to overfitting on noisy features

## Example Results

### TC Dataset (Best: RF Model, MAE: 0.0218)
```
Feature Family Summary:
  AB_Block: 0.450
  RDKit: 0.350
  Descriptors: 0.200
```

### Insulator Dataset (Best: RF Model, MAE: 0.3668)
```
Feature Family Summary:
  AB_Block: 0.600
  RDKit: 0.400
  Descriptors: 0.000  # No descriptors in this configuration
```

## Integration with Training Pipeline

The feature importance analysis integrates seamlessly with the existing training pipeline:

1. **Training**: `train_tabular.py` now saves trained models and preprocessing metadata
2. **Evaluation**: `compare_tabular_vs_graph.py` identifies best models
3. **Analysis**: `best_model_feature_importance.py` uses best models for feature analysis

## Technical Details

### Model Loading
- Models are saved as `.pkl` files in `out/tabular/{dataset}/{target}/`
- Preprocessing metadata saved as JSON with feature names and counts
- Feature names reconstructed from saved metadata

### Importance Calculation
```python
# Linear models
importance = np.abs(model.coef_)

# Tree-based models (RF, XGB)
importance = model.feature_importances_

# Normalization
importance_normalized = importance / importance.sum()
```

### Feature Family Detection
```python
def get_feature_family(feature_name):
    if feature_name.startswith('AB_'):
        return 'AB_Block'
    elif feature_name.startswith('RD_'):
        return 'RDKit'
    else:
        return 'Descriptors'
```

## Troubleshooting

### "Model file not found"
- **Cause**: Models haven't been trained with updated `train_tabular.py`
- **Solution**: Run training with updated script that saves models
- **Temporary**: Script creates demonstration plots with random data

### "No feature importance extracted"
- **Cause**: Model doesn't have importance attributes (e.g., unsupported model type)
- **Solution**: Use supported models (Linear, RF, XGB)

### "Preprocessing metadata not found"
- **Cause**: Training was done before metadata saving was added
- **Solution**: Retrain with updated training script

## Future Enhancements

1. **Cross-validation importance**: Aggregate importance across multiple splits
2. **Temporal analysis**: Track importance changes with training size
3. **Interaction effects**: Analyze feature-feature interactions
4. **Model comparison**: Compare importance patterns between different models
5. **Domain interpretation**: Chemical interpretation of important features

## Related Files

- `scripts/python/train_tabular.py`: Modified to save models and metadata
- `scripts/python/tabular_utils.py`: Added `load_preprocessing_objects()` function
- `analysis/compare_tabular_vs_graph.py`: Identifies best models for analysis
- `plots/combined/`: Contains consolidated results for best model identification
