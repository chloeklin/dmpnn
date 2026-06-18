# Architecture-Deviation R² Discrepancy Explanation

## The Discrepancy

- **Learning curve R²(Δy)**: ~0.35–0.54
- **Final Stage 2D R²(Δy)**: ~0.86–0.96

**Gap**: ~0.4–0.5 R² units

---

## Root Cause Analysis

The discrepancy arises from **three fundamental differences** between how the two analysis pipelines compute the metric:

### 1. Training Data Amount (Reduced vs Full)

| Pipeline | Training Data | Model Quality |
|----------|---------------|---------------|
| Learning curve | 5%–100% (fractions) | Lower capacity |
| Stage 2D final | 100% (full) | Higher capacity |

**Impact**: Models trained on reduced data have less capacity to learn architecture-specific patterns, resulting in poorer deviation predictions.

### 2. Group Definition (4-part vs 3-part Key)

| Pipeline | Group Key Definition | Group Granularity |
|----------|---------------------|-------------------|
| Learning curve | `A||B||fracA||fracB` | More granular |
| Stage 2D | `A|B|fracA` | Broader |

**Critical Finding**: 
- Learning curve uses 4-part group key (includes fracB)
- Stage 2D uses 3-part group key (excludes fracB, since fracB = 1 - fracA)

**Impact**: 
- 4-part key creates more groups with fewer samples each → noisier group means → lower R²
- 3-part key creates fewer, larger groups → more stable group means → higher R²

### 3. Aggregation Method (Per-file vs Pooled)

| Pipeline | Aggregation | Group Mean Computation |
|----------|-------------|----------------------|
| Learning curve | Per-file (each fraction separately) | Within-file only |
| Stage 2D | Pooled across all folds | Population mean |

**Impact**:
- Learning curve: Group means computed from limited test samples per fraction
- Stage 2D: Group means computed from full population (5-fold CV covers all data)

---

## Why Stage 2D Gets Higher R²

1. **Better models**: Full training (100%) vs reduced training (fractions)
2. **Broader groups**: 3-part key aggregates more samples per group → stable means
3. **Population means**: Pooled computation uses all data → better group mean estimates

## Why Learning Curve Gets Lower R²

1. **Weaker models**: Less training data → poorer architecture-aware predictions
2. **Granular groups**: 4-part key fragments groups → fewer samples per group
3. **Per-file computation**: Limited test set per fraction → noisy group means

---

## The One-Sentence Explanation

**The learning-curve architecture-deviation R² values are lower (~0.35–0.54) than final Stage 2D values (~0.86–0.96) because learning curves use (1) reduced-training checkpoints with limited model capacity, (2) a more granular 4-part group definition that fragments samples into smaller groups with noisier means, and (3) per-file aggregation that computes group means from limited test-set data rather than the full population, whereas Stage 2D uses full-training models, a broader 3-part group definition, and pooled aggregation across all folds for more stable group mean estimates.**

---

## Implications

1. **The metrics are not directly comparable** - they measure different things under different conditions
2. **Learning curve arch-dev R²** measures: "How well can the model predict architecture deviations when trained on limited data with noisy group means?"
3. **Stage 2D arch-dev R²** measures: "How well can the model predict architecture deviations when trained on full data with stable population means?"
4. **Both are valid** - they answer different questions about model behavior

---

## Recommendation

When reporting results:
- Clarify which metric is being reported
- Don't compare learning curve arch-dev R² directly to Stage 2D arch-dev R²
- For learning curves, focus on overall R² trends; for Stage 2D, arch-dev R² is meaningful

---

*Generated: June 18, 2026*
