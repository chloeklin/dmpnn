# Fusion Ablation — Manuscript Table Values

> Median R² is the actual median across 9 LOMO folds (not median Δ).
> Wins = number of folds where the variant outperformed Additive (2D1).
> p = paired Wilcoxon signed-rank test vs Additive. \* = p < 0.05.
> **Bold** = best median R² in that column.

| Model | EA R² med | EA wins | EA p | IP R² med | IP wins | IP p | EA ArchDev R² med | EA ArchDev wins | EA ArchDev p | IP ArchDev R² med | IP ArchDev wins | IP ArchDev p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Additive (2D1) | 0.692 | -- | -- | 0.497 | -- | -- | **0.924** | -- | -- | **0.896** | -- | -- |
| FiLM | **0.788** | 7/9 | 0.0547 | **0.871** | 7/9 | 0.0195\* | 0.630 | 1/9 | 0.0195\* | 0.779 | 1/9 | 0.0742 |
| NLMix | 0.743 | 6/9 | 0.1289 | 0.770 | 6/9 | 0.2031 | 0.759 | 2/9 | 0.0391\* | 0.793 | 2/9 | 0.1289 |
| FiLM+NLMix | 0.725 | 4/9 | 0.4258 | 0.799 | 8/9 | 0.0117\* | 0.614 | 1/9 | 0.0195\* | 0.818 | 1/9 | 0.0977 |
