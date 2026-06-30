# Experiment 3: Matched-Group Learning Curve

## Methodology

Since retraining with different data fractions is not available locally, we evaluate a **metric stability proxy**: computing arch-dev R² on increasing fractions (25%, 50%, 75%, 100%) of the test matched groups.

If the metric is stable across fractions, it suggests sufficient test coverage.
This is NOT a training learning curve — that requires cluster retraining.

## Results

| Variant | Target | 25% | 50% | 75% | 100% |
|---------|--------|-----|-----|-----|------|
| 2D0 | EA | 0.8417 | 0.8430 | 0.8431 | 0.8434 |
| 2D0 | IP | 0.9067 | 0.9061 | 0.9051 | 0.9060 |
| 2D1 | EA | 0.8629 | 0.8612 | 0.8626 | 0.8626 |
| 2D1 | IP | 0.9133 | 0.9142 | 0.9134 | 0.9140 |

## Interpretation

A proper training learning curve (training on 25/50/75/100% of matched groups) would require retraining on the cluster. The proxy analysis above only tests metric stability on the evaluation side.

**For a definitive answer on data sufficiency, retrain with subsampled training sets.**
