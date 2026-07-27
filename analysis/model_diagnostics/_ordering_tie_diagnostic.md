# Ordering Tie Diagnostic

## Exact prediction ties by model

| model              |   exact_pred_ties |   informative_pairs |   folds_with_ties |   maximum_fold_ties |   exact_tie_rate |
|:-------------------|------------------:|--------------------:|------------------:|--------------------:|-----------------:|
| hpg_hier_octamer   |                34 |               61380 |                16 |                   3 |      0.000553926 |
| hpg_hier           |                 0 |               61380 |                 0 |                   0 |      0           |
| hpg_hier_junction  |                 0 |               61380 |                 0 |                   0 |      0           |
| hpg_hier_junction1 |                 0 |               61380 |                 0 |                   0 |      0           |
| wdmpnn             |                 0 |               61380 |                 0 |                   0 |      0           |

## Exact prediction ties by model, target, and fold

| model              | target   |   fold |   exact_pred_ties |   informative_pairs |   exact_tie_rate |
|:-------------------|:---------|-------:|------------------:|--------------------:|-----------------:|
| hpg_hier           | EA       |      0 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      1 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      2 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      3 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      4 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      5 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      6 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      7 |                 0 |                3410 |      0           |
| hpg_hier           | EA       |      8 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      0 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      1 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      2 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      3 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      4 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      5 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      6 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      7 |                 0 |                3410 |      0           |
| hpg_hier           | IP       |      8 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      0 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      1 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      2 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      3 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      4 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      5 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      6 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      7 |                 0 |                3410 |      0           |
| wdmpnn             | EA       |      8 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      0 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      1 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      2 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      3 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      4 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      5 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      6 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      7 |                 0 |                3410 |      0           |
| wdmpnn             | IP       |      8 |                 0 |                3410 |      0           |
| hpg_hier_octamer   | EA       |      0 |                 0 |                3410 |      0           |
| hpg_hier_octamer   | EA       |      1 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | EA       |      2 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | EA       |      3 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | EA       |      4 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | EA       |      5 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | EA       |      6 |                 1 |                3410 |      0.000293255 |
| hpg_hier_octamer   | EA       |      7 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | EA       |      8 |                 1 |                3410 |      0.000293255 |
| hpg_hier_octamer   | IP       |      0 |                 0 |                3410 |      0           |
| hpg_hier_octamer   | IP       |      1 |                 1 |                3410 |      0.000293255 |
| hpg_hier_octamer   | IP       |      2 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | IP       |      3 |                 1 |                3410 |      0.000293255 |
| hpg_hier_octamer   | IP       |      4 |                 1 |                3410 |      0.000293255 |
| hpg_hier_octamer   | IP       |      5 |                 1 |                3410 |      0.000293255 |
| hpg_hier_octamer   | IP       |      6 |                 1 |                3410 |      0.000293255 |
| hpg_hier_octamer   | IP       |      7 |                 3 |                3410 |      0.000879765 |
| hpg_hier_octamer   | IP       |      8 |                 3 |                3410 |      0.000879765 |
| hpg_hier_junction  | EA       |      0 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      1 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      2 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      3 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      4 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      5 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      6 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      7 |                 0 |                3410 |      0           |
| hpg_hier_junction  | EA       |      8 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      0 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      1 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      2 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      3 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      4 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      5 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      6 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      7 |                 0 |                3410 |      0           |
| hpg_hier_junction  | IP       |      8 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      0 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      1 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      2 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      3 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      4 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      5 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      6 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      7 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | EA       |      8 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      0 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      1 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      2 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      3 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      4 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      5 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      6 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      7 |                 0 |                3410 |      0           |
| hpg_hier_junction1 | IP       |      8 |                 0 |                3410 |      0           |

## Tie conventions

The committed old inline metric used `sign_product > 0`; exact prediction ties therefore scored 0. The canonical module initially copied that rule. The selected canonical convention now gives an exact prediction tie 0.5 credit, representing expected accuracy under random tie breaking.

Old inline expression:

```python
scores.append(np.mean([(yt[i] - yt[j]) * (yp[i] - yp[j]) > 0 for i, j in pairs]))
```

Selected canonical expression:

```python
0.5 if pred_values[i] == pred_values[j]
else float((true_values[i] - true_values[j]) * (pred_values[i] - pred_values[j]) > 0)
```

## Octamer median ordering under each convention

| target   |   strict |   half_credit |   ties_correct |
|:---------|---------:|--------------:|---------------:|
| EA       | 0.818182 |      0.818263 |       0.818345 |
| IP       | 0.826979 |      0.827061 |       0.827142 |

The frozen Phase-1 reference used half credit: EA `0.818263` rounds to `0.81826`, and IP `0.827061` rounds to `0.82706`. The strict convention produced `0.818182` and `0.826979`. No other model has an exact prediction tie in these 90 seed-42 cells.
