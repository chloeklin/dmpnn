# Table 3: Generalization Comparison

| Model    | Split          | EA R²   | EA R²(Δ)   | IP R²   | IP R²(Δ)   |
|:---------|:---------------|:--------|:-----------|:--------|:-----------|
| Frac     | a_held_out     | 0.9741  | 0.9076     | 0.9642  | 0.8517     |
| 2D0-arch | a_held_out     | 0.9810  | 0.9579     | 0.9789  | 0.9523     |
| 2D1-arch | a_held_out     | 0.9820  | 0.9633     | 0.9794  | 0.9540     |
| wDMPNN   | a_held_out     | 0.9700  | 0.9276     | 0.9523  | 0.8797     |
| Frac     | group_disjoint | 0.9879  | -0.0000    | 0.9838  | -0.0000    |
| 2D0-arch | group_disjoint | 0.9971  | 0.8868     | 0.9980  | 0.9423     |
| 2D1-arch | group_disjoint | 0.9984  | 0.9381     | 0.9987  | 0.9649     |
| wDMPNN   | group_disjoint | NA      | NA         | NA      | NA         |
| Frac     | pair_disjoint  | 0.9878  | 0.0000     | 0.9831  | -0.0000    |
| 2D0-arch | pair_disjoint  | 0.9968  | 0.8862     | 0.9976  | 0.9406     |
| 2D1-arch | pair_disjoint  | 0.9979  | 0.9346     | 0.9981  | 0.9637     |
| wDMPNN   | pair_disjoint  | NA      | NA         | NA      | NA         |

*NA = wDMPNN generalization results not yet available*
