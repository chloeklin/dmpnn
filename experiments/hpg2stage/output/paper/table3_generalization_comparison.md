# Table 3: Generalization Comparison

| Model    | Split          |   EA R² |   EA R²(Δ) |   IP R² |   IP R²(Δ) |
|:---------|:---------------|--------:|-----------:|--------:|-----------:|
| Frac     | a_held_out     |  0.9741 |     0      |  0.9642 |    -0      |
| 2D0-arch | a_held_out     |  0.981  |     0.8467 |  0.9789 |     0.9077 |
| 2D1-arch | a_held_out     |  0.982  |     0.8655 |  0.9794 |     0.9167 |
| wDMPNN   | a_held_out     |  0.97   |     0.6736 |  0.9523 |     0.7092 |
| Frac     | group_disjoint |  0.9879 |    -0      |  0.9838 |    -0      |
| 2D0-arch | group_disjoint |  0.9971 |     0.8868 |  0.998  |     0.9423 |
| 2D1-arch | group_disjoint |  0.9984 |     0.9381 |  0.9987 |     0.9649 |
| wDMPNN   | group_disjoint |  0.9087 |     0.4585 |  0.8664 |     0.4942 |
| Frac     | pair_disjoint  |  0.9878 |     0      |  0.9831 |    -0      |
| 2D0-arch | pair_disjoint  |  0.9968 |     0.8862 |  0.9976 |     0.9406 |
| 2D1-arch | pair_disjoint  |  0.9979 |     0.9346 |  0.9981 |     0.9637 |
| wDMPNN   | pair_disjoint  |  0.9315 |     0.7068 |  0.8965 |     0.7377 |

