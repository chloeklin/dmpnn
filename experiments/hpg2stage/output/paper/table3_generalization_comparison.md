# Table 3: Generalization Comparison (ordered by increasing extrapolation difficulty)

| Model    | Split                        |   EA R² |   EA R²(Δ) |   IP R² |   IP R²(Δ) |
|:---------|:-----------------------------|--------:|-----------:|--------:|-----------:|
| Frac     | Group-disjoint               |  0.9879 |    -0      |  0.9838 |    -0      |
| 2D0-arch | Group-disjoint               |  0.9971 |     0.8868 |  0.998  |     0.9423 |
| 2D1-arch | Group-disjoint               |  0.9984 |     0.9381 |  0.9987 |     0.9649 |
| wDMPNN   | Group-disjoint               |  0.9087 |     0.4585 |  0.8664 |     0.4942 |
| Frac     | Pair-disjoint                |  0.9878 |     0      |  0.9831 |    -0      |
| 2D0-arch | Pair-disjoint                |  0.9968 |     0.8862 |  0.9976 |     0.9406 |
| 2D1-arch | Pair-disjoint                |  0.9979 |     0.9346 |  0.9981 |     0.9637 |
| wDMPNN   | Pair-disjoint                |  0.9315 |     0.7068 |  0.8965 |     0.7377 |
| Frac     | LOMO (Leave-One-Monomer-Out) |  0.7218 |    -0      |  0.7352 |    -0      |
| 2D0-arch | LOMO (Leave-One-Monomer-Out) |  0.6787 |     0.3743 |  0.5798 |     0.1699 |
| 2D1-arch | LOMO (Leave-One-Monomer-Out) |  0.6151 |     0.4344 |  0.5095 |     0.023  |
| wDMPNN   | LOMO (Leave-One-Monomer-Out) |  0.1776 |    -0.1709 |  0.215  |    -0.4835 |

