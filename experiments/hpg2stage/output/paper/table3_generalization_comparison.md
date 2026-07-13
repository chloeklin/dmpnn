# Table 3: Generalization Comparison

| Model      | Split          |   EA R2 |   EA R2(d) |   IP R2 |   IP R2(d) |
|:-----------|:---------------|--------:|-----------:|--------:|-----------:|
| Frac       | Group-disjoint |  0.9879 |     0      |  0.9838 |     0      |
| wDMPNN     | Group-disjoint |  0.9973 |     0.8759 |  0.9972 |     0.9207 |
| GlobalArch | Group-disjoint |  0.9971 |     0.8868 |  0.998  |     0.9423 |
| ChemArch   | Group-disjoint |  0.9984 |     0.9381 |  0.9987 |     0.9649 |
| Frac       | Pair-disjoint  |  0.9878 |     0      |  0.9831 |     0      |
| wDMPNN     | Pair-disjoint  |  0.9967 |     0.8739 |  0.9967 |     0.9186 |
| GlobalArch | Pair-disjoint  |  0.9968 |     0.8862 |  0.9976 |     0.9406 |
| ChemArch   | Pair-disjoint  |  0.9979 |     0.9346 |  0.9981 |     0.9637 |
| Frac       | LOMO           | -0.4697 |    -0      |  0.3884 |     0      |
| wDMPNN     | LOMO           |  0.9581 |     0.4715 |  0.8706 |     0.5703 |
| GlobalArch | LOMO           | -0.245  |     0.4972 |  0.3151 |     0.4326 |
| ChemArch   | LOMO           | -1.2881 |     0.5357 |  0.4653 |     0.5272 |
