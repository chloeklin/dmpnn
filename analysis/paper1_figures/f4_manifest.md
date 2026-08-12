# F4 — Split design

## Left panel
Unique smiles_B from `data/ea_ip.csv` (682 B monomers).
Murcko scaffold via `rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)`.
rdkit available: True.

## Right panel
`analysis/model_diagnostics/_octamer_k1_r3_results_fold_composition.csv`
Column `same_scaffold_share` = fraction of held-out B monomers whose Murcko scaffold
appears among training B monomers. S folds: [0, 1, 2, 3]. D folds: [4, 5, 6, 7, 8].

## Cells (right): 9 folds × 1 value each.