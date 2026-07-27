# A-heldout Seed-42 Reproduction Check

## Run

The `hpg_hier`, EA, fold 0, seed 42 model was retrained with 100 maximum epochs, patience 15, batch size 64, Stage-1 sum pooling, Stage-2 depth 2, full Stage-2 edges, transition-graph mode, and stoichiometric readout. The prediction was written under `/tmp/dmpnn_baseline_reproduction`; the canonical NPZ was not overwritten.

The training and prediction completed. Writing its provenance sidecar then failed because `Path` values in the resolved CLI arguments were not JSON serializable. The runner now serializes these values as strings for future runs.

## Comparison

| Check | Result |
| --- | --- |
| Test indices | Bitwise equal |
| True targets | Bitwise equal |
| Sample metadata | Bitwise equal |
| Predictions | Not bitwise equal |
| Maximum absolute prediction difference | 0.3642779589 eV |
| Prediction correlation | 0.9947367761 |
| Existing overall R² | 0.9241134900 |
| Reproduced overall R² | 0.9587659282 |
| Existing MAE | 0.1068073920 eV |
| Reproduced MAE | 0.0845148754 eV |
| Existing bias | -0.0984415220 eV |
| Reproduced bias | -0.0811927297 eV |

## Interpretation

The split, row ordering, targets, and sample metadata reproduce exactly, but the learned prediction vector does not. The old NPZ has no provenance sidecar, git SHA, environment record, epoch count, best validation loss, or deterministic-training record. Therefore the historical training environment and exact stopping trajectory cannot be reconstructed. The current runner seeds Python/NumPy/PyTorch through `set_seed`, but the Lightning trainer does not request deterministic algorithms, so bitwise equality across GPU training runs is not guaranteed. Model and featurizer code also evolved after the historical artifact was created. These facts explain why the split reproduces while the learned weights do not; they do not identify one uniquely provable source of the numerical difference.

The existing artifact should be treated as historically non-reproducible at bitwise prediction level. New B-heldout runs must retain the newly added provenance sidecars and should use an explicitly deterministic trainer configuration if bitwise reruns are required.
