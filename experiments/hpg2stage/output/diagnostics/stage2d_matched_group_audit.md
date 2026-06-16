# Experiment 1: Matched-Group Count Audit

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total polymers | 42,966 |
| Total matched groups (A, B, f_A, f_B) | 18,414 |
| Groups with 1 architecture | 0 |
| Groups with 2 architectures | 12,276 |
| Groups with 3 architectures | 6,138 |
| Unique monomer A | 9 |
| Unique monomer B | 682 |
| Unique monomer pairs (A, B) | 6,138 |
| Unique compositions f_A | 3 ([np.float64(0.25), np.float64(0.5), np.float64(0.75)]) |
| **Usable groups (≥2 arch)** | **18,414** |
| **Samples in usable groups** | **42,966** (100.0%) |

## Architecture Combinations in Usable Groups

| Combination | Count |
|-------------|-------|
| block + random | 12,276 |
| alternating + block + random | 6,138 |

## Architecture × Composition Breakdown

| fracA | alternating | block | random | total |
|-------|-------------|-------|--------|-------|
| 0.25 | 0 | 6,138 | 6,138 | 12,276 |
| 0.5 | 6,138 | 6,138 | 6,138 | 18,414 |
| 0.75 | 0 | 6,138 | 6,138 | 12,276 |

## Interpretation

**Dataset bottleneck NOT supported by count alone**: 18,414 usable matched groups containing 42,966 samples (100% of dataset).

The dataset provides extensive architecture contrast within matched groups. However, note that alternating architecture is only available at f_A = 0.5, limiting full 3-way comparisons to one composition.

**Key structural constraint**: Only 9 unique monomer A species exist. This limits chemical diversity in the a_held_out cross-validation (each fold holds out ~2 monomers).
