# Stage 2D Ablation Study: Fusion Mechanism Comparison

## Overview

This study tests whether alternative chemistry–architecture fusion
mechanisms improve upon the baseline 2D1 (additive residual) model.

- **Evaluation split**: Leave-One-Monomer-A-Out (LOMAO), 9 folds
- **Dataset**: ea_ip (42,966 copolymers, 9 unique monomer A)
- **Targets**: EA vs SHE (eV), IP vs SHE (eV)

## Model Descriptions

### 2D1

**Baseline**: Additive architecture residual.
h_poly = h_mix + α_arch · r_arch, where r_arch = MLP(z).


### 2D1_FiLM

**FiLM Architecture Modulation**: Multiplicative scaling of residual.
γ, β = MLP(e_arch); h_poly = h_mix + γ ⊙ (α_arch · r_arch) + β.
Tests: does architecture modulate rather than simply offset?


### 2D1_NonlinearMix

**Nonlinear Composition Pooling**: Nonlinear chemistry interaction.
h_mix_NL = f_A·h_A + f_B·h_B + f_A·f_B·g([h_A, h_B, |h_A−h_B|, h_A⊙h_B]).
Tests: does nonlinear monomer interaction improve prediction?


### 2D1_FiLM_NonlinearMix

**Combined**: Both FiLM modulation and nonlinear composition pooling.
Tests: do the two modifications complement each other?

- **Trainable parameters**: 1,474,197
- **Total training time**: 15896s (264.9min)

## Results

| Model | EA_R2 | IP_R2 | DeltaEA_R2 | DeltaIP_R2 | EA_MAE | IP_MAE | DeltaEA_MAE | DeltaIP_MAE |
|---|---|---|---|---|---|---|---|---|
| 2D1_FiLM_NonlinearMix | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Comparison with Baseline (2D1)

## Conclusions

### 1. Does multiplicative architecture modulation outperform additive fusion?

FiLM results not available.

### 2. Does nonlinear composition pooling improve prediction?

NonlinearMix results not available.

### 3. Do the two modifications complement each other?

The combined model does not outperform the best individual modification on EA R².

### 4. Which model should be carried forward?

