# Stage2D Initialization Audit

## Variant: 2d1_fixed

- **alpha_init**: 0.10000000149011612
- **alpha shape**: []
- **mlp_2d1 output weight norm**: 0.000000
- **mlp_2d1 output bias norm**: 0.000000
- **mlp_2d1 output layer zero-init**: YES

✅ **GRADIENT FLOWS**: alpha=0.10000000149011612 ≠ 0, even with zero-init MLP.
   ∂L/∂mlp_out = ∂L/∂h_poly · alpha ≠ 0 → MLP can learn.
   ∂L/∂alpha = ∂L/∂h_poly · r_arch = 0 initially, but once MLP updates, r_arch ≠ 0.

## Variant: 2d1_arch

- **alpha_init**: [0.10000000149011612, 0.10000000149011612, 0.10000000149011612]
- **alpha shape**: [3]
- **mlp_2d1 output weight norm**: 0.000000
- **mlp_2d1 output bias norm**: 0.000000
- **mlp_2d1 output layer zero-init**: YES

✅ **GRADIENT FLOWS**: alpha=0.10000000149011612 ≠ 0, even with zero-init MLP.
   ∂L/∂mlp_out = ∂L/∂h_poly · alpha ≠ 0 → MLP can learn.
   ∂L/∂alpha = ∂L/∂h_poly · r_arch = 0 initially, but once MLP updates, r_arch ≠ 0.

## Variant: 2d1_gate

- **alpha**: None (gate variant uses dynamic alpha)
- **mlp_2d1 output weight norm**: 0.000000
- **mlp_2d1 output bias norm**: 0.000000
- **mlp_2d1 output layer zero-init**: YES
- **gate_mlp output weight norm**: 0.000000
- **gate_mlp output bias**: -3.0000 (sigmoid → 0.0474)

✅ **GRADIENT FLOWS**: Gate uses sigmoid(bias≈-3) ≈ 0.047 → non-zero alpha from start.

## Variant: 2d0_fixed

- **alpha_init**: 0.10000000149011612
- **alpha shape**: []
- **mlp_2d1**: None (2D0 variant)

✅ **GRADIENT FLOWS**: arch_embedding is randomly initialized (std=0.02) → non-zero.

## Variant: 2d0_arch

- **alpha_init**: [0.10000000149011612, 0.10000000149011612, 0.10000000149011612]
- **alpha shape**: [3]
- **mlp_2d1**: None (2D0 variant)

✅ **GRADIENT FLOWS**: arch_embedding is randomly initialized (std=0.02) → non-zero.

## Variant: 2d0_gate

- **alpha**: None (gate variant uses dynamic alpha)
- **mlp_2d1**: None (2D0 variant)
- **gate_mlp output weight norm**: 0.000000
- **gate_mlp output bias**: -3.0000 (sigmoid → 0.0474)

✅ **GRADIENT FLOWS**: arch_embedding is randomly initialized (std=0.02) → non-zero.

## Summary

| Variant | alpha_init | mlp_output_zero | Gradients Flow? |
|---------|-----------|-----------------|-----------------|
| 2d1_fixed | 0.10000000149011612 | YES | ✅ YES |
| 2d1_arch | 0.10000000149011612 | YES | ✅ YES |
| 2d1_gate | N/A | YES | ✅ YES |
| 2d0_fixed | 0.10000000149011612 | N/A | ✅ YES |
| 2d0_arch | 0.10000000149011612 | N/A | ✅ YES |
| 2d0_gate | N/A | N/A | ✅ YES |
