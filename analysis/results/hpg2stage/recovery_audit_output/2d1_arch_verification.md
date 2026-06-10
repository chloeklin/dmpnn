# 2D1-arch Verification

Confirming 2d1_arch has independent per-architecture alpha parameters.

## Parameter Comparison

### 2d1_fixed
- alpha: 0.100000 (single scalar Parameter)
- alpha.shape: []
- alpha.requires_grad: True

### 2d1_arch
- alpha: [0.10000000149011612, 0.10000000149011612, 0.10000000149011612]
- alpha.shape: [3]
- alpha.requires_grad: True
- alpha_alt (arch=0): 0.100000
- alpha_rand (arch=1): 0.100000
- alpha_block (arch=2): 0.100000

## Gradient Independence Test

Train for 2 optimizer steps (to bootstrap MLP), then check per-arch alpha gradients:

- After step 0: MLP gets gradients, alpha grads = 0 (r_arch=0)
- After step 1:
  - grad(alpha_alt): -0.12949803
  - grad(alpha_rand): -0.13224250
  - grad(alpha_block): -0.11404505

- All gradients non-zero: True
- Gradients are architecture-dependent: True

✅ **Per-architecture alphas receive INDEPENDENT, NON-ZERO gradients.**
2d1_arch IS genuinely different from 2d1_fixed.

## After Two Optimizer Steps

- alpha_alt: 0.101295
- alpha_rand: 0.101322
- alpha_block: 0.101140

✅ **Alphas have diverged — 2d1_arch is NOT identical to 2d1_fixed.**

## Functional Difference Test

Same (h_A, h_B, f_A, f_B) with different arch labels after MLP produces non-zero output:

- pred(arch=alternating): [[0.10025858134031296, -0.06936192512512207]]
- pred(arch=random): [[0.10078571736812592, -0.055608272552490234]]
- pred(arch=block): [[0.09932688623666763, -0.08494018018245697]]

✅ **Different architectures produce different predictions.**
The per-arch alpha scaling creates architecture-dependent outputs when r_arch ≠ 0.
