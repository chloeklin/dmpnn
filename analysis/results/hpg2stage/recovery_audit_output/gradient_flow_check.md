# Gradient Flow Check

Two-step bootstrap verification with batch_size=8, d=64.

## Mechanism

With `alpha_init=0.1` and zero-init MLP output layer:

- **Step 0**: `r_arch=0` → `grad(alpha)=0`, but `grad(mlp)=dL/dh_poly·alpha≠0`
- **Step 1**: MLP updates → `r_arch≠0` → `grad(alpha)=dL/dh_poly·r_arch≠0`

The critical requirement is that **grad(mlp) ≠ 0 at Step 0** (symmetry broken).

## Variant: 2d1_fixed

### Step 0 (initialization)

- alpha = 0.10000000149011612
- r_arch norm (output): 0.0 (zero-init)
- ||grad(alpha)|| = 0.00000000 (expected: 0, because r_arch=0)
- ||grad(mlp_output_weight)|| = 0.94721061 ✅ NON-ZERO
- ||grad(mlp_output_bias)|| = 0.19693249 ✅ NON-ZERO

### Step 1 (after MLP update)

- alpha = 0.10000000149011612
- r_arch norm (output): 0.11783621 ✅ NON-ZERO
- ||grad(alpha)|| = 0.08506316 ✅ NON-ZERO
- ||grad(mlp_output_weight)|| = 1.36727703 ✅ NON-ZERO
- ||grad(mlp_output_bias)|| = 0.27967390 ✅ NON-ZERO

✅ **BOOTSTRAP CONFIRMED**: MLP learns at step 0, alpha follows at step 1.

## Variant: 2d1_arch

### Step 0 (initialization)

- alpha = [0.10000000149011612, 0.10000000149011612, 0.10000000149011612]
- r_arch norm (output): 0.0 (zero-init)
- ||grad(alpha)|| = 0.00000000 (expected: 0, because r_arch=0)
- ||grad(mlp_output_weight)|| = 0.94721061 ✅ NON-ZERO
- ||grad(mlp_output_bias)|| = 0.19693249 ✅ NON-ZERO

### Step 1 (after MLP update)

- alpha = [0.10000000149011612, 0.10000000149011612, 0.10000000149011612]
- r_arch norm (output): 0.11783621 ✅ NON-ZERO
- ||grad(alpha)|| = 0.05325027 ✅ NON-ZERO
- ||grad(mlp_output_weight)|| = 1.36727703 ✅ NON-ZERO
- ||grad(mlp_output_bias)|| = 0.27967390 ✅ NON-ZERO

✅ **BOOTSTRAP CONFIRMED**: MLP learns at step 0, alpha follows at step 1.

## Success Criteria

✅ **GRADIENT BOOTSTRAP CONFIRMED for both 2d1_fixed and 2d1_arch.**

The dead initialization bug is fixed. Both variants will train correctly:
1. Step 0: MLP receives gradients through non-zero alpha (0.1)
2. Step 1+: Alpha receives gradients through non-zero r_arch
3. Both parameters co-evolve during training