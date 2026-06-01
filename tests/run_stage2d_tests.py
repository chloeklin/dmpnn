#!/usr/bin/env python3
"""Standalone test runner for Stage2D (no pytest dependency)."""
import sys
sys.path.insert(0, ".")

import torch
from chemprop.nn.stage2d import Stage2Aggregator, VALID_STAGE2_VARIANTS

B, d, n_targets = 4, 64, 2
h_A = torch.randn(B, d)
h_B = torch.randn(B, d)
f_A = torch.tensor([0.6, 0.3, 0.5, 0.7])
f_B = torch.tensor([0.4, 0.7, 0.5, 0.3])
arch = torch.tensor([0, 1, 2, 0], dtype=torch.long)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  PASS: {msg}")


# ═══════════════════════════════════════════════════════════════
print("=== 1. Shape Tests ===")
for v in VALID_STAGE2_VARIANTS:
    agg = Stage2Aggregator(d=d, variant=v, n_targets=n_targets)
    preds, aux = agg(h_A, h_B, f_A, f_B, arch)
    check(preds.shape == (B, n_targets), f"{v}: preds shape {preds.shape}")
    check(aux["h_poly"].shape == (B, d), f"{v}: h_poly shape {aux['h_poly'].shape}")

# ═══════════════════════════════════════════════════════════════
print("\n=== 2. Alpha=0 → h_poly == h_mix ===")
h_mix = f_A.unsqueeze(1) * h_A + f_B.unsqueeze(1) * h_B
for v in ["frac", "2d0_fixed", "2d0_arch", "2d1_fixed", "2d1_arch"]:
    agg = Stage2Aggregator(d=d, variant=v, alpha_init=0.0, n_targets=n_targets)
    _, aux = agg(h_A, h_B, f_A, f_B, arch)
    diff = (aux["h_poly"] - h_mix).abs().max().item()
    check(diff < 1e-5, f"{v}: max_diff={diff:.2e}")

# ═══════════════════════════════════════════════════════════════
print("\n=== 3. Gate variants near frac at init ===")
for v in ["2d0_gate", "2d1_gate"]:
    agg = Stage2Aggregator(d=d, variant=v, alpha_init=0.0, n_targets=n_targets)
    _, aux = agg(h_A, h_B, f_A, f_B, arch)
    diff = (aux["h_poly"] - h_mix).abs().max().item()
    check(diff < 0.5, f"{v}: max_diff={diff:.2e} (should be <0.5)")

# ═══════════════════════════════════════════════════════════════
print("\n=== 4. Architecture sensitivity (alpha=1.0) ===")
for v in ["2d0_fixed", "2d0_arch"]:
    agg = Stage2Aggregator(d=d, variant=v, alpha_init=1.0, n_targets=n_targets)
    with torch.no_grad():
        agg.alpha.fill_(1.0)
    arch_0 = torch.zeros(B, dtype=torch.long)
    arch_2 = torch.full((B,), 2, dtype=torch.long)
    with torch.no_grad():
        _, a0 = agg(h_A, h_B, f_A, f_B, arch_0)
        _, a2 = agg(h_A, h_B, f_A, f_B, arch_2)
    differs = not torch.allclose(a0["h_poly"], a2["h_poly"], atol=1e-5)
    check(differs, f"{v}: different arch → different h_poly")

# For 2d1 variants, MLP output layer is zero-init → r_arch=0 at init.
# Must set output layer to non-zero to test architecture sensitivity.
for v in ["2d1_fixed", "2d1_arch"]:
    agg = Stage2Aggregator(d=d, variant=v, alpha_init=1.0, n_targets=n_targets)
    with torch.no_grad():
        agg.alpha.fill_(1.0)
        # Set MLP output layer to non-zero so r_arch != 0
        torch.nn.init.xavier_uniform_(agg.mlp_2d1[-1].weight)
    arch_0 = torch.zeros(B, dtype=torch.long)
    arch_2 = torch.full((B,), 2, dtype=torch.long)
    with torch.no_grad():
        _, a0 = agg(h_A, h_B, f_A, f_B, arch_0)
        _, a2 = agg(h_A, h_B, f_A, f_B, arch_2)
    differs = not torch.allclose(a0["h_poly"], a2["h_poly"], atol=1e-5)
    check(differs, f"{v}: different arch → different h_poly (non-zero MLP)")

# ═══════════════════════════════════════════════════════════════
print("\n=== 5. Gradient flow ===")
for v in VALID_STAGE2_VARIANTS:
    agg = Stage2Aggregator(d=d, variant=v, n_targets=n_targets)
    _h_A = h_A.detach().clone().requires_grad_(False)
    _h_B = h_B.detach().clone().requires_grad_(False)
    preds, _ = agg(_h_A, _h_B, f_A, f_B, arch)
    preds.sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in agg.parameters()
    )
    check(has_grad, f"{v}: gradients flow")

# ═══════════════════════════════════════════════════════════════
print("\n=== 6. CopolymerMPNN integration ===")
from chemprop.models.copolymer import CopolymerMPNN
for v in VALID_STAGE2_VARIANTS:
    mode = f"stage2d_{v}"
    check(mode in CopolymerMPNN.VALID_MODES, f"{mode} in VALID_MODES")

# ═══════════════════════════════════════════════════════════════
print("\n=== 7. Invalid variant raises ===")
try:
    Stage2Aggregator(d=64, variant="invalid_variant")
    check(False, "Should have raised ValueError")
except ValueError:
    check(True, "Invalid variant raises ValueError")

# ═══════════════════════════════════════════════════════════════
print("\n=== 8. Custom arch_emb_dim with projection ===")
agg = Stage2Aggregator(d=64, variant="2d0_fixed", arch_emb_dim=16)
check(agg.emb_proj is not None, "emb_proj exists when arch_emb_dim != d")
check(agg.arch_embedding.weight.shape == (3, 16), "embedding shape correct")

# ═══════════════════════════════════════════════════════════════
print("\n=== 9. Diagnostics logging ===")
for v in VALID_STAGE2_VARIANTS:
    agg = Stage2Aggregator(d=d, variant=v, n_targets=n_targets)
    _, aux = agg(h_A, h_B, f_A, f_B, arch)
    diag = agg.log_diagnostics(aux)
    check("stage2d/alpha_mean" in diag, f"{v}: alpha_mean logged")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} test(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
