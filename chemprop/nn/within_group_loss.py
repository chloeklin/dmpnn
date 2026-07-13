"""Within-group residual variance loss for architecture-aware copolymer models.

Mathematical definition
-----------------------
Given a batch of N samples where each sample i belongs to chemistry group g(i),
let e_i = y_i - ŷ_i be the prediction residual (in normalised / z-scored space).

The within-group residual variance loss is:

    L_within = Σ_g Σ_{i∈g} (e_i - ē_g)²
               ─────────────────────────────
               Σ_g Σ_{i∈g} (y_i - ȳ_g)²  +  ε

where ē_g = mean(e_i for i in g) and ȳ_g = mean(y_i for i in g) are computed
over the members of group g that are present in the current batch.

Normalisation by within-group target variance
---------------------------------------------
The denominator Σ_g Σ_{i∈g} (y_i - ȳ_g)² is the total within-group variance of
the targets.  Dividing by it makes L_within ≈ O(1) and gives λ a consistent
interpretation across targets (EA vs IP) regardless of their physical scale.

When the denominator is below ε the loss is set to zero to avoid division by
near-zero values (e.g., groups whose targets are identical in this batch, or
singleton groups that got through the group-aware sampler for some edge-case
reason).

Singleton handling
------------------
Groups with only one member present in the current batch contribute zero to
both numerator and denominator (variance of a single point is zero), so they
affect neither the loss value nor the gradients.  They still receive gradient
from L_overall.

Gradient property
-----------------
For a group g with members {i₁, …, iₖ}:

    ∂L_within/∂ŷ_i  ∝  -(e_i - ē_g) * (1 - 1/k)   for member i

Summing over the group:

    Σ_{i∈g} ∂L_within/∂ŷ_i  ∝  -Σ_i (e_i - ē_g)  =  0

The within-group gradients sum to zero within each group: the loss is invariant
to adding the same constant to all predictions inside a group (e.g., a global
chemistry-baseline shift), so it only penalises *relative* prediction errors
between architectures within the same chemistry group.
"""
from __future__ import annotations

import torch
from torch import Tensor


def within_group_residual_loss(
    targets: Tensor,
    preds: Tensor,
    group_ids: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, dict]:
    """Compute the normalised within-group residual variance loss.

    Parameters
    ----------
    targets : Tensor, shape (N, T)
        Normalised (z-scored) target values.  NaN entries are ignored.
    preds : Tensor, shape (N, T)
        Model predictions in the same normalised space.
    group_ids : Tensor, shape (N,), dtype long
        Integer group label for each sample.  Samples with the same label
        belong to the same chemistry group.
    eps : float
        Stability constant added to the denominator.

    Returns
    -------
    loss : Tensor, scalar
        Normalised within-group residual variance, averaged over targets.
    info : dict
        Diagnostic scalars:
        - "n_groups_active" : number of groups with ≥2 members in this batch
        - "avg_group_size"  : mean group size (over active groups)
        - "loss_per_target" : Tensor (T,) with per-target loss values
    """
    N, T = targets.shape
    device = targets.device

    residuals = targets - preds  # (N, T)

    unique_groups = group_ids.unique()
    n_groups_active = 0
    total_count = 0

    num = torch.zeros(T, device=device)
    den = torch.zeros(T, device=device)

    for g in unique_groups:
        mask_g = group_ids == g  # (N,) bool
        k = mask_g.sum().item()
        if k < 2:
            continue  # singleton: contributes nothing

        n_groups_active += 1
        total_count += k

        e_g = residuals[mask_g]   # (k, T)
        y_g = targets[mask_g]     # (k, T)

        # Per-target finite masks (NaN targets excluded from both num and den)
        finite_g = y_g.isfinite()  # (k, T)

        e_g_masked = torch.where(finite_g, e_g, torch.zeros_like(e_g))
        y_g_masked = torch.where(finite_g, y_g, torch.zeros_like(y_g))
        k_t = finite_g.float().sum(dim=0).clamp(min=1.0)  # (T,)

        e_mean = e_g_masked.sum(dim=0) / k_t   # (T,)
        y_mean = y_g_masked.sum(dim=0) / k_t   # (T,)

        e_dev = torch.where(finite_g, e_g - e_mean.unsqueeze(0), torch.zeros_like(e_g))
        y_dev = torch.where(finite_g, y_g - y_mean.unsqueeze(0), torch.zeros_like(y_g))

        num = num + (e_dev ** 2).sum(dim=0)
        den = den + (y_dev ** 2).sum(dim=0)

    # Normalise per target; zero out where denominator is too small
    safe_den = den + eps
    loss_per_target = torch.where(den > eps, num / safe_den, torch.zeros_like(num))
    loss = loss_per_target.mean()

    avg_group_size = (total_count / n_groups_active) if n_groups_active > 0 else 0.0

    return loss, {
        "n_groups_active": n_groups_active,
        "avg_group_size": float(avg_group_size),
        "loss_per_target": loss_per_target.detach(),
    }
