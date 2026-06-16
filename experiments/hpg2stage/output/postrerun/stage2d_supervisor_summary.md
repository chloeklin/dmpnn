# Stage 2D: Supervisor Summary

Stage 2D investigated whether incorporating copolymer architecture information (alternating, random, block) improves property predictions for electron affinity (EA) and ionisation potential (IP). We compared three model tiers: **Frac** (composition-only baseline), **2D0** (global architecture offset via a learned embedding), and **2D1** (chemistry-conditioned architecture modeling via an interaction MLP that conditions on both monomer identities and architecture). Seven variants were evaluated across 5-fold architecture-held-out cross-validation.

**Architecture matters.** All six architecture-aware variants significantly outperform Frac (p < 0.05 on every variant), improving overall R² by +0.006–0.008 for EA and +0.013–0.016 for IP.

**2D1 provides a modest but meaningful improvement over 2D0.** Overall R² differences are small and not statistically significant (EA: Δ = +0.0010, p = 0.51; IP: Δ = +0.0014, p = 0.12). However, on the more targeted architecture-deviation metric — which measures how well a model captures property differences *between* architectures sharing the same monomers — 2D1 significantly outperforms 2D0 for both targets (EA: Δ = +0.019, p = 0.027; IP: Δ = +0.015, p = 0.007).

**Recommended model: 2D1-arch** (per-architecture alpha × interaction MLP). It achieves the highest mean R² (0.9805) and significantly better architecture-deviation predictions. The additional complexity is justified when architecture-specific prediction accuracy matters, which is the core scientific question of this stage.
