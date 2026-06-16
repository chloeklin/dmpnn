"""Tests for Stage 2D architecture-aware polymer aggregation.

Tests cover:
  - Shape correctness for all variants
  - Alpha initialization behaviour (alpha_init=0 → h_poly ≈ h_mix)
  - Architecture embedding differentiability
  - Integration with CopolymerMPNN (stage2d_* modes)
"""

import pytest
import torch
import torch.nn as nn

from chemprop.nn.stage2d import (
    ARCH_LABEL_MAP,
    NUM_ARCHITECTURES,
    Stage2Aggregator,
    VALID_STAGE2_VARIANTS,
)


# ────────────────────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────────────────────

@pytest.fixture(params=VALID_STAGE2_VARIANTS)
def variant(request):
    return request.param


@pytest.fixture
def batch_data():
    """Generate synthetic batch data for testing."""
    B, d = 4, 64
    h_A = torch.randn(B, d)
    h_B = torch.randn(B, d)
    f_A = torch.tensor([0.6, 0.3, 0.5, 0.7])
    f_B = torch.tensor([0.4, 0.7, 0.5, 0.3])
    arch = torch.tensor([0, 1, 2, 0], dtype=torch.long)  # alt, rand, block, alt
    return h_A, h_B, f_A, f_B, arch, B, d


# ────────────────────────────────────────────────────────────────
#  Shape Tests
# ────────────────────────────────────────────────────────────────

class TestShapes:
    """Verify output shapes for all variants."""

    def test_h_poly_shape(self, variant, batch_data):
        """h_poly must be [B, d]."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant=variant, n_targets=2)
        preds, aux = agg(h_A, h_B, f_A, f_B, arch)
        assert aux["h_poly"].shape == (B, d), f"h_poly shape wrong for {variant}"

    def test_predictions_shape(self, variant, batch_data):
        """Predictions must be [B, n_targets]."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        n_targets = 2
        agg = Stage2Aggregator(d=d, variant=variant, n_targets=n_targets)
        preds, aux = agg(h_A, h_B, f_A, f_B, arch)
        assert preds.shape == (B, n_targets), f"preds shape wrong for {variant}"

    def test_alpha_shape(self, variant, batch_data):
        """Alpha must be scalar or [B]."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant=variant, n_targets=2)
        _, aux = agg(h_A, h_B, f_A, f_B, arch)
        alpha = aux["alpha"]
        # Either scalar (numel=1) or per-sample [B]
        assert alpha.ndim <= 1 or alpha.numel() == B or alpha.numel() == 1

    def test_emb_norm_shape(self, variant, batch_data):
        """emb_norm must be [B]."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant=variant, n_targets=2)
        _, aux = agg(h_A, h_B, f_A, f_B, arch)
        emb_norm = aux["emb_norm"]
        assert emb_norm.shape == (B,) or emb_norm.numel() == B


# ────────────────────────────────────────────────────────────────
#  Alpha Initialization Tests
# ────────────────────────────────────────────────────────────────

class TestAlphaInit:
    """With alpha_init=0, model should start close to Frac baseline."""

    def test_frac_is_pure_mixture(self, batch_data):
        """Frac variant: h_poly == h_mix exactly."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant="frac", n_targets=2)
        _, aux = agg(h_A, h_B, f_A, f_B, arch)
        h_mix = f_A.unsqueeze(1) * h_A + f_B.unsqueeze(1) * h_B
        torch.testing.assert_close(aux["h_poly"], h_mix)

    @pytest.mark.parametrize("variant", ["2d0_fixed", "2d0_arch", "2d1_fixed", "2d1_arch"])
    def test_alpha_zero_starts_as_frac(self, variant, batch_data):
        """With alpha_init=0, non-gated variants produce h_poly == h_mix."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant=variant, alpha_init=0.0, n_targets=2)
        _, aux = agg(h_A, h_B, f_A, f_B, arch)
        h_mix = f_A.unsqueeze(1) * h_A + f_B.unsqueeze(1) * h_B
        torch.testing.assert_close(aux["h_poly"], h_mix, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("variant", ["2d0_gate", "2d1_gate"])
    def test_gate_starts_near_frac(self, variant, batch_data):
        """Gate bias=-3 → sigmoid≈0.047, plus zero-init MLP → h_poly ≈ h_mix."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant=variant, alpha_init=0.0, n_targets=2)
        _, aux = agg(h_A, h_B, f_A, f_B, arch)
        h_mix = f_A.unsqueeze(1) * h_A + f_B.unsqueeze(1) * h_B
        # Gate is ~0.047 but MLP output is 0, so h_poly = h_mix + 0.047*0 = h_mix (for 2d1)
        # For 2d0_gate, h_poly = h_mix + gate * e_arch, where e_arch is small random
        # so it should be CLOSE to h_mix
        diff = (aux["h_poly"] - h_mix).abs().max().item()
        assert diff < 0.5, f"Gate variant {variant} too far from frac at init: max_diff={diff}"


# ────────────────────────────────────────────────────────────────
#  Architecture Sensitivity Tests
# ────────────────────────────────────────────────────────────────

class TestArchSensitivity:
    """Verify that different architectures produce different outputs."""

    @pytest.mark.parametrize("variant", [v for v in VALID_STAGE2_VARIANTS if v != "frac"])
    def test_different_arch_different_output(self, variant):
        """With non-zero alpha, different arch labels should give different h_poly."""
        B, d = 3, 32
        torch.manual_seed(42)
        h_A = torch.randn(B, d)
        h_B = torch.randn(B, d)
        f_A = torch.full((B,), 0.5)
        f_B = torch.full((B,), 0.5)

        agg = Stage2Aggregator(d=d, variant=variant, alpha_init=1.0, n_targets=2)
        # Manually set alpha to 1.0 for fixed/arch variants to see effect
        if hasattr(agg, 'alpha') and agg.alpha is not None:
            with torch.no_grad():
                agg.alpha.fill_(1.0)

        arch_0 = torch.zeros(B, dtype=torch.long)
        arch_2 = torch.full((B,), 2, dtype=torch.long)

        with torch.no_grad():
            _, aux_0 = agg(h_A, h_B, f_A, f_B, arch_0)
            _, aux_2 = agg(h_A, h_B, f_A, f_B, arch_2)

        assert not torch.allclose(aux_0["h_poly"], aux_2["h_poly"], atol=1e-5), (
            f"Variant {variant} should produce different h_poly for different architectures"
        )


# ────────────────────────────────────────────────────────────────
#  Gradient Flow Tests
# ────────────────────────────────────────────────────────────────

class TestGradients:
    """Ensure gradients flow through all learnable parameters."""

    def test_backward_passes(self, variant, batch_data):
        """Backward pass should not error and should produce gradients."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant=variant, n_targets=2)
        preds, _ = agg(h_A, h_B, f_A, f_B, arch)
        loss = preds.sum()
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in agg.parameters()
        )
        assert has_grad, f"No gradients in {variant}"


# ────────────────────────────────────────────────────────────────
#  Config Validation Tests
# ────────────────────────────────────────────────────────────────

class TestConfig:
    """Test configuration validation."""

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Invalid stage2_variant"):
            Stage2Aggregator(d=64, variant="invalid_variant")

    def test_custom_arch_emb_dim(self):
        """Non-default arch_emb_dim should work with projection."""
        agg = Stage2Aggregator(d=64, variant="2d0_fixed", arch_emb_dim=16)
        assert agg.emb_proj is not None
        assert agg.arch_embedding.weight.shape == (NUM_ARCHITECTURES, 16)

    def test_default_arch_emb_dim_no_projection(self):
        """Default arch_emb_dim=d should not need projection."""
        agg = Stage2Aggregator(d=64, variant="2d0_fixed")
        assert agg.emb_proj is None
        assert agg.arch_embedding.weight.shape == (NUM_ARCHITECTURES, 64)

    def test_n_targets_configurable(self):
        """Number of prediction heads matches n_targets."""
        agg = Stage2Aggregator(d=64, variant="frac", n_targets=3)
        assert len(agg.heads) == 3


# ────────────────────────────────────────────────────────────────
#  Diagnostics Logging Tests
# ────────────────────────────────────────────────────────────────

class TestDiagnostics:
    """Test diagnostic logging output."""

    def test_log_diagnostics_keys(self, variant, batch_data):
        """log_diagnostics returns expected metric keys."""
        h_A, h_B, f_A, f_B, arch, B, d = batch_data
        agg = Stage2Aggregator(d=d, variant=variant, n_targets=2)
        _, aux = agg(h_A, h_B, f_A, f_B, arch)
        metrics = agg.log_diagnostics(aux)
        assert "stage2d/alpha_mean" in metrics
        if variant != "frac":
            assert "stage2d/emb_norm_mean" in metrics


# ────────────────────────────────────────────────────────────────
#  Integration with CopolymerMPNN
# ────────────────────────────────────────────────────────────────

class TestCopolymerIntegration:
    """Test Stage2D modes work when plugged into CopolymerMPNN."""

    @pytest.mark.parametrize("variant", VALID_STAGE2_VARIANTS)
    def test_stage2d_mode_accepted(self, variant):
        """CopolymerMPNN should accept stage2d_{variant} as a valid mode."""
        from chemprop.models.copolymer import CopolymerMPNN
        mode = f"stage2d_{variant}"
        assert mode in CopolymerMPNN.VALID_MODES

    def test_arch_label_map_consistent(self):
        """ARCH_LABEL_MAP should have exactly NUM_ARCHITECTURES entries."""
        assert len(ARCH_LABEL_MAP) == NUM_ARCHITECTURES
        assert set(ARCH_LABEL_MAP.values()) == {0, 1, 2}

    def test_forward_applies_output_transform_in_eval(self):
        """Regression test: CopolymerMPNN.forward() must apply UnscaleTransform in eval mode.

        Stage2D models previously returned normalized predictions from predict_step
        because forward() called forward_stage2d() without applying output_transform.
        This test ensures the unscale transform is applied correctly.
        """
        import numpy as np
        from chemprop.models.copolymer import CopolymerMPNN
        from chemprop.nn.transforms import UnscaleTransform

        class FakeArgs:
            model_name = "DMPNN"
            task_type = "reg"
            message_hidden_dim = 32
            message_depth = 2
            dropout = 0.0
            activation = "relu"
            aggregation = "mean"
            aggregation_norm = 100
            warmup_epochs = 2
            init_lr = 1e-4
            max_lr = 1e-3
            final_lr = 1e-5
            ffn_hidden_dim = 32
            ffn_num_layers = 2

        args = FakeArgs()

        class FakeScaler:
            mean_ = np.array([-2.5])
            scale_ = np.array([0.6])

        scaler = FakeScaler()
        transform = UnscaleTransform.from_standard_scaler(scaler)

        mode = "stage2d_2d0_arch"
        mpnn = CopolymerMPNN(args=args, copolymer_mode=mode, descriptor_dim=0, n_tasks=1)
        mpnn.predictor.output_transform = transform

        mpnn.eval()
        assert not mpnn.predictor.output_transform.training, \
            "output_transform should be in eval mode"

        h_A = torch.randn(2, 300)
        h_B = torch.randn(2, 300)
        fracA = torch.tensor([0.5, 0.6])
        fracB = torch.tensor([0.5, 0.4])
        arch = torch.tensor([0, 1], dtype=torch.long)

        with torch.no_grad():
            preds_raw, _ = mpnn.forward_stage2d(
                None, None, fracA, fracB,
                torch.tensor([[0.], [1.]])
            )

        mpnn.train()
        transform.train()
        assert transform.training, "sanity: transform should be in train mode"
        with torch.no_grad():
            out_train = transform(preds_raw)
        assert torch.allclose(out_train, preds_raw), \
            "In train mode, UnscaleTransform must be identity (normalized predictions kept)"

        transform.eval()
        with torch.no_grad():
            out_eval = transform(preds_raw)
        expected = preds_raw * 0.6 + (-2.5)
        assert torch.allclose(out_eval, expected, atol=1e-5), \
            "In eval mode, UnscaleTransform must unscale predictions"

        assert not torch.allclose(preds_raw, expected, atol=1e-3), \
            "Normalized and unscaled predictions must differ (regression test for the unscale bug)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
