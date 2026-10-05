"""decoder_vit / VanillaViT: identity-only patch ViT baseline."""

import pytest
import torch

from aic_risk_modeling.train import models

H = W = 32          # small grid keeps the CPU forward pass cheap
PATCH = 8
EMBED = 32
STEPS = 6
CHANNELS = 7        # per spatial branch, time folded into channels


def _build_branches():
    """Two spatial image stacks (time folded into channels)."""
    return [
        models.get_identity([H, W, CHANNELS * STEPS], "im_annual"),
        models.get_identity([H, W, CHANNELS], "im_single"),
    ]


def _build_model():
    return models.decoder_vit(
        _build_branches(), embed_dim=EMBED, patch_size=PATCH, depth=1, num_heads=2)


def _inputs(batch=2):
    return {
        "im_annual": torch.randn(batch, H, W, CHANNELS * STEPS),
        "im_single": torch.randn(batch, H, W, CHANNELS),
    }


def test_branch_routing():
    model = _build_model()
    assert model.grid == (H // PATCH, W // PATCH)
    assert model.patch_embed.in_channels == CHANNELS * STEPS + CHANNELS


def test_forward_pass_shape():
    model = _build_model()
    out = model(_inputs(batch=2))
    assert out.shape == (2, H, W)
    assert torch.all((out >= 0) & (out <= 1))


def test_gradients_flow_to_patch_embed():
    model = _build_model()
    out = model(_inputs(batch=2))
    out.mean().backward()
    grad = model.patch_embed.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_encoder_branch_rejected():
    """A per-modality encoder (no input_shape) must be refused."""
    branches = [
        models.get_identity([H, W, CHANNELS], "im_annual"),
        models.get_pixel_mlp([H, W, CHANNELS], "im_single_cnn"),
    ]
    with pytest.raises(ValueError, match='identity'):
        models.decoder_vit(branches, embed_dim=EMBED,
                           patch_size=PATCH, depth=1, num_heads=2)


def test_requires_spatial_branch():
    branches = [models.get_identity([1, 2], "md_single")]  # non-spatial
    with pytest.raises(ValueError, match='spatial'):
        models.decoder_vit(branches, embed_dim=EMBED,
                           patch_size=PATCH, depth=1, num_heads=2)
