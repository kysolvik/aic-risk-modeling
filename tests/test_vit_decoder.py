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
    """One spatial image stack + one broadcast coordinate branch."""
    return [
        models.get_identity([H, W, CHANNELS * STEPS], "im_annual"),  # spatial
        models.get_identity([1, 2], "md_single"),                    # broadcast
    ]


def _build_model(num_classes=1):
    return models.decoder_vit(
        _build_branches(), num_classes=num_classes, embed_dim=EMBED,
        patch_size=PATCH, depth=1, num_heads=2)


def _inputs(batch=2):
    return {
        "im_annual": torch.randn(batch, H, W, CHANNELS * STEPS),
        "md_single": torch.randn(batch, 1, 2),
    }


def test_branch_routing():
    model = _build_model()
    assert model.broadcast_names == {"md_single"}
    assert (model.height, model.width) == (H, W)
    assert model.grid == (H // PATCH, W // PATCH)
    # in_channels = spatial (7*6) + broadcast (1*2)
    assert model.patch_embed.in_channels == CHANNELS * STEPS + 2


def test_forward_pass_shape_binary():
    model = _build_model(num_classes=1)
    out = model(_inputs(batch=2))
    assert out.shape == (2, H, W)
    assert torch.all((out >= 0) & (out <= 1))


def test_forward_pass_shape_multiclass():
    model = _build_model(num_classes=5)
    out = model(_inputs(batch=2))
    assert out.shape == (2, H, W, 5)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2, H, W), atol=1e-5)


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
        models.get_unet_lite([H, W, CHANNELS], "im_single_cnn"),
    ]
    with pytest.raises(ValueError, match='identity'):
        models.decoder_vit(branches, num_classes=1, embed_dim=EMBED,
                           patch_size=PATCH, depth=1, num_heads=2)


def test_requires_spatial_branch():
    branches = [models.get_identity([1, 2], "md_single")]  # broadcast only
    with pytest.raises(ValueError, match='spatial'):
        models.decoder_vit(branches, num_classes=1, embed_dim=EMBED,
                           patch_size=PATCH, depth=1, num_heads=2)
