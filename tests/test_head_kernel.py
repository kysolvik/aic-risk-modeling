"""FusionDecoder head_kernel: configurable head receptive field."""

import pytest
import torch

from aic_risk_modeling.train import models


def _branches(h=16, w=16, c=5):
    return [models.get_pixel_mlp([h, w, c], "im_px", out_channels=8)]


def test_default_is_unchanged():
    """Default must reproduce the pre-feature module exactly, so checkpoints load."""
    d = models.decoder_fusion(_branches())
    assert d.head_kernel == 3
    assert d.receptive_field == 9
    for name in ("conv1", "conv2", "conv3", "conv4"):
        conv = getattr(d, name)
        assert conv.kernel_size == (3, 3), name
        assert conv.padding == (1, 1), name
    assert d.out_conv.kernel_size == (1, 1)


def test_kernel_sets_receptive_field_and_shape_is_preserved():
    for k, rf in [(1, 1), (3, 9), (5, 17), (9, 33)]:
        d = models.decoder_fusion(_branches(), head_kernel=k)
        assert d.receptive_field == rf, f"k={k}"
        for name in ("conv1", "conv2", "conv3", "conv4"):
            conv = getattr(d, name)
            assert conv.kernel_size == (k, k) and conv.padding == (k // 2, k // 2)
        out = d({"im_px": torch.randn(2, 16, 16, 5)})
        assert tuple(out.shape) == (2, 16, 16), f"k={k} changed output shape"


def test_kernel_one_is_actually_pointwise():
    """With k=1 a single-pixel input perturbation must not move any other pixel."""
    torch.manual_seed(0)
    d = models.decoder_fusion(_branches(), head_kernel=1).eval()
    x = torch.randn(1, 16, 16, 5)
    base = d({"im_px": x})
    x2 = x.clone()
    x2[0, 8, 8, :] += 5.0
    moved = (d({"im_px": x2}) - base).abs()[0] > 1e-6
    assert moved[8, 8], "centre pixel should respond"
    moved[8, 8] = False
    assert not moved.any(), "k=1 leaked into neighbouring pixels"


def test_kernel_three_reaches_exactly_nine_pixels():
    torch.manual_seed(0)
    d = models.decoder_fusion(_branches(h=32, w=32), head_kernel=3).eval()
    x = torch.randn(1, 32, 32, 5)
    x2 = x.clone()
    x2[0, 16, 16, :] += 5.0
    moved = (d({"im_px": x2}) - d({"im_px": x})).abs()[0] > 1e-6
    rows = torch.nonzero(moved.any(1)).flatten()
    cols = torch.nonzero(moved.any(0)).flatten()
    # 1 + 4*(3-1) = 9 -> reaches 4 px either side of centre.
    assert (rows.min().item(), rows.max().item()) == (12, 20), f"rows {rows.min()}..{rows.max()}"
    assert (cols.min().item(), cols.max().item()) == (12, 20), f"cols {cols.min()}..{cols.max()}"


def test_rejects_even_and_nonpositive_kernels():
    for bad in (0, 2, 4, -1):
        with pytest.raises(ValueError):
            models.decoder_fusion(_branches(), head_kernel=bad)


def test_trains_end_to_end():
    d = models.decoder_fusion(_branches(), head_kernel=5)
    out = d({"im_px": torch.randn(2, 16, 16, 5)})
    torch.nn.functional.binary_cross_entropy(out, torch.zeros_like(out)).backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all()
             for p in d.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0
