"""decoder_film: FiLM (climate x location) conditioned fusion decoder."""

import pytest
import torch

from aic_risk_modeling.train import models

H = W = 32
N_CTX = 12
COND = 16


def _build_model(num_classes=1, location_input="md_single"):
    branches = [
        models.get_identity([N_CTX, 5], "md_monthly"),    # climate (FiLM cond)
        models.get_identity([H, W, 64], "im_single"),     # spatial (identity)
        models.get_unet_lite([H, W, 7], "im_single_cnn"),  # spatial (cnn)
    ]
    if location_input is not None:
        branches.insert(1, models.get_identity([1, 2], location_input))
    return models.decoder_film(branches, num_classes=num_classes, cond_dim=COND,
                               location_input=location_input)


def _inputs():
    return {
        "md_monthly": torch.randn(2, N_CTX, 5),
        "md_single": torch.randn(2, 1, 2),
        "im_single": torch.randn(2, H, W, 64),
        "im_single_cnn": torch.randn(2, H, W, 7),
    }


def test_film_is_identity_at_init():
    film = models.FiLM(COND, 4)
    x = torch.randn(2, 4, H, W)
    cond = torch.randn(2, COND)
    assert torch.allclose(film(x, cond), x), "FiLM should start as identity"


def test_gate_is_identity_at_init():
    model = _build_model(location_input="md_single")
    assert torch.count_nonzero(model.loc_gate.weight) == 0
    assert torch.count_nonzero(model.loc_gate.bias) == 0


def test_routing_separates_climate_location_and_spatial():
    model = _build_model(location_input="md_single")
    spatial_names = [b.input_name for b in model.spatial_branches]
    assert model.context_names == ["md_monthly"], model.context_names
    assert "md_single" not in model.context_names
    assert "md_single" not in spatial_names
    assert "im_single" in spatial_names and "im_single_cnn" in spatial_names, \
        spatial_names
    # Climate encoder consumes flattened (T, F) indices; head sees 64 + 16 chans.
    assert model.conditioner[0].in_features == N_CTX * 5
    assert model.conv1.in_channels == 64 + 16


def test_requires_climate_and_spatial():
    only_spatial = [models.get_identity([H, W, 8], "im_single")]
    with pytest.raises(ValueError):
        models.decoder_film(only_spatial)
    only_climate = [models.get_identity([N_CTX, 5], "md_monthly")]
    with pytest.raises(ValueError):
        models.decoder_film(only_climate)


def test_forward_shapes_with_and_without_location():
    inputs = _inputs()
    for loc in ("md_single", None):
        binary = _build_model(1, location_input=loc).eval()
        with torch.no_grad():
            out = binary(inputs)
        assert tuple(out.shape) == (2, H, W), (loc, out.shape)
        multi = _build_model(5, location_input=loc).eval()
        with torch.no_grad():
            out5 = multi(inputs)
        assert tuple(out5.shape) == (2, H, W, 5), (loc, out5.shape)
