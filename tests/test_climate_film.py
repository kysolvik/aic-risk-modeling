"""climate_film: climate x location FiLM head on the MTSViT fusion decoder."""

import os

import pytest
import torch

from aic_risk_modeling.train import models, trainer

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

# H, W must equal models.PATCH_SIZE (coord_fourier broadcasts to it).
H = W = models.PATCH_SIZE
PATCH = 8
EMBED = 32
STEPS = 4
N_CTX = 12
COND = 16


def _build_branches(context=True):
    branches = [
        models.get_identity([STEPS, H, W, 6], "im_annual"),
        models.get_identity([H, W, 8], "im_single"),
        models.get_coord_fourier([1, 2], "md_single"),
    ]
    if context:
        branches.insert(1, models.get_identity([N_CTX, 5], "md_monthly"))
    return branches


def _build_model(climate_film=True, film_location="md_single", context=True):
    return models.decoder_mtsvit(
        _build_branches(context), num_classes=1, embed_dim=EMBED,
        patch_size=PATCH, temporal_depth=1, spatial_depth=1, num_heads=2,
        climate_film=climate_film, film_location=film_location,
        film_cond_dim=COND)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "im_annual": torch.randn(2, STEPS, H, W, 6, generator=g),
        "md_monthly": torch.randn(2, N_CTX, 5, generator=g),
        "im_single": torch.randn(2, H, W, 8, generator=g),
        "md_single": torch.randn(2, 1, 2, generator=g),
    }


def test_identity_at_init():
    # Zero-init FiLM projections and gate: film-on with film-off weights is bitwise identical.
    torch.manual_seed(7)
    plain = _build_model(climate_film=False, film_location=None).eval()
    torch.manual_seed(7)
    filmed = _build_model().eval()
    missing, unexpected = filmed.load_state_dict(plain.state_dict(),
                                                 strict=False)
    assert not unexpected, unexpected  # film-off params are a strict subset
    inputs = _inputs()
    with torch.no_grad():
        assert torch.equal(plain(inputs), filmed(inputs))


def test_head_keys_unchanged():
    plain = _build_model(climate_film=False, film_location=None)
    filmed = _build_model()
    head_keys = lambda m: {k for k in m.state_dict() if k.startswith("head.")}
    assert head_keys(plain) == head_keys(filmed)


def test_climate_and_location_paths_are_live():
    torch.manual_seed(7)
    model = _build_model().eval()
    # Wake the zero-init gates so the paths are observable.
    with torch.no_grad():
        for film in model.films:
            film.proj.weight.normal_(0.0, 0.1)
        model.film_loc_gate.weight.normal_(0.0, 0.1)
    inputs = _inputs()
    with torch.no_grad():
        base = model(inputs)
        swapped = dict(inputs, md_monthly=torch.randn(2, N_CTX, 5))
        assert not torch.equal(model(swapped), base)
        moved = dict(inputs, md_single=inputs["md_single"] + 1.0)
        assert not torch.equal(model(moved), base)


def test_gradients_reach_film_and_conditioner():
    torch.manual_seed(7)
    model = _build_model()
    # At init the FiLM projections receive gradient themselves...
    out = model(_inputs())
    out.sum().backward()
    assert model.films[0].proj.weight.grad is not None
    assert model.films[0].proj.weight.grad.abs().sum() > 0
    # ...but pass none through until the FiLM weights move.
    assert model.film_conditioner[0].weight.grad.abs().sum() == 0
    model.zero_grad()
    with torch.no_grad():
        for film in model.films:
            film.proj.weight.normal_(0.0, 0.1)
    out = model(_inputs())
    out.sum().backward()
    assert model.film_conditioner[0].weight.grad.abs().sum() > 0


def test_errors():
    with pytest.raises(ValueError, match='temporal-context'):
        _build_model(context=False)
    with pytest.raises(ValueError, match='film_location'):
        _build_model(film_location="nope")


def test_decoder_config_plumbing():
    model = trainer.build_decoder(
        "mtsvit", _build_branches(),
        {"embed_dim": EMBED, "patch_size": PATCH, "temporal_depth": 1,
         "spatial_depth": 1, "num_heads": 2, "climate_film": True,
         "film_location": "md_single", "film_cond_dim": COND})
    out = model(_inputs())
    assert tuple(out.shape) == (2, H, W), out.shape


def test_real_config_builds(repo_config):
    config = repo_config("mtsvit_test_v25")
    real = trainer.build_decoder(
        config["decoder"],
        trainer.build_all_models(config["input_features"]),
        config.get("decoder_config"),
        num_classes=config.get("num_classes") or 1)
    assert real.climate_film and real.film_location == "md_single"
    assert real.head[0].in_channels == 96
    assert len(real.films) == 4


def test_old_checkpoint_still_loads():
    ckpt = os.path.join(_REPO_ROOT, "out", "mtsvit_test_v22.pt")
    if not os.path.exists(ckpt):
        pytest.skip("no local checkpoint")
    model = trainer.load_model(ckpt)
    assert not model.climate_film
