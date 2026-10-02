"""climate_loc_attn: location-conditioned climate cross-attention in MTSViT."""

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
LOC_DIM = 16
RANK = 4


def _build_branches(context=True):
    branches = [
        models.get_identity([STEPS, H, W, 6], "im_annual"),
        models.get_identity([H, W, 8], "im_single"),
        models.get_coord_fourier([1, 2], "md_single"),
    ]
    if context:
        branches.insert(1, models.get_identity([N_CTX, 5], "md_monthly"))
    return branches


def _build_model(climate_loc_attn=True, film_location="md_single",
                 context=True, **kwargs):
    return models.decoder_mtsvit(
        _build_branches(context), num_classes=1, embed_dim=EMBED,
        patch_size=PATCH, temporal_depth=1, spatial_depth=1, num_heads=2,
        climate_loc_attn=climate_loc_attn, film_location=film_location,
        loc_dim=LOC_DIM, loc_rank=RANK, **kwargs)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "im_annual": torch.randn(2, STEPS, H, W, 6, generator=g),
        "md_monthly": torch.randn(2, N_CTX, 5, generator=g),
        "im_single": torch.randn(2, H, W, 8, generator=g),
        "md_single": torch.randn(2, 1, 2, generator=g),
    }


def _wake(model, std=0.1):
    """Move the zero-init expansions so the conditioning becomes observable."""
    with torch.no_grad():
        for layer in model.temporal_layers:
            layer.loc_q_proj.weight.normal_(0.0, std)
            layer.loc_gate_proj.weight.normal_(0.0, std)


def test_identity_at_init():
    # Copy weights rather than seed-match: the flag adds modules and shifts the RNG stream.
    plain = _build_model(climate_loc_attn=False, film_location=None).eval()
    conditioned = _build_model().eval()
    missing, unexpected = conditioned.load_state_dict(plain.state_dict(),
                                                      strict=False)
    assert not unexpected, unexpected  # flag-off params are a strict subset
    inputs = _inputs()
    with torch.no_grad():
        assert torch.equal(plain(inputs), conditioned(inputs))


def test_existing_layer_keys_unchanged():
    plain = _build_model(climate_loc_attn=False, film_location=None)
    conditioned = _build_model()
    layer_keys = lambda m: {k for k in m.state_dict()
                            if k.startswith("temporal_layers.")}
    plain_keys, cond_keys = layer_keys(plain), layer_keys(conditioned)
    assert plain_keys <= cond_keys, plain_keys - cond_keys
    extra = cond_keys - plain_keys
    assert extra and all(".loc_" in k for k in extra), extra


def test_no_loc_leak_across_tiles():
    # (B, N, T, D) folds tile-major to (B*N, T, D); tile 0 must not see tile 1's location.
    torch.manual_seed(7)
    layer = models.CrossAttnTemporalLayer(EMBED, 2, loc_dim=LOC_DIM,
                                          loc_rank=RANK).eval()
    with torch.no_grad():
        layer.loc_q_proj.weight.normal_(0.0, 0.1)
        layer.loc_gate_proj.weight.normal_(0.0, 0.1)
    n, t = 3, STEPS
    x = torch.randn(2 * n, t, EMBED)
    context = torch.randn(2, N_CTX, EMBED)
    loc = torch.randn(2, LOC_DIM)
    with torch.no_grad():
        base = layer(x, context, loc)
        moved = layer(x, context, torch.stack([loc[0], loc[1] + 5.0]))
    assert torch.equal(base[:n], moved[:n])          # tile 0 untouched
    assert not torch.equal(base[n:], moved[n:])      # tile 1 moved


def test_climate_response_varies_by_location():
    # Same imagery and climate, different coords: the climate response must differ (stage 1).
    torch.manual_seed(7)
    model = _build_model().eval()
    _wake(model)
    inputs = _inputs()
    im = inputs["im_annual"][:1].expand(2, -1, -1, -1, -1).contiguous()
    ctx_a = dict(inputs, md_monthly=inputs["md_monthly"][:1].expand(2, -1, -1))
    ctx_b = dict(inputs, md_monthly=torch.randn(1, N_CTX, 5).expand(2, -1, -1))

    def response(m):
        with torch.no_grad():
            loc = m._loc_code(inputs) if m.climate_loc_attn else None
            a = m._encode_temporal(im, "im_annual", m._encode_context(ctx_a), loc)
            b = m._encode_temporal(im, "im_annual", m._encode_context(ctx_b), loc)
        return b - a

    delta = response(model)
    assert not torch.allclose(delta[0], delta[1]), \
        "climate response is identical at two different locations"

    plain = _build_model(climate_loc_attn=False, film_location=None).eval()
    delta_plain = response(plain)
    assert torch.equal(delta_plain[0], delta_plain[1]), \
        "unconditioned model should respond identically everywhere"


def test_gradients_reach_loc_encoder():
    torch.manual_seed(7)
    model = _build_model()
    # At init the zero-init expansions receive gradient themselves...
    out = model(_inputs())
    out.sum().backward()
    layer = model.temporal_layers[0]
    assert layer.loc_q_proj.weight.grad.abs().sum() > 0
    assert layer.loc_gate_proj.weight.grad.abs().sum() > 0
    # ...but pass none back to the shared encoder while still zero.
    assert model.loc_encoder[0].weight.grad.abs().sum() == 0
    assert layer.loc_down.weight.grad.abs().sum() == 0
    model.zero_grad()
    _wake(model)
    out = model(_inputs())
    out.sum().backward()
    assert model.loc_encoder[0].weight.grad.abs().sum() > 0
    assert layer.loc_down.weight.grad.abs().sum() > 0


def test_context_dropout():
    model = _build_model(context_dropout=1.0)
    inputs = _inputs()
    model.train()
    assert torch.count_nonzero(model._encode_context(inputs)) == 0
    model.eval()  # never drops at eval
    assert torch.count_nonzero(model._encode_context(inputs)) > 0
    off = _build_model(context_dropout=0.0)
    assert set(off.state_dict()) == set(model.state_dict())


def test_injection_flags():
    model = _build_model(loc_inject_q=False)
    layer = model.temporal_layers[0]
    assert layer.loc_q_proj is None
    assert layer.loc_gate_proj is not None
    assert not any("loc_q_proj" in k for k in model.state_dict())
    out = model(_inputs())
    assert tuple(out.shape) == (2, H, W), out.shape


def test_errors():
    with pytest.raises(ValueError, match='temporal-context'):
        _build_model(context=False)
    with pytest.raises(ValueError, match='film_location'):
        _build_model(film_location=None)
    with pytest.raises(ValueError, match='film_location'):
        _build_model(film_location="nope")


def test_strict_roundtrip():
    torch.manual_seed(7)
    a = _build_model()
    torch.manual_seed(11)
    b = _build_model()
    b.load_state_dict(a.state_dict())  # strict=True
    a.eval(), b.eval()
    inputs = _inputs()
    with torch.no_grad():
        assert torch.equal(a(inputs), b(inputs))


def test_decoder_config_plumbing():
    model = trainer.build_decoder(
        "mtsvit", _build_branches(),
        {"embed_dim": EMBED, "patch_size": PATCH, "temporal_depth": 1,
         "spatial_depth": 1, "num_heads": 2, "climate_loc_attn": True,
         "film_location": "md_single", "loc_dim": LOC_DIM, "loc_rank": RANK,
         "context_dropout": 0.2})
    out = model(_inputs())
    assert tuple(out.shape) == (2, H, W), out.shape


def test_real_config_builds(repo_config):
    config = repo_config("mtsvit_test_v29")
    real = trainer.build_decoder(
        config["decoder"],
        trainer.build_all_models(config["input_features"]),
        config.get("decoder_config"),
        num_classes=config.get("num_classes") or 1)
    assert real.climate_loc_attn and real.film_location == "md_single"
    assert not real.climate_film  # composable, but not co-enabled in v29
    assert len(real.temporal_layers) == 2
    layer = real.temporal_layers[0]
    assert layer.loc_down.in_features == real.loc_dim
    assert layer.loc_q_proj.out_features == config["decoder_config"]["embed_dim"]
    assert layer.loc_gate_proj.out_features == config["decoder_config"]["embed_dim"]
    assert real.context_names == ["md_monthly"], real.context_names
    assert "md_single" in [b.input_name for b in real.spatial_branches]


def test_old_checkpoint_still_loads():
    ckpt = os.path.join(_REPO_ROOT, "out", "mtsvit_test_v27.pt")
    if not os.path.exists(ckpt):
        pytest.skip("no local checkpoint")
    model = trainer.load_model(ckpt)
    assert not model.climate_loc_attn
