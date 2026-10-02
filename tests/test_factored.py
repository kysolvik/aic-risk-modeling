"""Factored model logit = gamma(t) + m(x,t) + s(x,t) + c(x,t): structural properties."""

import pytest
import torch

from aic_risk_modeling.train import factored, models, trainer

H = W = 32
OFFSETS = {2019: -0.10, 2020: 0.25, 2021: -0.15}


def _branches():
    return [
        models.get_pixel_temporal([4, H, W, 3], "im_annual", dim=8, depth=1, num_heads=2),
        models.get_pixel_mlp([H, W, 6], "im_single_cnn", out_channels=8),
        models.get_identity([12, 5], "md_monthly"),
        models.get_identity([1], "md_year"),
    ]


def _inputs(year=2020, batch=2, coords=False):
    x = {
        "im_annual": torch.randn(batch, 4, H, W, 3),
        "im_single_cnn": torch.randn(batch, H, W, 6),
        "md_monthly": torch.randn(batch, 12, 5),
        "md_year": torch.full((batch, 1), float(year)),
    }
    if coords:
        # md_single feeds year_gain directly; it need not be a routed branch.
        x["md_single"] = torch.randn(batch, 1, 2)
    return x


def _model(**kw):
    kw.setdefault("pixel_groups", ["im_annual", "im_single_cnn"])
    kw.setdefault("context_groups", ["md_monthly"])
    kw.setdefault("year_group", "md_year")
    kw.setdefault("year_offset", {"offsets": OFFSETS})
    return models.decoder_factored(_branches(), **kw)


def _gain_model(**kw):
    kw.setdefault("pixel_groups", ["im_annual", "im_single_cnn"])
    kw.setdefault("context_groups", ["md_monthly"])
    kw.setdefault("year_group", "md_year")
    kw.setdefault("year_offset", {"offsets": OFFSETS})
    kw.setdefault("year_gain_group", "md_single")
    kw.setdefault("year_gain", {})
    return models.decoder_factored(_branches(), **kw)


def test_forward_shape_and_range():
    out = _model()(_inputs())
    assert tuple(out.shape) == (2, H, W), out.shape
    assert torch.all((out >= 0) & (out <= 1))


def test_terms_are_additive():
    """sigmoid(gamma + year_gain + m + s + c) must BE the forward output, exactly."""
    m = _gain_model().eval()
    x = _inputs(coords=True)
    with torch.no_grad():
        terms = m.forward_terms(x)
        total = (terms["gamma"] + terms["year_gain"]
                 + terms["m"] + terms["s"] + terms["c"])
        assert torch.allclose(torch.sigmoid(total).squeeze(1), m(x), atol=1e-6)


def test_zeroing_a_term_moves_the_logit_by_exactly_that_term():
    """The property the per-pixel explanation rests on."""
    m = _gain_model().eval()
    x = _inputs(coords=True)
    with torch.no_grad():
        terms = m.forward_terms(x)
        full = sum(terms.values())
        for name in ("gamma", "year_gain", "m", "s", "c"):
            without = full - terms[name]
            assert torch.allclose(full - without, terms[name].expand_as(full), atol=1e-6), name


def test_susceptibility_is_strictly_pointwise():
    torch.manual_seed(0)
    s = factored.PixelSusceptibility(4, hidden=(8,)).eval()
    x = torch.randn(1, 4, H, W)
    x2 = x.clone()
    x2[0, :, 16, 16] += 5.0
    moved = (s(x2) - s(x)).abs()[0, 0] > 1e-6
    assert moved[16, 16], "centre must respond"
    moved[16, 16] = False
    assert not moved.any(), "s leaked into neighbours; it must be 1x1 only"


def test_local_context_excludes_the_centre_pixel():
    torch.manual_seed(0)
    c = factored.LocalContext(4, kernel=5, hidden=8).eval()
    torch.nn.init.normal_(c.out.weight, std=0.5)      # undo the zero-init no-op
    x = torch.randn(1, 4, H, W)
    x2 = x.clone()
    x2[0, :, 16, 16] += 5.0
    delta = (c(x2) - c(x)).abs()[0, 0]
    assert delta[16, 16] < 1e-6, f"centre leaked into c: {delta[16, 16]:.2e}"
    assert delta[16, 18] > 1e-6, "neighbours should respond"


def test_local_context_receptive_field_is_exactly_the_kernel():
    torch.manual_seed(0)
    for kernel in (3, 5, 9):
        c = factored.LocalContext(4, kernel=kernel, hidden=8).eval()
        torch.nn.init.normal_(c.out.weight, std=0.5)
        assert c.receptive_field == kernel
        x = torch.randn(1, 4, H, W)
        x2 = x.clone()
        x2[0, :, 16, 16] += 5.0
        moved = (c(x2) - c(x)).abs()[0, 0] > 1e-6
        rows = torch.nonzero(moved.any(1)).flatten()
        reach = kernel // 2
        assert (rows.min().item(), rows.max().item()) == (16 - reach, 16 + reach), \
            f"kernel {kernel}: rows {rows.min()}..{rows.max()}"


def test_local_kernel_one_is_the_pointwise_control():
    c = factored.LocalContext(4, kernel=1)
    assert c.receptive_field == 1
    assert torch.count_nonzero(c(torch.randn(2, 4, H, W))) == 0
    assert sum(p.numel() for p in c.parameters()) == 0, "k=1 must add no parameters"


def test_year_offset_lookup_and_frozen():
    y = factored.YearOffset(OFFSETS)
    got = y(torch.tensor([[2020.0], [2019.0], [2021.0]]))
    assert torch.allclose(got.flatten(), torch.tensor([0.25, -0.10, -0.15]), atol=1e-6)
    assert tuple(got.shape) == (3, 1, 1, 1)
    assert not any(p.requires_grad for p in y.parameters()), "gamma must be frozen by default"


def test_unknown_year_raises_rather_than_defaulting():
    """Silently returning 0.0 would mean 'average year' for a year we never fit."""
    y = factored.YearOffset(OFFSETS)
    for bad in (2012.0, 2026.0):
        with pytest.raises(KeyError):
            y(torch.tensor([[bad]]))


def test_gamma_shifts_predictions_between_years():
    m = _model().eval()
    with torch.no_grad():
        torch.manual_seed(0)
        base = _inputs(year=2019)
        low = m(base)
        high_in = dict(base, md_year=torch.full((2, 1), 2020.0))
        high = m(high_in)
    assert (high > low).all(), "the 2020 offset is larger, so every pixel must rise"


def test_year_gain_is_noop_at_init():
    m = _gain_model().eval()
    with torch.no_grad():
        term = m.forward_terms(_inputs(coords=True))["year_gain"]
    assert term.abs().max().item() == 0.0, "year_gain must start as an exact no-op"


def test_year_gain_is_mean_centered():
    m = _gain_model()
    torch.nn.init.normal_(m.year_gain.out.weight, std=0.5)     # undo the zero-init no-op
    torch.nn.init.normal_(m.year_gain.out.bias, std=0.5)

    m.train()
    g = m.year_gain(_inputs(coords=True, batch=8)["md_single"])
    assert g.mean().abs().item() < 1e-5, "g_res must be mean-zero across the batch in train mode"

    m.eval()
    coord = torch.randn(1, 1, 2)
    solo = m.year_gain(coord)
    batched = m.year_gain(torch.cat([coord, torch.randn(3, 1, 2)], dim=0))
    assert torch.allclose(solo[0], batched[0], atol=1e-6), \
        "at eval a chip's gain must not depend on batch composition"


def test_year_gain_uses_location_only():
    m = _gain_model().eval()
    torch.nn.init.normal_(m.year_gain.out.weight, std=0.5)     # make g_res non-trivial
    torch.nn.init.normal_(m.year_gain.out.bias, std=0.5)
    x = _inputs(coords=True)
    with torch.no_grad():
        base = m.forward_terms(x)["year_gain"]
        moved = m.forward_terms(dict(x, md_single=x["md_single"] + 1.0))["year_gain"]
        pixel = m.forward_terms(dict(x, im_annual=x["im_annual"] + 5.0))["year_gain"]
    assert (base - moved).abs().max() > 1e-6, "year_gain must respond to location"
    assert torch.allclose(base, pixel, atol=1e-6), "year_gain must not see pixel features"


def test_year_gain_requires_year_offset():
    """The gain multiplies gamma(t), so it is meaningless without a year offset."""
    with pytest.raises(ValueError, match='year_gain requires year_offset'):
        models.decoder_factored(
            _branches(),
            pixel_groups=["im_annual", "im_single_cnn"],
            context_groups=["md_monthly"], year_group="md_year",
            year_offset=None, year_gain_group="md_single", year_gain={})


def test_unrouted_group_raises():
    branches = _branches() + [models.get_pixel_mlp([H, W, 3], "im_extra", out_channels=4)]
    with pytest.raises(ValueError, match='im_extra'):
        models.decoder_factored(branches,
                                pixel_groups=["im_annual", "im_single_cnn"],
                                context_groups=["md_monthly"], year_group="md_year",
                                year_offset={"offsets": OFFSETS})


def test_extra_input_keys_are_ignored():
    m = _model().eval()
    x = _inputs()
    x["md_sidecar"] = torch.randn(2, 1, 2)
    with torch.no_grad():
        assert tuple(m(x).shape) == (2, H, W)


def test_trains_end_to_end():
    m = _model()
    out = m(_inputs())
    torch.nn.functional.binary_cross_entropy(out, torch.zeros_like(out)).backward()
    grads = [(n, p.grad) for n, p in m.named_parameters() if p.requires_grad]
    assert grads, "no trainable parameters"
    for n, g in grads:
        assert g is not None, f"{n} got no gradient"
        assert torch.isfinite(g).all(), f"{n} has non-finite gradient"


def test_builds_from_a_config_dict():
    """The trainer path: input_features -> branches -> decoder_config -> decoder."""
    branches = trainer.build_all_models({
        "im_annual": {"model_type": "pixel_temporal", "feature_names": ["a"] * 3,
                      "timesteps": [-2, -1], "shape": [H, W], "stack_timesteps": True,
                      "model_kwargs": {"dim": 8, "depth": 1, "num_heads": 2}},
        "im_monthly_coarse": {"model_type": "coarse_temporal", "feature_names": ["c"] * 6,
                              "timesteps": [-2, -1], "shape": [H, W], "stack_timesteps": True,
                              "model_kwargs": {"grid": 8, "dim": 8, "depth": 1, "num_heads": 2}},
        "im_single_cnn": {"model_type": "pixel_mlp", "feature_names": ["s"] * 6,
                          "timesteps": [], "shape": [H, W],
                          "model_kwargs": {"out_channels": 8}},
        "md_monthly": {"model_type": "identity", "feature_names": ["m"] * 5,
                       "timesteps": [], "shape": [12]},
        "md_year": {"model_type": "identity", "feature_names": ["md_year"],
                    "timesteps": [], "shape": [1]},
    })
    decoder = trainer.build_decoder("factored", branches, {
        "pixel_groups": ["im_annual", "im_monthly_coarse", "im_single_cnn"],
        "context_groups": ["md_monthly"],
        "year_group": "md_year",
        "local_kernel": 9, "coarse_grid": 4,
        "year_offset": {"offsets": OFFSETS},
        "year_gain_group": "md_single",
        "year_gain": {"num_freqs": 16, "sigma": 1.0, "hidden": 64},
    })
    assert decoder.receptive_field == 9
    assert decoder.year_gain is not None
    out = decoder({
        "im_annual": torch.randn(2, 2, H, W, 3),
        "im_monthly_coarse": torch.randn(2, 2, H, W, 6),
        "im_single_cnn": torch.randn(2, H, W, 6),
        "md_monthly": torch.randn(2, 12, 5),
        "md_year": torch.full((2, 1), 2020.0),
        "md_single": torch.randn(2, 1, 2),
    })
    assert tuple(out.shape) == (2, H, W), out.shape


def test_pixel_temporal_is_pointwise_and_coarse_temporal_is_not():
    torch.manual_seed(0)
    enc = models.get_pixel_temporal([4, H, W, 3], "x", dim=8, depth=1, num_heads=2).eval()
    assert enc.out_channels == 8
    x = torch.randn(1, 4, H, W, 3)
    x2 = x.clone()
    x2[0, :, 16, 16, :] += 5.0
    moved = (enc(x2) - enc(x)).abs()[0].sum(-1) > 1e-6
    assert moved[16, 16] and moved.sum() == 1, "pixel_temporal must not mix spatially"

    coarse = models.get_coarse_temporal([4, H, W, 3], "x", grid=8, dim=8, depth=1,
                                        num_heads=2).eval()
    out = coarse(x)
    assert tuple(out.shape) == (1, H, W, 8), out.shape
