"""eval/attribution: driver-spec resolution, baselines, copy-on-write occlusion, Shapley bands."""

import re
from collections import OrderedDict

import pytest
import torch

from aic_risk_modeling.eval import attribution
from aic_risk_modeling.train import losses


# One timestep-stacked group and one flat group with a transformed feature (like im_gov_type).
INPUT_FEATURES = {
    'g_time': {
        'feature_names': ['f_a', 'f_b'],
        'timesteps': [-2, -1],
        'shape': [4, 4],
        'normalize': True,
        'transforms': {},
    },
    'g_flat': {
        'feature_names': ['f_c', 'f_gt'],
        'timesteps': [],
        'shape': [4, 4],
        'normalize': True,
        'transforms': {'f_gt': 'gt0'},
    },
}
CONFIG = {'input_features': INPUT_FEATURES}

SPEC_JSON = {
    'drivers': {
        'drv_a': [['g_time', 'f_a']],
        'drv_b': [['g_flat', 'f_c']],
    },
    'baseline_overrides': {'g_flat/f_gt': 0.0},
}


def _inputs(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        'g_time': torch.rand((batch, 2, 4, 4, 2), generator=g) * 2 - 1,
        'g_flat': torch.rand((batch, 4, 4, 2), generator=g) * 2 - 1,
    }


class AdditiveModel(torch.nn.Module):
    """Probability-additive across features; coefficients keep p in (0, 1)."""

    def forward(self, inputs):
        gt = inputs['g_time']
        gf = inputs['g_flat']
        return (0.3
                + 0.10 * gt[..., 0].mean(dim=1)
                + 0.05 * gt[..., 1].mean(dim=1)
                + 0.08 * gf[..., 0]
                + 0.02 * gf[..., 1])


class InteractionModel(torch.nn.Module):
    """Adds an a*b interaction so OAT deltas double-count it."""

    def forward(self, inputs):
        a = inputs['g_time'][..., 0].mean(dim=1)
        b = inputs['g_flat'][..., 0]
        return 0.3 + 0.1 * a + 0.1 * b + 0.2 * a * b


class ConstModel(torch.nn.Module):
    def __init__(self, value, shape=()):
        super().__init__()
        self.value = value
        self.shape = shape

    def forward(self, inputs):
        batch = inputs['g_time'].shape[0]
        return torch.full((batch, 4, 4) + self.shape, self.value)


def _raises_value_error(fn, snippet):
    with pytest.raises(ValueError, match=re.escape(snippet)):
        fn()


def test_spec_resolves_indices():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    assert spec.drivers == OrderedDict(
        [('drv_a', [('g_time', 0)]), ('drv_b', [('g_flat', 0)])])
    assert spec.baseline_overrides == {('g_flat', 'f_gt'): 0.0}


def test_spec_rejects_unknown_and_duplicates():
    bad_group = {'drivers': {'d': [['nope', 'f_a']]}}
    _raises_value_error(
        lambda: attribution.resolve_driver_spec(bad_group, INPUT_FEATURES),
        "unknown input group")
    bad_feature = {'drivers': {'d': [['g_time', 'nope']]}}
    _raises_value_error(
        lambda: attribution.resolve_driver_spec(bad_feature, INPUT_FEATURES),
        "not in 'g_time'")
    dupe = {'drivers': {'d1': [['g_time', 'f_a']], 'd2': [['g_time', 'f_a']]}}
    _raises_value_error(
        lambda: attribution.resolve_driver_spec(dupe, INPUT_FEATURES),
        "claimed by both")


def test_default_spec_against_real_config(repo_config):
    config = repo_config('mtsvit_test_v22')
    spec = attribution.resolve_driver_spec(None, config['input_features'])
    assert list(spec.drivers) == list(attribution.DEFAULT_DRIVERS)
    baselines = attribution.resolve_baselines(spec, config)
    n_features = sum(len(c) for c in spec.drivers.values())
    assert len(baselines) == n_features
    gov_idx = config['input_features']['im_single_cnn'][
        'feature_names'].index('im_gov_type')
    assert baselines[('im_single_cnn', gov_idx)] == 0.0
    assert all(v == 0.0 for v in baselines.values())


def test_baselines_normalized_and_overrides():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    assert baselines == {('g_time', 0): 0.0, ('g_flat', 0): 0.0}

    with_gt = {'drivers': {'d': [['g_flat', 'f_gt']]}}
    spec = attribution.resolve_driver_spec(with_gt, INPUT_FEATURES)
    _raises_value_error(
        lambda: attribution.resolve_baselines(spec, CONFIG),
        "no baseline_overrides entry")

    with_gt['baseline_overrides'] = {'g_flat/f_gt': 0.25}
    spec = attribution.resolve_driver_spec(with_gt, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    assert baselines == {('g_flat', 1): 0.25}


def test_occlude_copy_on_write():
    inputs = _inputs()
    original = {k: v.clone() for k, v in inputs.items()}
    out = attribution.occlude(inputs, [('g_time', 1)], {('g_time', 1): 0.0})
    assert torch.all(out['g_time'][..., 1] == 0.0)
    assert torch.equal(out['g_time'][..., 0], original['g_time'][..., 0])
    for k in inputs:
        assert torch.equal(inputs[k], original[k])
    assert out['g_flat'] is inputs['g_flat']


def test_band_order_and_identity():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    inputs = _inputs()
    bands, names = attribution.attribution_bands(
        InteractionModel(), inputs, spec, baselines, pos_weight=1.0)
    assert names == ['risk', 'delta_drv_a', 'delta_drv_b',
                     'residual_interactions', 'risk_all_drivers_baseline']
    assert bands.shape == (2, 4, 4, 5)
    # risk - all_baseline == sum(deltas) + residual, exactly by construction.
    lhs = bands[..., 0] - bands[..., -1]
    rhs = bands[..., 1] + bands[..., 2] + bands[..., 3]
    assert torch.allclose(lhs, rhs, atol=1e-6)
    assert bands[..., 3].abs().max() > 1e-3


def test_additive_model_zero_residual():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    inputs = _inputs()
    bands, _ = attribution.attribution_bands(
        AdditiveModel(), inputs, spec, baselines, pos_weight=1.0)
    assert bands[..., 3].abs().max() < 1e-6
    expected_a = 0.10 * inputs['g_time'][..., 0].mean(dim=1)
    expected_b = 0.08 * inputs['g_flat'][..., 0]
    assert torch.allclose(bands[..., 1], expected_a, atol=1e-6)
    assert torch.allclose(bands[..., 2], expected_b, atol=1e-6)


def test_deflation_applied():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    bands, _ = attribution.attribution_bands(
        ConstModel(0.9), _inputs(), spec, baselines, pos_weight=9.0)
    expected = losses.deflate_probs(torch.tensor(0.9), 9.0)
    assert torch.allclose(bands[..., 0], expected.expand(2, 4, 4), atol=1e-6)
    assert torch.all(bands[..., 1] == 0.0)
    assert torch.all(bands[..., 2] == 0.0)


def test_rejects_multiclass_output():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    _raises_value_error(
        lambda: attribution.attribution_bands(
            ConstModel(0.5, shape=(3,)), _inputs(), spec, baselines),
        "binary")


def test_shapley_layout():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    bands, names = attribution.shapley_bands(
        InteractionModel(), _inputs(), spec, baselines, pos_weight=1.0)
    assert names == ['risk', 'shapley_drv_a', 'shapley_drv_b',
                     'residual_interactions', 'risk_all_drivers_baseline']
    assert bands.shape == (2, 4, 4, 5)


def test_shapley_equals_oat_for_additive_model():
    # No interactions -> Shapley value == single OAT delta.
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    inputs = _inputs()
    oat, _ = attribution.attribution_bands(
        AdditiveModel(), inputs, spec, baselines, pos_weight=1.0)
    shap, _ = attribution.shapley_bands(
        AdditiveModel(), inputs, spec, baselines, pos_weight=1.0)
    assert torch.allclose(shap[..., 0], oat[..., 0], atol=1e-6)   # risk
    assert torch.allclose(shap[..., 1], oat[..., 1], atol=1e-6)   # drv_a
    assert torch.allclose(shap[..., 2], oat[..., 2], atol=1e-6)   # drv_b
    assert torch.allclose(shap[..., -1], oat[..., -1], atol=1e-6)  # all_baseline
    assert shap[..., 3].abs().max() < 1e-6                        # residual ~0


def test_shapley_interaction_split():
    # 0.3 + 0.1a + 0.1b + 0.2ab: Shapley splits 0.2ab evenly; OAT gives it to each.
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    inputs = _inputs()
    bands, _ = attribution.shapley_bands(
        InteractionModel(), inputs, spec, baselines, pos_weight=1.0)
    a = inputs['g_time'][..., 0].mean(dim=1)
    b = inputs['g_flat'][..., 0]
    assert torch.allclose(bands[..., 1], 0.1 * a + 0.1 * a * b, atol=1e-6)
    assert torch.allclose(bands[..., 2], 0.1 * b + 0.1 * a * b, atol=1e-6)
    assert bands[..., 3].abs().max() < 1e-6  # residual ~0 despite interaction


def test_shapley_deflation_applied():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    bands, _ = attribution.shapley_bands(
        ConstModel(0.9), _inputs(), spec, baselines, pos_weight=9.0)
    expected = losses.deflate_probs(torch.tensor(0.9), 9.0)
    assert torch.allclose(bands[..., 0], expected.expand(2, 4, 4), atol=1e-6)
    assert torch.all(bands[..., 1] == 0.0)
    assert torch.all(bands[..., 2] == 0.0)
    assert torch.all(bands[..., 3] == 0.0)


def test_shapley_rejects_multiclass():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    _raises_value_error(
        lambda: attribution.shapley_bands(
            ConstModel(0.5, shape=(3,)), _inputs(), spec, baselines),
        "binary")


def test_shapley_sampling_efficiency_and_determinism():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    inputs = _inputs()
    bands, _ = attribution.shapley_bands(
        InteractionModel(), inputs, spec, baselines, pos_weight=1.0,
        samples=16, seed=0)
    # Sampled efficiency is exact: permutation marginals telescope.
    lhs = bands[..., 0] - bands[..., -1]
    rhs = bands[..., 1] + bands[..., 2]
    assert torch.allclose(lhs, rhs, atol=1e-6)
    assert bands[..., 3].abs().max() < 1e-6
    again, _ = attribution.shapley_bands(
        InteractionModel(), inputs, spec, baselines, pos_weight=1.0,
        samples=16, seed=0)
    assert torch.equal(bands, again)


def test_shapley_sampling_converges():
    spec = attribution.resolve_driver_spec(SPEC_JSON, INPUT_FEATURES)
    baselines = attribution.resolve_baselines(spec, CONFIG)
    inputs = _inputs()
    exact, _ = attribution.shapley_bands(
        InteractionModel(), inputs, spec, baselines, pos_weight=1.0)
    sampled, _ = attribution.shapley_bands(
        InteractionModel(), inputs, spec, baselines, pos_weight=1.0,
        samples=500, seed=0)
    assert torch.allclose(sampled, exact, atol=2e-2)


# Real FactoredFireModel with a per-location year gain: year term = gamma(t) * (1 + g_res).
FH = FW = 8
YEAR_OFFSETS = {2023: -0.35, 2024: 0.22}
YEAR_SPLIT = {'clim': {'2023': -0.30, '2024': 0.20},
              'fire': {'2023': -0.05, '2024': 0.02}}
FACTORED_FEATURES = {
    'im_annual': {'feature_names': ['a0', 'a1', 'a2']},
    'im_single_cnn': {'feature_names': ['s0', 's1', 's2', 's3']},
    'md_monthly': {'feature_names': ['m0', 'm1', 'm2']},
}
FACTORED_SPEC = {
    'drivers': {
        'clim': [['md_monthly', 'm0'], ['md_monthly', 'm1'], ['im_annual', 'a0']],
        'fire': [['im_single_cnn', 's0'], ['im_annual', 'a1']],
        'veg': [['im_single_cnn', 's1'], ['im_annual', 'a2']],
    },
    'year_terms': YEAR_SPLIT,
}


def _factored(seed=0):
    from aic_risk_modeling.train import models
    torch.manual_seed(seed)
    branches = [
        models.get_pixel_temporal([2, FH, FW, 3], 'im_annual', dim=8, depth=1, num_heads=2),
        models.get_pixel_mlp([FH, FW, 4], 'im_single_cnn', out_channels=8),
        models.get_identity([6, 3], 'md_monthly'),
        models.get_identity([1], 'md_year'),
    ]
    m = models.decoder_factored(
        branches, pixel_groups=['im_annual', 'im_single_cnn'],
        context_groups=['md_monthly'], year_group='md_year',
        year_offset={'offsets': YEAR_OFFSETS}, year_gain_group='md_single', year_gain={})
    torch.nn.init.normal_(m.year_gain.out.weight, std=0.5)   # undo the zero-init no-op
    torch.nn.init.normal_(m.year_gain.out.bias, std=0.5)
    return m.eval()


def _factored_inputs(years=(2023, 2024), seed=0):
    g = torch.Generator().manual_seed(seed)
    b = len(years)
    return {
        'im_annual': torch.randn((b, 2, FH, FW, 3), generator=g),
        'im_single_cnn': torch.randn((b, FH, FW, 4), generator=g),
        'md_monthly': torch.randn((b, 6, 3), generator=g),
        'md_year': torch.tensor([[float(y)] for y in years]),
        'md_single': torch.randn((b, 1, 2), generator=g),
    }


def _factored_setup(spec=FACTORED_SPEC):
    driver_spec = attribution.resolve_driver_spec(spec, FACTORED_FEATURES)
    baselines = {c: 0.0 for chans in driver_spec.drivers.values() for c in chans}
    return driver_spec, baselines


def test_year_terms_parse_and_reject_unknown_driver():
    spec, _ = _factored_setup()
    assert spec.year_terms['clim'] == {2023: -0.30, 2024: 0.20}
    bad = dict(FACTORED_SPEC, year_terms={'nope': {'2023': 0.0}})
    _raises_value_error(
        lambda: attribution.resolve_driver_spec(bad, FACTORED_FEATURES), 'nope')


def test_year_terms_absent_is_unchanged():
    """Without year_terms the factored path is the plain model(inputs) path."""
    no_year = {k: v for k, v in FACTORED_SPEC.items() if k != 'year_terms'}
    spec, baselines = _factored_setup(no_year)
    model, inputs = _factored(), _factored_inputs()
    bands, _ = attribution.shapley_bands(model, inputs, spec, baselines, pos_weight=10.0)
    with torch.no_grad():
        risk = losses.deflate_probs(model(inputs), 10.0)
    assert torch.equal(bands[..., 0], risk)


def test_year_terms_risk_baseline_and_efficiency():
    spec, baselines = _factored_setup()
    model, inputs = _factored(), _factored_inputs()
    w = 10.0
    for fn in (attribution.shapley_bands, attribution.attribution_bands):
        bands, names = fn(model, inputs, spec, baselines, pos_weight=w)
        with torch.no_grad():
            risk = losses.deflate_probs(model(inputs), w)
            occl = attribution.occlude(inputs, list(baselines), baselines)
            t = model.forward_terms(occl)
            no_year = losses.deflate_probs(
                torch.sigmoid((t['m'] + t['s'] + t['c']).squeeze(1)), w)
        assert torch.allclose(bands[..., 0], risk, atol=1e-6), fn.__name__
        assert torch.allclose(bands[..., -1], no_year, atol=1e-6), fn.__name__
        if fn is attribution.shapley_bands:
            assert bands[..., names.index('residual_interactions')].abs().max() < 1e-6


def test_year_terms_move_signal_into_owner():
    model, inputs = _factored(), _factored_inputs()
    with_year, baselines = _factored_setup()
    without = {k: v for k, v in FACTORED_SPEC.items() if k != 'year_terms'}
    plain, _ = _factored_setup(without)
    a, names = attribution.shapley_bands(model, inputs, with_year, baselines, pos_weight=1.0)
    b, _ = attribution.shapley_bands(model, inputs, plain, baselines, pos_weight=1.0)
    ci = names.index('shapley_clim')
    # 2024 has a positive clim component, so clim must gain on that chip
    assert (a[1, ..., ci] - b[1, ..., ci]).mean() > 0
    for bands in (a, b):
        assert torch.allclose(bands[..., 1:-2].sum(-1) + bands[..., -1], bands[..., 0], atol=1e-6)


def test_year_terms_mismatch_raises():
    model, inputs = _factored(), _factored_inputs()
    wrong = dict(FACTORED_SPEC, year_terms={'clim': {'2023': -0.35, '2024': 0.0}})
    spec, baselines = _factored_setup(wrong)
    _raises_value_error(
        lambda: attribution.shapley_bands(model, inputs, spec, baselines), 'another gamma fit')
    short = dict(FACTORED_SPEC, year_terms={'clim': {'2023': -0.35}})
    spec, baselines = _factored_setup(short)
    _raises_value_error(
        lambda: attribution.shapley_bands(model, inputs, spec, baselines), '2024')
