"""Post-hoc per-driver attribution maps via one-at-a-time (OAT) occlusion.

Explains a trained binary risk model's map: each named "driver" is a semantic
group of input features (possibly spanning several model input groups, e.g.
vegetation indices appear in im_annual, im_monthly, and im_single_cnn).
Replacing a driver's features with a grid-average baseline and re-running the
model gives its per-pixel contribution:

    delta_d = p(x) - p(x with driver d at baseline)

Positive delta = the driver's current condition raises risk relative to
grid-typical conditions. An all-drivers-at-baseline forward makes the
interaction mismatch explicit:

    residual = (p(x) - p(all drivers at baseline)) - sum_d delta_d

so band identity `risk - risk_all_baseline == sum(deltas) + residual` holds
exactly by construction.

All probabilities are deflated (`train.losses.deflate_probs`) before
differencing: models trained with weighted BCE (pos_weight w) predict the
inflated optimum q = w*p/(w*p+1-p).

Baselines: the tf.data pipeline standardizes each feature with global
training-set statistics, so the grid-average baseline of a normalized feature
is exactly 0.0 in tensor space. Features the normalizer skips because they
have a config transform (e.g. im_gov_type's gt0) have no recoverable
tensor-space mean -- mean(transform(x)) != transform(mean(x)) -- so they
require an explicit `baseline_overrides` entry; `resolve_baselines` raises
otherwise.

Caveats:
- OAT deltas are not Shapley values: correlated drivers each absorb their
  shared signal, so deltas can double-count; the residual band shows the
  total mismatch against the all-baseline jump.
- Occluded inputs are off-manifold (grid-average vegetation over real
  terrain never occurs); deltas reflect model extrapolation there.
- delta ~= 0 means the *model* does not use the driver, not that the driver
  is physically irrelevant.
"""

import dataclasses
from collections import OrderedDict

import torch

from aic_risk_modeling.train import data_norm
from aic_risk_modeling.train.losses import deflate_probs

# Driver name -> [(input group, feature name), ...]. Semantic groups that
# cross-cut the model's input branches; feature names must match the config's
# input_features exactly (v11-lineage configs). md_single (location) is
# deliberately excluded: "average location" is not a meaningful counterfactual.
DEFAULT_DRIVERS = OrderedDict([
    ('climate_indices', [
        ('md_monthly', 'md_mei'), ('md_monthly', 'md_oni'),
        ('md_monthly', 'md_soi'), ('md_monthly', 'md_tna'),
        ('md_monthly', 'md_amo')]),
    ('weather_drought', [
        ('im_monthly', 'im_pdsi'), ('im_monthly', 'im_tmmn'),
        ('im_monthly', 'im_tmmx'), ('im_monthly', 'im_vpd'),
        ('im_monthly', 'im_def'),
        ('im_single_cnn', 'im_def_-3'), ('im_single_cnn', 'im_pdsi_-3')]),
    ('vegetation', [
        ('im_annual', 'im_EVI'), ('im_annual', 'im_NDVI'),
        ('im_monthly', 'im_NDVI_monthly'), ('im_monthly', 'im_EVI_monthly'),
        ('im_single_cnn', 'im_EVI_-1'), ('im_single_cnn', 'im_NDVI_-1')]),
    ('fire_history', [
        ('im_annual', 'im_BurnDate'), ('im_single_cnn', 'im_BurnDate_-1')]),
    ('landuse_deforestation', [
        ('im_annual', 'im_ag'), ('im_annual', 'im_pasture'),
        ('im_annual', 'im_forest'),
        ('im_single_cnn', 'im_loss'), ('im_single_cnn', 'im_lossyear'),
        ('im_single_cnn', 'im_alert'), ('im_single_cnn', 'im_alertdate'),
        ('im_single_cnn', 'im_ag_-1'), ('im_single_cnn', 'im_pasture_-1'),
        ('im_single_cnn', 'im_forest_-1'), ('im_single_cnn', 'im_treecover2000')]), 
    ('access_governance', [
        ('im_single_cnn', 'im_accessibility'),
        ('im_single_cnn', 'im_gov_type')]),
    ('landscape_embedding',
        [('im_single', f'im_A{i:02d}_-2') for i in range(64)]),
])

# im_gov_type has the gt0 transform (protected-area flag), so it bypasses
# normalization and 0.0 in tensor space means "no protection designation" --
# a meaningful 'off' state, used as its removal baseline.
DEFAULT_BASELINE_OVERRIDES = {('im_single_cnn', 'im_gov_type'): 0.0}


@dataclasses.dataclass
class DriverSpec:
    """Resolved driver definitions for one model config.

    drivers: driver name -> [(group, index on the group's last/feature axis)].
        The feature axis is always the last axis of a group tensor with
        timesteps on a separate earlier axis, so one index selects a feature
        across all timesteps and pixels.
    baseline_overrides: (group, feature name) -> tensor-space baseline for
        features the normalizer skips (see module docstring).
    """
    drivers: 'OrderedDict[str, list]'
    baseline_overrides: dict


def resolve_driver_spec(spec, input_features):
    """Resolves a driver-spec JSON dict against a config's input_features.

    `spec` is either None (use DEFAULT_DRIVERS / DEFAULT_BASELINE_OVERRIDES)
    or a dict shaped like configs/attribution_drivers_default.json:
        {"drivers": {name: [[group, feature_name], ...]},
         "baseline_overrides": {"group/feature_name": value}}

    Raises ValueError on an unknown group or feature, or on a feature claimed
    twice (within or across drivers) -- overlapping drivers would make the
    all-drivers-at-baseline residual band ill-defined.
    """
    if spec is None:
        drivers = DEFAULT_DRIVERS
        overrides = dict(DEFAULT_BASELINE_OVERRIDES)
    else:
        drivers = OrderedDict(
            (name, [tuple(ref) for ref in refs])
            for name, refs in spec['drivers'].items())
        overrides = {}
        for key, value in spec.get('baseline_overrides', {}).items():
            group, _, feature = key.partition('/')
            overrides[(group, feature)] = float(value)

    if not drivers:
        raise ValueError('driver spec defines no drivers')

    resolved = OrderedDict()
    seen = {}
    for name, refs in drivers.items():
        channels = []
        for group, feature in refs:
            if group not in input_features:
                raise ValueError(
                    f"driver '{name}': unknown input group '{group}'")
            feature_names = input_features[group]['feature_names']
            if feature not in feature_names:
                raise ValueError(
                    f"driver '{name}': feature '{feature}' not in "
                    f"'{group}' feature_names")
            if (group, feature) in seen:
                raise ValueError(
                    f"feature '{group}/{feature}' claimed by both "
                    f"'{seen[(group, feature)]}' and '{name}'")
            seen[(group, feature)] = name
            channels.append((group, feature_names.index(feature)))
        if not channels:
            raise ValueError(f"driver '{name}' has no features")
        resolved[name] = channels

    return DriverSpec(drivers=resolved, baseline_overrides=overrides)


def resolve_baselines(driver_spec, config):
    """Tensor-space baseline value for every channel a driver touches.

    Returns {(group, index): float}: 0.0 for features the pipeline normalizes
    (grid mean maps to 0 under global standardization), the explicit override
    for transformed features, and raises for anything else so a new config
    with an uncovered transform fails loudly instead of silently attributing
    against a wrong baseline.
    """
    baselines = {}
    for name, channels in driver_spec.drivers.items():
        for group, idx in channels:
            group_cfg = config['input_features'][group]
            feature = group_cfg['feature_names'][idx]
            # The exact predicate the pipeline uses (skips transformed
            # features; appends _<timestep> suffixes for timestep groups).
            normalized = data_norm._normalize_single_features_dict(
                group_cfg, [])
            timesteps = group_cfg.get('timesteps') or []
            key = f'{feature}_{timesteps[0]}' if timesteps else feature
            if key in normalized:
                baselines[(group, idx)] = 0.0
            elif (group, feature) in driver_spec.baseline_overrides:
                baselines[(group, idx)] = (
                    driver_spec.baseline_overrides[(group, feature)])
            else:
                raise ValueError(
                    f"driver '{name}': '{group}/{feature}' is not normalized "
                    f"(transform or normalize=false) and has no "
                    f"baseline_overrides entry -- its grid-average tensor "
                    f"value cannot be assumed to be 0.0")
    return baselines


def occlude(inputs, channels, baselines):
    """Copy of `inputs` with each (group, index) channel set to its baseline.

    Copy-on-write: only groups that are touched get cloned; the caller's
    tensors are never mutated. A channel index selects the feature across all
    timesteps and pixels (feature axis is always last).
    """
    out = dict(inputs)
    cloned = set()
    for group, idx in channels:
        if group not in cloned:
            out[group] = out[group].clone()
            cloned.add(group)
        out[group][..., idx] = baselines[(group, idx)]
    return out


@torch.no_grad()
def attribution_bands(model, inputs, driver_spec, baselines, pos_weight=1.0):
    """OAT attribution for one batch: N+2 forwards, stacked as output bands.

    Runs the (eval-mode, binary) model on: the unmodified inputs, then once
    per driver with that driver at baseline, then once with every driver at
    baseline. All probabilities are deflated by `pos_weight` before
    differencing. Call outside autocast -- deltas can be ~1e-3 and should
    stay float32.

    Returns (bands, names): bands is (B, H, W, n_drivers + 3) stacked as
    ['risk', 'delta_<driver>' per driver in spec order,
     'residual_interactions', 'risk_all_drivers_baseline'], satisfying
    bands[..., 0] - bands[..., -1] == sum(deltas) + residual exactly.
    """
    base = deflate_probs(model(inputs), pos_weight)
    if base.ndim != 3:
        raise ValueError(
            f'attribution supports binary (B, H, W) outputs only, '
            f'got shape {tuple(base.shape)}')
    bands = [base]
    names = ['risk']
    total_delta = torch.zeros_like(base)
    all_channels = []
    for name, channels in driver_spec.drivers.items():
        all_channels.extend(channels)
        occluded = deflate_probs(
            model(occlude(inputs, channels, baselines)), pos_weight)
        delta = base - occluded
        total_delta += delta
        bands.append(delta)
        names.append(f'delta_{name}')
    all_baseline = deflate_probs(
        model(occlude(inputs, all_channels, baselines)), pos_weight)
    bands.append((base - all_baseline) - total_delta)
    names.append('residual_interactions')
    bands.append(all_baseline)
    names.append('risk_all_drivers_baseline')
    return torch.stack(bands, dim=-1), names
