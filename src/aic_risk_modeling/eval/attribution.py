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

`shapley_bands` is an alternative attribution mode over the same drivers:
Shapley values split the interaction term across drivers, but more intensive
(2^N runs where N is number of variable groups)

Year terms (optional, factored models only): the factored model's frozen year
offset `gamma(t)*(1 + g_res(location))` is added to the logit after the
network, so by default it is not a player and sits in the baseline band. A spec
`year_terms` block hands each driver a per-year logit COMPONENT of gamma (e.g.
the SOI part to climate, the previous-year-burn part to fire history); a driver
that is absent from a coalition then also loses its component (scaled by the
same per-chip `1 + g_res`), so the baseline becomes "average year" as well as
"grid-average pixel". The components must sum to the model's gamma table for
every year attributed (checked per batch). Costs no extra forwards: the year
term never enters the network, so it is subtracted from the summed
`forward_terms` logit.

Caveats:
- One-at-a-time deltas are not Shapley values: correlated drivers each absorb
  their shared signal, so deltas can double-count. The residual band shows 
  the total mismatch against the all-baseline run. Use `shapley_bands` for a
  decomposition that redistributes that mismatch fairly across drivers.
- Some variable deltas involve extrapolation, e.g. average vegation
  over real terrain never occurs.
- delta ~= 0 means the *model* does not use the driver, not that the driver
  is physically irrelevant.
"""

import dataclasses
import itertools
import math
import random
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
    year_terms: driver name -> {year: logit component of gamma(year)}; empty
        unless the spec has a `year_terms` block (see module docstring).
    """
    drivers: 'OrderedDict[str, list]'
    baseline_overrides: dict
    year_terms: dict = dataclasses.field(default_factory=dict)


def resolve_driver_spec(spec, input_features):
    """Resolves a driver-spec JSON dict against a config's input_features.

    `spec` is either None (use DEFAULT_DRIVERS / DEFAULT_BASELINE_OVERRIDES)
    or a dict shaped like configs/attribution_drivers_default.json:
        {"drivers": {name: [[group, feature_name], ...]},
         "baseline_overrides": {"group/feature_name": value},
         "year_terms": {name: {"<year>": logit component}}}   (optional)

    Raises ValueError on an unknown group or feature, or on a feature claimed
    twice (within or across drivers) -- overlapping drivers would make the
    all-drivers-at-baseline residual band ill-defined -- or on a year_terms
    entry for a driver the spec does not define.
    """
    year_terms = {}
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
        for name, per_year in spec.get('year_terms', {}).items():
            if name not in drivers:
                raise ValueError(
                    f"year_terms names driver '{name}', which the spec does "
                    f"not define")
            year_terms[name] = {int(y): float(v) for y, v in per_year.items()}

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

    return DriverSpec(drivers=resolved, baseline_overrides=overrides,
                      year_terms=year_terms)


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


def year_components(model, inputs, driver_spec):
    """driver -> (B, 1, 1) logit component of the year term, {} without year_terms.

    Each component is the spec's per-year value scaled by the chip's year gain
    `1 + g_res` (1 for models without one), i.e. that driver's share of the
    `gamma + year_gain` terms. Raises if a batch year is missing from a driver's
    table or if the components do not sum to the model's own gamma -- a spec
    built for another gamma fit would otherwise attribute silently wrong.
    """
    if not driver_spec.year_terms:
        return {}
    year_mod = getattr(model, 'year', None)
    if year_mod is None or not hasattr(model, 'forward_terms'):
        raise ValueError('year_terms needs a factored model with a year offset '
                         '(forward_terms + year)')
    raw = inputs[year_mod.input_name]
    years = torch.round(raw.reshape(raw.shape[0], -1)[:, 0]).long().tolist()
    gamma = year_mod(raw).reshape(-1)
    scale = torch.ones_like(gamma)
    gain = getattr(model, 'year_gain', None)
    if gain is not None:
        scale = scale + gain(inputs[gain.input_name]).reshape(-1)
    total = torch.zeros_like(gamma)
    comps = {}
    for name, per_year in driver_spec.year_terms.items():
        missing = sorted(set(years) - set(per_year))
        if missing:
            raise ValueError(f"year_terms['{name}'] has no entry for year(s) {missing}")
        c = torch.tensor([per_year[y] for y in years], dtype=gamma.dtype,
                         device=gamma.device)
        total += c
        comps[name] = (c * scale).reshape(-1, 1, 1)
    if not torch.allclose(total, gamma, atol=1e-4):
        raise ValueError(
            f'year_terms components sum to {total.tolist()} but the model gamma is '
            f'{gamma.tolist()} for years {years}: the spec was built for another '
            f'gamma fit')
    return comps


def _probs(model, inputs, removed=None):
    """Model probabilities, optionally with logit offset `removed` (B, 1, 1) taken out.

    `removed=None` is exactly `model(inputs)`. Otherwise the logit is rebuilt from
    `forward_terms` (the same sum `FactoredFireModel.forward` takes) so year
    components can be subtracted without an extra forward.
    """
    if removed is None:
        return model(inputs)
    t = model.forward_terms(inputs)
    logits = t['gamma'] + t['year_gain'] + t['m'] + t['s'] + t['c']
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return torch.sigmoid(logits.float().squeeze(1) - removed)


def _removed(comps, absent):
    """Summed year components of the `absent` drivers, or None without year_terms."""
    if not comps:
        return None
    return sum((comps[n] for n in absent if n in comps),
               torch.zeros_like(next(iter(comps.values()))))


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
    comps = year_components(model, inputs, driver_spec)
    base = deflate_probs(_probs(model, inputs, _removed(comps, [])), pos_weight)
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
            _probs(model, occlude(inputs, channels, baselines), _removed(comps, [name])),
            pos_weight)
        delta = base - occluded
        total_delta += delta
        bands.append(delta)
        names.append(f'delta_{name}')
    all_baseline = deflate_probs(
        _probs(model, occlude(inputs, all_channels, baselines),
               _removed(comps, driver_spec.drivers)), pos_weight)
    bands.append((base - all_baseline) - total_delta)
    names.append('residual_interactions')
    bands.append(all_baseline)
    names.append('risk_all_drivers_baseline')
    return torch.stack(bands, dim=-1), names


@torch.no_grad()
def shapley_bands(model, inputs, driver_spec, baselines, pos_weight=1.0,
                  samples=None, seed=0, max_exact_drivers=12):
    """Shapley-value attribution for one batch, same band layout as OAT.

    The driver groups are players in a cooperative game whose value function is
    the deflated model probability with every driver NOT in the coalition
    occluded to baseline:

        v(S) = deflate_probs(model(occlude(inputs, channels not in S)), w)

    so v(all) == the base 'risk' forward and v(none) == the all-drivers-baseline
    run. Each driver's Shapley value is the standard weighted average of its
    marginal contributions over coalitions. Efficiency gives sum(shapley) ==
    risk - risk_all_baseline exactly, so the residual band is ~0 (still included
    as a check). 

    samples: None or 0 -> exact enumeration of all 2^N coalitions (cached, so
        2^N forwards). >0 -> seeded permutation Monte-Carlo estimate (~N*samples
        forwards, cached); efficiency still holds exactly because each sampled
        permutation's marginals telescope to v(all) - v(none).
    max_exact_drivers: guard against 2^N blowing up on a fine-grained driver
        spec; exact mode raises above this and points at `samples`.

    Returns (bands, names): bands is (B, H, W, n_drivers + 3) stacked as
    ['risk', 'shapley_<driver>' per driver in spec order,
     'residual_interactions', 'risk_all_drivers_baseline'].
    """
    names = list(driver_spec.drivers.keys())
    n = len(names)
    comps = year_components(model, inputs, driver_spec)

    cache = {}  # frozenset[str] of drivers present -> (B, H, W) deflated probs

    def value(coalition):
        if coalition not in cache:
            absent = [name for name in names if name not in coalition]
            occluded = [c for name in absent for c in driver_spec.drivers[name]]
            cache[coalition] = deflate_probs(
                _probs(model, occlude(inputs, occluded, baselines),
                       _removed(comps, absent)), pos_weight)
        return cache[coalition]

    base = value(frozenset(names))  # occlude nothing -> the 'risk' forward
    if base.ndim != 3:
        raise ValueError(
            f'shapley supports binary (B, H, W) outputs only, '
            f'got shape {tuple(base.shape)}')
    all_baseline = value(frozenset())  # occlude everything

    shapley = OrderedDict((name, torch.zeros_like(base)) for name in names)
    if not samples:
        if n > max_exact_drivers:
            raise ValueError(
                f'exact Shapley over {n} drivers needs 2^{n} forwards; pass '
                f'samples>0 for a Monte-Carlo estimate or use a coarser '
                f'--drivers spec (max_exact_drivers={max_exact_drivers})')
        # weight[s] = |S|! (N-|S|-1)! / N! for a coalition S of size s.
        weight = [math.factorial(s) * math.factorial(n - s - 1)
                  / math.factorial(n) for s in range(n)]
        for name in names:
            others = [m for m in names if m != name]
            for size in range(len(others) + 1):
                for combo in itertools.combinations(others, size):
                    subset = frozenset(combo)
                    shapley[name] += weight[size] * (
                        value(subset | {name}) - value(subset))
    else:
        rng = random.Random(seed)
        perm = list(names)
        for _ in range(samples):
            rng.shuffle(perm)
            coalition = set()
            prev = all_baseline
            for name in perm:
                coalition.add(name)
                current = value(frozenset(coalition))
                shapley[name] += current - prev
                prev = current
        for name in names:
            shapley[name] /= samples

    bands = [base]
    names_out = ['risk']
    total = torch.zeros_like(base)
    for name in names:
        bands.append(shapley[name])
        names_out.append(f'shapley_{name}')
        total += shapley[name]
    bands.append((base - all_baseline) - total)
    names_out.append('residual_interactions')
    bands.append(all_baseline)
    names_out.append('risk_all_drivers_baseline')
    return torch.stack(bands, dim=-1), names_out
