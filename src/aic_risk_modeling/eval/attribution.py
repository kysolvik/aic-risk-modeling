"""Per-driver attribution maps: one-at-a-time occlusion or Shapley over driver groups.

A driver at baseline = its features set to the grid mean (0.0 after standardization).
Bands satisfy risk - risk_all_baseline == sum(deltas) + residual, on deflated probabilities."""

import dataclasses
import itertools
import math
import random
from collections import OrderedDict

import torch

from aic_risk_modeling.train import data_norm
from aic_risk_modeling.train.losses import deflate_probs

# md_single (location) is excluded: "average location" is not a meaningful counterfactual.
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

# gt0-transformed, so not normalized; 0.0 = no protection designation.
DEFAULT_BASELINE_OVERRIDES = {('im_single_cnn', 'im_gov_type'): 0.0}


@dataclasses.dataclass
class DriverSpec:
    """Drivers as {name: [(group, feature-axis index)]}, plus baseline overrides and year terms."""
    drivers: 'OrderedDict[str, list]'
    baseline_overrides: dict
    year_terms: dict = dataclasses.field(default_factory=dict)


def resolve_driver_spec(spec, input_features):
    """Resolve a driver-spec dict (None = defaults) against a config's input_features.

    Raises on unknown groups/features and on a feature claimed by two drivers."""
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
    """{(group, index): tensor-space baseline}; raises for an un-normalized feature without an override."""
    baselines = {}
    for name, channels in driver_spec.drivers.items():
        for group, idx in channels:
            group_cfg = config['input_features'][group]
            feature = group_cfg['feature_names'][idx]
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
    """Copy of `inputs` with each (group, index) channel set to its baseline; clones only touched groups."""
    out = dict(inputs)
    cloned = set()
    for group, idx in channels:
        if group not in cloned:
            out[group] = out[group].clone()
            cloned.add(group)
        out[group][..., idx] = baselines[(group, idx)]
    return out


def year_components(model, inputs, driver_spec):
    """driver -> (B, 1, 1) share of the year-offset logit; raises unless the shares sum to gamma."""
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
    """Model probabilities, with logit offset `removed` (B, 1, 1) subtracted if given."""
    if removed is None:
        return model(inputs)
    t = model.forward_terms(inputs)
    logits = t['gamma'] + t['year_gain'] + t['m'] + t['s'] + t['c']
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return torch.sigmoid(logits.float().squeeze(1) - removed)


def _removed(comps, absent):
    if not comps:
        return None
    return sum((comps[n] for n in absent if n in comps),
               torch.zeros_like(next(iter(comps.values()))))


@torch.no_grad()
def attribution_bands(model, inputs, driver_spec, baselines, pos_weight=1.0):
    """OAT attribution: N+2 forwards -> ((B, H, W, N+3) bands, names).

    Bands: risk, delta_<driver>..., residual_interactions, risk_all_drivers_baseline.
    Call outside autocast; deltas can be ~1e-3."""
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
    """Shapley attribution with the same band layout as attribution_bands.

    samples=None/0 enumerates all 2^N coalitions; samples>0 is a seeded permutation estimate."""
    names = list(driver_spec.drivers.keys())
    n = len(names)
    comps = year_components(model, inputs, driver_spec)

    cache = {}

    def value(coalition):
        if coalition not in cache:
            absent = [name for name in names if name not in coalition]
            occluded = [c for name in absent for c in driver_spec.drivers[name]]
            cache[coalition] = deflate_probs(
                _probs(model, occlude(inputs, occluded, baselines),
                       _removed(comps, absent)), pos_weight)
        return cache[coalition]

    base = value(frozenset(names))
    if base.ndim != 3:
        raise ValueError(
            f'shapley supports binary (B, H, W) outputs only, '
            f'got shape {tuple(base.shape)}')
    all_baseline = value(frozenset())

    shapley = OrderedDict((name, torch.zeros_like(base)) for name in names)
    if not samples:
        if n > max_exact_drivers:
            raise ValueError(
                f'exact Shapley over {n} drivers needs 2^{n} forwards; pass '
                f'samples>0 for a Monte-Carlo estimate or use a coarser '
                f'--drivers spec (max_exact_drivers={max_exact_drivers})')
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
