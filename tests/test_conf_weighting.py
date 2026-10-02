"""Confidence weighting: latent-class posterior q and confidence b from three fire products."""

import math

import numpy as np
import pytest
import tensorflow as tf
import torch

from aic_risk_modeling.train import data_loader, losses

POS_WEIGHT = 10.0

BANDS = ["im_BurnDate_0", "im_mod14_0", "im_viirs_snpp_0"]

# One stratum, no dilation, no dependence terms: checkable by hand.
FLAT_CFG = {
    "mode": "confidence",
    "soft_label": True,
    "prior": [0.05],
    "products": [
        {"name": "im_BurnDate_0", "dilate": 0, "sens": [0.30], "fpr": 0.005},
        {"name": "im_mod14_0", "dilate": 0, "sens": [0.25], "fpr": 0.002},
        {"name": "im_viirs_snpp_0", "dilate": 0, "sens": [0.50], "fpr": 0.003},
    ],
}


def _cfg(**overrides):
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in FLAT_CFG.items()}
    cfg["products"] = [dict(p) for p in FLAT_CFG["products"]]
    cfg.update(overrides)
    return cfg


def _expected_logit(pattern, cfg=FLAT_CFG, stratum=0):
    """Independent hand-computation of the posterior logit for one pattern."""
    pi = cfg["prior"][stratum]
    llr = math.log(pi / (1.0 - pi))
    for det, prod in zip(pattern, cfg["products"]):
        s, f = prod["sens"][stratum], prod["fpr"]
        llr += math.log(s / f) if det else math.log((1.0 - s) / (1.0 - f))
    return llr


def test_posterior_matches_hand_computed_llr():
    """Every one of the 8 detection patterns, against first-principles math."""
    patterns = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    example = {
        name: tf.constant([[float(p[i]) * 200.0 for p in patterns]])
        for i, name in enumerate(BANDS)
    }
    q, union = data_loader.build_confidence_posterior(example, FLAT_CFG)

    expected = np.array([[1.0 / (1.0 + math.exp(-_expected_logit(p)))
                          for p in patterns]], dtype=np.float32)
    assert np.allclose(q.numpy(), expected, atol=1e-6), (q.numpy(), expected)
    assert np.array_equal(union.numpy(),
                          np.array([[any(p) for p in patterns]]))


def test_detection_raises_q_and_nondetection_lowers_it():
    """Monotonicity: s > f means a detection is always evidence FOR fire."""
    patterns = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    example = {
        name: tf.constant([[float(p[i]) * 200.0 for p in patterns]])
        for i, name in enumerate(BANDS)
    }
    q = data_loader.build_confidence_posterior(example, FLAT_CFG)[0].numpy()[0]
    assert q[0] == q.min(), "all-negative must be the lowest-evidence pattern"
    assert q[4] == q.max(), "three-way agreement must be the highest"
    assert q[0] < FLAT_CFG["prior"][0] < q[1]


def test_nondetection_evidence_is_asymmetric_across_products():
    """The products differ mainly in what they MISS, not what they claim."""
    pos = [math.log(p["sens"][0] / p["fpr"]) for p in FLAT_CFG["products"]]
    neg = [math.log((1 - p["sens"][0]) / (1 - p["fpr"]))
           for p in FLAT_CFG["products"]]
    pos_spread = max(pos) / min(pos)
    neg_spread = max(neg, key=abs) / min(neg, key=abs)
    assert pos_spread < 1.4, f"detections should be comparable: {pos}"
    assert neg_spread > 2.0, f"non-detections should not be: {neg}"
    assert neg_spread > 1.7 * pos_spread, (pos_spread, neg_spread)


def test_stratified_sensitivity_changes_the_posterior():
    """A product that is blind in one stratum must not penalise it for silence."""
    cfg = _cfg2()
    # MCD64 sees almost nothing under canopy (stratum 1) but plenty in pasture.
    cfg["products"][0]["sens"] = [0.45, 0.05]
    cfg["products"][1]["sens"] = [0.25, 0.25]
    cfg["products"][2]["sens"] = [0.50, 0.50]

    zeros = tf.constant([[0.0, 0.0]])
    example = {name: zeros for name in BANDS}
    example["im_forest_-1"] = tf.constant([[0.1, 0.9]])  # pasture, forest
    q = data_loader.build_confidence_posterior(example, cfg)[0].numpy()[0]

    hand = [1.0 / (1.0 + math.exp(-_expected_logit((0, 0, 0), cfg, g)))
            for g in (0, 1)]
    assert np.allclose(q, hand, atol=1e-6)
    # Silence from a blind detector is weak evidence, so the forest pixel keeps more prior.
    assert q[1] / cfg["prior"][1] > q[0] / cfg["prior"][0]


def test_dilation_matches_a_hand_built_neighbourhood():
    cfg = _cfg()
    cfg["products"][1]["dilate"] = 1  # MOD14 is ~1.7 chip pixels wide
    example = {
        "im_BurnDate_0": tf.constant([[[0.0, 0.0, 0.0]] * 3]),
        "im_mod14_0": tf.constant([[[0.0, 0.0, 0.0],
                                    [0.0, 120.0, 0.0],
                                    [0.0, 0.0, 0.0]]]),
        "im_viirs_snpp_0": tf.constant([[[0.0, 0.0, 0.0]] * 3]),
    }
    q = data_loader.build_confidence_posterior(example, cfg)[0].numpy()[0]
    # A 1-px dilation of the centre hit covers the whole 3x3.
    assert (q > 0.5).all(), q
    # The union label is NOT dilated -- only the centre pixel is positive.
    union = data_loader.build_confidence_posterior(example, cfg)[1].numpy()[0]
    assert union.sum() == 1 and union[1, 1]


def test_pair_dependence_discounts_double_counted_agreement():
    """MCD64 is seeded by MOD14, so their agreement is partly one vote twice."""
    both_on = {name: tf.constant([[200.0]]) for name in BANDS[:2]}
    both_on["im_viirs_snpp_0"] = tf.constant([[0.0]])

    plain = data_loader.build_confidence_posterior(both_on, FLAT_CFG)[0].numpy()
    discounted = data_loader.build_confidence_posterior(
        both_on, _cfg(pair_llr=[{"a": 0, "b": 1, "both": -1.2}]))[0].numpy()
    assert discounted < plain


def test_doy_term_rewards_co_detection_in_the_same_week():
    """Two products firing 6 months apart may be two unrelated fires."""
    cfg = _cfg(doy_llr={"pairs": [[0, 2]], "edges": [8.0, 32.0],
                        "values": [1.2, 0.3, -0.4]})
    example = {
        "im_BurnDate_0": tf.constant([[200.0, 200.0]]),
        "im_mod14_0": tf.constant([[0.0, 0.0]]),
        "im_viirs_snpp_0": tf.constant([[203.0, 20.0]]),  # 3 days vs 180 days
    }
    q = data_loader.build_confidence_posterior(example, cfg)[0].numpy()[0]
    assert q[0] > q[1]
    base = data_loader.build_confidence_posterior(example, FLAT_CFG)[0].numpy()[0]
    assert q[0] > base[0] and q[1] < base[1]


def test_weight_and_target_expand_to_the_pseudo_count_loss():
    """(weight, target) must reconstruct b*[-P*q*log p - (1-q)*log(1-p)]."""
    q = np.array([[0.02, 0.35, 0.5, 0.87, 0.99]], dtype=np.float32)
    b = np.array([[1.0, 0.4, 2.0, 0.8, 1.3]], dtype=np.float32)
    P = POS_WEIGHT

    denom = P * q + (1.0 - q)
    weight = b * denom
    target = P * q / denom

    pred = torch.tensor([[0.1, 0.25, 0.5, 0.7, 0.95]], dtype=torch.float32)
    got = losses.weighted_bce(P)(torch.tensor(target), pred,
                                 torch.tensor(weight))

    p = pred.numpy()
    want = (b * (-P * q * np.log(p) - (1.0 - q) * np.log(1.0 - p))).mean()
    assert np.isclose(got.item(), want, atol=1e-5), (got.item(), want)


def test_reduces_to_plain_weighted_bce():
    """The superset proof, through the real pipeline."""
    raw = np.array([[0.0, 150.0, 0.0], [0.0, 0.0, 88.0], [12.0, 0.0, 0.0]],
                   dtype=np.float32)
    example = {"im_BurnDate_0": tf.constant(raw[None, ...]),
               "im_mod14_0": tf.constant(np.zeros_like(raw)[None, ...]),
               "im_viirs_snpp_0": tf.constant(np.zeros_like(raw)[None, ...])}
    cfg = _cfg(soft_label=False, confidence={"floor": 1.0})

    weights, soft = data_loader.build_confidence_weight_map(example, cfg, POS_WEIGHT)
    assert soft is None, "the hard arm must not emit a soft target"

    union = torch.tensor((raw > 0).astype(np.float32))[None, ...]
    pred = torch.full_like(union, 0.3)
    got = losses.weighted_bce(POS_WEIGHT)(
        union, pred, torch.tensor(weights.numpy()))
    want = losses.weighted_bce(POS_WEIGHT)(union, pred, None)
    assert torch.allclose(got, want, atol=1e-6), (got, want)


def test_deflate_probs_inverts_the_inflated_target():
    """area_ratio stays meaningful: deflate(target, P) returns q exactly."""
    q = torch.tensor([0.001, 0.05, 0.4, 0.9, 0.999], dtype=torch.float64)
    P = 12.0
    target = P * q / (P * q + 1.0 - q)
    assert torch.allclose(losses.deflate_probs(target, P), q, atol=1e-12)


def test_hard_arm_downweights_unreliable_label_bits():
    """b = P(the union bit is right): low-evidence positives lose weight."""
    example = {
        "im_BurnDate_0": tf.constant([[200.0, 0.0, 200.0, 0.0]]),
        "im_mod14_0": tf.constant([[0.0, 0.0, 200.0, 0.0]]),
        "im_viirs_snpp_0": tf.constant([[0.0, 200.0, 200.0, 0.0]]),
    }
    cfg = _cfg(soft_label=False)
    weights = data_loader.build_confidence_weight_map(
        example, cfg, POS_WEIGHT)[0].numpy()[0]
    # Positives carry pos_weight * P(correct): three-way agreement > single detection.
    assert weights[2] > weights[1] > weights[0]
    assert weights[2] < POS_WEIGHT
    # The negative's weight is P(no fire | nothing detected), just under 1.
    assert 0.0 < weights[3] < 1.0


_INPUT_CFG = {"im_main": {"feature_names": ["im_feat"], "transforms": {},
                          "timesteps": [], "stack_timesteps": False}}
_OUTPUT_CFG = {"feature_names": BANDS,
               "transforms": {b: "gt0_bool" for b in BANDS},
               "combine": "any", "timesteps": [], "stack_timesteps": False}


def _tiny_dataset(rows):
    """rows: list of (burndate, mod14, viirs) day-of-year triples."""
    arrs = {name: np.array([[r[i] for r in rows]], dtype=np.float32)
            for i, name in enumerate(BANDS)}
    arrs["im_feat"] = np.zeros_like(arrs[BANDS[0]])
    arrs["im_forest_-1"] = np.full_like(arrs[BANDS[0]], 0.9)
    ds = tf.data.Dataset.from_tensor_slices(
        {k: tf.constant(v) for k, v in arrs.items()})
    return ds.batch(1)


_ROWS = [(0, 0, 0), (200, 0, 0), (0, 0, 150), (210, 205, 208)]


def test_confidence_soft_config_emits_4tuple_with_hard_labels():
    ds = data_loader.select_bands_transform(
        _tiny_dataset(_ROWS), _INPUT_CFG, _OUTPUT_CFG,
        sample_weight_config=_cfg(soft_label=True), pos_weight=POS_WEIGHT)
    batch = next(iter(ds.as_numpy_iterator()))
    assert len(batch) == 4, "soft_label should add a 4th element"
    _, labels, weights, soft = batch
    assert np.array_equal(labels[0], np.array([any(r) for r in _ROWS]))
    assert labels.dtype == np.bool_
    assert (soft > 0).all() and (soft < 1).all()
    assert (weights > 0).all()


def test_confidence_hard_config_emits_3tuple():
    ds = data_loader.select_bands_transform(
        _tiny_dataset(_ROWS), _INPUT_CFG, _OUTPUT_CFG,
        sample_weight_config=_cfg(soft_label=False), pos_weight=POS_WEIGHT)
    batch = next(iter(ds.as_numpy_iterator()))
    assert len(batch) == 3, "the hard arm needs no soft target"


def test_existing_paths_are_untouched():
    """Backward compat: no config -> 2-tuple, type_weights -> the old 3-tuple."""
    ds = data_loader.select_bands_transform(
        _tiny_dataset(_ROWS), _INPUT_CFG, _OUTPUT_CFG)
    assert len(next(iter(ds.as_numpy_iterator()))) == 2

    type_cfg = {"feature_name": "im_BurnDate_0", "type_weights": {"200": 45.0}}
    ds = data_loader.select_bands_transform(
        _tiny_dataset(_ROWS), _INPUT_CFG, _OUTPUT_CFG,
        sample_weight_config=type_cfg, pos_weight=POS_WEIGHT)
    batch = next(iter(ds.as_numpy_iterator()))
    assert len(batch) == 3
    assert np.allclose(batch[2][0], [1.0, 45.0, 1.0, 10.0])


def test_trainer_batches_pass_through_both_arities():
    from aic_risk_modeling.train import trainer
    device = torch.device('cpu')
    for soft_label, arity in ((False, 3), (True, 4)):
        ds = data_loader.select_bands_transform(
            _tiny_dataset(_ROWS), _INPUT_CFG, _OUTPUT_CFG,
            sample_weight_config=_cfg(soft_label=soft_label),
            pos_weight=POS_WEIGHT)
        batch = next(iter(trainer._torch_batches(ds, device)))
        assert len(batch) == arity, (soft_label, len(batch))
        inputs, labels, weights, *rest = batch
        assert labels.dtype == torch.float32
        assert len(rest) == arity - 3


def test_validation_rejects_bad_measurement_models():
    def rejects(cfg, why):
        with pytest.raises(ValueError):
            data_loader._validate_confidence_config(cfg)

    rejects(_cfg(products=[]), "no products")
    rejects(_cfg(prior=[0.05, 0.1]), "prior length vs strata")
    rejects(_cfg(prior=[1.5]), "prior out of range")

    bad_fpr = _cfg()
    bad_fpr["products"][0]["fpr"] = 0.0
    rejects(bad_fpr, "zero fpr would divide by zero")

    inverted = _cfg()
    inverted["products"][0]["sens"] = [0.001]
    rejects(inverted, "sensitivity below fpr inverts the evidence")

    short = _cfg(prior=[0.03, 0.08],
                 stratify={"feature_name": "im_forest_-1", "edges": [0.5]})
    rejects(short, "sens vector shorter than the strata count")

    rejects(_cfg(doy_llr={"pairs": [[0, 1]], "edges": [8.0], "values": [1.0]}),
            "doy values must be one longer than edges")


def test_validation_accepts_the_shipped_shape():
    data_loader._validate_confidence_config(FLAT_CFG)
    data_loader._validate_confidence_config(
        _cfg(prior=[0.03, 0.08],
             stratify={"feature_name": "im_forest_-1", "edges": [0.5]},
             products=[dict(p, sens=[p["sens"][0], p["sens"][0] * 0.5])
                       for p in FLAT_CFG["products"]]))


# Stratifiers are normalized before the weight builder, so edges must be compared in z-space.

_STATS = {"features": {"im_forest_-1": {"mean": 0.6, "stddev": 0.25,
                                        "min": 0.0, "median": 0.7,
                                        "robust_scale": 0.3}}}


def _cfg2(**overrides):
    """Two forest-fraction strata, with a full sensitivity vector per product."""
    cfg = _cfg(prior=[0.03, 0.08],
               stratify={"feature_name": "im_forest_-1", "edges": [0.5]})
    for prod, sens in zip(cfg["products"], ([0.45, 0.08], [0.26, 0.21],
                                            [0.55, 0.47])):
        prod["sens"] = sens
    cfg.update(overrides)
    return cfg


def test_resolver_maps_edges_onto_the_normalized_scale():
    cfg = _cfg2()
    resolved = data_loader.resolve_stratifier_normalization(
        cfg, ["im_forest_-1"], [], _STATS)
    assert resolved["stratify"]["normalized"] == {"center": 0.6, "scale": 0.25}
    # cfg itself must not be mutated -- the trainer reuses it for val.
    assert "normalized" not in cfg["stratify"]

    # forest 0.9 -> (0.9-0.6)/0.25 = 1.2 (upper stratum); 0.1 -> -2.0 (lower).
    example = {name: tf.constant([[0.0, 0.0]]) for name in BANDS}
    example["im_forest_-1"] = tf.constant([[-2.0, 1.2]])
    q_norm = data_loader.build_confidence_posterior(example, resolved)[0].numpy()

    raw = {name: tf.constant([[0.0, 0.0]]) for name in BANDS}
    raw["im_forest_-1"] = tf.constant([[0.1, 0.9]])
    q_raw = data_loader.build_confidence_posterior(raw, cfg)[0].numpy()
    assert np.allclose(q_norm, q_raw), (q_norm, q_raw)


def test_unresolved_edges_put_pixels_in_the_wrong_stratum():
    """Pin the bug the resolver prevents, so it cannot quietly come back."""
    cfg = _cfg2()
    resolved = data_loader.resolve_stratifier_normalization(
        cfg, ["im_forest_-1"], [], _STATS)

    zscore = {name: tf.constant([[0.0]]) for name in BANDS}
    zscore["im_forest_-1"] = tf.constant([[(0.55 - 0.6) / 0.25]])
    natural = {name: tf.constant([[0.0]]) for name in BANDS}
    natural["im_forest_-1"] = tf.constant([[0.55]])

    wrong = data_loader.build_confidence_posterior(zscore, cfg)[0].numpy()
    right = data_loader.build_confidence_posterior(zscore, resolved)[0].numpy()
    truth = data_loader.build_confidence_posterior(natural, cfg)[0].numpy()

    assert np.allclose(right, truth), (right, truth)
    assert not np.allclose(wrong, truth), "expected the unresolved misassignment"


def test_resolver_uses_robust_constants_for_robust_bands():
    cfg = _cfg2()
    resolved = data_loader.resolve_stratifier_normalization(
        cfg, ["im_forest_-1"], ["im_forest_-1"], _STATS)
    assert resolved["stratify"]["normalized"] == {"center": 0.7, "scale": 0.3}


def test_resolver_is_a_noop_when_the_band_is_not_normalized():
    cfg = _cfg2()
    assert data_loader.resolve_stratifier_normalization(
        cfg, [], [], _STATS) is cfg
    assert data_loader.resolve_stratifier_normalization(None, [], [], _STATS) is None
    type_cfg = {"feature_name": "im_viirs_type", "type_weights": {}}
    assert data_loader.resolve_stratifier_normalization(
        type_cfg, [], [], _STATS) is type_cfg


def test_resolver_refuses_when_stats_lack_the_stratifier():
    cfg = _cfg2()
    with pytest.raises(ValueError, match='stats'):
        data_loader.resolve_stratifier_normalization(
            cfg, ["im_forest_-1"], [], {"features": {}})
