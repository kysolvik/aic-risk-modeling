"""trainer._BestTracker: configurable checkpoint and early-stopping metrics."""

import pytest

from aic_risk_modeling.train.trainer import _BestTracker, _monitoring


def _stop_epoch(values, metric, patience):
    tracker = _BestTracker(metric)
    epochs_since_improvement = 0
    for epoch, value in enumerate(values, 1):
        if tracker.improved({metric: value}):
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                return epoch
    return None


def test_maximized_metric():
    tracker = _BestTracker('pr_auc')
    assert tracker.improved({'pr_auc': 0.3})
    assert tracker.improved({'pr_auc': 0.4})
    assert not tracker.improved({'pr_auc': 0.35})
    assert not tracker.improved({'pr_auc': 0.4})  # tie is not an improvement


def test_loss_is_minimized():
    tracker = _BestTracker('loss')
    assert tracker.improved({'loss': 2.5})
    assert tracker.improved({'loss': 2.0})
    assert not tracker.improved({'loss': 2.2})
    assert not tracker.improved({'loss': 2.0})  # tie is not an improvement
    assert tracker.improved({'loss': 1.9})


def test_stop_epoch_on_loss():
    # v22's actual val_loss trajectory: improvements at epochs 1, 2, and 7.
    v22_val_loss = [2.524, 2.035, 2.405, 2.448, 2.190, 2.132, 1.651, 1.673,
                    2.037]
    # patience 4 stops at epoch 6, missing the epoch-7 best.
    assert _stop_epoch(v22_val_loss, 'loss', patience=4) == 6
    assert _stop_epoch(v22_val_loss, 'loss', patience=6) is None


def test_checkpoint_and_stop_metrics_decouple():
    # pr_auc improves at epochs 1-2, loss through 4: the loss-watcher keeps the run open.
    epochs = [
        {'pr_auc': 0.30, 'loss': 3.0},
        {'pr_auc': 0.40, 'loss': 2.5},
        {'pr_auc': 0.38, 'loss': 2.2},
        {'pr_auc': 0.39, 'loss': 2.0},
    ]
    checkpoint = _BestTracker('pr_auc')
    stopper = _BestTracker('loss')
    saved = [checkpoint.improved(e) for e in epochs]
    kept_open = [stopper.improved(e) for e in epochs]
    assert saved == [True, True, False, False]
    assert kept_open == [True, True, True, True]


def test_last_saves_every_epoch_and_never_stops():
    tracker = _BestTracker('last')
    assert all(tracker.improved({}) for _ in range(5))
    assert _stop_epoch([0.3, 0.2, 0.1, 0.05, 0.01], 'last', patience=2) is None


def test_monitoring_defaults_and_no_val_guard():
    val = {'val_data_dirs': ['gs://b/allpreds_2023/']}
    assert _monitoring(val) == (True, 'pr_auc', 'pr_auc')
    assert _monitoring({**val, 'checkpoint_metric': 'last'}) == (True, 'last', 'last')
    assert _monitoring({'checkpoint_metric': 'last'}) == (False, 'last', 'last')
    assert _monitoring({'val_data_dirs': [], 'checkpoint_metric': 'last'})[0] is False
    for bad in ({}, {'checkpoint_metric': 'pr_auc'},
                {'checkpoint_metric': 'last', 'early_stopping_metric': 'loss'}):
        with pytest.raises(ValueError):
            _monitoring(bad)
