"""Evaluation utilities: metrics, calibration, chip I/O, year offset and driver attribution."""

from .metrics import binary_metrics, calc_stats

__all__ = ["binary_metrics", "calc_stats"]
