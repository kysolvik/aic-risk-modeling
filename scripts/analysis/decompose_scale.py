"""Split a model's pooled PR-AUC into chip-level intensity vs within-chip localisation.

Pooled PR-AUC is dominated by getting each chip's overall burn RATE right, not by
ranking pixels inside it, so `decompose` reports both parts:

  pr_auc              pooled PR-AUC
  oracle_chip_only    true chip burn rate, flat inside each chip. Chip-level
                      information ALONE, no within-chip ranking at all.
  within_chip_only    predictions divided by their own chip mean, which strips the
                      model's chip-level signal and leaves pure localisation.
  oracle_chip_rescale predictions rescaled so each chip's mean matches truth. The
                      ceiling if the chip-intensity term were perfect.
  chip_r              corr(model's chip mean, true chip rate).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aic_risk_modeling.eval.eval import _binary_metrics  # noqa: E402

def _pr_auc(labels, scores):
    return float(_binary_metrics(labels, scores >= 0.5, scores=scores)["pr_auc"])


def decompose(scores, labels, chip):
    n_chips = int(chip.max()) + 1
    count = np.bincount(chip, minlength=n_chips).astype(np.float64)
    true_rate = np.bincount(chip, weights=labels, minlength=n_chips) / count
    pred_rate = np.maximum(np.bincount(chip, weights=scores, minlength=n_chips) / count, 1e-9)

    rescaled = np.clip(scores * (true_rate / pred_rate)[chip], 0.0, 1.0)

    # The complement of oracle_chip_rescale: PERFECT within-chip ranking at the
    # model's OWN chip intensity. This is the ceiling for a better-localising
    # architecture, and it is what oracle_chip_rescale does not tell you -- that
    # column holds localisation fixed and perfects the chip scalar; this one holds
    # the chip scalar fixed and perfects localisation.
    #
    # Built as a WITHIN-CHIP PERMUTATION of the model's own scores: inside each
    # chip the score multiset is untouched and simply re-assigned so the largest
    # values land on the positives. Chip mean, chip variance and the model's
    # within-chip dynamic range are therefore all preserved exactly, and only the
    # ordering is perfected.
    #
    # Do NOT instead nudge positives above negatives by a tiny epsilon. That
    # collapses within-chip dynamic range to ~0, so the global ranking degenerates
    # to chip order alone and pooled AP FALLS (measured: MLP 0.3566 -> 0.2332).
    # The result is an artifact of the construction, not a statement about
    # localisation.
    # `labels` arrives as a bool array; unary minus is undefined on numpy bools, so
    # cast before negating to get descending (positives-first) order.
    by_score = np.lexsort((-scores, chip))                    # per chip, score desc
    by_label = np.lexsort((-labels.astype(np.int8), chip))    # per chip, positives first
    oracle_within = np.empty_like(scores)
    oracle_within[by_label] = scores[by_score]

    return {
        "pr_auc": _pr_auc(labels, scores),
        "oracle_chip_only": _pr_auc(labels, true_rate[chip]),
        "within_chip_only": _pr_auc(labels, scores / pred_rate[chip]),
        "oracle_chip_rescale": _pr_auc(labels, rescaled),
        "oracle_within_chip": _pr_auc(labels, oracle_within),
        "chip_r": float(np.corrcoef(pred_rate, true_rate)[0, 1]),
        "prevalence": float(labels.mean()),
        "n_chips": n_chips,
        "n_pixels": int(labels.size),
    }
