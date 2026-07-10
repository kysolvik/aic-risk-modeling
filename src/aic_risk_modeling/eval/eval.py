"""Helpers for evaluating fire model outputs.

Predictions can be binary (single band of scores) or multiclass. For
multiclass, the prediction raster follows the layout written by
``scripts/predict/predict.py``: band 1 is the argmax (hard predicted class),
and the remaining bands are the per-class scores (softmax probabilities), so
band ``2 + c`` holds the score for class ``c``. Ground truth is a single band
of integer class labels.
"""

import numpy as np
from sklearn import metrics

from .calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_calibrator,
    plot_reliability_diagram,
    reliability_table_str,
)


def _print_metrics(title, stats):
    """Pretty-print a one-vs-rest metrics dict (as returned by _binary_metrics)."""
    print(title)
    print(f"  Accuracy: {stats['accuracy']:.4f}")
    print(f"  Precision: {stats['precision']:.4f}")
    print(f"  Recall: {stats['recall']:.4f}")
    print(f"  F1 Score: {stats['f1']:.4f}")
    print(f"  Cohen's Kappa: {stats['kappa']:.4f}")
    print(f"  PR AUC: {stats['pr_auc']:.4f}")
    if stats.get('ece') is not None:
        print(f"  ECE: {stats['ece']:.4f}")
        print(f"  MCE: {stats['mce']:.4f}")
    print(f"  N (truth): {stats['n_truth']}")
    print(f"  N (pred): {stats['n_pred']}")


def _binary_metrics(gt_binary, pred_binary, scores=None):
    """One-vs-rest metrics for a single class/label.

    ``pred_binary`` is the hard prediction (a thresholded score for binary
    models, or ``argmax == class`` for multiclass). ``scores`` are the
    continuous scores used for PR AUC; if ``None``, PR AUC is skipped.
    """
    gt_flat = np.asarray(gt_binary).flatten().astype(bool)
    pred_flat = np.asarray(pred_binary).flatten().astype(bool)
    n_truth = int(gt_flat.sum())
    n_pred = int(pred_flat.sum())
    stats = {k: 0.0 for k in ("accuracy", "precision", "recall", "f1", "kappa", "pr_auc")}
    stats["n_truth"] = n_truth
    stats["n_pred"] = n_pred
    if n_truth == 0:
        print("Warning: No positive cases in ground truth, metrics undefined.")
        return stats
    if n_pred == 0:
        print("Warning: No positive cases in predictions; precision/recall/F1 will be 0.")
    # Hard-label metrics from the thresholded/argmax predictions.
    stats["accuracy"] = metrics.accuracy_score(gt_flat, pred_flat)
    stats["precision"] = metrics.precision_score(gt_flat, pred_flat, zero_division=0)
    stats["recall"] = metrics.recall_score(gt_flat, pred_flat, zero_division=0)
    stats["f1"] = metrics.f1_score(gt_flat, pred_flat, zero_division=0)
    stats["kappa"] = metrics.cohen_kappa_score(gt_flat, pred_flat)
    # PR AUC uses the continuous class scores, not the hard labels.
    if scores is not None:
        stats["pr_auc"] = metrics.average_precision_score(gt_flat, np.asarray(scores).flatten())
    return stats


def _is_multiclass(predictions):
    """Multiclass predictions are band-first (>1 band): argmax + class scores."""
    return np.ndim(predictions) == 3 and np.shape(predictions)[0] > 1


def calc_stats_multiclass(predictions, ground_truth, class_names=None):
    """Per-class accuracy stats for multiclass predictions.

    Hard-label metrics (accuracy/precision/recall/F1/kappa) come from the
    argmax band; PR AUC uses the raw per-class scores. Also reports overall
    multiclass accuracy and kappa across all pixels.

    Args:
        predictions: array of shape ``(num_classes + 1, H, W)`` where band 0 is
            the argmax class and bands ``1 + c`` are the score for class ``c``.
        ground_truth: array of integer class labels.
        class_names: optional sequence mapping class index -> display name.

    Returns:
        dict with ``"overall"`` and ``"per_class"`` (keyed by class index) stats.
    """
    predictions = np.asarray(predictions)
    pred_labels = predictions[0].astype(int)  # argmax band
    scores = predictions[1:]                  # (num_classes, H, W)
    num_classes = scores.shape[0]

    gt_flat = np.asarray(ground_truth).flatten().astype(int)
    pred_flat = pred_labels.flatten()

    overall = {
        "accuracy": metrics.accuracy_score(gt_flat, pred_flat),
        "kappa": metrics.cohen_kappa_score(gt_flat, pred_flat),
        "macro_f1": metrics.f1_score(gt_flat, pred_flat, average="macro", zero_division=0),
    }
    print("Overall multiclass stats:")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    print(f"  Cohen's Kappa: {overall['kappa']:.4f}")
    print(f"  Macro F1: {overall['macro_f1']:.4f}")

    per_class = {}
    for c in range(num_classes):
        name = class_names[c] if class_names is not None else c
        stats = _binary_metrics(gt_flat == c, pred_flat == c, scores=scores[c])
        per_class[c] = stats
        _print_metrics(f"Stats for class {name}:", stats)

    return {"overall": overall, "per_class": per_class}


def calc_stats(predictions, ground_truth, grouped=False, threshold=0.5,
               class_names=None, reliability_plot=None, calibration_bins=15,
               calibration_binning='uniform', calibration_method='none',
               calibration_fit=None, temperature=None):
    """Calculate stats for predictions vs ground truth.

    Dispatches to per-class multiclass stats when ``predictions`` is a
    multi-band (band-first) array; otherwise computes binary stats by
    thresholding the single-band scores. For binary predictions, also reports
    calibration: Expected/Maximum Calibration Error plus a reliability table,
    and writes a reliability-diagram PNG to ``reliability_plot`` when given.

    ``calibration_binning`` selects ``'uniform'`` (equal-width) or ``'quantile'``
    (equal-count) ECE bins; quantile binning keeps the sparse high-probability
    region from being swamped by the near-zero mass under heavy class imbalance.

    Post-hoc calibration (binary only): ``calibration_method`` is one of
    ``'none'``/``'temperature'``/``'platt'``/``'isotonic'``. The calibrator is fit
    on ``calibration_fit`` (a ``(scores, ground_truth)`` held-out pair, an
    out-of-sample fit) if given, else in-sample on this set. ``temperature=T``
    applies a known scalar directly without fitting (overrides
    ``calibration_method``). When a calibrator is applied, the pre-calibration
    ECE/MCE are also printed.

    Returns ``(stats, calibrated)``: ``stats`` is the metrics dict, and
    ``calibrated`` is the calibrated prediction array (same shape as
    ``predictions``) when a calibrator was applied, else ``None``.
    """
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    if _is_multiclass(predictions):
        if calibration_method != 'none' or temperature is not None:
            print("Warning: post-hoc calibration is binary-only; ignoring for multiclass.")
        return calc_stats_multiclass(predictions, ground_truth, class_names=class_names), None

    labels = ground_truth > 0

    # Post-hoc calibration. Fitting and reporting on the same set (in-sample);
    # pass calibration_fit (a held-out split) for an out-of-sample estimate.
    transform = None
    cal_info = None
    if temperature is not None and temperature != 1.0:
        transform = lambda s: apply_temperature(s, temperature)  # noqa: E731
        cal_info = f"temperature (T={temperature:.4f}, supplied)"
    elif calibration_method != 'none':
        if calibration_fit is not None:
            fit_scores, fit_gt = calibration_fit
            fit_scores, fit_labels = np.asarray(fit_scores), np.asarray(fit_gt) > 0
            fit_src = "held-out calibration set"
        else:
            fit_scores, fit_labels = predictions, labels
            fit_src = "this set (in-sample)"
        transform, cal_info = fit_calibrator(calibration_method, fit_scores, fit_labels)
        cal_info += f", fit on {fit_src}"

    if transform is not None:
        ece_raw, mce_raw, _ = expected_calibration_error(
            predictions, labels, n_bins=calibration_bins, strategy=calibration_binning)
        predictions = transform(predictions)
        print(f"Applied post-hoc calibration: {cal_info}")
        print(f"  pre-calibration:  ECE={ece_raw:.4f}  MCE={mce_raw:.4f}")

    if grouped:
        unique_labels = np.unique(ground_truth)
        for label in unique_labels:
            if label != 0:  # Skip background
                label_mask = ground_truth == label
                stats = _binary_metrics(label_mask, predictions > threshold, scores=predictions)
                _print_metrics(f"Stats for group {label}:", stats)

    overall = _binary_metrics(labels, predictions > threshold, scores=predictions)

    # Calibration uses the continuous scores (not the thresholded labels).
    ece, mce, bins = expected_calibration_error(
        predictions, labels, n_bins=calibration_bins, strategy=calibration_binning)
    overall["ece"] = ece
    overall["mce"] = mce
    if cal_info is not None:
        overall["calibration"] = cal_info

    _print_metrics("Overall Stats:", overall)
    print(f"Reliability ({calibration_binning} bins, predicted probability vs "
          f"empirical frequency):")
    print(reliability_table_str(bins))
    if reliability_plot is not None:
        plot_reliability_diagram(bins, ece, mce, reliability_plot)

    return overall, (predictions if transform is not None else None)


def _fire_probability(predictions):
    """Per-pixel probability of fire (the pixel's expected burn value) as (H, W).

    Binary predictions are already P(fire). Multiclass predictions are
    band-first (band 0 is the argmax, band ``1 + c`` is the softmax score for
    class ``c``, class 0 = no-fire), so P(fire) is the complement of the class-0
    score, ``1 - predictions[1]``.
    """
    predictions = np.asarray(predictions)
    if _is_multiclass(predictions):
        return 1.0 - predictions[1]
    return predictions


def _burn_area_aggregate(actual, expected):
    """Aggregate actual-vs-expected burn-area stats over a set of regions.

    ``actual`` and ``expected`` are 1-D arrays of per-region burned-pixel counts
    (ground-truth count and summed probability, respectively). Returns totals,
    the expected/actual ratio, and per-region bias / MAE / RMSE, plus the Pearson
    correlation between actual and expected across regions. Shared by the tile
    and municipality analyses.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = expected - actual
    n = int(actual.size)
    total_actual = float(actual.sum())
    total_expected = float(expected.sum())

    # Pearson r is undefined with <2 regions or a constant side (zero variance).
    if n > 1 and actual.std() > 0 and expected.std() > 0:
        pearson_r = float(np.corrcoef(actual, expected)[0, 1])
    else:
        pearson_r = float('nan')

    return {
        'n': n,
        'total_actual': total_actual,
        'total_expected': total_expected,
        'ratio': (total_expected / total_actual) if total_actual else float('nan'),
        'bias': float(error.mean()) if n else 0.0,
        'mae': float(np.abs(error).mean()) if n else 0.0,
        'rmse': float(np.sqrt((error ** 2).mean())) if n else 0.0,
        'pearson_r': pearson_r,
    }


def _print_burn_summary(stats, header, count_label, per_label):
    """Print the aggregate burn-area stats (units = burned pixels).

    ``count_label`` labels the region count (e.g. ``"Tiles"``) and ``per_label``
    the per-region metrics (e.g. ``"tile"``).
    """
    print(header)
    print(f"  {count_label}: {stats['n']}")
    print(f"  Total actual burn:   {stats['total_actual']:.1f}")
    print(f"  Total expected burn: {stats['total_expected']:.1f}")
    print(f"  Expected/actual ratio: {stats['ratio']:.4f}")
    print(f"  Per-{per_label} bias (expected - actual): {stats['bias']:.4f}")
    print(f"  Per-{per_label} MAE:  {stats['mae']:.4f}")
    print(f"  Per-{per_label} RMSE: {stats['rmse']:.4f}")
    print(f"  Actual-vs-expected correlation (Pearson r): {stats['pearson_r']:.4f}")


def _write_burn_csv(columns, csv_path, what):
    """Write per-region burn areas (a dict of equal-length columns) to CSV."""
    import pandas as pd
    pd.DataFrame(columns).to_csv(csv_path, index=False)
    print(f"Wrote {what} to {csv_path}")


def _plot_burn_scatter(actual, expected, out_path, title,
                       highlight_mask=None, highlight_labels=None):
    """Save an expected-vs-actual burn-area scatter with a y=x reference line.

    matplotlib is imported lazily; if it is not installed, prints a notice and
    returns False instead of raising (mirrors ``plot_reliability_diagram``).

    Args:
        highlight_mask: optional boolean array over the points; where True the
            point is drawn in a distinct color on top of the base scatter.
        highlight_labels: optional labels (same length as the points) annotated
            next to the highlighted points.

    Returns:
        True if the figure was written, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend, no display required
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping burn-area scatter ({out_path}).")
        return False

    actual = np.asarray(actual)
    expected = np.asarray(expected)
    hi = float(max(actual.max(initial=0.0), expected.max(initial=0.0), 1.0))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, hi], [0, hi], '--', color='gray', label='y=x')
    ax.scatter(actual, expected, s=8, alpha=0.4, color='C0')
    if highlight_mask is not None:
        highlight_mask = np.asarray(highlight_mask, dtype=bool)
        ax.scatter(actual[highlight_mask], expected[highlight_mask],
                   s=40, color='C3', edgecolors='k', linewidths=0.5,
                   zorder=3, label='highlighted')
        if highlight_labels is not None:
            highlight_labels = np.asarray(highlight_labels)
            for x, y, lbl in zip(actual[highlight_mask], expected[highlight_mask],
                                 highlight_labels[highlight_mask]):
                ax.annotate(str(lbl), (x, y), textcoords='offset points',
                            xytext=(4, 4), fontsize=7, color='C3')
    ax.set_xlabel('actual burn area (pixels)')
    ax.set_ylabel('expected burn area (pixels)')
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_title(title)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def tile_burn_area_stats(predictions, ground_truth, tile_size=128,
                         csv_path=None, plot=None):
    """Compare actual vs expected burn area over non-overlapping tiles.

    Splits the scene into ``tile_size`` x ``tile_size`` tiles and, per tile,
    sums the ground-truth burned pixels (*actual* burn area) and the predicted
    fire probabilities (*expected* burn area -- the expected number of burned
    pixels, **not** a thresholded count). A spatially smeared but well-calibrated
    model lands near the y=x line here even when its per-pixel F1 is poor, so
    this isolates "does the model get the regional burn amount right?" from "does
    it put the fire in exactly the right pixels?".

    Actual and expected are in units of burned pixels. Partial edge tiles (when
    H/W are not multiples of ``tile_size``) are included; each tile's two sums
    are over the same pixel set so the comparison stays fair, but tiles differ in
    area (``n_pixels`` is recorded). Runs on ``predictions`` as passed in, before
    any post-hoc calibration applied by ``calc_stats``.

    Args:
        predictions: binary scores ``(H, W)`` or multiclass band-first array
            ``(num_classes + 1, H, W)``; converted to P(fire) via
            ``_fire_probability``.
        ground_truth: array of integer class labels; burned = ``> 0``.
        tile_size: side length of the square tiles.
        csv_path: if given, write per-tile rows there.
        plot: if given, write an expected-vs-actual scatter PNG there.

    Returns:
        dict of aggregate stats (``n_tiles``, ``total_actual``,
        ``total_expected``, ``ratio``, ``bias``, ``mae``, ``rmse``,
        ``pearson_r``) plus ``per_tile``, a dict of per-tile numpy arrays
        (``row``, ``col``, ``n_pixels``, ``actual``, ``expected``, ``error``).
    """
    prob = _fire_probability(predictions)
    labels = np.asarray(ground_truth) > 0
    if prob.shape != labels.shape:
        raise ValueError(f"prediction spatial shape {prob.shape} and "
                         f"ground-truth shape {labels.shape} differ")
    height, width = prob.shape

    rows, cols, n_pixels, actual, expected = [], [], [], [], []
    for r in range(0, height, tile_size):
        for c in range(0, width, tile_size):
            p_tile = prob[r:r + tile_size, c:c + tile_size]
            l_tile = labels[r:r + tile_size, c:c + tile_size]
            rows.append(r)
            cols.append(c)
            n_pixels.append(l_tile.size)
            actual.append(float(l_tile.sum()))
            expected.append(float(p_tile.sum()))

    actual = np.asarray(actual)
    expected = np.asarray(expected)
    agg = _burn_area_aggregate(actual, expected)
    per_tile = {
        'row': np.asarray(rows),
        'col': np.asarray(cols),
        'n_pixels': np.asarray(n_pixels),
        'actual': actual,
        'expected': expected,
        'error': expected - actual,
    }

    _print_burn_summary(
        agg, f"Tile burn-area analysis ({tile_size}x{tile_size} tiles, "
        f"units = burned pixels):", "Tiles", "tile")
    if csv_path is not None:
        _write_burn_csv(per_tile, csv_path, "per-tile burn areas")
    if plot is not None:
        _plot_burn_scatter(
            actual, expected, plot,
            f"{tile_size}x{tile_size} tiles")

    return {**agg, 'n_tiles': agg['n'], 'per_tile': per_tile}


def _print_extreme_municipalities(per_muni, top_n):
    """Print the municipalities whose expected burn most over/under-shoots actual."""
    error = np.asarray(per_muni['error'])
    if error.size == 0 or top_n <= 0:
        return
    name = np.asarray(per_muni['nm_mun'])
    uf = np.asarray(per_muni['sigla_uf'])
    actual = np.asarray(per_muni['actual'])
    expected = np.asarray(per_muni['expected'])
    order = np.argsort(error)  # most negative (under-predicted) first

    def line(i):
        return (f"    {name[i]} ({uf[i]}): actual={actual[i]:.1f} "
                f"expected={expected[i]:.1f} error={error[i]:+.1f}")

    print(f"  Most OVER-predicted (expected >> actual), top {top_n}:")
    for i in order[::-1][:top_n]:
        print(line(i))
    print(f"  Most UNDER-predicted (expected << actual), top {top_n}:")
    for i in order[:top_n]:
        print(line(i))


def municipality_burn_area_stats(predictions, ground_truth, transform, crs,
                                 shp_path, name_field='nm_mun',
                                 code_field='cd_mun', state_field='sigla_uf',
                                 top_n=5, csv_path=None, plot=None,
                                 highlight_cd_mun=None):
    """Compare actual vs expected burn area aggregated by Brazilian municipality.

    The zonal analogue of ``tile_burn_area_stats``: instead of a fixed grid, it
    rasterizes the municipality polygons from ``shp_path`` onto the prediction
    grid (given by ``transform`` / ``crs``) and, per municipality that overlaps
    the scene, sums the ground-truth burned pixels (*actual* burn area) and the
    predicted fire probabilities (*expected* burn area -- the expected number of
    burned pixels, **not** a thresholded count). Municipalities are administrative
    units fire managers act on, so this reports where the model over- or
    under-predicts the regional burn amount.

    Actual and expected are in units of burned pixels; only municipalities with
    at least one pixel inside the scene are kept. Runs on ``predictions`` as
    passed in, before any post-hoc calibration applied by ``calc_stats``.

    Args:
        predictions: binary scores ``(H, W)`` or multiclass band-first array;
            converted to P(fire) via ``_fire_probability``.
        ground_truth: array of integer class labels; burned = ``> 0``.
        transform: affine transform of the prediction grid (from the raster).
        crs: CRS of the prediction grid; polygons are reprojected to match.
        shp_path: path to the municipality shapefile.
        name_field/code_field/state_field: shapefile columns for the name, IBGE
            code, and state abbreviation.
        top_n: how many most over/under-predicted municipalities to print.
        csv_path: if given, write per-municipality rows there.
        plot: if given, write an expected-vs-actual scatter PNG there.
        highlight_cd_mun: optional iterable of ``cd_mun`` codes; matching
            municipalities are highlighted and name-annotated in the scatter
            plot and flagged in the CSV via a ``highlighted`` column.

    Returns:
        dict of aggregate stats (as ``_burn_area_aggregate``) plus
        ``n_municipalities`` and ``per_municipality`` (a dict of per-municipality
        arrays), or ``None`` if no municipality overlaps the scene.
    """
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import array_bounds

    prob = _fire_probability(predictions)
    labels = np.asarray(ground_truth) > 0
    if prob.shape != labels.shape:
        raise ValueError(f"prediction spatial shape {prob.shape} and "
                         f"ground-truth shape {labels.shape} differ")
    height, width = prob.shape

    gdf = gpd.read_file(shp_path)
    if crs is not None and gdf.crs is not None and str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)
    # Restrict to municipalities overlapping the raster's bounding box.
    left, bottom, right, top = array_bounds(height, width, transform)
    gdf = gdf.cx[left:right, bottom:top].reset_index(drop=True)
    if len(gdf) == 0:
        print("Municipality burn-area analysis: no municipalities overlap the "
              "prediction extent.")
        return None

    # Rasterize polygons to zone ids 1..N (0 = outside any polygon) on the
    # prediction grid, then take zonal sums via weighted bincount.
    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    zones = features.rasterize(shapes, out_shape=(height, width),
                               transform=transform, fill=0, dtype='int32')
    zones_flat = zones.ravel()
    n = len(gdf)
    counts = np.bincount(zones_flat, minlength=n + 1)[1:]
    actual_all = np.bincount(zones_flat, weights=labels.ravel().astype(np.float64),
                             minlength=n + 1)[1:]
    expected_all = np.bincount(zones_flat, weights=prob.ravel().astype(np.float64),
                               minlength=n + 1)[1:]

    present = counts > 0  # keep only municipalities with pixels in the scene
    gdf_p = gdf.loc[present]
    actual = actual_all[present]
    expected = expected_all[present]

    agg = _burn_area_aggregate(actual, expected)
    per_muni = {
        'cd_mun': gdf_p[code_field].to_numpy(),
        'nm_mun': gdf_p[name_field].to_numpy(),
        'sigla_uf': gdf_p[state_field].to_numpy(),
        'n_pixels': counts[present].astype(np.int64),
        'actual': actual,
        'expected': expected,
        'error': expected - actual,
    }

    highlight_mask = None
    if highlight_cd_mun:
        wanted = {str(c).strip() for c in highlight_cd_mun}
        codes_str = per_muni['cd_mun'].astype(str)
        highlight_mask = np.isin(codes_str, list(wanted))
        matched = codes_str[highlight_mask]
        missing = wanted - set(matched.tolist())
        print(f"  Highlighting {highlight_mask.sum()} municipalities by cd_mun.")
        if missing:
            print(f"  cd_mun not found in scene: {', '.join(sorted(missing))}")
        per_muni['highlighted'] = highlight_mask

    _print_burn_summary(
        agg, "Municipality burn-area analysis (units = burned pixels):",
        "Municipalities", "municipality")
    _print_extreme_municipalities(per_muni, top_n)
    if csv_path is not None:
        _write_burn_csv(per_muni, csv_path, "per-municipality burn areas")
    if plot is not None:
        _plot_burn_scatter(
            actual, expected, plot, "municipalities",
            highlight_mask=highlight_mask, highlight_labels=per_muni['nm_mun'])

    return {**agg, 'n_municipalities': agg['n'], 'per_municipality': per_muni}


def read_raster_geo(predictions_path):
    """Return ``(transform, crs)`` for a GeoTIFF path, or ``(None, None)`` for CSV.

    Municipality analysis needs the prediction grid's georeferencing, which
    ``load_preprocess_inputs`` drops when it returns bare arrays.
    """
    if predictions_path.endswith('.csv'):
        return None, None
    import rasterio as rio
    with rio.open(predictions_path) as src:
        return src.transform, src.crs


def load_preprocess_inputs(predictions_path, ground_truth_path):
    """Load and preprocess inputs for evaluation.

    For tif predictions, all bands are read. A single-band raster is squeezed to
    2D (binary scores); a multi-band raster is returned band-first
    ``(num_bands, H, W)`` (argmax band + class scores) and triggers the
    multiclass code path in ``calc_stats``.
    """
    if predictions_path.endswith('.csv'):
        if not ground_truth_path.endswith('.csv'):
            raise ValueError("Both predictions and ground truth must be in the same format (both csv or both tif)")
        import pandas as pd
        predictions = pd.read_csv(predictions_path)['pred'].values
        ground_truth = pd.read_csv(ground_truth_path)['label'].values
    else:
        if not ground_truth_path.endswith('.tif') and not predictions_path.endswith('.tif'):
            raise ValueError("Both predictions and ground truth must be in the same format (both csv or both tif")
        import rasterio as rio
        predictions = rio.open(predictions_path).read()  # (bands, H, W)
        if predictions.shape[0] == 1:
            predictions = predictions[0]  # binary: squeeze to (H, W)
        ground_truth = rio.open(ground_truth_path).read(1)
    return predictions, ground_truth


def write_calibrated_predictions(predictions_path, output_path, calibrated):
    """Write calibrated predictions back out, mirroring the input format.

    For a CSV input, copies the source frame and overwrites the ``pred`` column
    with the calibrated probabilities (other columns preserved). For a GeoTIFF
    input, copies the source raster's profile (georeferencing, compression) and
    writes the calibrated scores as a single float32 band. ``calibrated`` is the
    array returned by ``calc_stats`` and must match what ``load_preprocess_inputs``
    read for ``predictions_path`` (binary only).
    """
    calibrated = np.asarray(calibrated)
    if predictions_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(predictions_path)
        if len(df) != calibrated.size:
            raise ValueError(
                f"calibrated length {calibrated.size} does not match "
                f"{len(df)} rows in {predictions_path}")
        df['pred'] = calibrated.reshape(-1)
        df.to_csv(output_path, index=False)
    else:
        import rasterio as rio
        with rio.open(predictions_path) as src:
            profile = src.profile
        profile.update(count=1, dtype=rio.float32)
        with rio.open(output_path, 'w', **profile) as dst:
            dst.write(calibrated.astype('float32'), 1)
    print(f"Wrote calibrated predictions to {output_path}")
