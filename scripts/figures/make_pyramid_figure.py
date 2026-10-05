"""Fig 3: PR-AUC across the spatial pyramid (1 px to 128 px blocks) for the v3p CV archs + free baselines.

All series are scored on one fold's eval-year chips; a block is positive if any pixel burned.
Compare models at a fixed scale: block prevalence (and PR-AUC) rises with block size.
Usage: make_pyramid_figure.py [--score_pool mean|max] [--from_csv CSV]"""

import argparse
import csv
import os

import numpy as np

from aic_risk_modeling.eval.chips import chip_pairs, read_window
from aic_risk_modeling.eval.metrics import best_f1, best_kappa, binary_metrics
from style import INK_SECONDARY, SURFACE, save_figure, style_axes

DEFAULT_BLOCKS = [1, 2, 4, 8, 16, 32, 64, 128]
KM_PER_PIXEL = 0.463312716528

MODELS = [
    ("Factored",  "factored_v3p_union4_monthlyattn_wide_yeargain", "#0072b2", "-",  "o"),
    ("U-Net",     "unet_v3p_union4",     "#e69f00", ":",               "D"),
    ("ViT",       "vit_test_v3p_union4", "#cc79a7", (0, (3, 1, 1, 1)), "v"),
    ("MLP",       "mlp_v3p_union4_flat", "#d55e00", "--",              "s"),
    ("LSTM",      "lstm_v3p_union4",     "#009e73", "-.",              "^"),
]
BASELINES = [
    ("Burn frequency", "climatology", "#4d4d4d", (0, (4, 2)), "x"),
    ("Last-year burn", "last_year",   "#9a988f", (0, (1, 1.5)), "P"),
]
# No-skill floor: PR-AUC = block prevalence, ROC-AUC = 0.5.
RANDOM = ("Random", "#8a8880", "--", "")

DEFAULT_LABEL_DIR = "out/label_mosaics_v3p_union4"
DEFAULT_PREDS_ROOT = "out/cv/preds"

PANELS = [("pr_auc", "PR-AUC", (0.0, 1.0))]


def _block_max_pool(arr, block):
    """Non-overlapping block max-pool, zero-padding partial edge blocks (0 is the minimum)."""
    block = int(block)
    if block <= 1:
        return np.asarray(arr)
    arr = np.asarray(arr)
    height, width = arr.shape
    pad_h, pad_w = (-height) % block, (-width) % block
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=0)
    padded_h, padded_w = arr.shape
    return arr.reshape(padded_h // block, block,
                       padded_w // block, block).max(axis=(1, 3))


def _block_mean_pool(arr, block):
    """Non-overlapping block mean-pool; H and W must be multiples of block."""
    block = int(block)
    if block <= 1:
        return np.asarray(arr, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    height, width = arr.shape
    if height % block or width % block:
        raise ValueError(f"mean-pool needs H,W divisible by {block}; got {arr.shape}")
    return arr.reshape(height // block, block,
                       width // block, block).mean(axis=(1, 3))


def _pool(arr, block, how):
    return _block_max_pool(arr, block) if how == "max" else _block_mean_pool(arr, block)


def _inventory(spec):
    return [(o, m, str(year)) for chips_dir, year in spec for o, m in chip_pairs(chips_dir)]


def _levels_from_chips(score_iter, blocks, score_pool):
    acc = {b: {"pa_s": [], "f1_s": [], "y": []} for b in blocks}
    for score, label in score_iter:
        lab_f = (np.asarray(label) > 0).astype(np.float32)
        for b in blocks:
            acc[b]["pa_s"].append(_pool(score, b, score_pool).ravel())
            acc[b]["f1_s"].append(_pool(score, b, "max").ravel())
            acc[b]["y"].append((_block_max_pool(lab_f, b) > 0).ravel())
    levels = []
    for b in blocks:
        y = np.concatenate(acc[b]["y"])
        pa_s = np.concatenate(acc[b]["pa_s"])
        f1_s = np.concatenate(acc[b]["f1_s"])
        prev = float(y.mean())
        pa = binary_metrics(y, pa_s > 0.5, scores=pa_s)
        try:
            from sklearn.metrics import roc_auc_score
            roc = float(roc_auc_score(y, pa_s)) if 0.0 < prev < 1.0 else float("nan")
        except Exception:
            roc = float("nan")
        f1_val, f1_p, f1_r, f1_thr = best_f1(y, f1_s)
        kappa_val, kappa_thr = best_kappa(y, pa_s)
        levels.append({
            "block": b, "n_blocks": int(y.size),
            "prevalence": prev,
            "pr_auc": float(pa["pr_auc"]),
            "roc_auc": roc,
            "kappa": kappa_val, "kappa_threshold": kappa_thr,
            "lift": float(pa["pr_auc"]) / prev if prev > 0 else float("nan"),
            "f1": f1_val, "precision": f1_p, "recall": f1_r,
            "f1_threshold": f1_thr,
        })
        acc[b] = None
    return levels


def model_levels(spec, blocks, score_pool):
    import rasterio as rio

    def gen():
        for out_path, mask_path, _ in _inventory(spec):
            with rio.open(out_path) as s:
                score = s.read(1).astype(np.float32)
            with rio.open(mask_path) as m:
                yield score, m.read(1)
    return _levels_from_chips(gen(), blocks, score_pool)


def baseline_levels(spec, kind, label_dir, clim_path, blocks, score_pool):
    """Per-block metrics for 'last_year' or 'climatology', windowed onto the model chips."""
    import rasterio as rio

    handles = {}

    def _src(path):
        if path not in handles:
            handles[path] = rio.open(path)
        return handles[path]

    def gen():
        for out_path, mask_path, year in _inventory(spec):
            with rio.open(mask_path) as m:
                label = m.read(1)
                bounds = m.bounds
            if kind == "climatology":
                src = _src(clim_path)
            else:
                src = _src(os.path.join(label_dir, f"label_{int(year) - 1}.tif"))
            score = read_window(src, bounds, label.shape).astype(np.float32)
            if kind == "last_year":
                score = (score > 0).astype(np.float32)
            yield score, label
    try:
        return _levels_from_chips(gen(), blocks, score_pool)
    finally:
        for h in handles.values():
            h.close()


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ["model", "block", "km", "n_blocks", "prevalence", "pr_auc", "roc_auc",
            "kappa", "kappa_threshold", "lift", "f1", "precision", "recall",
            "f1_threshold"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"[pyramid] wrote {path}")


def _style_for(name):
    for nm, _p, col, ls, mk in MODELS:
        if nm == name:
            return col, ls, mk, True
    for nm, _k, col, ls, mk in BASELINES:
        if nm == name:
            return col, ls, mk, False
    if name == RANDOM[0]:
        return RANDOM[1], RANDOM[2], RANDOM[3], False
    return "#b6b4ab", "-", "o", False


def _append_random(series):
    base = series[0]["levels"]
    series.append({"name": RANDOM[0],
                   "levels": [{"pr_auc": lvl["prevalence"], "roc_auc": 0.5,
                               "kappa": 0.0}
                              for lvl in base]})


def plot(series, blocks, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(1.0 + 6.4 * len(PANELS), 6.4),
                             facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    xs = np.arange(len(blocks))
    for ax, (metric, ylabel, ylim) in zip(axes, PANELS):
        style_axes(ax)
        for s in series:
            col, ls, mk, is_model = _style_for(s["name"])
            is_random = s["name"] == RANDOM[0]
            y = [lvl[metric] for lvl in s["levels"]]
            ax.plot(xs, y, marker=mk, color=col,
                    linewidth=1.5 if is_random else 2.0, linestyle=ls,
                    markersize=6.0, markeredgecolor=SURFACE,
                    markeredgewidth=0.6, label=s["name"],
                    zorder=6 if is_model else (2 if is_random else 4))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b}\n{b * KM_PER_PIXEL:.3g} km" for b in blocks],
                           fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlabel("Block size (pixels per side / km)", fontsize=10.5,
                      color=INK_SECONDARY)
        ax.set_ylabel(ylabel, fontsize=11.5, color=INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
        ax.set_xlim(-0.35, len(blocks) - 1 + 0.35)
        ax.set_ylim(*ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=2.6)
    fig.tight_layout(rect=[0, 0.1, 1, 1.0])
    save_figure(fig, out_png, "pyramid_fig")
    plt.close(fig)


def build_climatology(label_dir, years, out_path):
    """Pixel-wise burn frequency over `years`, cached as a tif."""
    import rasterio as rio
    if os.path.exists(out_path):
        return out_path
    total, profile = None, None
    for y in years:
        with rio.open(os.path.join(label_dir, f"label_{y}.tif")) as s:
            lab = s.read(1) > 0
            if total is None:
                total, profile = np.zeros(lab.shape, np.uint16), s.profile
            elif s.transform != profile["transform"] or lab.shape != total.shape:
                raise ValueError(f"label_{y}.tif is not on the same grid")
        total += lab
    profile.update(dtype="float32", count=1, nodata=None, compress="deflate")
    with rio.open(out_path, "w", **profile) as d:
        d.write((total / len(years)).astype(np.float32), 1)
    print(f"[pyramid_fig] wrote {out_path}")
    return out_path


def load_from_csv(path):
    by, order = {}, []
    for r in csv.DictReader(open(path)):
        name = r["model"]
        if name not in by:
            by[name] = []
            order.append(name)
        by[name].append({"block": int(r["block"]),
                         "pr_auc": float(r["pr_auc"]),
                         "roc_auc": float(r["roc_auc"]),
                         "kappa": float(r["kappa"]),
                         "prevalence": float(r["prevalence"])})
    blocks = sorted({lvl["block"] for rows in by.values() for lvl in rows})
    series = [{"name": n, "levels": sorted(by[n], key=lambda d: d["block"])}
              for n in order]
    return series, blocks


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    ap.add_argument("--preds_root", default=DEFAULT_PREDS_ROOT)
    ap.add_argument("--fold", default="fwdpair_2022")
    ap.add_argument("--years", default="2022,2023", help="eval years of --fold")
    ap.add_argument("--clim_years", default="2013-2021",
                    help="the fold's train years (climatology)")
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--score_pool", choices=["mean", "max"], default="mean",
                    help="block pooling of every series' score")
    ap.add_argument("--from_csv", default=None,
                    help="restyle from a cached CSV")
    ap.add_argument("--out_png", default=None,
                    help="default out/figures/fig_pyramid_<fold>.png")
    args = ap.parse_args()
    suffix = "" if args.score_pool == "mean" else f"_{args.score_pool}pool"
    out_png = args.out_png or f"out/figures/fig_pyramid_{args.fold}{suffix}.png"
    out_csv = os.path.splitext(out_png)[0] + ".csv"

    if args.from_csv:
        series, blocks = load_from_csv(args.from_csv)
        _append_random(series)
        plot(series, blocks, out_png)
        return

    blocks = ([int(b) for b in args.blocks.split(",") if b.strip()]
              if args.blocks else DEFAULT_BLOCKS)
    years = [int(y) for y in args.years.split(",")]
    a, _, b = args.clim_years.partition("-")
    clim_years = list(range(int(a), int(b or a) + 1))
    if max(clim_years) >= min(years):
        raise ValueError(f"climatology years {args.clim_years} overlap eval years {years}")
    clim = build_climatology(
        args.label_dir, clim_years,
        os.path.join(args.label_dir, f"climatology_{clim_years[0]}_{clim_years[-1]}.tif"))

    def inventory(arch):
        return [(os.path.join(args.preds_root, arch, args.fold, str(y), "chips"), y)
                for y in years]

    series, rows = [], []
    for name, arch, *_ in MODELS:
        print(f"[pyramid_fig] {name}: {arch}/{args.fold} {years}", flush=True)
        levels = model_levels(inventory(arch), blocks, args.score_pool)
        series.append({"name": name, "levels": levels})
        rows += [{"model": name, **lvl} for lvl in levels]

    ref = inventory(MODELS[0][1])
    for name, kind, *_ in BASELINES:
        print(f"[pyramid_fig] baseline {name}", flush=True)
        levels = baseline_levels(ref, kind, args.label_dir, clim, blocks, args.score_pool)
        series.append({"name": name, "levels": levels})
        rows += [{"model": name, **lvl} for lvl in levels]

    for r in rows:
        r["km"] = round(r["block"] * KM_PER_PIXEL, 3)
    write_csv(rows, out_csv)
    _append_random(series)
    plot(series, blocks, out_png)


if __name__ == "__main__":
    main()
