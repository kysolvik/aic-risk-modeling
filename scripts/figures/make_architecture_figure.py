"""Fig 2: inputs + factored architecture (A) and the CV / final / forecast timeline (B).

Every number is read at runtime from the arch config, gamma JSON, Platt npz and protocol CSV;
check_layout() must print OK.
Usage: make_architecture_figure.py [--timeline_only]"""

import argparse
import csv
import json
import os

import numpy as np

from make_shapley_figure import GROUPS as DRIVER_GROUPS
from style import CALIBRATOR, INK_PRIMARY, INK_SECONDARY, SURFACE, save_figure

INPUT_FILL = "#ffffff"
ENCODER_FILL = "#f2f1ee"
TERM_FILL = "#f7f3ea"
OUTPUT_FILL = "#eef4f8"
EDGE = "#b9b7b0"
ARROW = "#7a7872"
# Panel B: training / CV evaluation / test / forecast
C_TRAIN = "#d9d8d3"
C_EVAL = "#0072b2"
C_TEST = "#d55e00"
C_FORECAST = "#52514e"

ARCH = "factored_v3p_union4_monthlyattn_wide_yeargain"
CONFIG = f"configs/cv/{ARCH}/final_all.json"
GAMMA = "out/cv/gamma/gamma_v3_patched_bd_2002_burn_final_all.json"
PROTOCOL = "out/cv/protocol.csv"
DRIVER_SPEC = "configs/attribution_drivers_v3p_yeargain_yearsplit.json"
PIXEL_M = 463.312716528
FORECAST_YEARS = [2026]           # predict-only, never scored (not in protocol.csv)

# Panel A canvas: 1 unit = 0.1 in, so text sizes in points map predictably.
W_UNITS, H_UNITS = 135.0, 73.0
Y_MIN = 8.6
COLS = {"inputs": (1, 34), "encoders": (39, 63), "stack": (67, 71),
        "terms": (77, 103), "output": (107, 134)}
HEADER_Y = 70.0

# Input rows: (key, config group, y0, y1). Encoders share the row centre.
INPUT_ROWS = [
    ("annual", "im_annual", 56.5, 66.0),
    ("weather", "im_monthly_coarse", 46.6, 54.9),
    ("veg", "im_monthly_fine", 39.5, 45.0),
    ("static", "im_single_cnn", 27.3, 37.9),
    ("location", "md_single", 20.2, 25.7),
]
# Head boxes in the terms column (top -> bottom), plus the two scalar inputs.
HEAD_ROWS = {
    "indices": (60.5, 66.0),
    "m": (50.5, 57.8),
    "s": (42.0, 48.3),
    "c": (32.5, 39.8),
    "gamma": (19.2, 29.8),
    "year": (9.2, 14.7),
}
OUTPUT_ROWS = {
    "legend": (46.0, 66.0),
    "sigmoid": (36.6, 42.0),
    "platt": (27.0, 34.0),
    "prob": (19.0, 24.5),
    "target": (9.2, 16.8),
}
GRES_Y = 17.3
SIGMA_XY = (109.0, 39.3)
SIGMA_R = 1.9

TITLE_PT, BODY_PT, HEADER_PT = 8.6, 7.2, 10.0
PAD_X, PAD_TOP = 1.0, 0.9


def load_facts():
    cfg = json.load(open(CONFIG))
    gam = json.load(open(GAMMA))
    cal = np.load(CALIBRATOR)
    spec = json.load(open(DRIVER_SPEC))
    dec = cfg["decoder_config"]
    feats = cfg["input_features"]
    chip_px = int(feats["im_annual"]["shape"][0])
    f = {
        "groups": {g: {"n": len(v["feature_names"]), "t": len(v.get("timesteps") or []),
                       "model": v.get("model_type"), "kw": v.get("model_kwargs", {}),
                       "shape": v.get("shape")}
                   for g, v in feats.items()},
        "chip_px": chip_px,
        "chip_km": chip_px * PIXEL_M / 1000,
        "kernel": int(dec["local_kernel"]),
        "kernel_km": int(dec["local_kernel"]) * PIXEL_M / 1000,
        "lattice": int(dec["coarse_grid"]),
        "lattice_km": chip_px / int(dec["coarse_grid"]) * PIXEL_M / 1000,
        "context": dec.get("context_groups", []),
        "b_soi": gam["coeffs"]["b_soi"], "b_prev": gam["coeffs"]["b_prev"],
        "gamma_fit": (min(gam["fit"]["fit_years"]), max(gam["fit"]["fit_years"])),
        "gamma_terms": gam["terms"],
        "platt_a": float(cal["a"]), "platt_b": float(cal["b"]),
        "platt_years": (int(cal["fit_years"].min()), int(cal["fit_years"].max())),
        "pos_weight": cfg["pos_weight"],
        "targets": cfg["output_features"]["feature_names"],
        "target_combine": cfg["output_features"].get("combine"),
    }
    weather = f["groups"]["im_monthly_coarse"]["kw"]
    f["weather_grid"] = int(weather["grid"])
    f["weather_km"] = chip_px / int(weather["grid"]) * PIXEL_M / 1000
    drivers = {}
    for name, refs in spec["drivers"].items():
        for group, _ in refs:
            drivers.setdefault(group, set()).add(name)
    drivers["gamma"] = set(spec.get("year_terms", {}))
    f["drivers"] = drivers
    rows = []
    with open(PROTOCOL) as fh:
        for r in csv.DictReader(fh):
            if r["arch"] == ARCH and r["stage"] in ("folds", "final"):
                rows.append({"fold": r["fold_id"], "stage": r["stage"],
                             "train": [int(y) for y in r["train_years"].split(";")],
                             "eval": [int(y) for y in r["eval_years"].split(";") if y]})
    f["jobs"] = sorted(rows, key=lambda r: (r["stage"] != "folds", r["fold"]))
    return f


def print_facts(f):
    print("[arch_fig] numbers shown in the figure:")
    for g, v in f["groups"].items():
        print(f"  {g:18s} {v['n']:2d} bands x {v['t']:2d} steps  {v['model']}  {v['kw']}")
    print(f"  chip {f['chip_px']} px = {f['chip_km']:.1f} km at {PIXEL_M:.0f} m; weather pool "
          f"{f['weather_grid']}x{f['weather_grid']} = {f['weather_km']:.1f} km; c kernel "
          f"{f['kernel']} = {f['kernel_km']:.1f} km; m lattice {f['lattice']}x{f['lattice']} = "
          f"{f['lattice_km']:.1f} km; m context {f['context']}")
    print(f"  gamma {f['gamma_terms']}: b_soi {f['b_soi']:+.3f}, b_prev {f['b_prev']:+.3f}, "
          f"fit {f['gamma_fit'][0]}-{f['gamma_fit'][1]}")
    print(f"  Platt a {f['platt_a']:.3f} b {f['platt_b']:+.3f} fit {f['platt_years']}; "
          f"pos_weight {f['pos_weight']}; target {f['target_combine']} of {f['targets']}")
    for j in f["jobs"]:
        print(f"  {j['fold']:14s} train {j['train'][0]}-{j['train'][-1]} ({len(j['train'])})  "
              f"eval {j['eval']}")
    for g, d in sorted(f["drivers"].items()):
        print(f"  drivers[{g}] = {sorted(d)}")


def num(x, fmt=".2f"):
    """Format with a true minus sign (U+2212), not a hyphen."""
    return format(x, fmt).replace("-", "−")


def signed(x, fmt=".2f"):
    """'− 0.12' / '+ 0.12': a spaced operator for a continued equation line."""
    return ("− " if x < 0 else "+ ") + format(abs(x), fmt)


def box_text(f):
    g = f["groups"]
    an, wx, vg, st = g["im_annual"], g["im_monthly_coarse"], g["im_monthly_fine"], g["im_single_cnn"]
    ix = g["md_monthly"]
    months = ix["shape"][0]
    return {
        "annual": (f"Annual · {an['t']} yr × {an['n']} bands",
                   "MapBiomas land use: agriculture, pasture,\n"
                   "forest, mining, urban, water\n"
                   "MCD64A1 burned area · MOD14 active fire\n"
                   "MODIS EVI, NDVI · CHIRPS water deficit"),
        "weather": (f"Monthly weather · {wx['t']} mo × {wx['n']} bands",
                    "AgERA5 temperature (mean, min, max),\n"
                    "VPD, precipitation · ERA5-Land evaporation,\n"
                    "precipitation, water deficit · CHIRPS deficit"),
        "veg": (f"Monthly vegetation · {vg['t']} mo × {vg['n']} bands",
                "MODIS NDVI, EVI"),
        "static": (f"Static + previous year · {st['n']} bands",
                   "SRTM elevation, slope · population\n"
                   "travel time to cities · Hansen tree cover\n"
                   "protected-area governance · night lights\n"
                   "year t−1: land use, burned area (MCD64A1,\n"
                   "VNP64A1), VIIRS active fire, EVI, NDVI, deficit"),
        "location": ("Location", "chip centre (x, y)"),
        "enc_annual": ("Per-pixel temporal transformer",
                       f"attention across {an['t']} years at each pixel\n"
                       f"{an['kw']['depth']} layers · {an['kw']['num_heads']} heads "
                       f"· d = {an['kw']['dim']}"),
        "enc_weather": ("Coarse temporal transformer",
                        f"pool to {f['weather_grid']}×{f['weather_grid']} "
                        f"({f['weather_km']:.1f} km) → attention\n"
                        f"across {wx['t']} months → upsample"),
        "enc_veg": ("Per-pixel temporal transformer",
                    f"across {vg['t']} months · d = {vg['kw']['dim']}"),
        "enc_static": ("Pixel MLP",
                       f"pointwise · {st['kw']['out_channels']} channels"),
        "enc_location": ("Fourier location code", "broadcast to every pixel"),
        "indices": (f"Climate indices · {ix['n']} × {months} months",
                    "MEI, ONI, SOI, TNA, AMO (previous 10 yr)"),
        "m": ("m · Coarse intensity (where)",
              f"{f['lattice']}×{f['lattice']} lattice ({f['lattice_km']:.0f} km cells),\n"
              "climate indices as context"),
        "s": ("s · Pixel susceptibility (ignition)",
              "1×1 MLP on the pixel's own features"),
        "c": ("c · Local spread",
              f"{f['kernel']}×{f['kernel']} kernel ({f['kernel_km']:.1f} km), "
              "centre pixel\nmasked: neighbours only"),
        "gamma": ("γ(t) · (1 + g_res) · Year term",
                  f"γ = {num(f['b_soi'], '.2f')} SOI (Oct–Dec, t−1)\n"
                  f"     {signed(f['b_prev'])} log basin burned area (t−1)\n"
                  f"standardized; frozen fit {f['gamma_fit'][0]}–{f['gamma_fit'][1]}\n"
                  "g_res: per-chip gain from location"),
        "year": ("Year t", "γ lookup"),
        "sigmoid": ("Sigmoid", None),
        "platt": ("Platt calibration",
                  f"a = {num(f['platt_a'], '.2f')}, b = {num(f['platt_b'], '.2f')}\n"
                  f"fit on CV years {f['platt_years'][0]}–{f['platt_years'][1]}"),
        "prob": ("Burn probability, year t", f"{PIXEL_M:.0f} m pixels"),
        "target": ("Training target",
                   "burned in any of MCD64A1, VNP64A1,\n"
                   "MOD14, VIIRS active fire\n"
                   f"weighted BCE (positives ×{f['pos_weight']:g})"),
    }


def draw_box(ax, key, x0, y0, x1, y1, title, body, fill, registry, texts,
             title_color=INK_PRIMARY, center=False, bold=True, lw=0.9):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                facecolor=fill, edgecolor=EDGE, linewidth=lw, zorder=2))
    registry[key] = (x0, y0, x1, y1)
    if center and body is None:
        t = ax.text((x0 + x1) / 2, (y0 + y1) / 2, title, ha="center", va="center",
                    fontsize=TITLE_PT, color=title_color,
                    fontweight="bold" if bold else "normal", zorder=4)
        texts.append((key, t))
        return
    t = ax.text(x0 + PAD_X, y1 - PAD_TOP, title, ha="left", va="top", fontsize=TITLE_PT,
                color=title_color, fontweight="bold" if bold else "normal", zorder=4)
    texts.append((key, t))
    if body:
        b = ax.text(x0 + PAD_X, y1 - PAD_TOP - 1.75, body, ha="left", va="top",
                    fontsize=BODY_PT, color=INK_SECONDARY, linespacing=1.3, zorder=4)
        texts.append((key, b))


def arrow(ax, pts, paths, dashed=False, color=ARROW):
    """Orthogonal polyline; arrowhead on the last segment. Registered for check_layout."""
    ls = (0, (3, 2)) if dashed else "-"
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, linewidth=1.0, linestyle=ls, zorder=1,
                solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.0, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    paths.append(pts)


def driver_dots(ax, x1, y1, groups, texts_key, dots):
    """Small dots (Fig 7 driver-group colours) in a box's top-right corner."""
    present = [(label, col) for band, label, col in DRIVER_GROUPS
               if band.replace("shapley_", "") in groups]
    for i, (_, col) in enumerate(present):
        cx = x1 - 1.3 - (len(present) - 1 - i) * 1.25
        ax.scatter([cx], [y1 - 1.35], s=16, color=col, edgecolor="white", linewidth=0.5,
                   zorder=5)
        dots.append((texts_key, cx))


def check_layout(fig, ax, texts, boxes, paths, pad_px=3):
    """Text inside its own box, touching no other box or text, and no arrow through text."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    disp = {k: Bbox.from_extents(*ax.transData.transform([(b[0], b[1]), (b[2], b[3])]).ravel())
            for k, b in boxes.items()}
    ext = [(k, t, t.get_window_extent(r)) for k, t in texts]
    bad = 0
    for k, t, e in ext:
        name = t.get_text().split("\n")[0]
        if k in disp:
            own = disp[k].padded(-pad_px)
            if not (own.x0 <= e.x0 and e.x1 <= own.x1 and own.y0 <= e.y0 and e.y1 <= own.y1):
                print(f"[arch_fig] LAYOUT: {name!r} crosses its {k} box edge")
                bad += 1
        for k2, b in disp.items():
            if k2 != k and b.overlaps(e):
                print(f"[arch_fig] LAYOUT: {name!r} touches the {k2} box")
                bad += 1
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            if ext[i][2].overlaps(ext[j][2]):
                print(f"[arch_fig] LAYOUT: {ext[i][1].get_text()[:30]!r} overlaps "
                      f"{ext[j][1].get_text()[:30]!r}")
                bad += 1
    for pts in paths:
        p = ax.transData.transform(pts)
        samples = np.concatenate([np.linspace(p[i], p[i + 1], 60) for i in range(len(p) - 1)])
        for k, t, e in ext:
            inside = ((samples[:, 0] > e.x0) & (samples[:, 0] < e.x1)
                      & (samples[:, 1] > e.y0) & (samples[:, 1] < e.y1))
            if inside.any():
                print(f"[arch_fig] LAYOUT: arrow {pts[0]}->{pts[-1]} crosses "
                      f"{t.get_text().split(chr(10))[0]!r}")
                bad += 1
    print(f"[arch_fig] layout check: {'OK' if not bad else f'{bad} problem(s)'}")
    return bad


def panel_a(fig, ax, f):
    T = box_text(f)
    boxes, texts, paths, dots = {}, [], [], []
    ax.set_xlim(0, W_UNITS)
    ax.set_ylim(Y_MIN, H_UNITS)
    ax.set_aspect("equal")
    ax.axis("off")

    headers = [("inputs", "Inputs"), ("encoders", "Encoders"),
               ("terms", "Factored head (log-odds)"), ("output", "Output")]
    for col, label in headers:
        x0, x1 = COLS[col]
        t = ax.text((x0 + x1) / 2, HEADER_Y, label, ha="center", va="bottom",
                    fontsize=HEADER_PT, color=INK_SECONDARY, fontweight="bold")
        texts.append((f"hdr_{col}", t))

    # inputs + encoders
    ix0, ix1 = COLS["inputs"]
    ex0, ex1 = COLS["encoders"]
    sx0, sx1 = COLS["stack"]
    centers = {}
    for key, group, y0, y1 in INPUT_ROWS:
        title, body = T[key]
        draw_box(ax, key, ix0, y0, ix1, y1, title, body, INPUT_FILL, boxes, texts)
        driver_dots(ax, ix1, y1, f["drivers"].get(group, set()), key, dots)
        yc = (y0 + y1) / 2
        centers[key] = yc
        eh = min(y1 - y0, 6.6) / 2
        etitle, ebody = T[f"enc_{key}"]
        draw_box(ax, f"enc_{key}", ex0, yc - eh, ex1, yc + eh, etitle, ebody,
                 ENCODER_FILL, boxes, texts)
        arrow(ax, [(ix1, yc), (ex0, yc)], paths)
        arrow(ax, [(ex1, yc), (sx0, yc)], paths)

    # pixel feature stack
    from matplotlib.patches import FancyBboxPatch
    st0, st1 = centers["location"] - 3.0, centers["annual"] + 3.0
    ax.add_patch(FancyBboxPatch((sx0, st0), sx1 - sx0, st1 - st0,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor="#e8e6df", edgecolor=EDGE, linewidth=0.9, zorder=2))
    boxes["stack"] = (sx0, st0, sx1, st1)
    t = ax.text((sx0 + sx1) / 2, (st0 + st1) / 2,
                f"Pixel feature stack · {f['chip_px']}×{f['chip_px']} px at "
                f"{PIXEL_M:.0f} m ({f['chip_km']:.0f} km chip)",
                rotation=90, ha="center", va="center", fontsize=BODY_PT + 0.4,
                color=INK_PRIMARY, zorder=4)
    texts.append(("stack", t))

    # head: scalar inputs + terms
    tx0, tx1 = COLS["terms"]
    tmid = (tx0 + tx1) / 2
    for key in ("indices", "year"):
        y0, y1 = HEAD_ROWS[key]
        title, body = T[key]
        draw_box(ax, key, tx0, y0, tx1, y1, title, body, INPUT_FILL, boxes, texts)
    driver_dots(ax, tx1, HEAD_ROWS["indices"][1], f["drivers"].get("md_monthly", set()),
                "indices", dots)
    for key in ("m", "s", "c", "gamma"):
        y0, y1 = HEAD_ROWS[key]
        title, body = T[key]
        draw_box(ax, key, tx0, y0, tx1, y1, title, body, TERM_FILL, boxes, texts, lw=1.1)
    driver_dots(ax, tx1, HEAD_ROWS["gamma"][1], f["drivers"].get("gamma", set()), "gamma", dots)

    for key in ("m", "s", "c"):
        y0, y1 = HEAD_ROWS[key]
        arrow(ax, [(sx1, (y0 + y1) / 2), (tx0, (y0 + y1) / 2)], paths)
    arrow(ax, [(tmid, HEAD_ROWS["indices"][0]), (tmid, HEAD_ROWS["m"][1])], paths)
    arrow(ax, [(tmid, HEAD_ROWS["year"][1]), (tmid, HEAD_ROWS["gamma"][0])], paths)
    # location -> year-term gain: under the encoder row, into gamma's left edge
    loc_y0 = INPUT_ROWS[-1][2]
    gy = HEAD_ROWS["gamma"][0] + 2.5
    lx = ix0 + 6.0
    arrow(ax, [(lx, loc_y0), (lx, GRES_Y), (tx0 - 2.2, GRES_Y), (tx0 - 2.2, gy), (tx0, gy)],
          paths, dashed=True)
    t = ax.text((ex0 + ex1) / 2, GRES_Y - 0.8, "location → g_res", ha="center", va="top",
                fontsize=BODY_PT, color=INK_SECONDARY, style="italic")
    texts.append(("lbl_gres", t))

    # output column
    from matplotlib.patches import Circle
    ox0, ox1 = COLS["output"]
    cx, cy = SIGMA_XY
    ax.add_patch(Circle((cx, cy), SIGMA_R, facecolor=SURFACE, edgecolor=INK_SECONDARY,
                        linewidth=1.1, zorder=3))
    t = ax.text(cx, cy, "Σ", ha="center", va="center", fontsize=11, color=INK_PRIMARY,
                zorder=4)
    texts.append(("sigma", t))
    boxes["sigma"] = (cx - SIGMA_R, cy - SIGMA_R, cx + SIGMA_R, cy + SIGMA_R)
    for key in ("m", "s", "c", "gamma"):
        y0, y1 = HEAD_ROWS[key]
        yc = (y0 + y1) / 2
        dx, dy = (cx - tx1), (cy - yc)
        n = np.hypot(dx, dy)
        arrow(ax, [(tx1, yc), (cx - dx / n * SIGMA_R, cy - dy / n * SIGMA_R)], paths)

    bx0 = ox0 + 5.0
    bmid = (bx0 + ox1) / 2
    for key in ("sigmoid", "platt", "prob", "target"):
        y0, y1 = OUTPUT_ROWS[key]
        title, body = T[key]
        fill = OUTPUT_FILL if key == "prob" else (SURFACE if key == "target" else ENCODER_FILL)
        draw_box(ax, key, bx0, y0, ox1, y1, title, body, fill, boxes, texts,
                 center=(key == "sigmoid"), bold=(key != "target"),
                 title_color=INK_PRIMARY if key != "target" else INK_SECONDARY)
    arrow(ax, [(cx + SIGMA_R, cy), (bx0, cy)], paths)
    arrow(ax, [(bmid, OUTPUT_ROWS["sigmoid"][0]), (bmid, OUTPUT_ROWS["platt"][1])], paths)
    arrow(ax, [(bmid, OUTPUT_ROWS["platt"][0]), (bmid, OUTPUT_ROWS["prob"][1])], paths)

    # driver-group legend (top right)
    ly0, ly1 = OUTPUT_ROWS["legend"]
    t = ax.text(bx0, ly1, "Driver group (Fig. 7)", ha="left", va="top", fontsize=BODY_PT + 0.4,
                color=INK_SECONDARY, fontweight="bold")
    texts.append(("legend", t))
    for i, (_, label, col) in enumerate(DRIVER_GROUPS):
        yy = ly1 - 3.2 - i * 2.4
        ax.scatter([bx0 + 0.7], [yy], s=22, color=col, edgecolor="white", linewidth=0.5)
        t = ax.text(bx0 + 2.0, yy, label, ha="left", va="center", fontsize=BODY_PT,
                    color=INK_SECONDARY)
        texts.append(("legend", t))

    t = ax.text(0.0, H_UNITS, "A", ha="left", va="top", fontsize=13, fontweight="bold",
                color=INK_PRIMARY)
    texts.append(("letter", t))
    return boxes, texts, paths


def panel_b(fig, ax, f, pt=9, letter=True):
    """Timeline; `pt` = tick/legend font size, `letter` draws the panel letter."""
    from matplotlib.patches import Patch, Rectangle
    years = list(range(min(j["train"][0] for j in f["jobs"]), max(FORECAST_YEARS) + 1))
    rows = f["jobs"]
    n = len(rows)
    for i, job in enumerate(rows):
        yy = n - 1 - i
        final = job["stage"] == "final"
        for y in years:
            hatch, edge = None, "none"
            if y in job["train"]:
                col = C_TRAIN
            elif y in job["eval"]:
                col = C_TEST if final else C_EVAL
            elif final and y in FORECAST_YEARS:
                col, hatch, edge = SURFACE, "////", C_FORECAST
            else:
                continue
            ax.add_patch(Rectangle((y - 0.44, yy - 0.36), 0.88, 0.72, facecolor=col,
                                   edgecolor=edge, hatch=hatch, linewidth=0.8 if hatch else 0))
    labels = [f"CV fold {i + 1}" if j["stage"] == "folds" else "Final model"
              for i, j in enumerate(rows)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=pt, color=INK_SECONDARY)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=pt, color=INK_SECONDARY)
    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("Year", fontsize=pt + 1, color=INK_SECONDARY)
    handles = [Patch(facecolor=C_TRAIN, label="Training"),
               Patch(facecolor=C_EVAL, label="CV evaluation"),
               Patch(facecolor=C_TEST, label="Test (scored once)"),
               Patch(facecolor=SURFACE, edgecolor=C_FORECAST, hatch="////",
                     label="Forecast (no labels)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4,
              frameon=False, fontsize=pt, labelcolor=INK_SECONDARY, handlelength=1.4,
              columnspacing=1.6, borderaxespad=0)
    if letter:
        ax.text(-0.075, 1.18, "B", transform=ax.transAxes, ha="left", va="bottom", fontsize=13,
                fontweight="bold", color=INK_PRIMARY)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_png", default="out/figures/fig_architecture.png")
    ap.add_argument("--timeline_only", action="store_true",
                    help="write only an enlarged panel B")
    a = ap.parse_args()

    f = load_facts()
    print_facts(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.timeline_only:
        fig = plt.figure(figsize=(10.0, 4.4), facecolor=SURFACE)
        ax = fig.add_axes([0.13, 0.15, 0.85, 0.68])
        panel_b(fig, ax, f, pt=13, letter=False)
        out = os.path.join(os.path.dirname(a.out_png), "fig_timeline.png")
        save_figure(fig, out, "arch_fig")
        plt.close(fig)
        return
    fig_w = W_UNITS / 10
    a_h = (H_UNITS - Y_MIN) / 10
    b_h = 2.6
    fig_h = a_h + b_h + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax_a = fig.add_axes([0, (b_h + 0.5) / fig_h, 1, a_h / fig_h])
    ax_a.set_facecolor(SURFACE)
    boxes, texts, paths = panel_a(fig, ax_a, f)
    ax_b = fig.add_axes([0.11, 0.45 / fig_h, 0.86, (b_h - 0.75) / fig_h])
    panel_b(fig, ax_b, f)
    check_layout(fig, ax_a, texts, boxes, paths)

    save_figure(fig, a.out_png, "arch_fig")
    plt.close(fig)


if __name__ == "__main__":
    main()
