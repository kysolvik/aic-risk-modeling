"""Plot calibrated expected vs actual burned pixels year-to-year.

Reads the CSV written by ``calibrated_year_totals.py`` (columns year,
expected, actual, expected_adj) and plots the calibrated expected burned
pixels (``expected_adj``) against the actual burned pixels. Vertical splits
mark the model's train / validation / test / forecast regions; predict-only
years (no actual label) draw the expected point and a "?" only.

House style (copied per-script; there is no shared plotting-utils module):
INK/SURFACE/GRID inks + the Okabe-Ito colourblind-safe palette, matching
``scatter_expected_actual.py`` / ``make_risk_figure_2024.py``.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

# --- House style (see scatter_expected_actual.py; copied, not imported) ------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"
# Okabe-Ito trio, aligned with CHIP_COLORS in make_risk_figure_2024.py
EXPECTED = "#0072b2"   # blue
ACTUAL = "#d55e00"     # vermillion
SPLIT = "#8a8880"      # split rules (matches ONE_TO_ONE grey)


def draw_splits(ax, boundaries, labels, start_year, end_year):
    """Vertical rules between year regions, with a small label per region.

    ``boundaries`` are the last year of each region except the last (e.g.
    [2022, 2024, 2025] -> rules drawn at 2022.5 / 2024.5 / 2025.5). ``labels``
    names each of the len(boundaries)+1 regions left-to-right.
    """
    edges = [start_year - 0.5] + [b + 0.5 for b in boundaries] + [end_year + 0.5]
    for b in boundaries:
        if start_year - 0.5 < b + 0.5 < end_year + 0.5:
            ax.axvline(b + 0.5, color=SPLIT, linewidth=1.0, linestyle=(0, (5, 4)),
                       zorder=1)
    for lo, hi, name in zip(edges[:-1], edges[1:], labels):
        mid = 0.5 * (max(lo, start_year - 0.5) + min(hi, end_year + 0.5))
        ax.text(mid, 0.975, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=INK_SECONDARY,
                clip_on=False, zorder=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="out/expected_actual.csv",
                        help="Input CSV path")
    parser.add_argument("--out", default="out/expected_actual_plot.png",
                        help="Output plot path")
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=150,
                        help="raster DPI (manuscript figs use 300)")
    # Split boundaries = last year of each region (train/val/test); everything
    # after the last is the forecast region.
    parser.add_argument("--train-end", type=int, default=2022)
    parser.add_argument("--val-end", type=int, default=2024)
    parser.add_argument("--test-end", type=int, default=2025)
    parser.add_argument("--ymax", type=float, default=2_000_000,
                        help="upper y-limit (headroom for the region labels)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[(df["year"] >= args.start_year) & (df["year"] <= args.end_year)]
    df = df.sort_values("year")

    # Rank years by burn within the displayed range (1 = highest burn).
    # Actual is NaN for predict-only years, so those get no actual rank.
    df["exp_rank"] = df["expected_adj"].rank(ascending=False, method="min")
    df["act_rank"] = df["actual"].rank(ascending=False, method="min")

    # Spearman rank correlation between expected and actual burn, over years
    # with an actual label.
    scored = df.dropna(subset=["actual"])
    rho_all = scored["expected_adj"].corr(scored["actual"], method="spearman")

    fig, (ax, ax_tbl) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [3, 1]},
        facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(df["year"], df["expected_adj"], marker="o", color=EXPECTED,
            linewidth=1.8, markersize=6, label="Expected", zorder=4)
    # Actual is missing for predict-only years (e.g. 2026); dropna keeps the
    # line from dipping to zero.
    actual = df.dropna(subset=["actual"])
    ax.plot(actual["year"], actual["actual"], marker="s", color=ACTUAL,
            linewidth=1.8, markersize=6, label="Actual", zorder=4)

    # Predict-only years have no actual label yet; mark them with a "?"
    # (anchored above the expected point) so the missing actual is explicit.
    missing = df[df["actual"].isna()]
    for _, row in missing.iterrows():
        ax.annotate("?", xy=(row["year"], row["expected_adj"]),
                    xytext=(0, 12), textcoords="offset points", color=ACTUAL,
                    fontsize=20, fontweight="bold", ha="center", va="center")

    draw_splits(ax, [args.train_end, args.val_end, args.test_end],
                ["Training", "Validation", "Test", "Forecast"],
                args.start_year, args.end_year)

    ax.set_xlabel("Year", fontsize=11.5, color=INK_SECONDARY)
    ax.set_ylabel("Burned Pixels", fontsize=11.5, color=INK_SECONDARY)
    # No figure title: the manuscript caption carries it (house style).
    ax.set_xticks(df["year"])
    ax.set_xlim(args.start_year - 0.5, args.end_year + 0.5)
    if args.ymax:
        ax.set_ylim(top=args.ymax)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9.5, length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend(loc="upper left", frameon=False, fontsize=10,
              labelcolor=INK_PRIMARY)

    ax.text(0.02, 0.82, f"Spearman ρ (Expected vs Actual): {rho_all:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            color=INK_SECONDARY,
            bbox=dict(boxstyle="round", facecolor=SURFACE, edgecolor=GRID,
                      alpha=0.95))

    # Rank table on the right: Exp vs Act rank per year (1 = highest burn).
    ax_tbl.axis("off")
    cell_text = []
    for _, r in df.iterrows():
        act = "—" if pd.isna(r["act_rank"]) else str(int(r["act_rank"]))
        cell_text.append([str(int(r["year"])), str(int(r["exp_rank"])), act])
    table = ax_tbl.table(cellText=cell_text,
                         colLabels=["Year", "Expected", "Actual"],
                         cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=INK_PRIMARY)
    # Bold the header row.
    for col in range(3):
        table[0, col].set_text_props(fontweight="bold", color=INK_PRIMARY)
    # Highlight years where expected and actual ranks disagree by >= 2 spots
    # (off-by-one is expected noise, so only flag the meaningful misses).
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        if pd.notna(r["act_rank"]) and abs(r["exp_rank"] - r["act_rank"]) >= 2:
            for col in range(3):
                table[i, col].set_facecolor("#f6ece1")
    ax_tbl.set_title(f"Rank (1 = Highest)\n out of {len(df)} Years",
                     fontsize=10, color=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"Wrote {args.out}")
    print(f"Spearman rho (Exp vs Act): {rho_all:.3f}")


if __name__ == "__main__":
    main()
