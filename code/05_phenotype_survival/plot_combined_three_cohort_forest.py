"""Create a publication-ready 1 x 3 forest plot for CFS, MrOS, and SHHS."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\SaberShuo\Documents\SDI")
STATS_PATH = (
    ROOT
    / "analysis"
    / "three_cohort_independent_sdi8_embedding8_outcomes"
    / "all_cohorts_forest_stats.csv"
)
OUTPUT_DIR = ROOT / "output" / "pdf"

FONT_SCALE = 1.8
FS_TITLE = 13 * FONT_SCALE
FS_LABEL = 12 * FONT_SCALE
FS_TICK = 9 * FONT_SCALE
FS_Y_TICK = 10 * FONT_SCALE
FS_ANNOT = 9 * FONT_SCALE

COHORTS = [("shhs", "SHHS"), ("mros", "MROS"), ("cfs", "CFS")]
CONTINUOUS_ROWS = [
    "Age",
    "BMI",
    "SE",
    "SL",
    "Features",
    "CWSD",
    "STDE",
    "TDTL",
    "SSTR",
    "BRAI",
    "TOS",
    "TCI",
    "SRI",
    "NE1",
    "NE2",
    "NE3",
    "NE4",
    "NE5",
    "NE6",
    "NE7",
    "NE8",
]
OUTCOME_ROWS = [
    "Outcomes",
    "Sleep Apnea",
    "Poor Sleep Quality",
    "Insomnia",
    "Diabetes",
    "CVD",
    "Hypertension",
]
ROW_ORDER = CONTINUOUS_ROWS + OUTCOME_ROWS
HEADERS = {"Features", "Outcomes"}
DISPLAY_LABELS = {"BRAI": "BBAI"}

BLUE = "#6BAED6"
ORANGE = "#E6550D"
GRAY = "#8C8C8C"


def get_row(stats: pd.DataFrame, cohort: str, variable: str) -> pd.Series | None:
    source_variable = "OSA" if variable == "Sleep Apnea" else variable
    matched = stats.loc[
        stats["Cohort"].eq(cohort) & stats["Variable"].eq(source_variable)
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def draw_panel(
    ax_or: plt.Axes,
    stats: pd.DataFrame,
    cohort: str,
    title: str,
    show_y_labels: bool,
    or_limit: float,
    or_ticks: list[float],
) -> None:
    ax_d = ax_or.twiny()
    y_lookup = {label: index for index, label in enumerate(ROW_ORDER)}

    for variable in ROW_ORDER:
        if variable in HEADERS:
            continue
        row = get_row(stats, cohort, variable)
        if row is None or row["Status"] != "ok":
            continue
        value = float(row["Val"])
        low = float(row["Low"])
        high = float(row["High"])
        y = y_lookup[variable]
        if row["Type"] == "OR":
            ax_or.errorbar(
                value,
                y,
                xerr=np.array([[value - low], [high - value]]),
                fmt="*",
                color=ORANGE,
                ecolor=ORANGE,
                markersize=7,
                capsize=2.5,
                zorder=4,
            )
        elif row["Type"] == "d":
            ax_d.errorbar(
                value,
                y,
                xerr=np.array([[value - low], [high - value]]),
                fmt="o",
                color=BLUE,
                ecolor=BLUE,
                markersize=5,
                capsize=2.5,
                zorder=4,
            )

    ymin, ymax = -0.55, len(ROW_ORDER) - 0.45
    separator_style = dict(color="black", linestyle="-", linewidth=1.6, alpha=0.9)
    for separator in ("Features", "Outcomes"):
        separator_y = y_lookup[separator]
        ax_or.axhline(separator_y, **separator_style)
        ax_d.axhline(separator_y, **separator_style)

    outcomes_y = y_lookup["Outcomes"]
    ax_d.plot(
        [0, 0], [ymin, outcomes_y], linestyle="--", linewidth=1,
        color=GRAY, alpha=0.7, zorder=1,
    )
    ax_or.plot(
        [1, 1], [outcomes_y, ymax], linestyle="--", linewidth=1,
        color=GRAY, alpha=0.7, zorder=1,
    )

    ax_or.set_xlim(0, or_limit)
    ax_or.set_xticks(or_ticks)
    ax_or.set_xlabel("Odds Ratio*", fontsize=FS_LABEL, fontweight="bold")
    ax_d.set_xlim(-3, 3)
    ax_d.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax_d.set_xlabel("Cohen's d", fontsize=FS_LABEL, fontweight="bold")
    ax_or.set_ylim(ymax, ymin)
    ax_or.set_yticks(np.arange(len(ROW_ORDER)))

    if show_y_labels:
        labels = [
            label if label in HEADERS else f"  {DISPLAY_LABELS.get(label, label)}"
            for label in ROW_ORDER
        ]
        ax_or.set_yticklabels(labels, fontsize=FS_Y_TICK)
        for tick, label in zip(ax_or.get_yticklabels(), ROW_ORDER):
            if label in HEADERS:
                tick.set_fontweight("bold")
    else:
        ax_or.tick_params(axis="y", labelleft=False, left=False)

    ax_or.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    ax_or.tick_params(axis="x", labelsize=FS_TICK, width=1.1, length=5)
    ax_d.tick_params(axis="x", labelsize=FS_TICK, width=1.1, length=5)
    ax_or.tick_params(axis="y", width=1.1, length=4)
    ax_or.grid(False)
    ax_d.grid(False)

    for axis in (ax_or, ax_d):
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.1)
        axis.spines["bottom"].set_linewidth(1.1)
        axis.spines["top"].set_linewidth(1.1)

    ax_or.annotate(
        "",
        xy=(0.08, -0.09),
        xycoords="axes fraction",
        xytext=(0.42, -0.09),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
        annotation_clip=False,
    )
    ax_or.text(
        0.17,
        -0.13,
        "Normal Sleep",
        transform=ax_or.transAxes,
        ha="center",
        va="top",
        fontsize=FS_ANNOT,
        fontweight="bold",
    )
    ax_or.annotate(
        "",
        xy=(0.92, -0.09),
        xycoords="axes fraction",
        xytext=(0.58, -0.09),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
        annotation_clip=False,
    )
    ax_or.text(
        0.83,
        -0.13,
        "Fragmented Sleep",
        transform=ax_or.transAxes,
        ha="center",
        va="top",
        fontsize=FS_ANNOT,
        fontweight="bold",
    )


def main() -> None:
    stats = pd.read_csv(STATS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(21, 12), sharey=True)
    or_scales = {
        "shhs": (4.0, [0, 1, 2, 3, 4]),
        "mros": (5.5, [0, 1, 2, 3, 4, 5]),
        "cfs": (8.5, [0, 2, 4, 6, 8]),
    }
    for index, ((cohort, title), axis) in enumerate(zip(COHORTS, axes)):
        or_limit, or_ticks = or_scales[cohort]
        draw_panel(
            axis,
            stats,
            cohort,
            title,
            show_y_labels=index == 0,
            or_limit=or_limit,
            or_ticks=or_ticks,
        )

    figure.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.20, wspace=0.12)
    figure.text(
        0.61,
        0.012,
        "* Odds-ratio axes use cohort-specific ranges; compare values by ticks, not horizontal position.",
        ha="center",
        va="bottom",
        fontsize=FS_ANNOT,
    )
    pdf_path = OUTPUT_DIR / "combined_forest_plot_shared_y.pdf"
    png_path = OUTPUT_DIR / "combined_forest_plot_shared_y.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
