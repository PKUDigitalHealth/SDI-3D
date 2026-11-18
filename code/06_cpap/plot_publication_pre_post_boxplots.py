from __future__ import annotations

import math
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


DESKTOP = Path(r"C:\Users\SaberShuo\Desktop")
PRE_PATH = DESKTOP / "\u6cbb\u7597\u524d.csv"
POST_PATH = DESKTOP / "\u6cbb\u7597\u540e.csv"
ANALYSIS_DIR = Path(r"C:\Users\SaberShuo\Documents\SDI\analysis")
STATS_PATH = ANALYSIS_DIR / "pre_post_feature_comparison_exact_pairs.csv"

# Ordered so row 1 is Body, row 2 is Brain, and row 3 completes Brain then dual-axis features.
FEATURES = [
    ("Body_CV", "Body CV"),
    ("Body_SK", "Body SK"),
    ("Body_RB", "Body RB"),
    ("Body_AP", "Body AP"),
    ("Body_APEn", "Body ApEn"),
    ("Body_DFA", "Body DFA"),
    ("Body_MDR", "Body MDR"),
    ("REM_PR", "Brain PR"),
    ("REM_CV", "Brain CV"),
    ("REM_SK", "Brain SK"),
    ("REM_RB", "Brain RB"),
    ("REM_AP", "Brain AP"),
    ("REM_APEn", "Brain ApEn"),
    ("REM_High_Activation_Ratio", "Brain high-score\nratio"),
    ("REM_High_Activation_Ratio_During_REM", "Brain high-score ratio\nwithin REM"),
    ("REM_NREM_Contrast", "Brain-score\nREM–NREM contrast"),
    ("Conf_Weighted_AP", "CWSD"),
    ("BB_Antagonistic_Ratio", "BBAI"),
    ("Path_Length_2D", "TDTL"),
    ("Transition_Rate_per_hour", "SSTR"),
    ("Directional_Entropy", "STDE"),
    ("TOS", "TOS"),
    ("TCI", "TCI"),
    ("SRI", "SRI"),
]

BEFORE_COLOR = "#377EB8"  # Colorblind-safe blue.
AFTER_COLOR = "#E69F00"   # Colorblind-safe orange.


def format_p(value: float) -> str:
    """Compact exact P value suitable for a dense multipanel figure."""
    if value >= 0.001:
        return f"P={value:.3f}"
    return f"P={value:.2e}"


def format_median(value: float) -> str:
    """Format raw medians compactly while retaining meaningful precision."""
    absolute = abs(value)
    if absolute >= 100:
        return f"{value:.1f}"
    if absolute >= 10:
        return f"{value:.2f}"
    if absolute >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the 4-by-6 publication boxplot")
    parser.add_argument("--pre", type=Path, default=PRE_PATH)
    parser.add_argument("--post", type=Path, default=POST_PATH)
    parser.add_argument("--stats", type=Path, default=STATS_PATH)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--expected-pairs", type=int, default=500)
    arguments = parser.parse_args()
    pre = pd.read_csv(arguments.pre, encoding="utf-8-sig")
    post = pd.read_csv(arguments.post, encoding="utf-8-sig")
    paired = pre.merge(post, on="source_file", how="inner", suffixes=("_pre", "_post"))
    if len(paired) != arguments.expected_pairs:
        raise ValueError(
            f"Expected {arguments.expected_pairs} exact-name pairs, found {len(paired)}"
        )
    stats = pd.read_csv(arguments.stats).set_index("feature")

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(4, 6, figsize=(7.45, 8.65))
    axes = axes.ravel()

    for index, (ax, (feature, label)) in enumerate(zip(axes, FEATURES)):
        before = pd.to_numeric(paired[f"{feature}_pre"], errors="coerce").dropna().to_numpy()
        after = pd.to_numeric(paired[f"{feature}_post"], errors="coerce").dropna().to_numpy()
        bp = ax.boxplot(
            [before, after],
            positions=[1, 2],
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.05},
            whiskerprops={"color": "#333333", "linewidth": 0.7},
            capprops={"color": "#333333", "linewidth": 0.7},
            boxprops={"edgecolor": "#333333", "linewidth": 0.7},
        )
        for box, color in zip(bp["boxes"], [BEFORE_COLOR, AFTER_COLOR]):
            box.set_facecolor(color)
            box.set_alpha(0.78)

        p_value = float(stats.at[feature, "wilcoxon_p"])
        effect = float(stats.at[feature, "rank_biserial"])
        pre_median = float(stats.at[feature, "pre_median"])
        post_median = float(stats.at[feature, "post_median"])
        ax.set_title(
            f"{label}\n{format_p(p_value)}\n$r_{{rb}}$={effect:+.2f}",
            fontsize=7.25,
            fontweight="normal",
            pad=2.5,
            linespacing=1.0,
        )
        ax.text(
            -0.13, 1.14, chr(65 + index), transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", ha="left", va="top",
        )
        ax.set_xticks(
            [1, 2],
            [
                f"Pre\n{format_median(pre_median)}",
                f"Post\n{format_median(post_median)}",
            ],
        )
        ax.tick_params(axis="both", labelsize=6.8, pad=1.5)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(FEATURES):]:
        ax.axis("off")

    legend_handles = [
        Patch(facecolor=BEFORE_COLOR, edgecolor="#333333", linewidth=0.6, label="Pre-treatment"),
        Patch(facecolor=AFTER_COLOR, edgecolor="#333333", linewidth=0.6, label="Post-treatment"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handlelength=1.7,
        columnspacing=1.8,
    )
    fig.text(
        0.5, 0.012,
        "Values below Pre and Post are the corresponding raw medians. Boxes indicate median and IQR; "
        "whiskers indicate 1.5×IQR.\n"
        "Two-sided paired Wilcoxon test; "
        f"effect size is paired rank-biserial r; exact paired n={len(paired)}. "
        "BH-FDR-adjusted q values are reported in the accompanying results table.",
        ha="center", va="bottom", fontsize=7.0, linespacing=1.25,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.935, bottom=0.12, wspace=0.66, hspace=1.08)

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    stem = arguments.out_dir / "publication_pre_post_boxplots_24_features_4x6_with_medians"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".tiff"))


if __name__ == "__main__":
    main()
