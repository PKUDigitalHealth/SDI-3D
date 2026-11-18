from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


BASE = Path(r"C:\Users\SaberShuo\Desktop")
PRE_PATH = BASE / "\u6cbb\u7597\u524d.csv"
POST_PATH = BASE / "\u6cbb\u7597\u540e.csv"
STATS_PATH = Path(
    r"C:\Users\SaberShuo\Documents\SDI\analysis\pre_post_feature_comparison_exact_pairs.csv"
)
OUT_DIR = Path(r"C:\Users\SaberShuo\Documents\SDI\analysis")

FEATURES = [
    "Body_CV", "Body_SK", "Body_RB", "Body_AP", "Body_APEn", "Body_DFA",
    "Body_MDR",
    "REM_PR", "REM_CV", "REM_SK", "REM_RB", "REM_AP", "REM_APEn",
    "REM_High_Activation_Ratio",
    "REM_High_Activation_Ratio_During_REM", "REM_NREM_Contrast",
    "Conf_Weighted_AP", "BB_Antagonistic_Ratio", "Path_Length_2D",
    "Transition_Rate_per_hour", "Directional_Entropy", "TOS", "TCI", "SRI",
]

COLORS = {"Before": "#4C78A8", "After": "#F28E2B"}


def main() -> None:
    pre = pd.read_csv(PRE_PATH, encoding="utf-8-sig")
    post = pd.read_csv(POST_PATH, encoding="utf-8-sig")
    paired = pre.merge(post, on="source_file", suffixes=("_pre", "_post"), how="inner")
    if len(paired) != 500:
        raise ValueError(f"Expected 500 exact-name pairs, found {len(paired)}")

    stats = pd.read_csv(STATS_PATH).set_index("feature")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(4, 6, figsize=(21, 14), constrained_layout=False)
    axes = axes.ravel()

    for ax, feature in zip(axes, FEATURES):
        before = pd.to_numeric(paired[f"{feature}_pre"], errors="coerce").dropna().to_numpy()
        after = pd.to_numeric(paired[f"{feature}_post"], errors="coerce").dropna().to_numpy()
        bp = ax.boxplot(
            [before, after],
            positions=[1, 2],
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#333333", "markersize": 4},
            medianprops={"color": "#1F1F1F", "linewidth": 1.6},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
        )
        for box, color in zip(bp["boxes"], [COLORS["Before"], COLORS["After"]]):
            box.set_facecolor(color)
            box.set_alpha(0.78)
            box.set_edgecolor("#333333")

        p_value = float(stats.at[feature, "wilcoxon_p"])
        effect = float(stats.at[feature, "rank_biserial"])
        ax.set_title(f"{feature}\np={p_value:.2e}, r={effect:+.2f}", pad=5)
        ax.set_xticks([1, 2], ["Before", "After"])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(FEATURES):]:
        ax.axis("off")

    legend = [
        Patch(facecolor=COLORS["Before"], edgecolor="#333333", label="Before treatment"),
        Patch(facecolor=COLORS["After"], edgecolor="#333333", label="After treatment"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Pre-post distributions of dual-axis sleep features (exact paired n=500)",
                 fontsize=17, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.012,
        "Boxes: IQR; center line: median; diamond: mean; whiskers: 1.5×IQR. "
        "p = raw two-sided paired Wilcoxon p-value; all displayed features remain significant after BH-FDR (q<0.05). "
        "r = paired rank-biserial effect size; outliers hidden for readability.",
        ha="center", va="bottom", fontsize=9.5,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.055, hspace=0.52, wspace=0.32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "pre_post_boxplots_without_REM_DFA_REM_MRA_with_p_values.png"
    pdf_path = OUT_DIR / "pre_post_boxplots_without_REM_DFA_REM_MRA_with_p_values.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
