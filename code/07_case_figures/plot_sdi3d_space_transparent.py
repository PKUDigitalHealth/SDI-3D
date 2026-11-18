"""Create a plot-only transparent SDI-3D trajectory as SVG and PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


DEFAULT_INPUT = Path(r"C:\Users\SaberShuo\Desktop\sdi3d_predictions.csv")
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\SaberShuo\Documents\SDI\output\figures\desktop_sdi3d"
)

QUADRANTS = {
    "Q1": ("#D62728", "Q1 (Body+, Brain+)"),
    "Q2": ("#169C9C", "Q2 (Body−, Brain+)"),
    "Q3": ("#55A868", "Q3 (Body−, Brain−)"),
    "Q4": ("#D99A00", "Q4 (Body+, Brain−)"),
}


def quadrant(body: float, brain: float) -> str:
    if body >= 0.5 and brain >= 0.5:
        return "Q1"
    if body < 0.5 and brain >= 0.5:
        return "Q2"
    if body < 0.5 and brain < 0.5:
        return "Q3"
    return "Q4"


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {
        "elapsed_seconds",
        "body_score_normalized",
        "rem_score_normalized",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    values = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Input contains missing or non-finite values")
    frame = frame.sort_values("elapsed_seconds").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("At least two rows are required")
    if not np.allclose(np.diff(frame["elapsed_seconds"]), 30.0):
        raise ValueError("Expected a continuous 30-s time grid")
    body = frame["body_score_normalized"].to_numpy(float)
    brain = frame["rem_score_normalized"].to_numpy(float)
    if np.any((body < 0) | (body > 1) | (brain < 0) | (brain > 1)):
        raise ValueError("Normalized Body/Brain scores must lie in [0,1]")
    time_h = (
        frame["elapsed_seconds"].to_numpy(float)
        - float(frame["elapsed_seconds"].iloc[0])
    ) / 3600.0
    return time_h, body, brain


def make_plot(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    time_h, body, brain = load_data(csv_path)
    q = np.array([quadrant(b, r) for b, r in zip(body, brain)])
    colors = [QUADRANTS[name][0] for name in q[:-1]]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(12.0, 5.8), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("none")
    ax.view_init(elev=20, azim=-52)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    try:
        ax.set_box_aspect((3.4, 1.0, 1.0), zoom=1.08)
    except TypeError:
        ax.set_box_aspect((3.4, 1.0, 1.0))

    points = np.column_stack([time_h, body, brain]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = Line3DCollection(segments, colors=colors, linewidths=1.05, alpha=0.88)
    ax.add_collection3d(lines)

    # Projection on the start plane and two 0.5 decision planes.
    projection_colors = [QUADRANTS[name][0] for name in q]
    ax.scatter(
        np.zeros_like(time_h),
        body,
        brain,
        c=projection_colors,
        s=7,
        alpha=0.48,
        depthshade=False,
        linewidths=0,
    )
    tmax = float(time_h[-1])
    xx, zz = np.meshgrid([0.0, tmax], [0.0, 1.0])
    ax.plot_surface(
        xx, np.full_like(xx, 0.5), zz,
        color="#27C6C8", alpha=0.09, shade=False, linewidth=0,
    )
    xx, yy = np.meshgrid([0.0, tmax], [0.0, 1.0])
    ax.plot_surface(
        xx, yy, np.full_like(xx, 0.5),
        color="#C35DC8", alpha=0.09, shade=False, linewidth=0,
    )
    ax.plot(
        [0.0, tmax], [0.5, 0.5], [0.5, 0.5],
        color="#4E5962", linewidth=1.2, linestyle="-.", alpha=0.8,
    )

    ax.scatter(
        time_h[0], body[0], brain[0], marker="*", s=120,
        color="#2E9D50", edgecolor="#111111", linewidth=0.8, depthshade=False,
    )
    ax.scatter(
        time_h[-1], body[-1], brain[-1], marker="X", s=105,
        color="#E41A1C", edgecolor="#111111", linewidth=0.8, depthshade=False,
    )

    ax.set_xlim(0.0, tmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("Elapsed time (h)", labelpad=8)
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title("Full-night SDI-3D trajectory", pad=18)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_zticks(np.linspace(0, 1, 6))
    ax.tick_params(colors="#27323A", labelsize=9, pad=1)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.35, 0.42, 0.47, 0.22)
        axis._axinfo["grid"]["linestyle"] = "--"
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((0.42, 0.48, 0.52, 0.38))

    handles = [
        Line2D([0], [0], color=color, lw=2.4, label=label)
        for color, label in QUADRANTS.values()
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#2E9D50",
                   markeredgecolor="#111111", markersize=10, label="Start"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#E41A1C",
                   markeredgecolor="#111111", markersize=9, label="End"),
        ]
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=6,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
        fontsize=8.5,
    )

    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.11, top=0.88)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "sdi3d_trajectory_space_transparent.svg"
    png_path = output_dir / "sdi3d_trajectory_space_transparent.png"
    save_kwargs = {
        "transparent": True,
        "facecolor": "none",
        "edgecolor": "none",
        "bbox_inches": "tight",
        "pad_inches": 0.05,
    }
    fig.savefig(svg_path, format="svg", **save_kwargs)
    fig.savefig(png_path, format="png", dpi=400, **save_kwargs)
    plt.close(fig)
    return svg_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    svg_path, png_path = make_plot(args.input, args.output_dir)
    print(f"SVG: {svg_path}")
    print(f"PNG preview: {png_path}")


if __name__ == "__main__":
    main()
