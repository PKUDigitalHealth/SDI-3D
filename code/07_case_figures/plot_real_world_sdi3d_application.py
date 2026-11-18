"""Publication figure for a real-world ambulatory ECG SDI-3D deployment case."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection


DEFAULT_CASE_DIR = Path(
    r"C:\Users\SaberShuo\Desktop\ECG-3D\问心无恙40578926743916544"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\SaberShuo\Documents\SDI\output\figures\real_world_sdi3d"
)

INK = "#17212B"
MUTED = "#5B6872"
GRID = "#DCE4EA"
BLUE = "#2878B5"
ORANGE = "#D97732"
PURPLE = "#7B4FA3"
TEAL = "#2A9D8F"

REQUIRED_COLUMNS = {
    "original_epoch_index",
    "elapsed_seconds",
    "wake_probability",
    "rem_probability",
    "body_score_normalized",
    "rem_score_normalized",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#8C99A3",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if frame.empty:
        raise ValueError("Prediction file is empty")
    numeric = frame[list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Prediction file contains missing or non-finite values")
    if frame["original_epoch_index"].duplicated().any():
        raise ValueError("Prediction file contains duplicate epoch indices")
    if len(frame) > 1 and not np.allclose(np.diff(frame["elapsed_seconds"]), 30.0):
        raise ValueError("Predictions are not on a continuous 30-s grid")
    for column in (
        "wake_probability",
        "rem_probability",
        "body_score_normalized",
        "rem_score_normalized",
    ):
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} has values outside [0, 1]")
    frame = frame.sort_values("elapsed_seconds").reset_index(drop=True)
    frame["elapsed_hours"] = frame["elapsed_seconds"] / 3600.0
    return frame


def load_raw_case(case_dir: Path):
    bin_files = sorted(case_dir.glob("*.BIN3_16"))
    if len(bin_files) != 1:
        raise ValueError(f"Expected one BIN3_16 file in {case_dir}, found {len(bin_files)}")
    info = np.load(case_dir / "info.bin.npy", allow_pickle=False)
    r_peaks = np.load(case_dir / "r_peaks.bin.npy", allow_pickle=False).astype(np.int64)
    fs = float(np.ravel(info)[0])
    raw = np.memmap(bin_files[0], dtype=">i2", mode="r")
    if raw.size % 3:
        raise ValueError("Raw sample count is not divisible by three channels")
    signal = raw.reshape(-1, 3)
    valid_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(signal))]
    if len(valid_peaks) < 2:
        raise ValueError("Fewer than two valid R peaks")
    return bin_files[0], signal, fs, valid_peaks


def representative_excerpt(signal, fs: float, r_peaks: np.ndarray) -> tuple[int, int]:
    """Choose a reproducible 10-s excerpt near the median plausible heart rate."""
    rr = np.diff(r_peaks) / fs
    valid = (rr >= 0.3) & (rr <= 2.0)
    peak_times = r_peaks[1:] / fs
    duration = len(signal) / fs
    centers = np.arange(300.0, max(301.0, duration - 300.0), 30.0)
    medians = np.full(len(centers), np.nan)
    for i, center in enumerate(centers):
        mask = (peak_times >= center - 15.0) & (peak_times < center + 15.0) & valid
        if np.any(mask):
            medians[i] = np.median(60.0 / rr[mask])
    target = np.nanmedian(medians)
    candidates = np.flatnonzero(np.isfinite(medians))
    chosen = candidates[np.argmin(np.abs(medians[candidates] - target))]
    start = int(round((centers[chosen] - 5.0) * fs))
    stop = start + int(round(10.0 * fs))
    return start, stop


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
    )


def add_pipeline_box(ax, x: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, 0.19),
        0.255,
        0.62,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=color,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, 0.67, title, transform=ax.transAxes, weight="bold", color=INK)
    ax.text(
        x + 0.018,
        0.49,
        body,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7.1,
        linespacing=1.35,
        va="top",
    )


def draw_pipeline(ax, duration_h: float, n_outputs: int) -> None:
    ax.set_axis_off()
    panel_label(ax, "A")
    ax.set_title("End-to-end deployment on a portable ECG recording", loc="left", pad=7)
    add_pipeline_box(ax, 0.00, "Ambulatory ECG", f"3 channels, 200 Hz\n{duration_h:.2f} h recording", BLUE)
    add_pipeline_box(ax, 0.365, "Model-ready epochs", "100 Hz, 0.5-40 Hz\n30-s windows + 39 features", TEAL)
    add_pipeline_box(ax, 0.73, "SDI-3D outputs", f"{n_outputs:,} consecutive epochs\nBody and REM-associated axes", ORANGE)
    for left, right in ((0.257, 0.362), (0.622, 0.727)):
        ax.add_patch(
            FancyArrowPatch(
                (left, 0.50),
                (right, 0.50),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#83909A",
            )
        )


def draw_outputs(fig, cell, frame: pd.DataFrame) -> None:
    sub = cell.subgridspec(3, 1, hspace=0.08)
    axes = [fig.add_subplot(sub[i, 0]) for i in range(3)]
    time = frame["elapsed_hours"]
    window = 10  # 5 min at 30 s per epoch
    series = [
        ("wake_probability", "Wake probability", ORANGE),
        ("body_score_normalized", "Body-axis score", BLUE),
        ("rem_score_normalized", "REM-axis score", PURPLE),
    ]
    for i, (column, label, color) in enumerate(series):
        raw = frame[column]
        smooth = raw.rolling(window, center=True, min_periods=1).median()
        axes[i].plot(time, raw, color=color, alpha=0.22, linewidth=0.45)
        axes[i].plot(time, smooth, color=color, linewidth=1.15)
        axes[i].set_ylim(-0.03, 1.03)
        axes[i].set_yticks([0, 0.5, 1])
        axes[i].set_ylabel(label, color=color, fontsize=7.4)
        axes[i].grid(axis="y", color=GRID, linewidth=0.5)
        axes[i].spines[["top", "right"]].set_visible(False)
        axes[i].set_xlim(time.iloc[0], time.iloc[-1])
        if i < 2:
            axes[i].tick_params(axis="x", labelbottom=False)
        else:
            axes[i].set_xlabel("Elapsed recording time (h)")
    axes[0].set_title("Continuous 30-s model outputs across the night", loc="left", pad=9)
    axes[0].text(
        1.0,
        1.08,
        "thin: epoch output   thick: 5-min median",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=MUTED,
    )
    panel_label(axes[0], "B")


def draw_state_space_3d(ax, frame: pd.DataFrame) -> None:
    # Use every consecutive 30-s output so the 3-D panel represents the full
    # overnight trajectory rather than a temporally aggregated summary.
    body = frame["body_score_normalized"].to_numpy()
    rem = frame["rem_score_normalized"].to_numpy()
    time = frame["elapsed_hours"].to_numpy()
    points = np.column_stack([time, body, rem]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mpl.colors.Normalize(time.min(), time.max())
    collection = Line3DCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidth=0.52,
        alpha=0.68,
    )
    collection.set_array(time[:-1])
    ax.add_collection3d(collection)

    t0, t1 = float(time.min()), float(time.max())
    xx, zz = np.meshgrid([t0, t1], [0.0, 1.0])
    ax.plot_surface(xx, np.full_like(xx, 0.5), zz, color="#29C5C8", alpha=0.08, shade=False)
    xx, yy = np.meshgrid([t0, t1], [0.0, 1.0])
    ax.plot_surface(xx, yy, np.full_like(xx, 0.5), color="#C05BC7", alpha=0.08, shade=False)
    ax.plot([t0, t1], [0.5, 0.5], [0.5, 0.5], color="#4D5962", linewidth=0.7, linestyle="--")

    # Projection onto the starting plane mirrors the reference 3-D figure.
    scatter = ax.scatter(
        np.full_like(time, t0),
        body,
        rem,
        c=time,
        cmap="viridis",
        norm=norm,
        s=2.2,
        alpha=0.30,
        depthshade=False,
    )
    ax.scatter(time[0], body[0], rem[0], s=42, marker="o", facecolor="white", edgecolor=INK, linewidth=0.8)
    ax.scatter(time[-1], body[-1], rem[-1], s=42, marker="s", facecolor=INK, edgecolor="white", linewidth=0.5)

    ax.set_xlim(t0, t1)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("Time (h)", labelpad=3)
    ax.set_ylabel("Body axis", labelpad=3)
    ax.set_zlabel("REM axis", labelpad=3)
    ax.set_title("Full-night 30-s SDI-3D trajectory", loc="left", pad=8)
    ax.view_init(elev=20, azim=-52)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    try:
        ax.set_box_aspect((1.8, 1.0, 1.0), zoom=1.05)
    except TypeError:
        ax.set_box_aspect((1.8, 1.0, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.45, 0.52, 0.57, 0.18)
        axis._axinfo["grid"]["linestyle"] = "--"
        axis.set_pane_color((0.96, 0.97, 0.98, 0.26))
    ax.tick_params(labelsize=6.4, pad=0)
    ax.text2D(-0.10, 1.08, "C", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="top")
    colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02, shrink=0.72)
    colorbar.set_label("Elapsed time (h)", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.8)
    ax.text2D(0.02, 0.08, "start: circle   end: square", transform=ax.transAxes, fontsize=6.3, color=MUTED)


def draw_ecg(ax, signal, fs: float, r_peaks: np.ndarray, start: int, stop: int) -> None:
    time = np.arange(stop - start) / fs
    offsets = np.array([2.2, 1.1, 0.0])
    colors = [BLUE, TEAL, ORANGE]
    for channel in range(3):
        values = np.asarray(signal[start:stop, channel], dtype=float)
        values -= np.median(values)
        scale = np.percentile(np.abs(values), 99.5)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        values = np.clip(values / scale, -1.0, 1.0) * 0.42 + offsets[channel]
        ax.plot(time, values, color=colors[channel], linewidth=0.62)
        ax.text(0.05, offsets[channel] + 0.34, f"Ch {channel + 1}", color=colors[channel], fontsize=7.0, weight="bold")
    local_peaks = r_peaks[(r_peaks >= start) & (r_peaks < stop)]
    for peak in local_peaks:
        ax.axvline((peak - start) / fs, color="#7C8790", linewidth=0.35, alpha=0.30)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.58, 2.78)
    ax.set_yticks([])
    ax.set_xlabel("Time within excerpt (s)")
    ax.set_title("Representative 10-s ECG excerpt", loc="left", pad=9)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.spines[["top", "right", "left"]].set_visible(False)
    panel_label(ax, "D")
    ax.text(
        1.0,
        0.98,
        "gray: device-provided R peaks",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=MUTED,
    )


def make_figure(case_dir: Path, predictions_csv: Path, output_dir: Path) -> dict[str, object]:
    configure_style()
    frame = load_predictions(predictions_csv)
    raw_path, signal, fs, r_peaks = load_raw_case(case_dir)
    duration_h = len(signal) / fs / 3600.0
    start, stop = representative_excerpt(signal, fs, r_peaks)

    figure = plt.figure(figsize=(7.45, 9.2), constrained_layout=False)
    outer = figure.add_gridspec(
        4,
        2,
        height_ratios=[0.72, 1.32, 3.15, 2.45],
        width_ratios=[1.12, 0.88],
        left=0.095,
        right=0.975,
        top=0.975,
        bottom=0.065,
        hspace=0.46,
        wspace=0.42,
    )
    header = figure.add_subplot(outer[0, :])
    header.set_axis_off()
    header.text(
        0,
        0.90,
        "Real-world ambulatory ECG deployment of SDI-3D",
        fontsize=12.2,
        weight="bold",
        color=INK,
        va="top",
    )
    header.text(
        0,
        0.18,
        "Single overnight recording acquired with a portable three-channel device",
        fontsize=7.8,
        color=MUTED,
        va="bottom",
    )

    pipeline = figure.add_subplot(outer[1, :])
    draw_pipeline(pipeline, duration_h, len(frame))
    draw_outputs(figure, outer[2, :], frame)
    state = figure.add_subplot(outer[3, 0], projection="3d")
    draw_state_space_3d(state, frame)
    ecg = figure.add_subplot(outer[3, 1])
    draw_ecg(ecg, signal, fs, r_peaks, start, stop)

    figure.text(
        0.095,
        0.020,
        (
            f"Complete outputs: {len(frame):,}/{len(frame):,}; model interval "
            f"{frame['elapsed_hours'].iloc[0]:.2f}-{frame['elapsed_hours'].iloc[-1]:.2f} h. "
            "Single unlabeled case: this demonstrates end-to-end feasibility, not diagnostic accuracy."
        ),
        fontsize=6.6,
        color=MUTED,
        ha="left",
        va="bottom",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "real_world_sdi3d_application.pdf"
    png_path = output_dir / "real_world_sdi3d_application.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)

    return {
        "raw_file": str(raw_path),
        "prediction_file": str(predictions_csv),
        "rows": int(len(frame)),
        "duration_hours": duration_h,
        "model_start_hour": float(frame["elapsed_hours"].iloc[0]),
        "model_end_hour": float(frame["elapsed_hours"].iloc[-1]),
        "wake_probability_over_0.5": float((frame["wake_probability"] > 0.5).mean()),
        "rem_probability_over_0.5": float((frame["rem_probability"] > 0.5).mean()),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    predictions = args.predictions or args.case_dir / "sdi3d_predictions.csv"
    summary = make_figure(args.case_dir, predictions, args.output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
