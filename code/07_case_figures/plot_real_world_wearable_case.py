"""Create a publication figure for one real-world ambulatory ECG recording.

The source recording is a 3-channel, 200-Hz, big-endian int16 stream. The
companion NumPy files provide the sampling rate, R-peak sample indices, and a
device-derived SQA quantity. This figure deliberately stops at technical input
feasibility: it does not fabricate SDI-3D outputs when frozen model weights are
not available locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_INPUT = Path(
    r"C:\Users\SaberShuo\Desktop\ECG-3D\问心无恙40578926743916544"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\SaberShuo\Documents\SDI\output\pdf\real_world_wearable_ecg_feasibility.pdf"
)

BLUE = "#1769AA"
CYAN = "#2A9D8F"
ORANGE = "#E07A3F"
INK = "#17212B"
MUTED = "#536471"
LIGHT = "#E9F0F5"
GREEN = "#2E8B57"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#8A99A6",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_recording(input_dir: Path):
    signal_path = next(input_dir.glob("*.BIN3_16"), None)
    if signal_path is None:
        raise FileNotFoundError(f"No *.BIN3_16 file found in {input_dir}")

    info = np.load(input_dir / "info.bin.npy")
    r_peaks = np.load(input_dir / "r_peaks.bin.npy").astype(np.int64)
    sqa = np.load(input_dir / "sqa.bin.npy").astype(float)
    fs = float(info[0])
    if fs <= 0:
        raise ValueError(f"Invalid sampling rate: {fs}")

    raw = np.memmap(signal_path, dtype=">i2", mode="r")
    if raw.size % 3:
        raise ValueError("BIN3_16 sample count is not divisible by 3 channels")
    signal = raw.reshape(-1, 3)
    if len(r_peaks) != len(sqa):
        raise ValueError("R-peak and SQA arrays have different lengths")
    if np.any(np.diff(r_peaks) <= 0) or r_peaks[-1] >= len(signal):
        raise ValueError("R-peak indices are not strictly increasing and in bounds")
    return signal_path, signal, fs, r_peaks, sqa


def derive_metrics(signal, fs: float, r_peaks: np.ndarray):
    duration_s = len(signal) / fs
    epoch_samples = int(round(30 * fs))
    n_epochs = len(signal) // epoch_samples
    remainder_s = (len(signal) % epoch_samples) / fs

    rr_s = np.diff(r_peaks) / fs
    hr_bpm = 60.0 / rr_s
    plausible = (rr_s >= 0.3) & (rr_s <= 2.0)

    bins = np.arange(n_epochs + 1, dtype=np.int64) * epoch_samples
    beats_per_epoch = np.histogram(r_peaks, bins=bins)[0]

    # Five-minute summaries reduce visual overplotting while retaining the
    # overnight trend. Only physiologically plausible RR intervals are used.
    five_min_s = 300.0
    n_five = int(duration_s // five_min_s)
    t5, med5, q25_5, q75_5 = [], [], [], []
    second_peak = r_peaks[1:]
    for i in range(n_five):
        lo = int(round(i * five_min_s * fs))
        hi = int(round((i + 1) * five_min_s * fs))
        mask = (second_peak >= lo) & (second_peak < hi) & plausible
        values = hr_bpm[mask]
        if values.size:
            t5.append((i + 0.5) * five_min_s / 3600.0)
            med5.append(np.median(values))
            q25_5.append(np.percentile(values, 25))
            q75_5.append(np.percentile(values, 75))

    # Select a reproducible representative epoch: closest to the record-wide
    # median 30-s HR; ties are resolved by proximity to the record midpoint.
    epoch_hr = np.full(n_epochs, np.nan)
    for i in range(n_epochs):
        lo, hi = bins[i], bins[i + 1]
        mask = (second_peak >= lo) & (second_peak < hi) & plausible
        if np.any(mask):
            epoch_hr[i] = np.median(hr_bpm[mask])
    target = np.nanmedian(epoch_hr)
    distance = np.abs(epoch_hr - target)
    candidates = np.flatnonzero(distance == np.nanmin(distance))
    midpoint = (n_epochs - 1) / 2.0
    representative_epoch = int(candidates[np.argmin(np.abs(candidates - midpoint))])

    saturated = int(np.sum((signal == -32768) | (signal == 32767)))
    return {
        "duration_s": duration_s,
        "n_epochs": n_epochs,
        "remainder_s": remainder_s,
        "rr_s": rr_s,
        "hr_bpm": hr_bpm,
        "plausible": plausible,
        "beats_per_epoch": beats_per_epoch,
        "t5": np.asarray(t5),
        "med5": np.asarray(med5),
        "q25_5": np.asarray(q25_5),
        "q75_5": np.asarray(q75_5),
        "representative_epoch": representative_epoch,
        "saturated": saturated,
    }


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.035,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )


def add_box(ax, xy, wh, title, body, edge, fill="#FFFFFF", dashed=False):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
        linestyle="--" if dashed else "-",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.035,
        y + h - 0.07,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + 0.035,
        y + h - 0.22,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=MUTED,
        linespacing=1.35,
    )


def draw_pipeline(ax, duration_h: float, n_peaks: int, n_epochs: int) -> None:
    ax.set_axis_off()
    panel_label(ax, "A")
    ax.set_title("Real-world ECG-to-SDI-3D deployment path", loc="left", pad=9)
    add_box(
        ax,
        (0.00, 0.26),
        (0.285, 0.55),
        "Ambulatory acquisition",
        f"3 channels  |  200 Hz\n{duration_h:.2f} h continuous recording",
        BLUE,
        fill="#F3F8FC",
    )
    add_box(
        ax,
        (0.355, 0.26),
        (0.285, 0.55),
        "Technical preprocessing",
        f"{n_peaks:,} detected R peaks\n{n_epochs:,} complete 30-s windows",
        CYAN,
        fill="#F1FAF8",
    )
    add_box(
        ax,
        (0.71, 0.26),
        (0.285, 0.55),
        "Frozen SDI-3D inference",
        "Body + REM-associated scores\nOvernight trajectory + biomarkers",
        ORANGE,
        fill="#FFF7F2",
        dashed=True,
    )
    for x0, x1 in ((0.287, 0.352), (0.642, 0.707)):
        ax.add_patch(
            FancyArrowPatch(
                (x0, 0.535),
                (x1, 0.535),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#7C8B96",
            )
        )
    ax.text(
        0.995,
        0.10,
        "Dashed: intended deployment step; inference was not run for this case.",
        transform=ax.transAxes,
        fontsize=6.9,
        color=ORANGE,
        ha="right",
        va="bottom",
    )


def draw_integrity(ax, metrics, n_peaks: int) -> None:
    ax.set_axis_off()
    panel_label(ax, "B")
    ax.set_title("Technical input integrity", loc="left", pad=9)

    valid_n = int(np.sum(metrics["plausible"]))
    valid_pct = 100.0 * np.mean(metrics["plausible"])
    all_epoch_beats = int(np.sum(metrics["beats_per_epoch"] > 0))
    cards = [
        (f"{metrics['duration_s'] / 3600:.2f} h", "recording duration"),
        (f"{n_peaks:,}", "provided R peaks"),
        (f"{all_epoch_beats:,}/{metrics['n_epochs']:,}", "30-s windows with R peaks"),
        (f"{valid_pct:.3f}%", f"RR in 0.3-2.0 s ({valid_n:,})"),
    ]
    xs = [0.00, 0.255, 0.51, 0.765]
    for (value, label), x in zip(cards, xs):
        patch = FancyBboxPatch(
            (x, 0.31),
            0.225,
            0.46,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0.8,
            edgecolor="#CAD6DF",
            facecolor="#F8FAFC",
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.1125,
            0.60,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=BLUE,
        )
        ax.text(
            x + 0.1125,
            0.40,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.7,
            color=MUTED,
        )
    ax.text(
        0.00,
        0.12,
        f"No int16 saturation detected across 3 channels; {metrics['remainder_s']:.1f} s remainder after complete windows.",
        transform=ax.transAxes,
        fontsize=7.1,
        color=MUTED,
        ha="left",
        va="bottom",
    )


def draw_overnight_hr(ax, metrics) -> None:
    panel_label(ax, "C")
    ax.set_title("Overnight heart-rate dynamics from provided R peaks", loc="left", pad=10)
    t = metrics["t5"]
    band = ax.fill_between(
        t,
        metrics["q25_5"],
        metrics["q75_5"],
        color=BLUE,
        alpha=0.18,
        linewidth=0,
        label="interquartile range",
    )
    line, = ax.plot(t, metrics["med5"], color=BLUE, linewidth=1.3, label="5-min median")
    selected_h = metrics["representative_epoch"] / 120.0
    ax.axvline(selected_h, color=ORANGE, linewidth=1.0, linestyle="--", zorder=4)
    ymax = max(105.0, float(np.nanmax(metrics["q75_5"]) + 5))
    ax.text(
        selected_h + 0.08,
        ymax - 3,
        "waveform excerpt",
        fontsize=7.1,
        color=ORANGE,
        va="top",
    )
    ax.set_xlim(0, metrics["duration_s"] / 3600.0)
    ax.set_ylim(50, ymax)
    ax.set_xlabel("Elapsed recording time (h)")
    ax.set_ylabel("Heart rate (beats/min)")
    ax.grid(axis="y", color="#DCE5EB", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[line, band],
        labels=["5-min median", "interquartile range"],
        loc="upper right",
        frameon=False,
        fontsize=7.2,
    )
    median = np.median(metrics["hr_bpm"][metrics["plausible"]])
    q1, q3 = np.percentile(metrics["hr_bpm"][metrics["plausible"]], [25, 75])
    ax.text(
        0.01,
        0.06,
        f"Beat-level median {median:.1f} beats/min (IQR {q1:.1f}-{q3:.1f})",
        transform=ax.transAxes,
        fontsize=7.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )


def draw_waveforms(fig, gs_cell, signal, fs: float, r_peaks: np.ndarray, representative_epoch: int) -> None:
    sub = gs_cell.subgridspec(3, 1, hspace=0.08)
    axes = [fig.add_subplot(sub[i, 0]) for i in range(3)]
    epoch_start = representative_epoch * int(round(30 * fs))
    start = epoch_start + int(round(10 * fs))
    stop = start + int(round(10 * fs))
    time = np.arange(stop - start) / fs
    local_peaks = r_peaks[(r_peaks >= start) & (r_peaks < stop)]
    local_t = (local_peaks - start) / fs
    colors = [BLUE, CYAN, ORANGE]

    for i, ax in enumerate(axes):
        y = np.asarray(signal[start:stop, i], dtype=float)
        center = np.median(y)
        scale = np.percentile(np.abs(y - center), 99.7)
        if scale == 0:
            scale = 1.0
        y = (y - center) / scale
        ax.plot(time, y, color=colors[i], linewidth=0.65)
        for tp in local_t:
            ax.axvline(tp, color="#7D8790", linewidth=0.35, alpha=0.35)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.15, 1.15)
        ax.set_yticks([])
        ax.text(
            0.006,
            0.78,
            f"Channel {i + 1}",
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color=colors[i],
            ha="left",
            va="top",
        )
        ax.grid(axis="x", color="#E5EBEF", linewidth=0.45)
        ax.spines[["top", "right", "left"]].set_visible(False)
        if i < 2:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel("Time within excerpt (s)")
    axes[0].text(
        -0.035,
        1.32,
        "D",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    axes[0].set_title(
        "Representative 10-s three-channel ECG excerpt",
        loc="left",
        pad=12,
    )
    axes[0].text(
        1.0,
        1.19,
        "Gray lines: provided R-peak positions",
        transform=axes[0].transAxes,
        fontsize=6.8,
        color=MUTED,
        ha="right",
        va="bottom",
    )
def make_figure(input_dir: Path, output_pdf: Path, preview_png: Path | None = None) -> dict:
    configure_style()
    signal_path, signal, fs, r_peaks, _sqa = load_recording(input_dir)
    metrics = derive_metrics(signal, fs, r_peaks)

    fig = plt.figure(figsize=(7.35, 9.55), constrained_layout=False)
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.50, 1.38, 1.08, 2.15, 2.65],
        left=0.085,
        right=0.975,
        bottom=0.070,
        top=0.985,
        hspace=0.48,
    )
    header = fig.add_subplot(outer[0, 0])
    header.set_axis_off()
    header.text(
        0.0,
        0.92,
        "Real-world ambulatory ECG: technical feasibility for SDI-3D input",
        fontsize=11.8,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    header.text(
        0.0,
        0.14,
        "Single overnight case from a portable three-channel ECG recorder",
        fontsize=7.8,
        color=MUTED,
        ha="left",
        va="bottom",
    )

    ax_a = fig.add_subplot(outer[1, 0])
    ax_b = fig.add_subplot(outer[2, 0])
    draw_pipeline(
        ax_a,
        metrics["duration_s"] / 3600.0,
        len(r_peaks),
        metrics["n_epochs"],
    )
    draw_integrity(ax_b, metrics, len(r_peaks))

    ax_c = fig.add_subplot(outer[3, 0])
    draw_overnight_hr(ax_c, metrics)
    draw_waveforms(
        fig,
        outer[4, 0],
        signal,
        fs,
        r_peaks,
        metrics["representative_epoch"],
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.text(
        0.975,
        0.018,
        "Channels normalized separately for display; physical calibration was not provided.",
        fontsize=6.5,
        color=MUTED,
        ha="right",
        va="bottom",
    )
    fig.savefig(output_pdf, format="pdf")
    if preview_png is not None:
        preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_png, format="png", dpi=240)
    plt.close(fig)

    return {
        "source": str(signal_path),
        "output": str(output_pdf),
        "duration_h": metrics["duration_s"] / 3600.0,
        "sampling_rate_hz": fs,
        "channels": 3,
        "r_peaks": int(len(r_peaks)),
        "complete_30s_epochs": int(metrics["n_epochs"]),
        "epochs_with_r_peaks": int(np.sum(metrics["beats_per_epoch"] > 0)),
        "plausible_rr_fraction": float(np.mean(metrics["plausible"])),
        "saturated_samples": int(metrics["saturated"]),
        "representative_epoch": int(metrics["representative_epoch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    summary = make_figure(args.input_dir, args.output, args.preview)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
