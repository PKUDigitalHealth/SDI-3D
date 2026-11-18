from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "case_study"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "case_figures_body_brain"
DEFAULT_IDS = ("200419", "201575", "201674", "205218")


def validate_and_load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).sort_values("window_center").reset_index(drop=True)
    required = {"window_center", "wake_depth_score", "rem_prob_rem", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=sorted(required)).copy()
    if len(df) < 2:
        raise ValueError(f"{csv_path.name} has fewer than two complete epochs")
    return df


def plot_two_panel(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    df = validate_and_load(csv_path)
    normalized = MinMaxScaler().fit_transform(
        df[["wake_depth_score", "rem_prob_rem"]]
    )
    body = normalized[:, 0]
    brain = normalized[:, 1]
    time_hours = (df["window_center"].to_numpy(dtype=float) * 30.0) / 3600.0
    time_hours = time_hours - time_hours.min()

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.85], "hspace": 0.28},
    )

    body_color = "#F27873"
    brain_color = "#59B98C"
    stage_color = "#4A5568"

    ax1.fill_between(time_hours, body, 0, color=body_color, alpha=0.08, linewidth=0)
    ax1.fill_between(time_hours, brain, 0, color=brain_color, alpha=0.08, linewidth=0)
    ax1.plot(time_hours, body, color=body_color, linewidth=0.75, label="Body sleep depth")
    ax1.plot(time_hours, brain, color=brain_color, linewidth=0.75, label="Brain sleep depth")
    ax1.set_title("Body and Brain Sleep-Depth Coordinates", pad=30)
    ax1.set_ylabel("Normalized score (0–1)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.grid(True, linestyle="-", linewidth=0.5, alpha=0.22)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
        borderaxespad=0,
    )

    stage_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
    labels = df["label"].astype(int).to_numpy()
    unknown = sorted(set(labels).difference(stage_map))
    if unknown:
        raise ValueError(f"{csv_path.name} contains unknown sleep-stage labels: {unknown}")
    stages = np.asarray([stage_map[value] for value in labels])
    ax2.step(time_hours, stages, where="post", linewidth=1.0, color=stage_color)
    ax2.set_title("Sleep Stage", pad=7)
    ax2.set_yticks([0, 1, 2, 3], ["Wake", "N1+N2", "N3", "REM"])
    ax2.set_ylim(-0.2, 3.2)
    ax2.set_xlabel("Time (hours)")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.24)

    for axis in (ax1, ax2):
        axis.margins(x=0)
        for spine in axis.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.9)

    fig.subplots_adjust(left=0.12, right=0.985, top=0.86, bottom=0.12)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{csv_path.stem}_2panel_body_brain"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def plot_three_dimensional(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    env = os.environ.copy()
    env["SHOW_3D_CSV_PATH"] = str(csv_path.resolve())
    env["SHOW_3D_OUTPUT_DIR"] = str(output_dir.resolve())
    env["SHOW_3D_NO_SHOW"] = "1"
    subprocess.run(
        [sys.executable, str(ROOT / "show_3D_white.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    stem = output_dir / f"{csv_path.stem}_3D_white"
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D and two-panel Body–Brain figures for four SHHS cases."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for subject_id in args.ids:
        csv_path = args.data_dir / f"shhs1-{subject_id}_features_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = validate_and_load(csv_path)
        two_png, two_pdf = plot_two_panel(csv_path, args.output_dir)
        three_png, three_pdf = plot_three_dimensional(csv_path, args.output_dir)
        duration_hours = (
            (df["window_center"].max() - df["window_center"].min()) * 30.0 / 3600.0
        )
        rows.append(
            {
                "subject_id": subject_id,
                "epochs": len(df),
                "duration_hours": round(float(duration_hours), 4),
                "two_panel_png": two_png.name,
                "two_panel_pdf": two_pdf.name,
                "three_dimensional_png": three_png.name,
                "three_dimensional_pdf": three_pdf.name,
            }
        )

    manifest = args.output_dir / "case_figure_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(manifest)


if __name__ == "__main__":
    main()
