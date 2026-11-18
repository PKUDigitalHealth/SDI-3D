"""Update terminology in vector PDF figures without changing plotted data.

The source assets are retained as vector pages. Small white overlays replace
only the legacy text, after which the revised labels are drawn with embedded
Arial fonts. Histogram bins, curves, estimates, confidence intervals, and page
dimensions remain unchanged.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pypdf import PdfReader, PdfWriter


def make_overlay(width: float, height: float, painter) -> object:
    buffer = BytesIO()
    figure = plt.figure(figsize=(width / 72, height / 72), dpi=72)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.axis("off")
    painter(axis)
    figure.savefig(buffer, format="pdf", transparent=True, dpi=72)
    plt.close(figure)
    buffer.seek(0)
    return PdfReader(buffer).pages[0]


def merge_overlay(source: Path, target: Path, painter, legacy_literals: tuple[bytes, ...]) -> None:
    reader = PdfReader(source)
    page = reader.pages[0]
    form = page["/Resources"]["/XObject"]["/RecoveredFigure"].get_object()
    data = form.get_data()
    for literal in legacy_literals:
        token = b"(" + literal + b")"
        if token not in data:
            raise ValueError(f"Expected legacy label not found: {literal!r}")
        data = data.replace(token, b"()")
    form.set_data(data)
    overlay = make_overlay(float(page.mediabox.width), float(page.mediabox.height), painter)
    page.merge_page(overlay)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(page)
    with target.open("wb") as handle:
        writer.write(handle)


def draw_centered_vertical(
    axis,
    text: str,
    *,
    x: float,
    center_y: float,
    size: float,
) -> None:
    axis.text(
        x,
        center_y,
        text,
        rotation=90,
        ha="center",
        va="center",
        family="Arial",
        weight="bold",
        fontsize=size,
        color="#222222",
    )


def paint_stage(axis) -> None:
    # The two legacy y-axis labels occupy a white margin outside the axes.
    axis.add_patch(Rectangle((0, 0), 67, 883.125, facecolor="white", edgecolor="none"))

    upper_center = (514.546189 + 830.64375) / 2
    lower_center = (53.04375 + 369.141311) / 2
    draw_centered_vertical(
        axis,
        "Body sleep-depth score",
        x=27.1,
        center_y=upper_center,
        size=24,
    )
    draw_centered_vertical(
        axis,
        "(epoch count)",
        x=54.1,
        center_y=upper_center,
        size=24,
    )
    draw_centered_vertical(
        axis,
        "Brain sleep-depth score",
        x=22.4,
        center_y=lower_center,
        size=20,
    )
    draw_centered_vertical(
        axis,
        "(epoch count)",
        x=44.8,
        center_y=lower_center,
        size=20,
    )


def paint_survival(axis) -> None:
    # Replace the second legend entry in each of the three panels.
    for x in (373.9, 951.7, 1529.5):
        axis.add_patch(Rectangle((x - 3, 505), 151, 25, facecolor="white", edgecolor="none"))
        axis.text(
            x,
            509.5,
            "Fragmented Sleep",
            ha="left",
            va="bottom",
            family="Arial",
            fontsize=17,
            color="#222222",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-source", type=Path, required=True)
    parser.add_argument("--stage-target", type=Path, required=True)
    parser.add_argument("--survival-source", type=Path, required=True)
    parser.add_argument("--survival-target", type=Path, required=True)
    args = parser.parse_args()

    merge_overlay(
        args.stage_source,
        args.stage_target,
        paint_stage,
        (b"Body Depth Score", b"REM-associated activation score", b"\\(epoch count\\)"),
    )
    merge_overlay(
        args.survival_source,
        args.survival_target,
        paint_survival,
        (b"Disturbed Sleep",),
    )
    print(args.stage_target)
    print(args.survival_target)


if __name__ == "__main__":
    main()
