"""Overlay consistent Body/Brain and Fragmented labels on manuscript diagrams."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
PICTURE_DIR = ROOT / "output" / "tex" / "picture"
OUTPUT_DIR = ROOT / "output" / "pdf"


def overlay_page(source: Path, target: Path, painter) -> None:
    reader = PdfReader(source)
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    buffer = BytesIO()
    fig = plt.figure(figsize=(width / 72.0, height / 72.0), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    painter(ax)
    fig.savefig(buffer, format="pdf", transparent=True, dpi=72)
    plt.close(fig)
    buffer.seek(0)
    page.merge_page(PdfReader(buffer).pages[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(page)
    with target.open("wb") as handle:
        writer.write(handle)


def paint_overview(ax) -> None:
    # Output labels at the right of the two coordinate traces.
    ax.add_patch(Rectangle((474, 282), 75, 78, facecolor="white", edgecolor="none"))
    ax.text(
        510,
        341,
        "Body sleep-depth\nscore",
        ha="center",
        va="center",
        family="Arial",
        weight="bold",
        fontsize=8.2,
        linespacing=1.05,
        color="#111111",
    )
    ax.text(
        510,
        300,
        "Brain sleep-depth\nscore",
        ha="center",
        va="center",
        family="Arial",
        weight="bold",
        fontsize=8.2,
        linespacing=1.05,
        color="#111111",
    )

    # Phenotype labels in the clustering schematic and survival legend.
    ax.add_patch(Rectangle((332, 137), 48, 13, facecolor="white", edgecolor="none"))
    ax.text(333, 141, "Fragmented", ha="left", va="bottom", family="Arial", fontsize=5.0, color="#f05a28")
    ax.add_patch(Rectangle((353, 97), 48, 12, facecolor="white", edgecolor="none"))
    ax.text(356, 100, "Fragmented", ha="left", va="bottom", family="Arial", fontsize=5.8, color="#f05a28")


def paint_architecture(ax) -> None:
    # Preserve the purple head box and replace only its legacy label.
    ax.add_patch(Rectangle((817, 41), 171, 55, facecolor="#ebe7f7", edgecolor="none"))
    ax.text(
        902.5,
        68,
        "Brain sleep-depth\nHead",
        ha="center",
        va="center",
        family="Arial",
        weight="bold",
        fontsize=15.5,
        linespacing=1.05,
        color="#111111",
    )


def main() -> None:
    overview = OUTPUT_DIR / "model_brain_depth.pdf"
    architecture = OUTPUT_DIR / "new_model_brain_depth.pdf"
    overlay_page(PICTURE_DIR / "model.pdf", overview, paint_overview)
    overlay_page(PICTURE_DIR / "new_model.pdf", architecture, paint_architecture)
    print(overview)
    print(architecture)


if __name__ == "__main__":
    main()
