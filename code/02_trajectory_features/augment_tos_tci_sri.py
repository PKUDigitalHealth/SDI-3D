"""Add TOS, TCI, and SRI to an existing participant-night feature table.

This avoids recomputing the slower one-dimensional SDI/REM features when a
legacy aggregate CSV already contains them. Raw files are matched by the
``source_file`` column and are never modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_body_rem_dual_features import (  # noqa: E402
    minmax,
    state_recurrence_index,
    trajectory_curvature_index,
    trajectory_occupancy_surface,
)


def calculate(raw_path: Path, radius: float, theiler_window: int) -> dict[str, float]:
    frame = pd.read_csv(raw_path)
    body = pd.to_numeric(frame["wake_depth_score"], errors="coerce").to_numpy(float)
    rem = pd.to_numeric(frame["rem_logit_rem"], errors="coerce").to_numpy(float)
    finite = np.isfinite(body) & np.isfinite(rem)
    body, rem = minmax(body[finite]), minmax(rem[finite])
    return {
        "TOS": trajectory_occupancy_surface(body, rem),
        "TCI": trajectory_curvature_index(body, rem),
        "SRI": state_recurrence_index(body, rem, radius, theiler_window),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Append TOS, TCI, and SRI")
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--sri-radius", type=float, default=0.05)
    parser.add_argument("--sri-theiler-window", type=int, default=20)
    arguments = parser.parse_args()

    table = pd.read_csv(arguments.feature_csv, encoding="utf-8-sig")
    errors = []
    rows = []
    for source_file in table["source_file"].astype(str):
        raw_path = arguments.raw_dir / source_file
        if not raw_path.exists() and raw_path.name.endswith("_predictions.csv"):
            raw_path = raw_path.with_name(
                raw_path.name.removesuffix("_predictions.csv") + ".csv"
            )
        try:
            rows.append(
                calculate(
                    raw_path,
                    arguments.sri_radius,
                    arguments.sri_theiler_window,
                )
            )
        except Exception as error:
            rows.append({"TOS": np.nan, "TCI": np.nan, "SRI": np.nan})
            errors.append(
                {"source_file": source_file, "error": f"{type(error).__name__}: {error}"}
            )

    additions = pd.DataFrame(rows, index=table.index)
    for feature in additions.columns:
        table[feature] = additions[feature]
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.output_csv, index=False, encoding="utf-8-sig")
    error_path = arguments.output_csv.with_name(arguments.output_csv.stem + "_errors.csv")
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")
    elif error_path.exists():
        error_path.unlink()
    print(f"Wrote {len(table)} rows; new-feature failures: {len(errors)}")


if __name__ == "__main__":
    main()
