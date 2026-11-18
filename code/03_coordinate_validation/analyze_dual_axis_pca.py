"""Participant-level standardized PCA of the Body and REM axes.

For two within-participant z-standardized variables, the correlation matrix is
[[1, r], [r, 1]]. Its ordered eigenvalues are 1 + |r| and 1 - |r|, so the
explained-variance ratios are (1 + |r|) / 2 and (1 - |r|) / 2. This permits an
exact PCA variance-retention analysis from participant-level correlations even
when the original epoch files are not locally available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSES = {
    "All sleep": "body_rem_pearson_sleep",
    "NREM sleep": "body_rem_pearson_nrem",
}


def bootstrap_median_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int = 10_000,
) -> tuple[float, float]:
    """Return a percentile bootstrap 95% CI for the median."""
    values = values[np.isfinite(values)]
    sample_size = len(values)
    medians = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = values[rng.integers(0, sample_size, sample_size)]
        medians[index] = np.median(sample)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def analyse(input_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(input_path)
    missing = set(ANALYSES.values()).difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")

    rng = np.random.default_rng(20260821)
    participant_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for subset, correlation_column in ANALYSES.items():
        correlations = pd.to_numeric(frame[correlation_column], errors="coerce")
        valid = correlations.notna() & correlations.between(-1, 1)
        r = correlations.loc[valid].to_numpy(float)
        absolute_r = np.abs(r)
        pc1 = (1 + absolute_r) / 2
        pc2 = (1 - absolute_r) / 2

        participant_tables.append(
            pd.DataFrame(
                {
                    "subject_id": frame.loc[valid, "subject_id"].to_numpy(),
                    "epoch_subset": subset,
                    "body_rem_pearson_r": r,
                    "pc1_explained_variance_ratio": pc1,
                    "pc2_explained_variance_ratio": pc2,
                    "pc1_loading_body_abs": np.full(len(r), 1 / np.sqrt(2)),
                    "pc1_loading_rem_abs": np.full(len(r), 1 / np.sqrt(2)),
                }
            )
        )

        pc1_ci = bootstrap_median_ci(pc1, rng)
        pc2_ci = bootstrap_median_ci(pc2, rng)
        summary_rows.append(
            {
                "epoch_subset": subset,
                "n_participants": len(r),
                "median_body_rem_r": float(np.median(r)),
                "median_absolute_r": float(np.median(absolute_r)),
                "pc1_median": float(np.median(pc1)),
                "pc1_q1": float(np.quantile(pc1, 0.25)),
                "pc1_q3": float(np.quantile(pc1, 0.75)),
                "pc1_median_ci_low": pc1_ci[0],
                "pc1_median_ci_high": pc1_ci[1],
                "pc2_median": float(np.median(pc2)),
                "pc2_q1": float(np.quantile(pc2, 0.25)),
                "pc2_q3": float(np.quantile(pc2, 0.75)),
                "pc2_median_ci_low": pc2_ci[0],
                "pc2_median_ci_high": pc2_ci[1],
                "pct_pc1_ge_70": float(np.mean(pc1 >= 0.70) * 100),
                "pct_pc1_ge_80": float(np.mean(pc1 >= 0.80) * 100),
                "pct_pc1_ge_90": float(np.mean(pc1 >= 0.90) * 100),
                "pct_pc2_ge_30": float(np.mean(pc2 >= 0.30) * 100),
                "pct_pc2_ge_40": float(np.mean(pc2 >= 0.40) * 100),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(participant_tables, ignore_index=True).to_csv(
        output_dir / "participant_standardized_pca.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "standardized_pca_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyse(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
