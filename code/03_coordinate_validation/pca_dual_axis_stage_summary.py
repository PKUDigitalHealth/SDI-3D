"""PCA of participant-by-stage Body and REM summary coordinates.

Primary analysis uses N1/N2/N3/REM stage medians and correlation-matrix PCA.
Participant-cluster bootstrap preserves the repeated stage observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STAGES = ("Wake", "N1", "N2", "N3", "REM")


def long_stage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in STAGES:
        selected = frame[
            ["subject_id", f"n_{stage}", f"body_median_{stage}", f"rem_median_{stage}"]
        ].copy()
        selected.columns = ["subject_id", "n_epochs", "body", "rem"]
        selected["stage"] = stage
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).dropna(subset=["body", "rem"])


def pca_two_axes(values: np.ndarray, standardize: bool) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=0)
    scales = centered.std(axis=0, ddof=0) if standardize else np.ones(2)
    transformed = centered / scales
    covariance = transformed.T @ transformed / len(transformed)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Deterministic signs: PC1 has positive REM loading; PC2 has positive Body loading.
    if eigenvectors[1, 0] < 0:
        eigenvectors[:, 0] *= -1
    if eigenvectors[0, 1] < 0:
        eigenvectors[:, 1] *= -1

    return {
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": (eigenvalues / eigenvalues.sum()).tolist(),
        "loadings_rows_body_rem_cols_pc1_pc2": eigenvectors.tolist(),
        "body_rem_pearson_r": float(np.corrcoef(values.T)[0, 1]),
        "n_observations": int(len(values)),
    }


def cluster_bootstrap(
    data: pd.DataFrame, subject_ids: np.ndarray, repetitions: int, seed: int
) -> dict[str, list[float]]:
    grouped = {
        subject_id: group[["body", "rem"]].to_numpy(float)
        for subject_id, group in data.groupby("subject_id")
    }
    available = np.array([subject_id for subject_id in subject_ids if subject_id in grouped])
    rng = np.random.default_rng(seed)
    statistics = np.empty((repetitions, 3))
    for index in range(repetitions):
        sampled = rng.choice(available, size=len(available), replace=True)
        values = np.vstack([grouped[subject_id] for subject_id in sampled])
        standardized = pca_two_axes(values, standardize=True)
        unscaled = pca_two_axes(values, standardize=False)
        statistics[index] = [
            standardized["body_rem_pearson_r"],
            standardized["explained_variance_ratio"][0],
            unscaled["explained_variance_ratio"][0],
        ]
    names = ("body_rem_pearson_r", "pc1_standardized", "pc1_unscaled")
    return {
        name: np.quantile(statistics[:, column], [0.025, 0.5, 0.975]).tolist()
        for column, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    participants = pd.read_csv(args.input_csv)
    stage_data = long_stage_table(participants)
    scopes = {
        "sleep_N1_N2_N3_REM": ("N1", "N2", "N3", "REM"),
        "all_stages": STAGES,
        "NREM_N1_N2_N3": ("N1", "N2", "N3"),
    }
    results: dict[str, object] = {
        "source": str(args.input_csv),
        "n_participants": int(participants["subject_id"].nunique()),
        "unit_of_analysis": "participant-by-stage median coordinate",
        "analyses": {},
    }
    for name, stages in scopes.items():
        selected = stage_data[stage_data["stage"].isin(stages)]
        values = selected[["body", "rem"]].to_numpy(float)
        results["analyses"][name] = {
            "n_participants": int(selected["subject_id"].nunique()),
            "stages": list(stages),
            "standardized": pca_two_axes(values, standardize=True),
            "unscaled": pca_two_axes(values, standardize=False),
        }

    primary = stage_data[stage_data["stage"].isin(scopes["sleep_N1_N2_N3_REM"])]
    results["primary_cluster_bootstrap_95ci"] = cluster_bootstrap(
        primary,
        participants["subject_id"].to_numpy(),
        repetitions=args.bootstrap,
        seed=args.seed,
    )
    results["bootstrap_repetitions"] = args.bootstrap
    results["bootstrap_seed"] = args.seed

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
