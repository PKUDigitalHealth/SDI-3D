"""Unified clustering pipeline for the last eight SDI features.

The clustering stage is intentionally outcome-free. Clinical outcomes can be
merged later using ``subject_id`` or the original ``NSRRID`` column.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURES = ["SSTR", "CWSD", "TDTL", "BRAI", "STDE", "TOS", "TCI", "SRI"]
DEFAULT_INPUTS = {
    "cfs": Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14\cfs_SDI_14_Features.csv"),
    "mros": Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14\mros_SDI_14_Features.csv"),
    "shhs": Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14\shhs_SDI_14_Features.csv"),
}
ID_PATTERNS = {
    "cfs": re.compile(r"cfs-visit5-([^_]+)", re.IGNORECASE),
    "mros": re.compile(r"mros-visit1-([^_]+)", re.IGNORECASE),
    "shhs": re.compile(r"shhs1-([^_]+)", re.IGNORECASE),
}


@dataclass
class CohortFit:
    cohort: str
    source: Path
    full_data: pd.DataFrame
    analysis_data: pd.DataFrame
    scaler: StandardScaler
    model: GaussianMixture
    x_scaled: np.ndarray
    raw_labels: np.ndarray
    probabilities: np.ndarray
    model_selection: pd.DataFrame
    stability: pd.DataFrame
    duplicate_ids: int
    excluded_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster CFS, MROS and SHHS using eight SDI features only."
    )
    for cohort, path in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{cohort}", type=Path, default=path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "sdi_8feature_clustering",
    )
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--selection-n-init", type=int, default=10)
    parser.add_argument("--stability-runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.70)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def extract_subject_id(value: object, cohort: str) -> str | None:
    if pd.isna(value):
        return None
    match = ID_PATTERNS[cohort].search(str(value))
    return match.group(1) if match else None


def load_and_validate(path: Path, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"{cohort}: input file not found: {path}")

    data = pd.read_csv(path)
    required = ["NSRRID", *FEATURES]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{cohort}: missing required columns: {missing_columns}")

    data = data.copy()
    data.insert(0, "cohort", cohort)
    data.insert(
        2,
        "subject_id",
        data["NSRRID"].map(lambda value: extract_subject_id(value, cohort)),
    )
    duplicate_ids = int(data["NSRRID"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"{cohort}: found {duplicate_ids} duplicated NSRRID values")

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data.loc[~np.isfinite(data[feature]), feature] = np.nan

    complete = data.dropna(subset=FEATURES).copy()
    excluded_rows = len(data) - len(complete)
    if len(complete) <= 2:
        raise ValueError(f"{cohort}: too few complete rows for clustering")
    return data, complete, duplicate_ids, excluded_rows


def evaluate_component_counts(
    x_scaled: np.ndarray,
    max_components: int,
    n_init: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    upper = min(max_components, len(x_scaled) - 1)
    for k in range(1, upper + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=n_init,
            reg_covar=1e-6,
            random_state=seed,
        ).fit(x_scaled)
        labels = model.predict(x_scaled)
        silhouette = np.nan
        if k >= 2 and len(np.unique(labels)) >= 2:
            silhouette = silhouette_score(
                x_scaled,
                labels,
                sample_size=min(2000, len(x_scaled)),
                random_state=seed,
            )
        rows.append(
            {
                "n_components": k,
                "aic": model.aic(x_scaled),
                "bic": model.bic(x_scaled),
                "silhouette": silhouette,
                "converged": int(model.converged_),
            }
        )
    return pd.DataFrame(rows)


def assess_initialization_stability(
    x_scaled: np.ndarray,
    reference_labels: np.ndarray,
    n_components: int,
    runs: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for run in range(runs):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=1,
            reg_covar=1e-6,
            random_state=seed + run,
        ).fit(x_scaled)
        labels = model.predict(x_scaled)
        rows.append(
            {
                "run": run + 1,
                "random_state": seed + run,
                "adjusted_rand_index": adjusted_rand_score(reference_labels, labels),
                "converged": int(model.converged_),
            }
        )
    return pd.DataFrame(rows)


def fit_cohort(
    cohort: str,
    source: Path,
    n_components: int,
    max_components: int,
    n_init: int,
    selection_n_init: int,
    stability_runs: int,
    seed: int,
) -> CohortFit:
    full_data, analysis_data, duplicate_ids, excluded_rows = load_and_validate(source, cohort)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(analysis_data[FEATURES])
    model_selection = evaluate_component_counts(
        x_scaled, max_components, selection_n_init, seed
    )
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        reg_covar=1e-6,
        random_state=seed,
    ).fit(x_scaled)
    raw_labels = model.predict(x_scaled)
    probabilities = model.predict_proba(x_scaled)
    stability = assess_initialization_stability(
        x_scaled, raw_labels, n_components, stability_runs, seed
    )
    return CohortFit(
        cohort=cohort,
        source=source,
        full_data=full_data,
        analysis_data=analysis_data,
        scaler=scaler,
        model=model,
        x_scaled=x_scaled,
        raw_labels=raw_labels,
        probabilities=probabilities,
        model_selection=model_selection,
        stability=stability,
        duplicate_ids=duplicate_ids,
        excluded_rows=excluded_rows,
    )


def component_order(fit: CohortFit) -> np.ndarray:
    """Order one cohort's components along its own PC1 without outcomes."""
    pca = PCA(n_components=1, random_state=0).fit(fit.x_scaled)
    direction = pca.components_[0].copy()
    if direction[0] < 0:  # Anchor the sign to SSTR for reproducibility.
        direction *= -1
    projections = fit.model.means_ @ direction
    return np.argsort(projections)


def independent_component_mapping(fit: CohortFit) -> dict[int, int]:
    """Create deterministic labels using only data from the current cohort."""
    order = component_order(fit)
    return {int(raw): int(aligned) for aligned, raw in enumerate(order)}


def add_aligned_assignments(
    fit: CohortFit,
    mapping: dict[int, int],
    ambiguity_threshold: float,
) -> pd.DataFrame:
    result = fit.analysis_data.copy()
    result["cluster"] = [mapping[int(label)] for label in fit.raw_labels]
    result["cluster_probability"] = fit.probabilities.max(axis=1)
    result["cluster_ambiguous"] = result["cluster_probability"] < ambiguity_threshold
    return result


def save_plots(
    output_dir: Path,
    fits: dict[str, CohortFit],
    assignments: dict[str, pd.DataFrame],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Re-run with --no-plots or install matplotlib."
        ) from exc

    fig, axes = plt.subplots(1, len(fits), figsize=(15, 4.5), sharey=False)
    for ax, (cohort, fit) in zip(np.atleast_1d(axes), fits.items()):
        table = fit.model_selection
        ax.plot(table["n_components"], table["bic"], marker="o")
        ax.set_title(cohort.upper())
        ax.set_xlabel("Number of components")
        ax.set_ylabel("BIC (lower is better)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_selection_bic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for cohort, fit in fits.items():
        assigned = assignments[cohort]
        profile = assigned.groupby("cluster")[FEATURES].mean()
        profile_z = pd.DataFrame(
            fit.scaler.transform(profile),
            index=profile.index,
            columns=FEATURES,
        )
        fig, ax = plt.subplots(figsize=(9, 3.2 + 0.45 * len(profile_z)))
        image = ax.imshow(profile_z, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
        ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
        ax.set_yticks(range(len(profile_z)), [f"Cluster {i}" for i in profile_z.index])
        ax.set_title(f"{cohort.upper()} standardized cluster profiles")
        for row, column in np.ndindex(profile_z.shape):
            ax.text(column, row, f"{profile_z.iloc[row, column]:.2f}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label="Within-cohort z score")
        fig.tight_layout()
        fig.savefig(output_dir / f"{cohort}_cluster_profiles.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_components < 2:
        raise ValueError("--n-components must be at least 2 for subtype analysis")
    if args.max_components < args.n_components:
        raise ValueError("--max-components must be >= --n-components")
    if not 0 < args.ambiguity_threshold < 1:
        raise ValueError("--ambiguity-threshold must be between 0 and 1")

    inputs = {cohort: getattr(args, cohort) for cohort in DEFAULT_INPUTS}
    fits = {
        cohort: fit_cohort(
            cohort=cohort,
            source=path,
            n_components=args.n_components,
            max_components=args.max_components,
            n_init=args.n_init,
            selection_n_init=args.selection_n_init,
            stability_runs=args.stability_runs,
            seed=args.seed,
        )
        for cohort, path in inputs.items()
    }
    mappings = {
        cohort: independent_component_mapping(fit) for cohort, fit in fits.items()
    }
    assignments = {
        cohort: add_aligned_assignments(fit, mappings[cohort], args.ambiguity_threshold)
        for cohort, fit in fits.items()
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_selection_frames = []
    stability_frames = []
    summary_rows = []
    centroid_rows = []
    quality = {
        "features": FEATURES,
        "n_components_used": args.n_components,
        "final_model_n_init": args.n_init,
        "model_selection_n_init": args.selection_n_init,
        "stability_runs": args.stability_runs,
        "cohort_strategy": "independent clustering and independent scaling",
        "label_rule": "within-cohort PC1 order, sign anchored to positive SSTR loading",
        "ambiguity_threshold": args.ambiguity_threshold,
        "cohorts": {},
    }

    for cohort, fit in fits.items():
        assigned = assignments[cohort]
        assigned.to_csv(output_dir / f"{cohort}_cluster_assignments.csv", index=False)

        selection = fit.model_selection.assign(cohort=cohort)
        model_selection_frames.append(selection)
        stability_frames.append(fit.stability.assign(cohort=cohort))

        for cluster, group in assigned.groupby("cluster", sort=True):
            row = {
                "cohort": cohort,
                "cluster": int(cluster),
                "n": int(len(group)),
                "proportion": len(group) / len(assigned),
                "ambiguous_n": int(group["cluster_ambiguous"].sum()),
                "mean_probability": group["cluster_probability"].mean(),
            }
            row.update({f"mean_{feature}": group[feature].mean() for feature in FEATURES})
            summary_rows.append(row)

        raw_centers = fit.scaler.inverse_transform(fit.model.means_)
        for raw_cluster, aligned_cluster in mappings[cohort].items():
            row = {"cohort": cohort, "cluster": aligned_cluster}
            row.update(
                {f"z_{feature}": fit.model.means_[raw_cluster, i] for i, feature in enumerate(FEATURES)}
            )
            row.update(
                {f"raw_{feature}": raw_centers[raw_cluster, i] for i, feature in enumerate(FEATURES)}
            )
            centroid_rows.append(row)

        best_bic_k = int(selection.loc[selection["bic"].idxmin(), "n_components"])
        silhouette_rows = selection.dropna(subset=["silhouette"])
        best_silhouette_k = int(
            silhouette_rows.loc[silhouette_rows["silhouette"].idxmax(), "n_components"]
        )
        used_bic = float(
            selection.loc[
                selection["n_components"] == args.n_components, "bic"
            ].iloc[0]
        )
        best_bic = float(selection["bic"].min())
        quality["cohorts"][cohort] = {
            "source": str(fit.source),
            "input_rows": int(len(fit.full_data)),
            "analysis_rows": int(len(fit.analysis_data)),
            "excluded_incomplete_rows": int(fit.excluded_rows),
            "duplicate_ids": int(fit.duplicate_ids),
            "bic_preferred_components": best_bic_k,
            "silhouette_preferred_components": best_silhouette_k,
            "bic_delta_for_used_components": used_bic - best_bic,
            "gmm_converged": bool(fit.model.converged_),
            "initialization_stability_median_ari": float(
                fit.stability["adjusted_rand_index"].median()
            ),
            "initialization_stability_min_ari": float(
                fit.stability["adjusted_rand_index"].min()
            ),
            "ambiguous_rows": int(assigned["cluster_ambiguous"].sum()),
        }

    pd.concat(model_selection_frames, ignore_index=True).to_csv(
        output_dir / "model_selection.csv", index=False
    )
    pd.concat(stability_frames, ignore_index=True).to_csv(
        output_dir / "initialization_stability.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(output_dir / "cluster_summary.csv", index=False)
    pd.DataFrame(centroid_rows).sort_values(["cohort", "cluster"]).to_csv(
        output_dir / "cluster_centroids.csv", index=False
    )
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)

    if not args.no_plots:
        save_plots(output_dir, fits, assignments)

    print(json.dumps(quality, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
