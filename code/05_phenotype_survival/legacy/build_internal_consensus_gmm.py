"""Build a pooled internal CFS/MROS GMM and apply it to external SHHS.

Clinical outcomes are never read here. CFS and MROS are the only cohorts used
to estimate mixture parameters. SHHS receives frozen-model assignments only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Internal CFS/MROS consensus GMM with external SHHS application."
    )
    for cohort, path in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{cohort}", type=Path, default=path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "sdi_8feature_internal_consensus_gmm",
    )
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.70)
    return parser.parse_args()


def extract_subject_id(value: object, cohort: str) -> str | None:
    if pd.isna(value):
        return None
    match = ID_PATTERNS[cohort].search(str(value))
    return match.group(1).lower() if match else None


def load_features(path: Path, cohort: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = ["NSRRID", *FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{cohort}: missing columns {missing}")
    data = data.copy()
    data.insert(0, "cohort", cohort)
    data.insert(
        2,
        "subject_id",
        data["NSRRID"].map(lambda value: extract_subject_id(value, cohort)),
    )
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data.loc[~np.isfinite(data[feature]), feature] = np.nan
    if data["NSRRID"].duplicated().any():
        raise ValueError(f"{cohort}: duplicated NSRRID values")
    complete = data.dropna(subset=FEATURES).copy()
    if len(complete) != len(data):
        raise ValueError(f"{cohort}: incomplete feature rows are not allowed in this model")
    return complete


def fit_native_model(
    data: pd.DataFrame, n_init: int, seed: int
) -> tuple[StandardScaler, np.ndarray, GaussianMixture]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[FEATURES])
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=n_init,
        reg_covar=1e-6,
        random_state=seed,
    ).fit(x_scaled)
    return scaler, x_scaled, model


def build_pooled_internal_model(
    cfs_scaled: np.ndarray, mros_scaled: np.ndarray, n_init: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], GaussianMixture]:
    """Fit one model to all internal participants after cohort-wise z scoring."""
    pooled = np.vstack([cfs_scaled, mros_scaled])
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=n_init,
        reg_covar=1e-6,
        random_state=seed,
    ).fit(pooled)
    anchor = model.means_[:, FEATURES.index("SSTR")] + model.means_[:, FEATURES.index("TDTL")]
    order = np.argsort(anchor)
    weights = model.weights_[order]
    means = model.means_[order]
    covariances = model.covariances_[order]
    alignment = {
        "final_order_by_sstr_plus_tdtl": order.tolist(),
        "label_rule": "Cluster 1 has the higher pooled-internal SSTR + TDTL centroid",
    }
    return weights, means, covariances, alignment, model


def predict_consensus(
    x_scaled: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    log_joint = np.column_stack(
        [
            np.log(weights[index])
            + multivariate_normal.logpdf(
                x_scaled,
                mean=means[index],
                cov=covariances[index],
                allow_singular=False,
            )
            for index in range(len(weights))
        ]
    )
    probabilities = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    labels = probabilities.argmax(axis=1)
    return labels, probabilities


def profile_rows(
    data: pd.DataFrame, x_scaled: np.ndarray, labels: np.ndarray, cohort: str
) -> list[dict[str, object]]:
    rows = []
    for cluster in [0, 1]:
        mask = labels == cluster
        row: dict[str, object] = {
            "cohort": cohort,
            "role": "external" if cohort == "shhs" else "internal",
            "cluster": cluster,
            "n": int(mask.sum()),
            "proportion": float(mask.mean()),
        }
        for feature_index, feature in enumerate(FEATURES):
            row[f"mean_raw_{feature}"] = float(data.loc[mask, feature].mean())
            row[f"mean_z_{feature}"] = float(x_scaled[mask, feature_index].mean())
        rows.append(row)
    return rows


def plot_profile_differences(profile: pd.DataFrame, output_dir: Path) -> None:
    difference_rows = []
    for cohort in ["cfs", "mros", "shhs"]:
        table = profile.loc[profile["cohort"] == cohort].set_index("cluster")
        difference_rows.append(
            [
                table.loc[1, f"mean_z_{feature}"] - table.loc[0, f"mean_z_{feature}"]
                for feature in FEATURES
            ]
        )
    differences = np.asarray(difference_rows)
    limit = max(1.5, float(np.ceil(np.abs(differences).max() * 2) / 2))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    image = ax.imshow(differences, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha="right")
    ax.set_yticks(range(3), ["CFS (internal)", "MROS (internal)", "SHHS (external)"])
    for row, column in np.ndindex(differences.shape):
        ax.text(column, row, f"{differences[row, column]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Frozen internal consensus GMM: Cluster 1 minus Cluster 0 profiles", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Difference in within-cohort z-score means")
    fig.tight_layout()
    fig.savefig(output_dir / "internal_external_profile_consistency.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "internal_external_profile_consistency.pdf", bbox_inches="tight")
    plt.close(fig)


def scaler_dict(scaler: StandardScaler) -> dict[str, list[float]]:
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.ambiguity_threshold < 1:
        raise ValueError("--ambiguity-threshold must be between 0 and 1")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        cohort: load_features(getattr(args, cohort), cohort)
        for cohort in ["cfs", "mros", "shhs"]
    }

    scalers: dict[str, StandardScaler] = {}
    scaled: dict[str, np.ndarray] = {}
    native_models: dict[str, GaussianMixture] = {}
    for index, cohort in enumerate(["cfs", "mros"]):
        scaler, x_scaled, model = fit_native_model(
            data[cohort], args.n_init, args.seed + index
        )
        scalers[cohort] = scaler
        scaled[cohort] = x_scaled
        native_models[cohort] = model

    weights, means, covariances, alignment, pooled_model = build_pooled_internal_model(
        scaled["cfs"], scaled["mros"], args.n_init, args.seed
    )

    shhs_scaler = StandardScaler()
    scalers["shhs"] = shhs_scaler
    scaled["shhs"] = shhs_scaler.fit_transform(data["shhs"][FEATURES])

    profile_records: list[dict[str, object]] = []
    quality: dict[str, object] = {
        "features": FEATURES,
        "development_cohorts": ["cfs", "mros"],
        "external_validation_cohort": "shhs",
        "gmm_components": 2,
        "gmm_n_init_internal": args.n_init,
        "internal_model_fit": "single GMM fit on all CFS and MROS participants after cohort-wise z scoring",
        "internal_sample_weighting": "participant-level pooling; no outcome-based sampling or balancing",
        "external_gmm_refit": False,
        "external_outcomes_used": False,
        "preprocessing": "within-cohort z-score fixed before outcome analysis",
        "alignment": alignment,
        "pooled_internal_bic": float(pooled_model.bic(np.vstack([scaled["cfs"], scaled["mros"]]))),
        "pooled_internal_aic": float(pooled_model.aic(np.vstack([scaled["cfs"], scaled["mros"]]))),
        "cohorts": {},
    }

    for cohort in ["cfs", "mros", "shhs"]:
        labels, probabilities = predict_consensus(
            scaled[cohort], weights, means, covariances
        )
        assignments = data[cohort].copy()
        assignments["cluster"] = labels
        assignments["cluster_probability"] = probabilities.max(axis=1)
        assignments["cluster_ambiguous"] = (
            assignments["cluster_probability"] < args.ambiguity_threshold
        )
        assignments["model_role"] = (
            "external_frozen_model_application"
            if cohort == "shhs"
            else "internal_consensus_model_application"
        )
        assignments.to_csv(output_dir / f"{cohort}_cluster_assignments.csv", index=False)
        profile_records.extend(profile_rows(data[cohort], scaled[cohort], labels, cohort))

        cohort_quality: dict[str, object] = {
            "rows": len(assignments),
            "cluster0": int((labels == 0).sum()),
            "cluster1": int((labels == 1).sum()),
            "ambiguous": int(assignments["cluster_ambiguous"].sum()),
            "mean_assignment_probability": float(probabilities.max(axis=1).mean()),
        }
        if cohort in native_models:
            cohort_quality["ari_native_vs_consensus"] = float(
                adjusted_rand_score(native_models[cohort].predict(scaled[cohort]), labels)
            )
        quality["cohorts"][cohort] = cohort_quality

    profile = pd.DataFrame(profile_records)
    profile.to_csv(output_dir / "cluster_profiles.csv", index=False)
    centroid_rows = []
    for cluster in [0, 1]:
        row = {"cluster": cluster, "weight": weights[cluster]}
        row.update({f"mean_z_{feature}": means[cluster, i] for i, feature in enumerate(FEATURES)})
        centroid_rows.append(row)
    pd.DataFrame(centroid_rows).to_csv(output_dir / "frozen_consensus_centroids.csv", index=False)

    model_artifact = {
        "features": FEATURES,
        "weights": weights.tolist(),
        "means": means.tolist(),
        "covariances": covariances.tolist(),
        "scalers": {cohort: scaler_dict(scalers[cohort]) for cohort in scalers},
        "provenance": quality,
    }
    with (output_dir / "frozen_internal_consensus_gmm.json").open("w", encoding="utf-8") as handle:
        json.dump(model_artifact, handle, ensure_ascii=False, indent=2)
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)
    plot_profile_differences(profile, output_dir)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
