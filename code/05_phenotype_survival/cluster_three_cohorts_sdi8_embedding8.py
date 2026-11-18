"""Independently cluster CFS, MROS, and SHHS using SDI8 + embedding8 features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


SDI_FEATURES = ["SSTR", "CWSD", "TDTL", "BRAI", "STDE", "TOS", "TCI", "SRI"]
EMBED_SOURCE = [f"feature_{index}" for index in range(1, 9)]
EMBED_FEATURES = [f"NE{index}" for index in range(1, 9)]
FEATURES = SDI_FEATURES + EMBED_FEATURES
ROOT = Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14")
CONFIG = {
    "cfs": {
        "sdi": ROOT / "cfs_SDI_14_Features.csv",
        "embedding": ROOT / "cfs_embeddings_8d.csv",
        "pattern": re.compile(r"cfs-visit5-([^_]+)", re.IGNORECASE),
    },
    "mros": {
        "sdi": ROOT / "mros_SDI_14_Features.csv",
        "embedding": ROOT / "mros_embeddings_8d.csv",
        "pattern": re.compile(r"mros-visit1-([^_]+)", re.IGNORECASE),
    },
    "shhs": {
        "sdi": ROOT / "shhs_SDI_14_Features.csv",
        "embedding": ROOT / "shhs_external_embeddings_8d.csv",
        "pattern": re.compile(r"shhs1-([^_]+)", re.IGNORECASE),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("analysis/three_cohort_independent_sdi8_embedding8_gmm"),
    )
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.70)
    return parser.parse_args()


def subject_id(value: object, pattern: re.Pattern[str]) -> str | None:
    if pd.isna(value):
        return None
    match = pattern.search(str(value))
    return match.group(1).lower() if match else None


def load_cohort(cohort: str) -> pd.DataFrame:
    config = CONFIG[cohort]
    sdi = pd.read_csv(config["sdi"])
    embedding = pd.read_csv(config["embedding"])
    required_sdi = ["NSRRID", *SDI_FEATURES]
    required_embedding = ["night_id", "split", *EMBED_SOURCE]
    missing_sdi = [column for column in required_sdi if column not in sdi.columns]
    missing_embedding = [column for column in required_embedding if column not in embedding.columns]
    if missing_sdi or missing_embedding:
        raise ValueError(f"{cohort}: missing SDI={missing_sdi}, embedding={missing_embedding}")
    sdi = sdi[required_sdi].copy()
    embedding = embedding[required_embedding].copy()
    pattern = config["pattern"]
    sdi["subject_id"] = sdi["NSRRID"].map(lambda value: subject_id(value, pattern))
    embedding["subject_id"] = embedding["night_id"].map(lambda value: subject_id(value, pattern))
    embedding = embedding.rename(columns=dict(zip(EMBED_SOURCE, EMBED_FEATURES)))
    for frame, source in [(sdi, "SDI"), (embedding, "embedding")]:
        if frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
            raise ValueError(f"{cohort} {source}: invalid or duplicated subject IDs")
    data = sdi.merge(embedding, on="subject_id", how="inner", validate="one_to_one")
    if len(data) != len(sdi) or len(data) != len(embedding):
        raise ValueError(
            f"{cohort}: ID mismatch, SDI={len(sdi)}, embedding={len(embedding)}, merged={len(data)}"
        )
    data.insert(0, "cohort", cohort)
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data.loc[~np.isfinite(data[feature]), feature] = np.nan
    if data[FEATURES].isna().any().any():
        raise ValueError(f"{cohort}: missing/non-finite feature values")
    return data


def fit_cohort(
    data: pd.DataFrame, n_init: int, seed: int, ambiguity_threshold: float
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[FEATURES])
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=n_init,
        reg_covar=1e-6,
        random_state=seed,
    ).fit(x_scaled)
    native_probabilities = model.predict_proba(x_scaled)
    anchor = model.means_[:, FEATURES.index("SSTR")] + model.means_[:, FEATURES.index("TDTL")]
    order = np.argsort(anchor)
    probabilities = native_probabilities[:, order]
    labels = probabilities.argmax(axis=1)
    assignments = data.copy()
    assignments["cluster"] = labels
    assignments["cluster_probability"] = probabilities.max(axis=1)
    assignments["cluster_ambiguous"] = assignments["cluster_probability"] < ambiguity_threshold
    assignments["group"] = assignments["cluster"].map({0: "Normal Sleep", 1: "Disturbed Sleep"})
    assignments["model_role"] = "cohort-specific 16-feature GMM refit"

    profile_rows: list[dict[str, object]] = []
    for cluster in [0, 1]:
        mask = labels == cluster
        row: dict[str, object] = {
            "cohort": str(data["cohort"].iloc[0]),
            "cluster": cluster,
            "group": "Normal Sleep" if cluster == 0 else "Disturbed Sleep",
            "n": int(mask.sum()),
            "proportion": float(mask.mean()),
        }
        for index, feature in enumerate(FEATURES):
            row[f"mean_raw_{feature}"] = float(data.loc[mask, feature].mean())
            row[f"mean_z_{feature}"] = float(x_scaled[mask, index].mean())
        profile_rows.append(row)

    artifact = {
        "rows": len(data),
        "cluster0": int((labels == 0).sum()),
        "cluster1": int((labels == 1).sum()),
        "ambiguous": int(assignments["cluster_ambiguous"].sum()),
        "mean_assignment_probability": float(assignments["cluster_probability"].mean()),
        "split_counts": assignments["split"].value_counts().to_dict(),
        "component_order": order.tolist(),
        "weights": model.weights_[order].tolist(),
        "means": model.means_[order].tolist(),
        "covariances": model.covariances_[order].tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "bic": float(model.bic(x_scaled)),
        "aic": float(model.aic(x_scaled)),
    }
    return assignments, profile_rows, artifact


def plot_profiles(profile: pd.DataFrame, output_dir: Path) -> None:
    differences = []
    for cohort in ["cfs", "mros", "shhs"]:
        table = profile.loc[profile["cohort"].eq(cohort)].set_index("cluster")
        differences.append([
            table.loc[1, f"mean_z_{feature}"] - table.loc[0, f"mean_z_{feature}"]
            for feature in FEATURES
        ])
    values = np.asarray(differences)
    limit = max(1.0, float(np.ceil(np.abs(values).max() * 2) / 2))
    fig, ax = plt.subplots(figsize=(14, 4.6))
    image = ax.imshow(values, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=40, ha="right")
    ax.set_yticks(range(3), ["CFS", "MROS", "SHHS"])
    for row, column in np.ndindex(values.shape):
        ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    ax.axvline(7.5, color="black", linewidth=2)
    ax.set_title("Independent 16-feature GMM profiles: Disturbed minus Normal", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Difference in within-cohort standardized means")
    fig.tight_layout()
    fig.savefig(output_dir / "three_cohort_16feature_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "three_cohort_16feature_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < args.ambiguity_threshold < 1:
        raise ValueError("ambiguity threshold must be between 0 and 1")

    profiles: list[dict[str, object]] = []
    quality: dict[str, object] = {
        "features": FEATURES,
        "sdi_features": SDI_FEATURES,
        "embedding_features": EMBED_FEATURES,
        "strategy": "each cohort standardized and clustered independently",
        "clusters": 2,
        "label_rule": "Cluster 1 has higher SSTR + TDTL centroid; outcomes not used",
        "cohorts": {},
    }
    for index, cohort in enumerate(["cfs", "mros", "shhs"]):
        data = load_cohort(cohort)
        assignments, profile_rows, artifact = fit_cohort(
            data, args.n_init, args.seed + index, args.ambiguity_threshold
        )
        assignments.to_csv(output_dir / f"{cohort}_cluster_assignments.csv", index=False)
        profiles.extend(profile_rows)
        quality["cohorts"][cohort] = artifact

    profile = pd.DataFrame(profiles)
    profile.to_csv(output_dir / "cluster_profiles.csv", index=False)
    plot_profiles(profile, output_dir)
    with (output_dir / "independent_16feature_gmm_models.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "strategy": quality["strategy"],
        "label_rule": quality["label_rule"],
        "cohorts": {
            cohort: {key: value for key, value in artifact.items()
                     if key in ["rows", "cluster0", "cluster1", "ambiguous", "mean_assignment_probability", "split_counts"]}
            for cohort, artifact in quality["cohorts"].items()
        },
    }, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
