"""Cluster SHHS1 using eight SDI features plus eight night-embedding features."""

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


SDI_FEATURES = ["SSTR", "CWSD", "TDTL", "BRAI", "STDE", "TOS", "TCI", "SRI"]
EMBEDDING_SOURCE = [f"feature_{index}" for index in range(1, 9)]
EMBEDDING_FEATURES = [f"NE{index}" for index in range(1, 9)]
ALL_FEATURES = SDI_FEATURES + EMBEDDING_FEATURES
DEFAULT_SDI = Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14\shhs_SDI_14_Features.csv")
DEFAULT_EMBEDDING = Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14\shhs_night_embeddings_8d.csv")
DEFAULT_OLD_ASSIGNMENTS = Path("analysis/sdi_8feature_shhs_reclustered_gmm/shhs_cluster_assignments.csv")
DEFAULT_OUTPUT = Path("analysis/shhs_sdi8_embedding8_reclustered_gmm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdi", type=Path, default=DEFAULT_SDI)
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--old-assignments", type=Path, default=DEFAULT_OLD_ASSIGNMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.70)
    return parser.parse_args()


def extract_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.extract(r"shhs1-([^_]+)", flags=re.IGNORECASE, expand=False).str.lower()


def plot_profile(profile: pd.DataFrame, output_dir: Path) -> None:
    values = profile.set_index("cluster")
    difference = np.array([
        values.loc[1, f"mean_z_{feature}"] - values.loc[0, f"mean_z_{feature}"]
        for feature in ALL_FEATURES
    ])
    limit = max(1.0, float(np.ceil(np.abs(difference).max() * 2) / 2))
    fig, ax = plt.subplots(figsize=(13, 3.2))
    image = ax.imshow(difference.reshape(1, -1), cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(ALL_FEATURES)), ALL_FEATURES, rotation=40, ha="right")
    ax.set_yticks([0], ["Disturbed minus Normal"])
    for column, value in enumerate(difference):
        ax.text(column, 0, f"{value:.2f}", ha="center", va="center", fontsize=9)
    ax.axvline(7.5, color="black", linewidth=2)
    ax.set_title("SHHS1 16-feature GMM profile (SDI 8 + Night Embedding 8)", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Difference in standardized means")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_profile_16features.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "cluster_profile_16features.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sdi = pd.read_csv(args.sdi)
    embedding = pd.read_csv(args.embedding)
    missing_sdi = [column for column in ["NSRRID", *SDI_FEATURES] if column not in sdi.columns]
    missing_embedding = [column for column in ["night_id", "split", *EMBEDDING_SOURCE] if column not in embedding.columns]
    if missing_sdi or missing_embedding:
        raise ValueError(f"Missing SDI columns={missing_sdi}; embedding columns={missing_embedding}")
    sdi = sdi[["NSRRID", *SDI_FEATURES]].copy()
    embedding = embedding[["night_id", "split", *EMBEDDING_SOURCE]].copy()
    sdi["subject_id"] = extract_id(sdi["NSRRID"])
    embedding["subject_id"] = extract_id(embedding["night_id"])
    embedding = embedding.rename(columns=dict(zip(EMBEDDING_SOURCE, EMBEDDING_FEATURES)))
    for frame, name in [(sdi, "SDI"), (embedding, "embedding")]:
        if frame["subject_id"].isna().any() or frame["subject_id"].duplicated().any():
            raise ValueError(f"{name}: invalid or duplicated subject IDs")
    data = sdi.merge(embedding, on="subject_id", how="inner", validate="one_to_one")
    if len(data) != len(sdi) or len(data) != len(embedding):
        raise ValueError(f"ID mismatch: SDI={len(sdi)}, embedding={len(embedding)}, merged={len(data)}")
    for feature in ALL_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data.loc[~np.isfinite(data[feature]), feature] = np.nan
    if data[ALL_FEATURES].isna().any().any():
        raise ValueError("Missing/non-finite values in the combined 16-feature matrix")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[ALL_FEATURES])
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=args.n_init,
        reg_covar=1e-6,
        random_state=args.seed,
    ).fit(x_scaled)
    native_probabilities = model.predict_proba(x_scaled)
    anchor = model.means_[:, SDI_FEATURES.index("SSTR")] + model.means_[:, SDI_FEATURES.index("TDTL")]
    order = np.argsort(anchor)
    probabilities = native_probabilities[:, order]
    labels = probabilities.argmax(axis=1)
    data["cluster"] = labels
    data["cluster_probability"] = probabilities.max(axis=1)
    data["cluster_ambiguous"] = data["cluster_probability"] < args.ambiguity_threshold
    data["model_role"] = "SHHS1 cohort-specific 16-feature GMM refit"
    data.to_csv(output_dir / "shhs_cluster_assignments.csv", index=False)

    profiles: list[dict[str, object]] = []
    for cluster in [0, 1]:
        mask = labels == cluster
        row: dict[str, object] = {
            "cluster": cluster,
            "group": "Normal Sleep" if cluster == 0 else "Disturbed Sleep",
            "n": int(mask.sum()),
            "proportion": float(mask.mean()),
        }
        for index, feature in enumerate(ALL_FEATURES):
            row[f"mean_raw_{feature}"] = float(data.loc[mask, feature].mean())
            row[f"mean_z_{feature}"] = float(x_scaled[mask, index].mean())
        profiles.append(row)
    profile = pd.DataFrame(profiles)
    profile.to_csv(output_dir / "cluster_profiles.csv", index=False)
    plot_profile(profile, output_dir)

    comparison: dict[str, object] = {}
    if args.old_assignments.exists():
        old = pd.read_csv(args.old_assignments, usecols=["subject_id", "cluster"])
        old["subject_id"] = old["subject_id"].astype("string").str.lower()
        matched = data[["subject_id", "cluster"]].merge(
            old.rename(columns={"cluster": "old_cluster"}), on="subject_id", validate="one_to_one"
        )
        comparison = {
            "old_model": "SHHS1 eight-SDI-feature GMM",
            "matched_rows": len(matched),
            "adjusted_rand_index": float(adjusted_rand_score(matched["old_cluster"], matched["cluster"])),
            "normalized_mutual_information": float(normalized_mutual_info_score(matched["old_cluster"], matched["cluster"])),
            "same_label_proportion": float((matched["old_cluster"] == matched["cluster"]).mean()),
        }

    artifact = {
        "cohort": "SHHS1",
        "sdi_features": SDI_FEATURES,
        "embedding_features": EMBEDDING_FEATURES,
        "embedding_source_columns": EMBEDDING_SOURCE,
        "input_dimensions": 16,
        "model": "two-component full-covariance Gaussian mixture",
        "n_init": args.n_init,
        "seed": args.seed,
        "label_rule": "Cluster 1 has higher SDI SSTR + TDTL centroid; outcomes not used",
        "component_order": order.tolist(),
        "weights": model.weights_[order].tolist(),
        "means": model.means_[order].tolist(),
        "covariances": model.covariances_[order].tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "split_counts": data["split"].value_counts().to_dict(),
        "bic": float(model.bic(x_scaled)),
        "aic": float(model.aic(x_scaled)),
        "rows": len(data),
        "cluster0": int((labels == 0).sum()),
        "cluster1": int((labels == 1).sum()),
        "ambiguous": int(data["cluster_ambiguous"].sum()),
        "mean_assignment_probability": float(data["cluster_probability"].mean()),
        "comparison_to_8feature_model": comparison,
    }
    with (output_dir / "shhs_16feature_gmm.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
