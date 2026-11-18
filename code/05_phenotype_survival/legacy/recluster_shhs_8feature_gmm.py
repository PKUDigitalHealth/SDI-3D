"""Refit a two-component GMM within SHHS1 using the eight prespecified SDI features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


FEATURES = ["SSTR", "CWSD", "TDTL", "BRAI", "STDE", "TOS", "TCI", "SRI"]
DEFAULT_INPUT = Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14\shhs_SDI_14_Features.csv")
DEFAULT_OUTPUT = Path("analysis/sdi_8feature_shhs_reclustered_gmm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.70)
    return parser.parse_args()


def extract_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"shhs1-([^_]+)", str(value), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    missing = [column for column in ["NSRRID", *FEATURES] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data.copy()
    data.insert(1, "subject_id", data["NSRRID"].map(extract_id))
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data.loc[~np.isfinite(data[feature]), feature] = np.nan
    if data["subject_id"].isna().any() or data["subject_id"].duplicated().any():
        raise ValueError("Invalid or duplicated SHHS1 subject identifiers")
    if data[FEATURES].isna().any().any():
        raise ValueError("Incomplete eight-feature rows are not allowed")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[FEATURES])
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=args.n_init,
        reg_covar=1e-6,
        random_state=args.seed,
    ).fit(x_scaled)
    native_probabilities = model.predict_proba(x_scaled)

    anchor = model.means_[:, FEATURES.index("SSTR")] + model.means_[:, FEATURES.index("TDTL")]
    order = np.argsort(anchor)
    probabilities = native_probabilities[:, order]
    labels = probabilities.argmax(axis=1)
    data["cluster"] = labels
    data["cluster_probability"] = probabilities.max(axis=1)
    data["cluster_ambiguous"] = data["cluster_probability"] < args.ambiguity_threshold
    data["model_role"] = "SHHS1 cohort-specific GMM refit"
    data.to_csv(output_dir / "shhs_cluster_assignments.csv", index=False)

    profiles = []
    for cluster in [0, 1]:
        mask = labels == cluster
        row: dict[str, object] = {
            "cluster": cluster,
            "group": "Normal Sleep" if cluster == 0 else "Disturbed Sleep",
            "n": int(mask.sum()),
            "proportion": float(mask.mean()),
        }
        for index, feature in enumerate(FEATURES):
            row[f"mean_raw_{feature}"] = float(data.loc[mask, feature].mean())
            row[f"mean_z_{feature}"] = float(x_scaled[mask, index].mean())
        profiles.append(row)
    pd.DataFrame(profiles).to_csv(output_dir / "cluster_profiles.csv", index=False)

    artifact = {
        "features": FEATURES,
        "cohort": "SHHS1",
        "model": "two-component full-covariance Gaussian mixture",
        "n_init": args.n_init,
        "seed": args.seed,
        "label_rule": "Cluster 1 has higher SSTR + TDTL centroid; outcomes not used",
        "component_order": order.tolist(),
        "weights": model.weights_[order].tolist(),
        "means": model.means_[order].tolist(),
        "covariances": model.covariances_[order].tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "bic": float(model.bic(x_scaled)),
        "aic": float(model.aic(x_scaled)),
        "rows": len(data),
        "cluster0": int((labels == 0).sum()),
        "cluster1": int((labels == 1).sum()),
        "ambiguous": int(data["cluster_ambiguous"].sum()),
        "mean_assignment_probability": float(data["cluster_probability"].mean()),
    }
    with (output_dir / "shhs_reclustered_gmm.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
