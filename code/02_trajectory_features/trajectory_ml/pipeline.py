from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import NightRecord, build_analysis_table, load_nights
from .encoder import extract_embeddings, train_masked_encoder
from .evaluation import (
    classification_metrics,
    fit_predict_xgboost,
    grouped_bootstrap_auc_difference,
    grouped_bootstrap_metric_intervals,
    make_group_splits,
)
from .features import TrajectoryScaler, extract_handcrafted_features


MODEL_FEATURE_SETS = ["clinical", "handcrafted", "embedding", "fusion"]


def _write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=float)


def _features_for_sequences(
    sequences: list[np.ndarray],
    records: list[NightRecord],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        extract_handcrafted_features(
            sequence,
            record.stage,
            float(config["data"]["epoch_seconds"]),
            config["handcrafted"],
        )
        for sequence, record in zip(sequences, records)
    ]
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def _embedding_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        columns=[f"embedding_{index:03d}" for index in range(values.shape[1])],
    )


def _assemble_model_frame(
    metadata: pd.DataFrame,
    handcrafted: pd.DataFrame,
    embedding: pd.DataFrame,
    feature_set: str,
    continuous_covariates: list[str],
    categorical_covariates: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    clinical_columns = [*continuous_covariates, *categorical_covariates]
    parts = [metadata[clinical_columns].reset_index(drop=True)]
    numeric_columns = list(continuous_covariates)
    if feature_set in {"handcrafted", "fusion"}:
        parts.append(handcrafted.reset_index(drop=True))
        numeric_columns.extend(handcrafted.columns.tolist())
    if feature_set in {"embedding", "fusion"}:
        parts.append(embedding.reset_index(drop=True))
        numeric_columns.extend(embedding.columns.tolist())
    frame = pd.concat(parts, axis=1)
    return frame, numeric_columns, list(categorical_covariates)


def _fit_gmm(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
    outcomes: np.ndarray,
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> None:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(embeddings)
    component_count = min(10, standardized.shape[1], standardized.shape[0] - 1)
    pca = PCA(n_components=component_count, random_state=seed)
    reduced = pca.fit_transform(standardized)
    rows = []
    models: dict[int, GaussianMixture] = {}
    minimum = int(config["minimum_components"])
    maximum = min(int(config["maximum_components"]), len(embeddings) - 1)
    for components in range(minimum, maximum + 1):
        model = GaussianMixture(
            n_components=components,
            covariance_type=config["covariance_type"],
            n_init=int(config["repetitions"]),
            random_state=seed,
        ).fit(reduced)
        labels = model.predict(reduced)
        silhouette = (
            float(silhouette_score(reduced, labels))
            if len(np.unique(labels)) > 1
            else float("nan")
        )
        rows.append(
            {
                "components": components,
                "bic": float(model.bic(reduced)),
                "aic": float(model.aic(reduced)),
                "silhouette": silhouette,
                "smallest_cluster": int(np.bincount(labels).min()),
            }
        )
        models[components] = model
    selection = pd.DataFrame(rows)
    selection.to_csv(output_dir / "gmm_selection.csv", index=False)
    best_components = int(selection.loc[selection["bic"].idxmin(), "components"])
    best_model = models[best_components]
    probabilities = best_model.predict_proba(reduced)
    assignments = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "outcome": outcomes,
            "subtype": best_model.predict(reduced),
            "subtype_probability": probabilities.max(axis=1),
        }
    )
    assignments.to_csv(output_dir / "subtype_assignments.csv", index=False)
    summary = (
        assignments.groupby("subtype")
        .agg(n=("outcome", "size"), events=("outcome", "sum"), event_rate=("outcome", "mean"))
        .reset_index()
    )
    summary.to_csv(output_dir / "subtype_outcome_summary_exploratory.csv", index=False)
    joblib.dump(
        {"scaler": scaler, "pca": pca, "gmm": best_model},
        output_dir / "gmm_model.joblib",
    )


def run_pipeline(config: dict[str, Any]) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["_config_path"], output_dir / "analysis_config.yaml")

    records, trajectory_audit = load_nights(config)
    trajectory_audit.to_csv(output_dir / "trajectory_qc.csv", index=False)
    analysis, merge_audit = build_analysis_table(records, config)
    merge_audit.to_csv(output_dir / "cohort_assembly_audit.csv", index=False)
    analysis.to_csv(output_dir / "analysis_cohort.csv", index=False)

    eligible_records = [records[index] for index in analysis["record_index"].astype(int)]
    outcome = analysis["outcome"].to_numpy(int)
    groups = analysis["subject_id"].astype(str).to_numpy()
    validation = config["validation"]
    seed = int(validation["random_seed"])
    outer_splits = make_group_splits(
        analysis,
        outcome,
        groups,
        int(validation["outer_splits"]),
        seed,
    )

    meta_config = config["metadata"]
    continuous = list(meta_config["continuous_covariates"])
    categorical = list(meta_config["categorical_covariates"])
    oof_rows = analysis[
        ["subject_id", "visit", "source_file", "outcome"]
    ].copy()
    for feature_set in MODEL_FEATURE_SETS:
        oof_rows[f"probability_{feature_set}"] = np.nan
    fold_metric_rows = []
    tuning_rows = []
    oof_handcrafted: list[pd.DataFrame] = []
    oof_embeddings: list[pd.DataFrame] = []

    for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        fold_seed = seed + fold
        train_records = [eligible_records[index] for index in train_index]
        test_records = [eligible_records[index] for index in test_index]
        scaler = TrajectoryScaler.from_config(config["scaling"]).fit(train_records)
        train_sequences = [scaler.transform(record) for record in train_records]
        test_sequences = [scaler.transform(record) for record in test_records]
        train_handcrafted = _features_for_sequences(train_sequences, train_records, config)
        test_handcrafted = _features_for_sequences(test_sequences, test_records, config)

        if not config["encoder"]["enabled"]:
            raise ValueError("This experiment requires encoder.enabled: true")
        encoder, history, device = train_masked_encoder(
            train_sequences, groups[train_index], config["encoder"], fold_seed
        )
        pd.DataFrame(history).to_csv(
            output_dir / f"encoder_history_fold_{fold}.csv", index=False
        )
        train_embedding = _embedding_frame(
            extract_embeddings(encoder, train_sequences, config["encoder"], device)
        )
        test_embedding = _embedding_frame(
            extract_embeddings(encoder, test_sequences, config["encoder"], device)
        )

        held_hand = test_handcrafted.copy()
        held_hand.insert(0, "row_index", test_index)
        held_hand.insert(1, "fold", fold)
        oof_handcrafted.append(held_hand)
        held_embed = test_embedding.copy()
        held_embed.insert(0, "row_index", test_index)
        held_embed.insert(1, "fold", fold)
        oof_embeddings.append(held_embed)

        for feature_set in MODEL_FEATURE_SETS:
            train_frame, numeric_columns, categorical_columns = _assemble_model_frame(
                analysis.iloc[train_index],
                train_handcrafted,
                train_embedding,
                feature_set,
                continuous,
                categorical,
            )
            test_frame, _, _ = _assemble_model_frame(
                analysis.iloc[test_index],
                test_handcrafted,
                test_embedding,
                feature_set,
                continuous,
                categorical,
            )
            probability, _, best_parameters, inner_auc = fit_predict_xgboost(
                train_frame,
                test_frame,
                outcome[train_index],
                groups[train_index],
                numeric_columns,
                categorical_columns,
                validation,
                config["xgboost"],
                fold_seed,
            )
            oof_rows.loc[test_index, f"probability_{feature_set}"] = probability
            fold_metrics = classification_metrics(outcome[test_index], probability)
            fold_metrics.update({"fold": fold, "feature_set": feature_set})
            fold_metric_rows.append(fold_metrics)
            tuning_rows.append(
                {
                    "fold": fold,
                    "feature_set": feature_set,
                    "inner_cv_roc_auc": inner_auc,
                    "best_parameters": json.dumps(best_parameters, sort_keys=True),
                }
            )

    if oof_rows.filter(like="probability_").isna().any().any():
        raise RuntimeError("Some subjects did not receive out-of-fold predictions.")
    oof_rows.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(output_dir / "tuning_audit.csv", index=False)
    pd.concat(oof_handcrafted).sort_values("row_index").to_csv(
        output_dir / "oof_handcrafted_features.csv", index=False
    )
    pd.concat(oof_embeddings).sort_values("row_index").to_csv(
        output_dir / "oof_embeddings_fold_specific_coordinates.csv", index=False
    )

    summary_rows = []
    for feature_set in MODEL_FEATURE_SETS:
        metrics = classification_metrics(
            oof_rows["outcome"].to_numpy(),
            oof_rows[f"probability_{feature_set}"].to_numpy(),
        )
        metrics.update(
            grouped_bootstrap_metric_intervals(
                oof_rows,
                f"probability_{feature_set}",
                int(validation["bootstrap_iterations"]),
                seed + MODEL_FEATURE_SETS.index(feature_set),
            )
        )
        metrics["feature_set"] = feature_set
        summary_rows.append(metrics)
    pd.DataFrame(summary_rows).to_csv(output_dir / "oof_summary_metrics.csv", index=False)

    comparison_rows = []
    for feature_set in ["clinical", "embedding", "fusion"]:
        comparison = grouped_bootstrap_auc_difference(
            oof_rows,
            "probability_handcrafted",
            f"probability_{feature_set}",
            int(validation["bootstrap_iterations"]),
            seed,
        )
        comparison.update(
            {"reference": "handcrafted", "comparison": feature_set}
        )
        comparison_rows.append(comparison)
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "bootstrap_auc_comparisons.csv", index=False
    )

    final_scaler = TrajectoryScaler.from_config(config["scaling"]).fit(eligible_records)
    final_sequences = [final_scaler.transform(record) for record in eligible_records]
    final_encoder, final_history, device = train_masked_encoder(
        final_sequences, groups, config["encoder"], seed + 10000
    )
    pd.DataFrame(final_history).to_csv(
        output_dir / "encoder_history_final.csv", index=False
    )
    final_embeddings = extract_embeddings(
        final_encoder, final_sequences, config["encoder"], device
    )
    final_embedding_frame = _embedding_frame(final_embeddings)
    final_embedding_frame.insert(0, "subject_id", groups)
    final_embedding_frame.to_csv(output_dir / "final_embeddings.csv", index=False)
    final_handcrafted = _features_for_sequences(
        final_sequences, eligible_records, config
    )
    final_handcrafted_export = final_handcrafted.copy()
    final_handcrafted_export.insert(0, "subject_id", groups)
    final_handcrafted_export.to_csv(
        output_dir / "final_handcrafted_features.csv", index=False
    )
    _write_json(final_scaler.to_dict(), output_dir / "final_trajectory_scaler.json")
    torch.save(
        {
            "state_dict": final_encoder.cpu().state_dict(),
            "encoder_config": config["encoder"],
            "scaler": final_scaler.to_dict(),
        },
        output_dir / "final_encoder.pt",
    )
    final_model_rows = []
    final_embedding_features = _embedding_frame(final_embeddings)
    for feature_set in MODEL_FEATURE_SETS:
        final_frame, numeric_columns, categorical_columns = _assemble_model_frame(
            analysis,
            final_handcrafted,
            final_embedding_features,
            feature_set,
            continuous,
            categorical,
        )
        _, fitted_model, best_parameters, inner_auc = fit_predict_xgboost(
            final_frame,
            final_frame,
            outcome,
            groups,
            numeric_columns,
            categorical_columns,
            validation,
            config["xgboost"],
            seed + 20000,
        )
        joblib.dump(
            {
                "model": fitted_model,
                "feature_set": feature_set,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
            },
            output_dir / f"final_xgboost_{feature_set}.joblib",
        )
        final_model_rows.append(
            {
                "feature_set": feature_set,
                "inner_cv_roc_auc": inner_auc,
                "best_parameters": json.dumps(best_parameters, sort_keys=True),
            }
        )
    pd.DataFrame(final_model_rows).to_csv(
        output_dir / "final_model_tuning.csv", index=False
    )
    if config["gmm"]["enabled"]:
        _fit_gmm(
            final_embeddings,
            groups,
            outcome,
            config["gmm"],
            seed,
            output_dir,
        )

    run_summary = {
        "eligible_nights": len(analysis),
        "independent_subjects": int(analysis["subject_id"].nunique()),
        "events": int(analysis["outcome"].sum()),
        "outer_folds": len(outer_splits),
        "note": (
            "OOF embedding coordinates are fold-specific and must not be pooled for GMM. "
            "GMM uses the final outcome-agnostic encoder fitted to the full cohort."
        ),
    }
    _write_json(run_summary, output_dir / "run_summary.json")
