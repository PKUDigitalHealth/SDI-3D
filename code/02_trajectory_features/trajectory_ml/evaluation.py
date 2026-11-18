from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def determine_split_count(
    outcome: np.ndarray, groups: np.ndarray, requested: int
) -> int:
    frame = pd.DataFrame({"outcome": outcome, "group": groups})
    group_outcomes = frame.groupby("group")["outcome"].agg(["min", "max"])
    if np.any(group_outcomes["min"] != group_outcomes["max"]):
        raise ValueError("The same subject has inconsistent binary outcomes across visits.")
    class_counts = group_outcomes["max"].value_counts()
    if len(class_counts) != 2:
        raise ValueError("Both outcome classes must occur among independent subjects.")
    possible = int(class_counts.min())
    splits = min(int(requested), possible)
    if splits < 2:
        raise ValueError("At least two independent subjects per class are required.")
    return splits


def make_group_splits(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    groups: np.ndarray,
    requested: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = determine_split_count(outcome, groups, requested)
    group_frame = pd.DataFrame({"group": groups, "outcome": outcome}).drop_duplicates()
    if group_frame["group"].duplicated().any():
        raise ValueError("A subject maps to more than one outcome.")
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    row_indices = np.arange(len(frame))
    result = []
    for train_group_index, test_group_index in splitter.split(
        group_frame["group"], group_frame["outcome"]
    ):
        train_groups = set(group_frame.iloc[train_group_index]["group"])
        test_groups = set(group_frame.iloc[test_group_index]["group"])
        train_rows = row_indices[np.isin(groups, list(train_groups))]
        test_rows = row_indices[np.isin(groups, list(test_groups))]
        result.append((train_rows, test_rows))
    return result


def _model_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    positive_weight: float,
    seed: int,
    n_jobs: int,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    processor = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        # Parallelism is controlled by RandomizedSearchCV to avoid nested
        # oversubscription on shared servers.
        n_jobs=1,
        scale_pos_weight=positive_weight,
    )
    return Pipeline([("preprocess", processor), ("model", model)])


def fit_predict_xgboost(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_outcome: np.ndarray,
    train_groups: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
    validation_config: dict[str, Any],
    xgboost_config: dict[str, list[Any]],
    seed: int,
) -> tuple[np.ndarray, Pipeline, dict[str, Any], float]:
    positives = float(np.sum(train_outcome == 1))
    negatives = float(np.sum(train_outcome == 0))
    positive_weight = negatives / max(positives, 1.0)
    pipeline = _model_pipeline(
        numeric_columns,
        categorical_columns,
        positive_weight,
        seed,
        int(validation_config["n_jobs"]),
    )
    inner_splits = make_group_splits(
        train_frame,
        train_outcome,
        train_groups,
        int(validation_config["inner_splits"]),
        seed,
    )
    parameters = {f"model__{key}": values for key, values in xgboost_config.items()}
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=int(validation_config["tuning_iterations"]),
        scoring="roc_auc",
        cv=inner_splits,
        refit=True,
        random_state=seed,
        n_jobs=int(validation_config["n_jobs"]),
        return_train_score=False,
    )
    search.fit(train_frame, train_outcome)
    probability = search.best_estimator_.predict_proba(test_frame)[:, 1]
    best_parameters = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    return probability, search.best_estimator_, best_parameters, float(search.best_score_)


def classification_metrics(
    outcome: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(outcome, predicted, labels=[0, 1]).ravel()
    return {
        "n": float(len(outcome)),
        "events": float(np.sum(outcome)),
        "roc_auc": float(roc_auc_score(outcome, probability)),
        "average_precision": float(average_precision_score(outcome, probability)),
        "brier_score": float(brier_score_loss(outcome, probability)),
        "accuracy_at_0_5": float(accuracy_score(outcome, predicted)),
        "sensitivity_at_0_5": float(tp / max(tp + fn, 1)),
        "specificity_at_0_5": float(tn / max(tn + fp, 1)),
    }


def grouped_bootstrap_auc_difference(
    frame: pd.DataFrame,
    reference_probability: str,
    comparison_probability: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    group_to_indices = {
        group: indices.to_numpy()
        for group, indices in frame.groupby("subject_id").groups.items()
    }
    groups = np.asarray(list(group_to_indices))
    differences = []
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate(
            [group_to_indices[group] for group in sampled_groups]
        )
        sampled = frame.iloc[sampled_indices]
        if sampled["outcome"].nunique() != 2:
            continue
        reference_auc = roc_auc_score(
            sampled["outcome"], sampled[reference_probability]
        )
        comparison_auc = roc_auc_score(
            sampled["outcome"], sampled[comparison_probability]
        )
        differences.append(comparison_auc - reference_auc)
    if not differences:
        raise RuntimeError("No valid grouped bootstrap replicates were generated.")
    values = np.asarray(differences)
    observed = roc_auc_score(frame["outcome"], frame[comparison_probability]) - roc_auc_score(
        frame["outcome"], frame[reference_probability]
    )
    lower, upper = np.quantile(values, [0.025, 0.975])
    probability_nonpositive = (np.sum(values <= 0) + 1) / (len(values) + 1)
    probability_nonnegative = (np.sum(values >= 0) + 1) / (len(values) + 1)
    return {
        "delta_auc": float(observed),
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "bootstrap_p_two_sided": float(
            min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        ),
        "valid_replicates": float(len(values)),
    }


def grouped_bootstrap_metric_intervals(
    frame: pd.DataFrame,
    probability_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    group_to_indices = {
        group: indices.to_numpy()
        for group, indices in frame.groupby("subject_id").groups.items()
    }
    groups = np.asarray(list(group_to_indices))
    bootstrap_rows = []
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate(
            [group_to_indices[group] for group in sampled_groups]
        )
        sampled = frame.iloc[sampled_indices]
        if sampled["outcome"].nunique() != 2:
            continue
        bootstrap_rows.append(
            [
                roc_auc_score(sampled["outcome"], sampled[probability_column]),
                average_precision_score(
                    sampled["outcome"], sampled[probability_column]
                ),
                brier_score_loss(
                    sampled["outcome"], sampled[probability_column]
                ),
            ]
        )
    if not bootstrap_rows:
        raise RuntimeError("No valid grouped bootstrap replicates were generated.")
    values = np.asarray(bootstrap_rows)
    result: dict[str, float] = {"valid_replicates": float(len(values))}
    for column_index, metric in enumerate(
        ["roc_auc", "average_precision", "brier_score"]
    ):
        lower, upper = np.quantile(values[:, column_index], [0.025, 0.975])
        result[f"{metric}_ci_2_5"] = float(lower)
        result[f"{metric}_ci_97_5"] = float(upper)
    return result
