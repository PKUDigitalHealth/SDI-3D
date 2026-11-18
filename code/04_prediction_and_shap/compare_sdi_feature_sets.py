"""Compare SDI-6 with handcrafted and learned SDI-3D representations.

Primary analysis: cohort-specific five-fold stratified out-of-fold prediction.
External analysis: fit on pooled CFS + MrOS and evaluate once in SHHS.
All feature sets use identical XGBoost settings and outcome-specific samples.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier


SEED = 2026
N_BOOT = 2000
FEATURE_ROOT = Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14")
OUTPUT_DIR = Path(r"C:\Users\SaberShuo\Documents\SDI\analysis\sdi3d_vs_sdi6_20260822")
CLINICAL = {
    "CFS": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\cfs\cfs-visit5-dataset-0.7.0.csv"),
    "MrOS": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\mros\mros-visit1-dataset-0.6.0.csv"),
    "SHHS": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs1-dataset-0.20.0.csv"),
}
SHHS_CVD = Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs-cvd-summary-dataset-0.19.0.csv")

SDI6 = ["AP", "RB", "MDR", "PR", "CV", "SK"]
HAND8 = ["SSTR", "CWSD", "TDTL", "BRAI", "STDE", "TOS", "TCI", "SRI"]
EMBED8 = [f"feature_{index}" for index in range(1, 9)]
FEATURE_SETS = {
    "SDI-6": SDI6,
    "SDI-3D handcrafted-8": HAND8,
    "SDI-3D embedding-8": EMBED8,
    "SDI-3D fused-16": HAND8 + EMBED8,
    "SDI-6 + SDI-3D-16": SDI6 + HAND8 + EMBED8,
}
MAIN_COMPARATORS = [
    "SDI-3D handcrafted-8",
    "SDI-3D embedding-8",
    "SDI-3D fused-16",
    "SDI-6 + SDI-3D-16",
]
OUTCOME_ORDER = ["OSA", "Insomnia", "Hypertension", "Diabetes", "CVD"]


def normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().str.replace(r"\.0$", "", regex=True)


def extract_feature_id(series: pd.Series, cohort: str) -> pd.Series:
    patterns = {
        "CFS": r"cfs-visit5-([^_\.]+)",
        "MrOS": r"mros-visit1-([^_\.]+)",
        "SHHS": r"shhs1-([^_\.]+)",
    }
    result = normalize(series).str.extract(patterns[cohort], expand=False)
    if result.isna().any():
        raise ValueError(f"{cohort}: could not parse {int(result.isna().sum())} feature IDs")
    return result


def numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(np.isfinite(values))


def strict_binary(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.where(values.isin([0, 1]))


def threshold_binary(series: pd.Series, threshold: float) -> pd.Series:
    values = numeric(series)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[values.notna()] = (values.loc[values.notna()] >= threshold).astype(float)
    return result


def load_labels(cohort: str) -> pd.DataFrame:
    if cohort == "CFS":
        cols = ["nsrrid", "cvd", "htn", "diadiag", "ahi_a0h4", "easlp", "difbak"]
        source = pd.read_csv(CLINICAL[cohort], usecols=cols, low_memory=False)
        result = pd.DataFrame({"subject_id": normalize(source["nsrrid"])})
        result["CVD"] = strict_binary(source["cvd"])
        result["Hypertension"] = strict_binary(source["htn"])
        result["Diabetes"] = strict_binary(source["diadiag"])
        result["OSA"] = threshold_binary(source["ahi_a0h4"], 5.0)
        easy = numeric(source["easlp"])
        return_difficulty = numeric(source["difbak"])
        result["Insomnia"] = np.nan
        result.loc[(easy <= 3) & (return_difficulty == 0), "Insomnia"] = 0.0
        result.loc[(easy > 4) & (return_difficulty == 1), "Insomnia"] = 1.0
        return result

    if cohort == "MrOS":
        cols = ["nsrrid", "mhbp", "mhdiab", "poahi3", "poxfall"]
        source = pd.read_csv(CLINICAL[cohort], usecols=cols, low_memory=False)
        result = pd.DataFrame({"subject_id": normalize(source["nsrrid"])})
        result["Hypertension"] = strict_binary(source["mhbp"])
        result["Diabetes"] = strict_binary(source["mhdiab"])
        result["OSA"] = threshold_binary(source["poahi3"], 5.0)
        latency = numeric(source["poxfall"])
        result["Insomnia"] = np.nan
        result.loc[latency.notna(), "Insomnia"] = (latency.loc[latency.notna()] > 40).astype(float)
        return result

    cols = ["nsrrid", "htnderv_s1", "parrptdiab", "ahi_a0h4", "diffa10"]
    source = pd.read_csv(CLINICAL[cohort], usecols=cols, low_memory=False)
    cvd = pd.read_csv(SHHS_CVD, usecols=["nsrrid", "any_cvd"], low_memory=False)
    source["subject_id"] = normalize(source["nsrrid"])
    cvd["subject_id"] = normalize(cvd["nsrrid"])
    source = source.merge(cvd[["subject_id", "any_cvd"]], on="subject_id", how="left", validate="one_to_one")
    result = pd.DataFrame({"subject_id": source["subject_id"]})
    result["CVD"] = strict_binary(source["any_cvd"])
    result["Hypertension"] = strict_binary(source["htnderv_s1"])
    result["Diabetes"] = strict_binary(source["parrptdiab"])
    result["OSA"] = threshold_binary(source["ahi_a0h4"], 5.0)
    result["Insomnia"] = strict_binary(source["diffa10"])
    return result


def load_cohort(cohort: str) -> tuple[pd.DataFrame, dict[str, object]]:
    file_names = {
        "CFS": ("cfs_SDI_14_Features.csv", "cfs_embeddings_8d.csv"),
        "MrOS": ("mros_SDI_14_Features.csv", "mros_embeddings_8d.csv"),
        "SHHS": ("shhs_SDI_14_Features.csv", "shhs_external_embeddings_8d.csv"),
    }
    sdi_name, embedding_name = file_names[cohort]
    sdi = pd.read_csv(FEATURE_ROOT / sdi_name, low_memory=False)
    embedding = pd.read_csv(FEATURE_ROOT / embedding_name, low_memory=False)
    sdi["subject_id"] = extract_feature_id(sdi["NSRRID"], cohort)
    embedding["subject_id"] = extract_feature_id(embedding["night_id"], cohort)
    labels = load_labels(cohort)
    for name, data in [("SDI", sdi), ("embedding", embedding), ("clinical", labels)]:
        if data["subject_id"].duplicated().any():
            raise ValueError(f"{cohort}: duplicate subject IDs in {name}")
    merged_features = sdi.merge(
        embedding[["subject_id", "dataset", "split", *EMBED8]],
        on="subject_id", how="inner", validate="one_to_one",
    )
    merged = merged_features.merge(labels, on="subject_id", how="inner", validate="one_to_one")
    audit = {
        "cohort": cohort,
        "sdi_rows": len(sdi),
        "embedding_rows": len(embedding),
        "clinical_rows": len(labels),
        "feature_matched_rows": len(merged_features),
        "final_matched_rows": len(merged),
        "missing_feature_cells": int(merged[[*SDI6, *HAND8, *EMBED8]].isna().sum().sum()),
    }
    return merged, audit


def build_classifier(positive_weight: float, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=positive_weight,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def cross_fitted_predictions(
    data: pd.DataFrame,
    outcome: str,
    feature_columns: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    x = data[feature_columns].to_numpy(float)
    y = data[outcome].to_numpy(int)
    predictions = np.full(len(data), np.nan, dtype=float)
    for fold, (train_index, test_index) in enumerate(splits):
        imputer = SimpleImputer(strategy="median")
        train_x = imputer.fit_transform(x[train_index])
        test_x = imputer.transform(x[test_index])
        train_y = y[train_index]
        positives = int(train_y.sum())
        negatives = len(train_y) - positives
        positive_weight = negatives / positives if positives else 1.0
        model = build_classifier(positive_weight, seed + fold)
        model.fit(train_x, train_y)
        predictions[test_index] = model.predict_proba(test_x)[:, 1]
    if np.isnan(predictions).any():
        raise RuntimeError("Cross-fitting left missing predictions")
    return predictions


def bootstrap_metric_and_deltas(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    seed: int,
    n_boot: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    model_names = list(predictions)
    auc_samples = {name: [] for name in model_names}
    delta_samples = {name: [] for name in model_names if name != "SDI-6"}
    n = len(y)
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        sampled_y = y[index]
        if np.unique(sampled_y).size < 2:
            continue
        boot_aucs = {
            name: roc_auc_score(sampled_y, values[index])
            for name, values in predictions.items()
        }
        for name, value in boot_aucs.items():
            auc_samples[name].append(value)
        baseline = boot_aucs["SDI-6"]
        for name in delta_samples:
            delta_samples[name].append(boot_aucs[name] - baseline)
    auc_intervals = {
        name: tuple(np.percentile(values, [2.5, 97.5]))
        for name, values in auc_samples.items()
    }
    delta_stats: dict[str, dict[str, float]] = {}
    for name, values in delta_samples.items():
        array = np.asarray(values)
        low, high = np.percentile(array, [2.5, 97.5])
        count = len(array)
        lower_tail = (np.sum(array <= 0) + 1) / (count + 1)
        upper_tail = (np.sum(array >= 0) + 1) / (count + 1)
        delta_stats[name] = {
            "delta_ci_low": float(low),
            "delta_ci_high": float(high),
            "p_value": float(min(1.0, 2 * min(lower_tail, upper_tail))),
            "bootstrap_valid": count,
        }
    return auc_intervals, delta_stats


def analyze_within_cohort(
    cohort: str,
    data: pd.DataFrame,
    outcome: str,
    outcome_index: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    analysis = data.loc[data[outcome].notna(), ["subject_id", outcome, *SDI6, *HAND8, *EMBED8]].copy()
    analysis[outcome] = analysis[outcome].astype(int)
    y = analysis[outcome].to_numpy(int)
    n_splits = min(5, int(y.sum()), int((1 - y).sum()))
    if n_splits < 3:
        raise ValueError(f"{cohort} {outcome}: too few cases for stratified CV")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + outcome_index)
    splits = list(splitter.split(np.zeros(len(y)), y))
    predictions: dict[str, np.ndarray] = {}
    for set_index, (name, columns) in enumerate(FEATURE_SETS.items()):
        predictions[name] = cross_fitted_predictions(
            analysis, outcome, columns, splits, SEED + 1000 * outcome_index + 100 * set_index
        )
    auc_intervals, delta_stats = bootstrap_metric_and_deltas(
        y, predictions, SEED + 10000 + outcome_index, N_BOOT
    )
    metric_rows = []
    comparison_rows = []
    baseline_auc = roc_auc_score(y, predictions["SDI-6"])
    for name, values in predictions.items():
        auc = roc_auc_score(y, values)
        low, high = auc_intervals[name]
        metric_rows.append({
            "analysis": "cohort-specific 5-fold OOF",
            "cohort": cohort,
            "outcome": outcome,
            "feature_set": name,
            "n": len(y),
            "events": int(y.sum()),
            "event_rate": float(y.mean()),
            "auc": auc,
            "auc_ci_low": low,
            "auc_ci_high": high,
            "average_precision": average_precision_score(y, values),
            "brier_score": brier_score_loss(y, values),
            "n_features": len(FEATURE_SETS[name]),
        })
        if name != "SDI-6":
            delta = auc - baseline_auc
            comparison_rows.append({
                "analysis": "cohort-specific 5-fold OOF",
                "cohort": cohort,
                "outcome": outcome,
                "comparator": name,
                "baseline": "SDI-6",
                "n": len(y),
                "events": int(y.sum()),
                "baseline_auc": baseline_auc,
                "comparator_auc": auc,
                "delta_auc": delta,
                "delta_auc_pp": 100 * delta,
                "relative_gain_pct": 100 * delta / baseline_auc,
                **delta_stats[name],
            })
    prediction_frame = pd.DataFrame({
        "analysis": "cohort-specific 5-fold OOF",
        "cohort": cohort,
        "outcome": outcome,
        "subject_id": analysis["subject_id"].to_numpy(),
        "label": y,
        **{name: values for name, values in predictions.items()},
    })
    return metric_rows, comparison_rows, prediction_frame


def fit_external_predictions(
    development: pd.DataFrame,
    external: pd.DataFrame,
    outcome: str,
    feature_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = development.loc[development[outcome].notna()].copy()
    test = external.loc[external[outcome].notna()].copy()
    train_y = train[outcome].astype(int).to_numpy()
    test_y = test[outcome].astype(int).to_numpy()
    imputer = SimpleImputer(strategy="median")
    train_x = imputer.fit_transform(train[feature_columns].to_numpy(float))
    test_x = imputer.transform(test[feature_columns].to_numpy(float))
    positives = int(train_y.sum())
    positive_weight = (len(train_y) - positives) / positives if positives else 1.0
    model = build_classifier(positive_weight, seed)
    model.fit(train_x, train_y)
    return test["subject_id"].to_numpy(), test_y, model.predict_proba(test_x)[:, 1]


def analyze_external(
    development: pd.DataFrame,
    shhs: pd.DataFrame,
    outcome: str,
    outcome_index: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    predictions: dict[str, np.ndarray] = {}
    subject_ids = None
    y = None
    for set_index, (name, columns) in enumerate(FEATURE_SETS.items()):
        ids, labels, values = fit_external_predictions(
            development, shhs, outcome, columns, SEED + 50000 + 1000 * outcome_index + set_index
        )
        subject_ids, y = ids, labels
        predictions[name] = values
    assert subject_ids is not None and y is not None
    auc_intervals, delta_stats = bootstrap_metric_and_deltas(
        y, predictions, SEED + 60000 + outcome_index, N_BOOT
    )
    metric_rows = []
    comparison_rows = []
    baseline_auc = roc_auc_score(y, predictions["SDI-6"])
    development_valid = int(development[outcome].notna().sum())
    development_events = int(development.loc[development[outcome].notna(), outcome].sum())
    for name, values in predictions.items():
        auc = roc_auc_score(y, values)
        low, high = auc_intervals[name]
        metric_rows.append({
            "analysis": "CFS+MrOS development to SHHS external test",
            "cohort": "SHHS external",
            "outcome": outcome,
            "feature_set": name,
            "development_n": development_valid,
            "development_events": development_events,
            "n": len(y),
            "events": int(y.sum()),
            "event_rate": float(y.mean()),
            "auc": auc,
            "auc_ci_low": low,
            "auc_ci_high": high,
            "average_precision": average_precision_score(y, values),
            "brier_score": brier_score_loss(y, values),
            "n_features": len(FEATURE_SETS[name]),
        })
        if name != "SDI-6":
            delta = auc - baseline_auc
            comparison_rows.append({
                "analysis": "CFS+MrOS development to SHHS external test",
                "cohort": "SHHS external",
                "outcome": outcome,
                "comparator": name,
                "baseline": "SDI-6",
                "development_n": development_valid,
                "development_events": development_events,
                "n": len(y),
                "events": int(y.sum()),
                "baseline_auc": baseline_auc,
                "comparator_auc": auc,
                "delta_auc": delta,
                "delta_auc_pp": 100 * delta,
                "relative_gain_pct": 100 * delta / baseline_auc,
                **delta_stats[name],
            })
    prediction_frame = pd.DataFrame({
        "analysis": "CFS+MrOS development to SHHS external test",
        "cohort": "SHHS external",
        "outcome": outcome,
        "subject_id": subject_ids,
        "label": y,
        **{name: values for name, values in predictions.items()},
    })
    return metric_rows, comparison_rows, prediction_frame


def add_fdr(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["q_value_bh"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result["fdr_significant"] = result["q_value_bh"] < 0.05
    result["direction"] = np.where(result["delta_auc"] > 0, "higher", np.where(result["delta_auc"] < 0, "lower", "equal"))
    return result


def plot_delta_heatmap(comparisons: pd.DataFrame, title: str, output_stem: Path) -> None:
    cohorts = list(dict.fromkeys(comparisons["cohort"]))
    outcomes = [outcome for outcome in OUTCOME_ORDER if outcome in set(comparisons["outcome"])]
    comparators = [name for name in MAIN_COMPARATORS if name in set(comparisons["comparator"])]
    row_keys = [(cohort, outcome) for cohort in cohorts for outcome in outcomes if ((comparisons["cohort"] == cohort) & (comparisons["outcome"] == outcome)).any()]
    matrix = np.full((len(row_keys), len(comparators)), np.nan)
    annotations = np.full(matrix.shape, "", dtype=object)
    for i, (cohort, outcome) in enumerate(row_keys):
        for j, comparator in enumerate(comparators):
            subset = comparisons.loc[
                (comparisons["cohort"] == cohort)
                & (comparisons["outcome"] == outcome)
                & (comparisons["comparator"] == comparator)
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            matrix[i, j] = row["delta_auc_pp"]
            stars = "*" if row["q_value_bh"] < 0.05 else ""
            annotations[i, j] = f"{row['delta_auc_pp']:+.1f}{stars}"
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    fig_height = max(4.2, 0.45 * len(row_keys) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                color = "white" if abs(matrix[i, j]) > 0.55 * limit else "black"
                ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(comparators)), [name.replace("SDI-3D ", "") for name in comparators], rotation=20, ha="right")
    ax.set_yticks(range(len(row_keys)), [f"{cohort} | {outcome}" for cohort, outcome in row_keys])
    ax.set_title(title, weight="bold", pad=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("AUC difference vs SDI-6 (percentage points)")
    fig.text(0.01, 0.015, "* Benjamini-Hochberg q<0.05", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_fused_forest(comparisons: pd.DataFrame, title: str, output_stem: Path) -> None:
    data = comparisons.loc[comparisons["comparator"] == "SDI-3D fused-16"].copy()
    cohort_order = list(dict.fromkeys(data["cohort"]))
    data["cohort_order"] = data["cohort"].map({name: i for i, name in enumerate(cohort_order)})
    data["outcome_order"] = data["outcome"].map({name: i for i, name in enumerate(OUTCOME_ORDER)})
    data = data.sort_values(["cohort_order", "outcome_order"]).reset_index(drop=True)
    y = np.arange(len(data))[::-1]
    estimate = data["delta_auc_pp"].to_numpy(float)
    low = 100 * data["delta_ci_low"].to_numpy(float)
    high = 100 * data["delta_ci_high"].to_numpy(float)
    colors = np.where(data["q_value_bh"].to_numpy(float) < 0.05, "#B2182B", "#7F8C8D")
    fig, ax = plt.subplots(figsize=(8.6, max(3.8, 0.42 * len(data) + 1.4)))
    for index in range(len(data)):
        ax.errorbar(
            estimate[index], y[index],
            xerr=[[estimate[index] - low[index]], [high[index] - estimate[index]]],
            fmt="o", color=colors[index], ecolor=colors[index], capsize=3,
            markersize=5.5, linewidth=1.2,
        )
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{row.cohort} | {row.outcome}" for row in data.itertuples()])
    ax.set_xlabel("AUC difference: SDI-3D fused-16 minus SDI-6 (percentage points)")
    ax.set_title(title, weight="bold", pad=10)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.015, "Red: Benjamini-Hochberg q<0.05; gray: q>=0.05", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohorts: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, object]] = []
    for cohort in ["CFS", "MrOS", "SHHS"]:
        cohorts[cohort], audit = load_cohort(cohort)
        audits.append(audit)
        print(f"Loaded {cohort}: {len(cohorts[cohort])} matched participants", flush=True)

    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    outcome_index = 0
    for cohort in ["CFS", "MrOS", "SHHS"]:
        for outcome in OUTCOME_ORDER:
            if outcome not in cohorts[cohort].columns:
                continue
            print(f"Cross-fitting {cohort} | {outcome}", flush=True)
            metrics, comparisons, predictions = analyze_within_cohort(
                cohort, cohorts[cohort], outcome, outcome_index
            )
            metric_rows.extend(metrics)
            comparison_rows.extend(comparisons)
            prediction_frames.append(predictions)
            outcome_index += 1

    within_metrics = pd.DataFrame(metric_rows)
    within_comparisons = add_fdr(pd.DataFrame(comparison_rows))
    within_predictions = pd.concat(prediction_frames, ignore_index=True)
    within_metrics.to_csv(OUTPUT_DIR / "within_cohort_oof_metrics.csv", index=False)
    within_comparisons.to_csv(OUTPUT_DIR / "within_cohort_auc_comparisons_vs_sdi6.csv", index=False)
    within_predictions.to_csv(OUTPUT_DIR / "within_cohort_oof_predictions.csv", index=False)

    development = pd.concat([cohorts["CFS"], cohorts["MrOS"]], ignore_index=True, sort=False)
    external_metric_rows: list[dict[str, object]] = []
    external_comparison_rows: list[dict[str, object]] = []
    external_prediction_frames: list[pd.DataFrame] = []
    for index, outcome in enumerate(["OSA", "Insomnia", "Hypertension", "Diabetes"]):
        print(f"External testing CFS+MrOS -> SHHS | {outcome}", flush=True)
        metrics, comparisons, predictions = analyze_external(
            development, cohorts["SHHS"], outcome, index
        )
        external_metric_rows.extend(metrics)
        external_comparison_rows.extend(comparisons)
        external_prediction_frames.append(predictions)
    external_metrics = pd.DataFrame(external_metric_rows)
    external_comparisons = add_fdr(pd.DataFrame(external_comparison_rows))
    external_predictions = pd.concat(external_prediction_frames, ignore_index=True)
    external_metrics.to_csv(OUTPUT_DIR / "external_shhs_metrics.csv", index=False)
    external_comparisons.to_csv(OUTPUT_DIR / "external_shhs_auc_comparisons_vs_sdi6.csv", index=False)
    external_predictions.to_csv(OUTPUT_DIR / "external_shhs_predictions.csv", index=False)

    audit_frame = pd.DataFrame(audits)
    outcome_audit = []
    for cohort, data in cohorts.items():
        for outcome in OUTCOME_ORDER:
            if outcome not in data:
                continue
            valid = data[outcome].dropna()
            outcome_audit.append({
                "cohort": cohort,
                "outcome": outcome,
                "n": len(valid),
                "events": int(valid.sum()),
                "event_rate": float(valid.mean()),
                "missing_labels": int(data[outcome].isna().sum()),
            })
    audit_frame.to_csv(OUTPUT_DIR / "input_matching_audit.csv", index=False)
    pd.DataFrame(outcome_audit).to_csv(OUTPUT_DIR / "outcome_audit.csv", index=False)

    plot_delta_heatmap(
        within_comparisons,
        "Cohort-specific cross-validated AUC differences vs SDI-6",
        OUTPUT_DIR / "within_cohort_auc_delta_heatmap",
    )
    plot_delta_heatmap(
        external_comparisons,
        "External SHHS AUC differences vs SDI-6",
        OUTPUT_DIR / "external_shhs_auc_delta_heatmap",
    )
    plot_fused_forest(
        within_comparisons,
        "SDI-3D fused representation vs SDI-6",
        OUTPUT_DIR / "within_cohort_fused16_auc_delta_forest",
    )
    plot_fused_forest(
        external_comparisons,
        "External SHHS: SDI-3D fused representation vs SDI-6",
        OUTPUT_DIR / "external_shhs_fused16_auc_delta_forest",
    )

    summary = {
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "classifier": {
            "type": "XGBClassifier",
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "class_weighting": "outer-training negative/positive ratio",
        },
        "within_cohort_validation": "5-fold stratified OOF; identical folds across feature sets",
        "external_validation": "fit on pooled CFS+MrOS and test once in SHHS",
        "feature_sets": FEATURE_SETS,
        "fdr_scope": {
            "within_cohort": len(within_comparisons),
            "external_shhs": len(external_comparisons),
        },
    }
    with (OUTPUT_DIR / "analysis_specification.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Outputs written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
