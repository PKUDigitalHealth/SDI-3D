"""Outcome associations and forest plots for SDI cluster assignments.

Each cohort's outcomes are analyzed separately. Missing/unknown clinical codes
remain missing. No clinical outcome is used to create or orient clusters.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning


BASE_FEATURES = ["SSTR", "CWSD", "TDTL", "BRAI", "STDE", "TOS", "TCI", "SRI"]
EMBEDDING_FEATURES = [f"NE{index}" for index in range(1, 9)]
FEATURES = BASE_FEATURES.copy()
FEATURE_LABELS = {feature: feature for feature in FEATURES}
FEATURE_SECTION_LABEL = "SDI Features"
CONTINUOUS_LABELS = {
    "age_clean": "Age",
    "bmi_clean": "BMI",
    "se": "SE",
    "sl": "SL",
}
OUTCOME_LABELS = {
    "osa": "OSA",
    "poor_sleep": "Poor Sleep Quality",
    "insomnia": "Insomnia",
    "diabetes": "Diabetes",
    "cvd": "CVD",
    "hypertension": "Hypertension",
}

DEFAULT_CLUSTER_DIR = Path("analysis") / "sdi_8feature_clustering"
DEFAULT_OUTPUT_DIR = Path("analysis") / "sdi_8feature_outcomes"
DEFAULT_CLINICAL_FILES = {
    "cfs": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\cfs\cfs-visit5-dataset-0.7.0.csv"),
    "mros": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\mros\mros-visit1-dataset-0.6.0.csv"),
    "shhs": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs1-dataset-0.20.0.csv"),
    "shhs_cvd": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs-cvd-summary-dataset-0.19.0.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Associate independent two-cluster SDI models with clinical outcomes."
    )
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cfs-clinical", type=Path, default=DEFAULT_CLINICAL_FILES["cfs"])
    parser.add_argument("--mros-clinical", type=Path, default=DEFAULT_CLINICAL_FILES["mros"])
    parser.add_argument("--shhs-clinical", type=Path, default=DEFAULT_CLINICAL_FILES["shhs"])
    parser.add_argument("--shhs-cvd", type=Path, default=DEFAULT_CLINICAL_FILES["shhs_cvd"])
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-embedding-features",
        action="store_true",
        help="Include NE1-NE8 in feature-effect rows of the forest plots.",
    )
    parser.add_argument(
        "--cluster-strategy",
        default="three cohorts clustered independently",
        help="Provenance label written to the quality report.",
    )
    parser.add_argument(
        "--exclude-ambiguous",
        action="store_true",
        help="Exclude cluster assignments with posterior probability below 0.70.",
    )
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").astype(float)
    return result.where(np.isfinite(result))


def normalize_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.str.replace(r"\.0$", "", regex=True)


def strict_binary(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.where(values.isin([0, 1]))


def threshold_binary(series: pd.Series, threshold: float) -> pd.Series:
    values = numeric(series)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[values.notna()] = (values.loc[values.notna()] >= threshold).astype(float)
    return result


def strict_sleep_quality(
    first: pd.Series,
    second: pd.Series,
    good_max: float | None = None,
    good_min: float | None = None,
    poor_min_first: float | None = None,
    poor_min_second: float | None = None,
    poor_max: float | None = None,
) -> pd.Series:
    first_num, second_num = numeric(first), numeric(second)
    result = pd.Series(np.nan, index=first.index, dtype=float)
    valid = first_num.notna() & second_num.notna()
    if good_max is not None:
        good = valid & (first_num <= good_max) & (second_num <= good_max)
    else:
        good = valid & (first_num >= good_min) & (second_num >= good_min)
    if poor_max is not None:
        poor = valid & (first_num <= poor_max) & (second_num <= poor_max)
    else:
        poor = valid & (first_num >= poor_min_first) & (second_num >= poor_min_second)
    result.loc[good] = 0.0
    result.loc[poor] = 1.0
    return result


def load_cfs(path: Path) -> pd.DataFrame:
    columns = [
        "nsrrid", "age", "bmi", "sex", "cvd", "htn", "diadiag", "ahi_a0h4",
        "slpqua", "desslp", "easlp", "difbak", "slpeffp", "slplatp",
    ]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    result = pd.DataFrame({"subject_id": normalize_id(data["nsrrid"])})
    result["age_clean"] = numeric(data["age"])
    result["bmi_clean"] = numeric(data["bmi"])
    result["sex_clean"] = strict_binary(data["sex"])
    result["se"] = numeric(data["slpeffp"])
    result["sl"] = numeric(data["slplatp"])
    result["cvd"] = strict_binary(data["cvd"])
    result["hypertension"] = strict_binary(data["htn"])
    result["diabetes"] = strict_binary(data["diadiag"])
    result["osa"] = threshold_binary(data["ahi_a0h4"], 5.0)
    result["poor_sleep"] = strict_sleep_quality(
        data["slpqua"], data["desslp"],
        good_max=2, poor_min_first=5, poor_min_second=4,
    )
    easy, return_difficulty = numeric(data["easlp"]), numeric(data["difbak"])
    result["insomnia"] = np.nan
    result.loc[(easy <= 3) & (return_difficulty == 0), "insomnia"] = 0.0
    result.loc[(easy > 4) & (return_difficulty == 1), "insomnia"] = 1.0
    return result


def load_mros(path: Path) -> pd.DataFrame:
    columns = [
        "nsrrid", "vsage1", "hwbmi", "gender", "mhbp", "mhdiab", "poahi3",
        "poxqual1", "poxqual3", "poxfall", "poslpeff", "posllatp",
    ]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    result = pd.DataFrame({"subject_id": normalize_id(data["nsrrid"])})
    result["age_clean"] = numeric(data["vsage1"])
    result["bmi_clean"] = numeric(data["hwbmi"])
    gender = numeric(data["gender"])
    result["sex_clean"] = gender.map({1.0: 0.0, 2.0: 1.0})
    result["se"] = numeric(data["poslpeff"])
    result["sl"] = numeric(data["posllatp"])
    result["hypertension"] = strict_binary(data["mhbp"])
    result["diabetes"] = strict_binary(data["mhdiab"])
    result["osa"] = threshold_binary(data["poahi3"], 5.0)
    result["poor_sleep"] = strict_sleep_quality(
        data["poxqual1"], data["poxqual3"], good_min=4, poor_max=2,
    )
    fall_latency = numeric(data["poxfall"])
    result["insomnia"] = np.nan
    result.loc[fall_latency.notna(), "insomnia"] = (
        fall_latency.loc[fall_latency.notna()] > 40
    ).astype(float)
    return result


def load_shhs(path: Path, cvd_path: Path) -> pd.DataFrame:
    columns = [
        "nsrrid", "age_s1", "bmi_s1", "gender", "htnderv_s1", "parrptdiab",
        "ahi_a0h4", "ltdp10", "rest10", "diffa10", "slpeffp", "slplatp",
    ]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    cvd = pd.read_csv(cvd_path, usecols=["nsrrid", "any_cvd"], low_memory=False)
    data["subject_id"] = normalize_id(data["nsrrid"])
    cvd["subject_id"] = normalize_id(cvd["nsrrid"])
    if data["subject_id"].duplicated().any() or cvd["subject_id"].duplicated().any():
        raise ValueError("SHHS clinical or CVD data contain duplicate subject IDs")
    data = data.merge(cvd[["subject_id", "any_cvd"]], on="subject_id", how="left", validate="one_to_one")
    result = pd.DataFrame({"subject_id": data["subject_id"]})
    result["age_clean"] = numeric(data["age_s1"])
    result["bmi_clean"] = numeric(data["bmi_s1"])
    gender = numeric(data["gender"])
    result["sex_clean"] = gender.map({1.0: 1.0, 2.0: 0.0})
    result["se"] = numeric(data["slpeffp"])
    result["sl"] = numeric(data["slplatp"])
    result["cvd"] = strict_binary(data["any_cvd"])
    result["hypertension"] = strict_binary(data["htnderv_s1"])
    result["diabetes"] = strict_binary(data["parrptdiab"])
    result["osa"] = threshold_binary(data["ahi_a0h4"], 5.0)
    result["poor_sleep"] = strict_sleep_quality(
        data["ltdp10"], data["rest10"], good_min=4, poor_max=2,
    )
    result["insomnia"] = strict_binary(data["diffa10"])
    return result


def load_clusters(cluster_dir: Path, cohort: str, exclude_ambiguous: bool) -> pd.DataFrame:
    path = cluster_dir / f"{cohort}_cluster_assignments.csv"
    data = pd.read_csv(path, low_memory=False)
    required = ["subject_id", "cluster", "cluster_probability", "cluster_ambiguous", *FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{cohort}: cluster assignment file is missing {missing}")
    data = data.copy()
    data["subject_id"] = normalize_id(data["subject_id"])
    if data["subject_id"].duplicated().any():
        raise ValueError(f"{cohort}: duplicated subject IDs in cluster assignments")
    if exclude_ambiguous:
        ambiguous = data["cluster_ambiguous"].astype(str).str.lower().isin(["true", "1"])
        data = data.loc[~ambiguous].copy()
    return data


def merge_cohort(cluster_data: pd.DataFrame, clinical_data: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if clinical_data["subject_id"].duplicated().any():
        raise ValueError(f"{cohort}: duplicated subject IDs in clinical data")
    merged = cluster_data.merge(clinical_data, on="subject_id", how="inner", validate="one_to_one")
    merged.insert(0, "analysis_cohort", cohort)
    return merged


def cohens_d(group1: np.ndarray, group0: np.ndarray) -> float:
    n1, n0 = len(group1), len(group0)
    if n1 < 2 or n0 < 2:
        return np.nan
    pooled_variance = (
        (n1 - 1) * np.var(group1, ddof=1) + (n0 - 1) * np.var(group0, ddof=1)
    ) / (n1 + n0 - 2)
    if pooled_variance <= 0 or not np.isfinite(pooled_variance):
        return np.nan
    return (np.mean(group1) - np.mean(group0)) / np.sqrt(pooled_variance)


def bootstrap_cohens_d(
    data: pd.DataFrame,
    variable: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    group1 = numeric(data.loc[data["cluster"] == 1, variable]).dropna().to_numpy()
    group0 = numeric(data.loc[data["cluster"] == 0, variable]).dropna().to_numpy()
    estimate = cohens_d(group1, group0)
    bootstrap_values = []
    for _ in range(n_boot):
        sample1 = rng.choice(group1, size=len(group1), replace=True)
        sample0 = rng.choice(group0, size=len(group0), replace=True)
        value = cohens_d(sample1, sample0)
        if np.isfinite(value):
            bootstrap_values.append(value)
    if bootstrap_values:
        low, high = np.percentile(bootstrap_values, [2.5, 97.5])
    else:
        low = high = np.nan
    return {
        "Variable": CONTINUOUS_LABELS.get(variable, FEATURE_LABELS.get(variable, variable)),
        "SourceVariable": variable,
        "Type": "d",
        "Val": estimate,
        "Low": low,
        "High": high,
        "N": len(group1) + len(group0),
        "Events": np.nan,
        "Cluster0Rate": np.nan,
        "Cluster1Rate": np.nan,
        "Covariates": "",
        "BootstrapValid": len(bootstrap_values),
        "Status": "ok" if np.isfinite(estimate) else "not_estimable",
    }


def fit_logistic(data: pd.DataFrame, outcome: str, covariates: list[str]):
    columns = [outcome, "cluster", *covariates]
    temp = data[columns].copy()
    for column in columns:
        temp[column] = numeric(temp[column])
    temp = temp.dropna()
    active_covariates = [
        column for column in covariates if temp[column].nunique(dropna=True) > 1
    ]
    temp = temp[[outcome, "cluster", *active_covariates]]
    if len(temp) < 20 or temp[outcome].nunique() < 2 or temp["cluster"].nunique() < 2:
        return None, temp, active_covariates
    design = sm.add_constant(temp[["cluster", *active_covariates]], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        model = sm.GLM(temp[outcome], design, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
    return model, temp, active_covariates


def bootstrap_adjusted_or(
    data: pd.DataFrame,
    outcome: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    covariates = ["age_clean", "bmi_clean", "sex_clean"]
    model, temp, active_covariates = fit_logistic(data, outcome, covariates)
    label = OUTCOME_LABELS[outcome]
    if model is None:
        return {
            "Variable": label, "SourceVariable": outcome, "Type": "OR",
            "Val": np.nan, "Low": np.nan, "High": np.nan, "N": len(temp),
            "Events": int(temp[outcome].sum()) if outcome in temp else 0,
            "Cluster0Rate": np.nan, "Cluster1Rate": np.nan,
            "Covariates": ",".join(active_covariates), "BootstrapValid": 0,
            "Status": "not_estimable",
        }
    log_or = float(model.params["cluster"])
    bootstrap_log_ors: list[float] = []
    for _ in range(n_boot):
        sample = temp.iloc[rng.integers(0, len(temp), size=len(temp))].copy()
        if sample[outcome].nunique() < 2 or sample["cluster"].nunique() < 2:
            continue
        design = sm.add_constant(sample[["cluster", *active_covariates]], has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = sm.GLM(
                    sample[outcome], design, family=sm.families.Binomial()
                ).fit(maxiter=60, disp=0)
            value = float(fitted.params["cluster"])
            if np.isfinite(value):
                bootstrap_log_ors.append(value)
        except Exception:
            continue
    if len(bootstrap_log_ors) >= max(50, int(0.5 * n_boot)):
        low_log, high_log = np.percentile(bootstrap_log_ors, [2.5, 97.5])
        status = "ok"
    else:
        low_log, high_log = model.conf_int().loc["cluster"].to_numpy()
        status = "model_ci_fallback"
    rates = temp.groupby("cluster")[outcome].mean()
    return {
        "Variable": label,
        "SourceVariable": outcome,
        "Type": "OR",
        "Val": float(np.exp(log_or)),
        "Low": float(np.exp(low_log)),
        "High": float(np.exp(high_log)),
        "N": len(temp),
        "Events": int(temp[outcome].sum()),
        "Cluster0Rate": float(rates.get(0, np.nan)),
        "Cluster1Rate": float(rates.get(1, np.nan)),
        "Covariates": ",".join(active_covariates),
        "BootstrapValid": len(bootstrap_log_ors),
        "Status": status,
    }


def analyze_cohort(data: pd.DataFrame, cohort: str, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    variables = ["age_clean", "bmi_clean", "se", "sl", *FEATURES]
    for index, variable in enumerate(variables):
        rows.append(
            bootstrap_cohens_d(data, variable, n_boot, np.random.default_rng(seed + index))
        )
    outcomes = ["osa", "poor_sleep", "insomnia", "diabetes", "hypertension"]
    if cohort != "mros":
        outcomes.insert(4, "cvd")
    for index, outcome in enumerate(outcomes):
        rows.append(
            bootstrap_adjusted_or(
                data, outcome, n_boot, np.random.default_rng(seed + 100 + index)
            )
        )
    result = pd.DataFrame(rows)
    result.insert(0, "Cohort", cohort)
    return result


def ordered_plot_rows(stats: pd.DataFrame, cohort: str) -> list[dict[str, object]]:
    lookup = {row["Variable"]: row.to_dict() for _, row in stats.iterrows()}
    rows: list[dict[str, object]] = []
    outcome_order = ["Hypertension"]
    if cohort != "mros":
        outcome_order.append("CVD")
    outcome_order += ["Diabetes", "Insomnia", "Poor Sleep Quality", "OSA"]
    for label in outcome_order:
        if label in lookup:
            rows.append(lookup[label])
    rows.append({"Variable": "Outcomes", "Type": "header"})
    for label in reversed(FEATURES):
        rows.append(lookup[label])
    rows.append({"Variable": FEATURE_SECTION_LABEL, "Type": "header"})
    for label in ["SL", "SE", "BMI", "Age"]:
        if label in lookup:
            rows.append(lookup[label])
    return rows


def plot_forest(stats: pd.DataFrame, cohort: str, output_dir: Path) -> None:
    rows = ordered_plot_rows(stats, cohort)
    plot_data = pd.DataFrame(rows)
    y = np.arange(len(plot_data))
    fig, ax_or = plt.subplots(figsize=(8, 10.5))
    ax_d = ax_or.twiny()

    or_mask = plot_data["Type"].eq("OR") & plot_data["Val"].notna()
    d_mask = plot_data["Type"].eq("d") & plot_data["Val"].notna()
    if or_mask.any():
        or_values = plot_data.loc[or_mask, "Val"].astype(float)
        or_low = plot_data.loc[or_mask, "Low"].astype(float)
        or_high = plot_data.loc[or_mask, "High"].astype(float)
        ax_or.errorbar(
            or_values,
            y[or_mask],
            xerr=np.vstack([or_values - or_low, or_high - or_values]),
            fmt="*",
            color="#E6550D",
            ecolor="#E6550D",
            markersize=10,
            capsize=3,
            linewidth=1.2,
            label="Adjusted Odds Ratio",
        )
    if d_mask.any():
        d_values = plot_data.loc[d_mask, "Val"].astype(float)
        d_low = plot_data.loc[d_mask, "Low"].astype(float)
        d_high = plot_data.loc[d_mask, "High"].astype(float)
        ax_d.errorbar(
            d_values,
            y[d_mask],
            xerr=np.vstack([d_values - d_low, d_high - d_values]),
            fmt="o",
            color="#6BAED6",
            ecolor="#6BAED6",
            markersize=6,
            capsize=3,
            linewidth=1.2,
            label="Cohen's d",
        )

    finite_or_high = pd.to_numeric(plot_data.loc[or_mask, "High"], errors="coerce")
    or_max = max(5.0, float(finite_or_high.max()) * 1.10) if len(finite_or_high) else 5.0
    ax_or.set_xlim(0, or_max)
    ax_or.set_xlabel("Adjusted Odds Ratio (Cluster 1 vs Cluster 0)", fontsize=12)
    ax_d.set_xlim(-3, 3)
    ax_d.set_xlabel("Cohen's d (Cluster 1 minus Cluster 0)", fontsize=12)
    ax_or.set_yticks(y)
    labels = []
    for _, row in plot_data.iterrows():
        label = str(row["Variable"])
        labels.append(label if row["Type"] == "header" else f"  {label}")
    ax_or.set_yticklabels(labels, fontsize=11)
    for tick, row_type in zip(ax_or.get_yticklabels(), plot_data["Type"]):
        if row_type == "header":
            tick.set_fontweight("bold")

    outcome_positions = y[plot_data["Type"].eq("OR")]
    d_positions = y[plot_data["Type"].eq("d")]
    ymin, ymax = -0.5, len(plot_data) - 0.5
    if len(outcome_positions):
        ax_or.axvline(
            1,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            ymin=(outcome_positions.min() - ymin) / (ymax - ymin),
            ymax=(outcome_positions.max() - ymin) / (ymax - ymin),
        )
    if len(d_positions):
        ax_d.axvline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            ymin=(d_positions.min() - ymin) / (ymax - ymin),
            ymax=(d_positions.max() - ymin) / (ymax - ymin),
        )

    ax_or.annotate(
        "", xy=(0.05, -0.055), xycoords="axes fraction", xytext=(0.30, -0.055),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )
    ax_or.text(0.175, -0.085, "Cluster 0", transform=ax_or.transAxes, ha="center", fontsize=12, fontweight="bold")
    ax_or.annotate(
        "", xy=(0.95, -0.055), xycoords="axes fraction", xytext=(0.70, -0.055),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    )
    ax_or.text(0.825, -0.085, "Cluster 1", transform=ax_or.transAxes, ha="center", fontsize=12, fontweight="bold")
    ax_or.set_ylim(ymin, ymax)
    ax_or.set_title(cohort.upper(), fontsize=16, fontweight="bold", pad=18)
    ax_or.grid(False)
    ax_d.grid(False)
    ax_or.spines["right"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    plt.subplots_adjust(left=0.28, right=0.95, top=0.92, bottom=0.12)
    fig.savefig(output_dir / f"{cohort}_forest_plot_full.png", dpi=800, bbox_inches="tight")
    fig.savefig(output_dir / f"{cohort}_forest_plot_full.pdf", bbox_inches="tight")
    plt.close(fig)


def audit_rows(data: pd.DataFrame, cohort: str, stats: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for outcome, label in OUTCOME_LABELS.items():
        if outcome not in data.columns or (cohort == "mros" and outcome == "cvd"):
            continue
        valid = data[[outcome, "cluster"]].dropna()
        rates = valid.groupby("cluster")[outcome].mean()
        rows.append(
            {
                "cohort": cohort,
                "outcome": outcome,
                "label": label,
                "merged_rows": len(data),
                "valid_outcome_rows": len(valid),
                "missing_outcome_rows": int(data[outcome].isna().sum()),
                "events": int(valid[outcome].sum()),
                "cluster0_rate": rates.get(0, np.nan),
                "cluster1_rate": rates.get(1, np.nan),
            }
        )
    return rows


def main() -> None:
    global FEATURES, FEATURE_LABELS, FEATURE_SECTION_LABEL
    args = parse_args()
    if args.include_embedding_features:
        FEATURES = BASE_FEATURES + EMBEDDING_FEATURES
        FEATURE_LABELS = {feature: feature for feature in FEATURES}
        FEATURE_SECTION_LABEL = "SDI + Embedding Features"
    if args.n_boot < 50:
        raise ValueError("--n-boot must be at least 50")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = args.cluster_dir.resolve()

    clinical = {
        "cfs": load_cfs(args.cfs_clinical),
        "mros": load_mros(args.mros_clinical),
        "shhs": load_shhs(args.shhs_clinical, args.shhs_cvd),
    }
    all_stats = []
    all_audit = []
    quality: dict[str, object] = {
        "features": FEATURES,
        "cluster_strategy": args.cluster_strategy,
        "or_contrast": "Cluster 1 versus Cluster 0",
        "adjustment": "age, BMI, and sex when non-constant",
        "bootstrap_replicates": args.n_boot,
        "exclude_ambiguous": args.exclude_ambiguous,
        "cohorts": {},
    }

    for index, cohort in enumerate(["cfs", "mros", "shhs"]):
        clusters = load_clusters(cluster_dir, cohort, args.exclude_ambiguous)
        merged = merge_cohort(clusters, clinical[cohort], cohort)
        stats = analyze_cohort(merged, cohort, args.n_boot, args.seed + 1000 * index)
        stats.to_csv(output_dir / f"{cohort}_forest_stats.csv", index=False)
        merged.to_csv(output_dir / f"{cohort}_merged_analysis.csv", index=False)
        plot_forest(stats, cohort, output_dir)
        all_stats.append(stats)
        all_audit.extend(audit_rows(merged, cohort, stats))
        quality["cohorts"][cohort] = {
            "cluster_rows": len(clusters),
            "clinical_rows": len(clinical[cohort]),
            "merged_rows": len(merged),
            "unmatched_cluster_rows": len(clusters) - len(merged),
            "cluster0_rows": int((merged["cluster"] == 0).sum()),
            "cluster1_rows": int((merged["cluster"] == 1).sum()),
        }

    combined_stats = pd.concat(all_stats, ignore_index=True)
    combined_stats.to_csv(output_dir / "all_cohorts_forest_stats.csv", index=False)
    pd.DataFrame(all_audit).to_csv(output_dir / "outcome_definition_audit.csv", index=False)
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
