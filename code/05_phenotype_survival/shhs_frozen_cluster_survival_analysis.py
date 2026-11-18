"""SHHS survival analysis using prespecified cluster-assignment files.

This script does not fit clustering itself. Cluster 0 is labelled Normal Sleep
and Cluster 1 Fragmented Sleep, following the prespecified SSTR+TDTL direction.
Outcomes are not used for labelling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


FS_AXIS_LABEL = 20
FS_TITLE = 24
FS_TICK = 18
FS_STATISTICS = 16
FS_LEGEND = 17
FS_PANEL = 26


DEFAULT_BASE = Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs")
DEFAULT_CLUSTERS = Path("analysis/sdi_8feature_internal_consensus_gmm/shhs_cluster_assignments.csv")
DEFAULT_OUTPUT = Path("analysis/sdi_8feature_internal_consensus_survival")
OUTCOMES = {
    "allcause": ("event_allcause", "All-cause Mortality"),
    "chd": ("event_chd", "Coronary Heart Disease Mortality"),
    "cvd": ("event_cvd", "Cardiovascular Disease Mortality"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-ambiguous", action="store_true")
    parser.add_argument(
        "--cluster-model-refit-in-shhs",
        action="store_true",
        help="Record that the supplied assignment file came from an SHHS-specific refit.",
    )
    return parser.parse_args()


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().str.replace(r"\.0$", "", regex=True)


def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error  # type: ignore[misc]


def binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.isin([0, 1]))


def coalesce_column(data: pd.DataFrame, name: str) -> pd.Series:
    """Return an unsuffixed merged column or coalesce its _x/_y variants."""
    if name in data.columns:
        return data[name]
    if f"{name}_x" in data.columns and f"{name}_y" in data.columns:
        return data[f"{name}_x"].combine_first(data[f"{name}_y"])
    if f"{name}_x" in data.columns:
        return data[f"{name}_x"]
    if f"{name}_y" in data.columns:
        return data[f"{name}_y"]
    raise KeyError(f"Missing required column: {name}")


def load_analysis_data(base_dir: Path, cluster_path: Path, exclude_ambiguous: bool) -> pd.DataFrame:
    baseline = pd.read_csv(base_dir / "shhs1-dataset-0.20.0.csv", low_memory=False)
    mortality = pd.read_csv(base_dir / "shhs-cvd-summary-dataset-0.19.0.csv", low_memory=False)
    harmonized = read_csv_robust(base_dir / "shhs-harmonized-dataset-0.20.0.csv", low_memory=False)
    clusters = pd.read_csv(cluster_path, low_memory=False)

    for frame in [baseline, mortality, harmonized]:
        frame["subject_id"] = normalize_id(frame["nsrrid"])
    clusters["subject_id"] = normalize_id(clusters["subject_id"])
    if exclude_ambiguous:
        clusters = clusters.loc[~clusters["cluster_ambiguous"].astype(bool)].copy()
    clusters = clusters[["subject_id", "cluster", "cluster_probability", "cluster_ambiguous"]]
    if clusters["subject_id"].duplicated().any():
        raise ValueError("Duplicated subject_id values in cluster assignments")

    harmonized = harmonized.loc[
        pd.to_numeric(harmonized["visitnumber"], errors="coerce").eq(1),
        ["subject_id", "nsrr_ahi_hp4u_aasm15"],
    ].drop_duplicates("subject_id")
    data = baseline.merge(mortality.drop(columns=["nsrrid"]), on="subject_id", how="inner")
    data = data.merge(harmonized, on="subject_id", how="left")
    data = data.merge(clusters, on="subject_id", how="inner")
    data["subtype"] = pd.to_numeric(data["cluster"], errors="raise").astype(int)
    data["group"] = data["subtype"].map({0: "Normal Sleep", 1: "Fragmented Sleep"})

    data["age_clean"] = pd.to_numeric(coalesce_column(data, "age_s1"), errors="coerce")
    data["bmi_clean"] = pd.to_numeric(coalesce_column(data, "bmi_s1"), errors="coerce")
    gender = pd.to_numeric(coalesce_column(data, "gender"), errors="coerce")
    data["sex_clean"] = (gender == 1).astype(float)
    data.loc[gender.isna(), "sex_clean"] = np.nan
    data["race_clean"] = pd.to_numeric(coalesce_column(data, "race"), errors="coerce")
    data["diabetes_binary"] = binary(data["parrptdiab"])
    data["ahi_value"] = pd.to_numeric(data["nsrr_ahi_hp4u_aasm15"], errors="coerce")

    if "evsmok15" in data.columns:
        data["smoking_binary"] = binary(data["evsmok15"])
    else:
        smoke = pd.to_numeric(data.get("smokstat_s1"), errors="coerce")
        data["smoking_binary"] = smoke.where(smoke.isna(), (smoke > 0).astype(float))

    stage_cols = ["timest1", "timest2", "timest34", "timerem"]
    stage_data = data[stage_cols].apply(pd.to_numeric, errors="coerce")
    data["tst_value"] = stage_data.sum(axis=1, min_count=len(stage_cols))

    vital = binary(data["vital"])
    data["event_allcause"] = 1 - vital
    # Missing cause among living participants or deaths of another/unknown cause
    # is treated as no target event, as in the supplied script.
    data["event_chd"] = binary(data["chd_death"]).fillna(0)
    data["event_cvd"] = binary(data["cvd_death"]).fillna(0)
    censdate = pd.to_numeric(data["censdate"], errors="coerce")
    data["time_years"] = censdate / 365.25 if censdate.max() > 1000 else censdate
    data = data.loc[data["time_years"].ge(0) | data["time_years"].isna()].copy()
    return data


def add_race_dummies(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    race = pd.get_dummies(frame["race_clean"], prefix="race", drop_first=True, dtype=float)
    return pd.concat([frame.drop(columns=["race_clean"]), race], axis=1), race.columns.tolist()


def fit_cox(data: pd.DataFrame, event_col: str, variables: list[str]) -> dict[str, float | int]:
    model_data = data[["time_years", event_col, *variables, "race_clean"]].copy()
    model_data, race_vars = add_race_dummies(model_data)
    variables = [*variables, *race_vars]
    model_data = model_data[["time_years", event_col, *variables]].dropna()
    constant = [column for column in variables if model_data[column].nunique() < 2]
    variables = [column for column in variables if column not in constant]
    model_data = model_data[["time_years", event_col, *variables]]
    cph = CoxPHFitter()
    cph.fit(model_data, duration_col="time_years", event_col=event_col)
    row = cph.summary.loc["subtype"]
    return {
        "n": int(len(model_data)),
        "events": int(model_data[event_col].sum()),
        "hr": float(row["exp(coef)"]),
        "ci_low": float(row["exp(coef) lower 95%"]),
        "ci_high": float(row["exp(coef) upper 95%"]),
        "p_value": float(row["p"]),
    }


def analyze_outcome(data: pd.DataFrame, key: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    event_col, title = OUTCOMES[key]
    surv = data.dropna(subset=["subtype", "time_years", event_col]).copy()
    normal = surv.loc[surv["subtype"].eq(0)]
    disturbed = surv.loc[surv["subtype"].eq(1)]
    if normal.empty or disturbed.empty:
        raise ValueError(f"{key}: one cluster is empty")

    km_normal = KaplanMeierFitter(label="Normal Sleep").fit(
        normal["time_years"], normal[event_col]
    )
    km_disturbed = KaplanMeierFitter(label="Fragmented Sleep").fit(
        disturbed["time_years"], disturbed[event_col]
    )
    logrank = logrank_test(
        normal["time_years"], disturbed["time_years"],
        event_observed_A=normal[event_col], event_observed_B=disturbed[event_col],
    )

    base = ["subtype", "age_clean", "sex_clean"]
    model1 = fit_cox(surv, event_col, base)
    model2 = fit_cox(surv, event_col, base + ["bmi_clean", "smoking_binary", "diabetes_binary"])
    model3 = fit_cox(
        surv, event_col,
        base + ["bmi_clean", "smoking_binary", "diabetes_binary", "ahi_value", "tst_value"],
    )
    result: dict[str, object] = {
        "key": key,
        "title": title,
        "event_col": event_col,
        "km_normal": km_normal,
        "km_disturbed": km_disturbed,
        "logrank_p": float(logrank.p_value),
        "model1": model1,
        "model2": model2,
        "model3": model3,
    }
    rows: list[dict[str, object]] = []
    for subtype, label in [(0, "Normal Sleep"), (1, "Fragmented Sleep")]:
        group = surv.loc[surv["subtype"].eq(subtype)]
        rows.append({
            "outcome": title,
            "group": label,
            "n": len(group),
            "events": int(group[event_col].sum()),
            "event_rate": float(group[event_col].mean()),
            "median_followup_years": float(group["time_years"].median()),
            "survival_5y": float((km_normal if subtype == 0 else km_disturbed).predict(5)),
            "survival_10y": float((km_normal if subtype == 0 else km_disturbed).predict(10)),
            "survival_15y": float((km_normal if subtype == 0 else km_disturbed).predict(15)),
        })
    for index, model in enumerate([model1, model2, model3], start=1):
        rows.append({
            "outcome": title,
            "group": f"Cox Model {index}: Fragmented vs Normal",
            "n": model["n"],
            "events": model["events"],
            "hr": model["hr"],
            "ci_low": model["ci_low"],
            "ci_high": model["ci_high"],
            "p_value": model["p_value"],
            "logrank_p": float(logrank.p_value),
        })
    return result, rows


def plot_single(ax: plt.Axes, result: dict[str, object]) -> None:
    colors = {"Normal Sleep": "#2166ac", "Fragmented Sleep": "#b2182b"}
    for label, key in [("Normal Sleep", "km_normal"), ("Fragmented Sleep", "km_disturbed")]:
        km = result[key]
        sf = km.survival_function_
        ci = km.confidence_interval_
        timeline = sf.index.to_numpy()
        survival = sf.iloc[:, 0].to_numpy()
        ci_low = ci.iloc[:, 0].to_numpy()
        ci_high = ci.iloc[:, 1].to_numpy()
        if timeline[-1] < 15:
            timeline = np.append(timeline, 15)
            survival = np.append(survival, survival[-1])
            ci_low = np.append(ci_low, ci_low[-1])
            ci_high = np.append(ci_high, ci_high[-1])
        ax.step(timeline, survival, where="post", color=colors[label], linewidth=3, label=label)
        ax.fill_between(timeline, ci_low, ci_high, step="post", alpha=0.20, color=colors[label])

    ax.set_xlabel("Time to Event (Years)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_ylabel("Survival Probability (%)", fontsize=FS_AXIS_LABEL, fontweight="bold")
    ax.set_title(str(result["title"]), fontsize=FS_TITLE, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 15)
    ax.tick_params(axis="both", labelsize=FS_TICK, width=1.2, length=5)
    if result["key"] == "allcause":
        ax.set_ylim(0.55, 1.0)
        ticks = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    else:
        ax.set_ylim(0.85, 1.0)
        ticks = [0.85, 0.88, 0.91, 0.94, 0.97, 1.0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{value * 100:.0f}" for value in ticks])

    p_value = float(result["logrank_p"])
    p_text = f"p={p_value:.4f}" if p_value >= 0.001 else "p<0.001"
    texts = [f"Log-rank test: {p_text}"]
    for index in range(1, 4):
        model = result[f"model{index}"]
        texts.append(
            f"Model {index}: HR={model['hr']:.2f} "
            f"(95% CI: {model['ci_low']:.2f}-{model['ci_high']:.2f})"
        )
    ax.text(
        0.02,
        0.025,
        "\n".join(texts),
        transform=ax.transAxes,
        fontsize=FS_STATISTICS,
        linespacing=1.35,
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            alpha=0.92,
            edgecolor="gray",
            linewidth=1.1,
        ),
    )
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=True)


def save_figures(results: list[dict[str, object]], output_dir: Path) -> None:
    for result in results:
        fig, ax = plt.subplots(figsize=(12, 8.5))
        plot_single(ax, result)
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            fig.savefig(output_dir / f"shhs_km_{result['key']}.{suffix}", dpi=800 if suffix == "png" else None,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9.5))
    for index, (ax, result) in enumerate(zip(axes, results)):
        plot_single(ax, result)
        ax.text(
            -0.02,
            1.08,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            fontsize=FS_PANEL,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.subplots_adjust(bottom=0.12, top=0.86, left=0.06, right=0.99, wspace=0.28)
    fig.savefig(output_dir / "shhs_km_curves_combined_3plots.png", dpi=800,
                bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "shhs_km_curves_combined_3plots.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(args.base_dir, args.clusters.resolve(), args.exclude_ambiguous)
    data.to_csv(output_dir / "shhs_survival_merged.csv", index=False)

    results = []
    summary_rows: list[dict[str, object]] = []
    for key in ["allcause", "chd", "cvd"]:
        result, rows = analyze_outcome(data, key)
        results.append(result)
        summary_rows.extend(rows)
    pd.DataFrame(summary_rows).to_csv(output_dir / "shhs_survival_summary.csv", index=False)
    save_figures(results, output_dir)

    quality = {
        "cluster_source": str(args.clusters.resolve()),
        "cluster_model_refit_in_shhs": args.cluster_model_refit_in_shhs,
        "group_definition": {"0": "Normal Sleep", "1": "Fragmented Sleep"},
        "exclude_ambiguous": args.exclude_ambiguous,
        "merged_rows": len(data),
        "normal_rows": int(data["subtype"].eq(0).sum()),
        "disturbed_rows": int(data["subtype"].eq(1).sum()),
        "outcomes": {
            str(result["key"]): {
                "logrank_p": result["logrank_p"],
                **{f"model{i}": result[f"model{i}"] for i in range(1, 4)},
            }
            for result in results
        },
    }
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
