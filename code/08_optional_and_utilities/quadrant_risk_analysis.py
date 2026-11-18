"""
SHHS 四象限占比与不良结局风险关联分析
========================================

本脚本只需要修改下方“硬编码路径与分析配置”区域，然后直接运行：

    python quadrant_risk_analysis.py

分析步骤：
1. 批量读取 shhs1-<nsrrid>_features_predictions.csv。
2. 每位受试者内部对 wake_depth_score 和 rem_prob_rem 分别做 Min-Max 归一化。
3. 使用 0.5 阈值计算 Q1-Q4 的整夜 epoch 占比。
4. 合并 SHHS1 harmonized 基线协变量与 SHHS CVD 随访结局。
5. 默认排除基线已有 CVD 的受试者，分析 incident any_cvd。
6. 输出组间比较、Logistic 回归、五折交叉验证和论文图。

重要解释：
这里的 0.5 是“受试者个体内相对位置”，不是跨患者统一的绝对生理阈值。
因此结果应表述为 exploratory/proof-of-concept association，不能直接表述为
已经完成临床验证的诊断或预后工具。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================================
# 硬编码路径与分析配置：如文件移动，只修改这里
# =====================================================================
PREDICTION_DIR = Path(
    r"C:\Users\SaberShuo\Desktop\SDI\SDI_all_data\csv_exports"
)
HARMONIZED_CSV = Path(
    r"C:\Users\SaberShuo\Desktop\SDI\nsrr_data\shhs-harmonized-dataset-0.20.0.csv"
)
OUTCOME_CSV = Path(
    r"C:\Users\SaberShuo\Desktop\shhs-cvd-summary-dataset-0.19.0.csv"
)
OUTPUT_DIR = Path(
    r"C:\Users\SaberShuo\Documents\SDI\quadrant_risk_results"
)

BODY_COL = "wake_depth_score"
BRAIN_COL = "rem_prob_rem"

# 主要结局：SHHS CVD summary 中 0=未发生，1=发生，NaN=结局未知。
OUTCOME_COL = "any_cvd"

# incident CVD 分析通常应排除基线已有相关事件者。
EXCLUDE_PREVALENT_CVD = True
PREVALENT_CVD_COLS = [
    "prev_mi",
    "prev_mip",
    "prev_stk",
    "prev_chf",
    "prev_revpro",
    "prev_ang",
]

RANDOM_SEED = 20261002
N_FOLDS = 5
N_BOOTSTRAP = 1000

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
Q_COLORS = {
    "Q1": "#d62728",
    "Q2": "#169c9c",
    "Q3": "#55a868",
    "Q4": "#d99a00",
}

BASELINE_NUMERIC = ["nsrr_age", "nsrr_bmi", "log1p_ahi"]
BASELINE_CATEGORICAL = ["nsrr_sex", "nsrr_race", "nsrr_current_smoker"]
ADJUSTMENT_TERMS = (
    "nsrr_age + nsrr_bmi + log1p_ahi + "
    "C(nsrr_sex) + C(nsrr_race) + C(nsrr_current_smoker)"
)


def validate_paths() -> None:
    """检查三个输入路径，避免处理数千文件后才发现配置错误。"""
    errors = []
    if not PREDICTION_DIR.is_dir():
        errors.append(f"预测文件夹不存在：{PREDICTION_DIR}")
    if not HARMONIZED_CSV.is_file():
        errors.append(f"Harmonized 文件不存在：{HARMONIZED_CSV}")
    if not OUTCOME_CSV.is_file():
        errors.append(f"结局文件不存在：{OUTCOME_CSV}")
    if errors:
        raise FileNotFoundError("\n".join(errors))


def _minmax(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """个体内 Min-Max；常数列按 0 处理并返回 constant=True。"""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=float), True
    return (values - vmin) / (vmax - vmin), False


def extract_nsrrid(path: Path) -> int:
    match = re.fullmatch(r"shhs1-(\d+)_features_predictions\.csv", path.name)
    if not match:
        raise ValueError(f"文件名不符合预期格式：{path.name}")
    return int(match.group(1))


def calculate_subject_quadrants(path: Path) -> dict:
    """计算单位受试者 Q1-Q4 的 epoch 数和占比。"""
    nsrrid = extract_nsrrid(path)
    df = pd.read_csv(path, usecols=[BODY_COL, BRAIN_COL])
    df[BODY_COL] = pd.to_numeric(df[BODY_COL], errors="coerce")
    df[BRAIN_COL] = pd.to_numeric(df[BRAIN_COL], errors="coerce")
    df = df.dropna(subset=[BODY_COL, BRAIN_COL])
    if df.empty:
        raise ValueError("Body/Brain 无有效数据")

    body_raw = df[BODY_COL].to_numpy(dtype=float)
    brain_raw = df[BRAIN_COL].to_numpy(dtype=float)
    body, body_constant = _minmax(body_raw)
    brain, brain_constant = _minmax(brain_raw)

    masks = {
        "Q1": (body >= 0.5) & (brain >= 0.5),
        "Q2": (body < 0.5) & (brain >= 0.5),
        "Q3": (body < 0.5) & (brain < 0.5),
        "Q4": (body >= 0.5) & (brain < 0.5),
    }
    n_epochs = len(df)
    row = {
        "nsrrid": nsrrid,
        "source_file": path.name,
        "n_valid_epochs": n_epochs,
        "body_raw_min": float(body_raw.min()),
        "body_raw_max": float(body_raw.max()),
        "brain_raw_min": float(brain_raw.min()),
        "brain_raw_max": float(brain_raw.max()),
        "body_constant": body_constant,
        "brain_constant": brain_constant,
    }
    for q in QUADRANTS:
        count = int(masks[q].sum())
        row[f"{q}_count"] = count
        row[f"{q}_prop"] = count / n_epochs
        row[f"{q}_pct"] = count / n_epochs * 100.0
    return row


def build_quadrant_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量计算全部受试者四象限特征，并记录失败文件。"""
    files = sorted(PREDICTION_DIR.glob("shhs1-*_features_predictions.csv"))
    if not files:
        raise RuntimeError(f"预测文件夹中未找到 CSV：{PREDICTION_DIR}")

    rows = []
    failures = []
    print(f"[1/6] 计算 Q1-Q4：发现 {len(files):,} 个预测文件")
    for i, path in enumerate(files, start=1):
        try:
            rows.append(calculate_subject_quadrants(path))
        except Exception as exc:
            failures.append({"source_file": path.name, "error": str(exc)})
        if i % 500 == 0 or i == len(files):
            print(f"      已处理 {i:,}/{len(files):,}")

    q_df = pd.DataFrame(rows).sort_values("nsrrid").reset_index(drop=True)
    fail_df = pd.DataFrame(failures, columns=["source_file", "error"])
    if q_df.empty:
        raise RuntimeError("没有任何受试者成功计算 Q1-Q4。")

    prop_sum = q_df[[f"{q}_prop" for q in QUADRANTS]].sum(axis=1)
    if not np.allclose(prop_sum, 1.0, atol=1e-10):
        raise RuntimeError("Q1-Q4 占比之和不等于 1，象限计算存在错误。")
    return q_df, fail_df


def load_clinical_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 SHHS1 基线协变量和随访结局。"""
    print("[2/6] 读取并检查 harmonized 基线与 CVD outcome")
    harmonized = pd.read_csv(HARMONIZED_CSV, low_memory=False)
    outcome = pd.read_csv(OUTCOME_CSV, low_memory=False)

    required_h = {
        "nsrrid",
        "visitnumber",
        "nsrr_age",
        "nsrr_sex",
        "nsrr_race",
        "nsrr_bmi",
        "nsrr_current_smoker",
        "nsrr_ahi_hp3u",
    }
    required_o = {"nsrrid", OUTCOME_COL, *PREVALENT_CVD_COLS}
    missing_h = required_h.difference(harmonized.columns)
    missing_o = required_o.difference(outcome.columns)
    if missing_h:
        raise ValueError(f"Harmonized 文件缺少字段：{sorted(missing_h)}")
    if missing_o:
        raise ValueError(f"Outcome 文件缺少字段：{sorted(missing_o)}")

    baseline = harmonized.loc[harmonized["visitnumber"].eq(1)].copy()
    baseline = baseline[
        [
            "nsrrid",
            "nsrr_age",
            "nsrr_sex",
            "nsrr_race",
            "nsrr_bmi",
            "nsrr_current_smoker",
            "nsrr_ahi_hp3u",
        ]
    ]
    baseline = baseline.drop_duplicates("nsrrid", keep="first")

    if "visitnumber" in outcome.columns:
        outcome = outcome.loc[outcome["visitnumber"].eq(3)].copy()
    outcome = outcome[
        ["nsrrid", OUTCOME_COL, *PREVALENT_CVD_COLS]
    ].drop_duplicates("nsrrid", keep="first")

    for col in [OUTCOME_COL, *PREVALENT_CVD_COLS]:
        outcome[col] = pd.to_numeric(outcome[col], errors="coerce")
    outcome["prevalent_cvd"] = (
        outcome[PREVALENT_CVD_COLS].fillna(0).gt(0).any(axis=1).astype(int)
    )
    return baseline, outcome


def make_analysis_cohort(
    q_df: pd.DataFrame,
    baseline: pd.DataFrame,
    outcome: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """合并数据并生成队列流图所需计数。"""
    print("[3/6] 合并受试者、基线协变量与随访结局")
    flow = []
    flow.append({"step": "预测文件成功计算 Q1-Q4", "n": len(q_df)})

    merged = q_df.merge(baseline, on="nsrrid", how="left", validate="one_to_one")
    flow.append(
        {
            "step": "可匹配 SHHS1 harmonized 基线",
            "n": int(merged["nsrr_age"].notna().sum()),
        }
    )
    merged = merged.merge(outcome, on="nsrrid", how="left", validate="one_to_one")
    flow.append(
        {
            "step": f"有 {OUTCOME_COL} 随访结局",
            "n": int(merged[OUTCOME_COL].notna().sum()),
        }
    )

    cohort = merged.loc[merged[OUTCOME_COL].isin([0, 1])].copy()
    if EXCLUDE_PREVALENT_CVD:
        cohort = cohort.loc[cohort["prevalent_cvd"].eq(0)].copy()
        flow.append({"step": "排除基线已有 CVD 后", "n": len(cohort)})

    cohort[OUTCOME_COL] = cohort[OUTCOME_COL].astype(int)
    cohort["log1p_ahi"] = np.log1p(
        pd.to_numeric(cohort["nsrr_ahi_hp3u"], errors="coerce").clip(lower=0)
    )
    cohort["nsrr_current_smoker"] = cohort["nsrr_current_smoker"].replace(
        {"not reported": np.nan, "unknown": np.nan}
    )
    for q in QUADRANTS:
        cohort[f"{q}_per10"] = cohort[f"{q}_prop"] * 10.0

    if cohort[OUTCOME_COL].nunique() != 2:
        raise RuntimeError(f"{OUTCOME_COL} 在分析队列中不同时包含 0 和 1。")
    return cohort.reset_index(drop=True), pd.DataFrame(flow)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR 校正，保留原索引。"""
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def compare_outcome_groups(cohort: pd.DataFrame) -> pd.DataFrame:
    """比较有/无结局组的 Q1-Q4 分布。"""
    rows = []
    no_event = cohort.loc[cohort[OUTCOME_COL].eq(0)]
    event = cohort.loc[cohort[OUTCOME_COL].eq(1)]
    for q in QUADRANTS:
        col = f"{q}_pct"
        x0 = no_event[col].dropna().to_numpy()
        x1 = event[col].dropna().to_numpy()
        test = mannwhitneyu(x1, x0, alternative="two-sided")
        rank_biserial = 2.0 * test.statistic / (len(x1) * len(x0)) - 1.0
        rows.append(
            {
                "quadrant": q,
                "n_no_event": len(x0),
                "no_event_median_pct": np.median(x0),
                "no_event_q1_pct": np.quantile(x0, 0.25),
                "no_event_q3_pct": np.quantile(x0, 0.75),
                "n_event": len(x1),
                "event_median_pct": np.median(x1),
                "event_q1_pct": np.quantile(x1, 0.25),
                "event_q3_pct": np.quantile(x1, 0.75),
                "rank_biserial_event_vs_no_event": rank_biserial,
                "p_value": test.pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["p_fdr"] = benjamini_hochberg(result["p_value"])
    return result


def _fit_logit(formula: str, data: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.logit(formula, data=data, missing="drop").fit(
            disp=False,
            method="lbfgs",
            maxiter=2000,
        )


def _extract_or_row(
    fit,
    term: str,
    quadrant: str,
    model_name: str,
    formula: str,
) -> dict:
    ci = fit.conf_int().loc[term]
    return {
        "model": model_name,
        "quadrant": quadrant,
        "interpretation": "OR per 10 percentage-point increase",
        "n": int(fit.nobs),
        "events": int(np.asarray(fit.model.endog).sum()),
        "odds_ratio": float(np.exp(fit.params[term])),
        "ci95_low": float(np.exp(ci.iloc[0])),
        "ci95_high": float(np.exp(ci.iloc[1])),
        "p_value": float(fit.pvalues[term]),
        "converged": bool(fit.mle_retvals.get("converged", False)),
        "formula": formula,
    }


def run_logistic_models(cohort: pd.DataFrame) -> pd.DataFrame:
    """运行单象限模型和以 Q4 为参照的组成模型。"""
    print("[4/6] Logistic 风险关联分析")
    adjusted_terms = ADJUSTMENT_TERMS
    rows = []

    # 每个象限分别建模：便于制作直观森林图。
    for q in QUADRANTS:
        term = f"{q}_per10"
        for model_name, rhs in [
            ("unadjusted_single_quadrant", term),
            ("adjusted_single_quadrant", f"{term} + {adjusted_terms}"),
        ]:
            formula = f"{OUTCOME_COL} ~ {rhs}"
            try:
                fit = _fit_logit(formula, cohort)
                rows.append(_extract_or_row(fit, term, q, model_name, formula))
            except Exception as exc:
                rows.append(
                    {
                        "model": model_name,
                        "quadrant": q,
                        "interpretation": "OR per 10 percentage-point increase",
                        "n": np.nan,
                        "events": np.nan,
                        "odds_ratio": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "p_value": np.nan,
                        "converged": False,
                        "formula": formula,
                        "error": str(exc),
                    }
                )

    # Q1-Q3 同时进入模型、Q4 省略：系数可理解为相对 Q4 的 10% 时间再分配。
    for model_name, extra in [
        ("quadrant_composition_q4_reference", ""),
        ("adjusted_composition_q4_reference", f" + {adjusted_terms}"),
    ]:
        rhs = "Q1_per10 + Q2_per10 + Q3_per10" + extra
        formula = f"{OUTCOME_COL} ~ {rhs}"
        try:
            fit = _fit_logit(formula, cohort)
            for q in ["Q1", "Q2", "Q3"]:
                rows.append(
                    _extract_or_row(fit, f"{q}_per10", q, model_name, formula)
                )
        except Exception as exc:
            for q in ["Q1", "Q2", "Q3"]:
                rows.append(
                    {
                        "model": model_name,
                        "quadrant": q,
                        "interpretation": "OR per 10 percentage-point reallocation from Q4",
                        "n": np.nan,
                        "events": np.nan,
                        "odds_ratio": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "p_value": np.nan,
                        "converged": False,
                        "formula": formula,
                        "error": str(exc),
                    }
                )

    result = pd.DataFrame(rows)
    nonzero_pct = {
        q: float(cohort[f"{q}_prop"].gt(0).mean() * 100.0) for q in QUADRANTS
    }
    result["subjects_with_nonzero_pct"] = result["quadrant"].map(nonzero_pct)
    result["sparse_quadrant_warning"] = result["subjects_with_nonzero_pct"].lt(20.0)
    mask = result["model"].eq("adjusted_single_quadrant") & result["p_value"].notna()
    result.loc[mask, "p_fdr_across_q1_q4"] = benjamini_hochberg(
        result.loc[mask, "p_value"]
    )
    return result


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, numeric_cols),
            ("categorical", categorical, categorical_cols),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def bootstrap_auc_ci(
    y: np.ndarray,
    predictions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = []
    n = len(y)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        values.append(roc_auc_score(y[idx], predictions[idx]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def cross_validated_performance(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比较 Q-only、基线临床模型和基线+Q 模型。"""
    print("[5/6] 五折交叉验证：Q-only、临床基线、临床基线+Q")
    y = cohort[OUTCOME_COL].to_numpy(dtype=int)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    models = {
        "quadrants_only": (
            ["Q1_prop", "Q2_prop", "Q3_prop"],
            [],
        ),
        "clinical_baseline": (
            BASELINE_NUMERIC,
            BASELINE_CATEGORICAL,
        ),
        "clinical_plus_quadrants": (
            BASELINE_NUMERIC + ["Q1_prop", "Q2_prop", "Q3_prop"],
            BASELINE_CATEGORICAL,
        ),
    }

    predictions = pd.DataFrame({"nsrrid": cohort["nsrrid"], OUTCOME_COL: y})
    metric_rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for model_name, (numeric_cols, categorical_cols) in models.items():
        feature_cols = numeric_cols + categorical_cols
        x = cohort[feature_cols].copy()
        oof = np.full(len(cohort), np.nan, dtype=float)
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
            pipe = make_pipeline(numeric_cols, categorical_cols)
            pipe.fit(x.iloc[train_idx], y[train_idx])
            pred = pipe.predict_proba(x.iloc[test_idx])[:, 1]
            oof[test_idx] = pred
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "auc": roc_auc_score(y[test_idx], pred),
                    "brier": brier_score_loss(y[test_idx], pred),
                    "n_test": len(test_idx),
                    "events_test": int(y[test_idx].sum()),
                }
            )

        predictions[f"pred_{model_name}"] = oof
        auc = roc_auc_score(y, oof)
        ci_low, ci_high = bootstrap_auc_ci(y, oof, rng)
        metric_rows.append(
            {
                "model": model_name,
                "fold": "overall_oof",
                "auc": auc,
                "auc_ci95_low": ci_low,
                "auc_ci95_high": ci_high,
                "brier": brier_score_loss(y, oof),
                "n_test": len(y),
                "events_test": int(y.sum()),
            }
        )
        metric_rows.extend(fold_rows)

    return pd.DataFrame(metric_rows), predictions


def create_figure(
    cohort: pd.DataFrame,
    regression: pd.DataFrame,
    performance: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """生成白底四联图：占比、OR、ROC、校准。"""
    print("[6/6] 绘制风险关联综合图")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), dpi=160)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A：有/无结局组的均值及 95% CI。
    positions = np.arange(len(QUADRANTS))
    width = 0.36
    for offset, outcome_value, label, color in [
        (-width / 2, 0, "No incident CVD", "#7f8c8d"),
        (width / 2, 1, "Incident CVD", "#c0392b"),
    ]:
        group = cohort.loc[cohort[OUTCOME_COL].eq(outcome_value)]
        means = np.array([group[f"{q}_pct"].mean() for q in QUADRANTS])
        sem = np.array([group[f"{q}_pct"].sem() for q in QUADRANTS])
        ax_a.bar(
            positions + offset,
            means,
            width,
            yerr=1.96 * sem,
            capsize=3,
            label=label,
            color=color,
            alpha=0.86,
        )
    ax_a.set_xticks(positions, QUADRANTS)
    ax_a.set_ylabel("Night-time occupancy (%)")
    ax_a.set_title("A  Quadrant occupancy by outcome")
    ax_a.legend(frameon=False)

    # B：校正后的单象限 OR。
    forest = regression.loc[
        regression["model"].eq("adjusted_single_quadrant")
    ].set_index("quadrant").reindex(QUADRANTS)
    y_pos = np.arange(len(QUADRANTS))[::-1]
    for yv, q in zip(y_pos, QUADRANTS):
        row = forest.loc[q]
        if pd.notna(row["odds_ratio"]):
            ax_b.errorbar(
                row["odds_ratio"],
                yv,
                xerr=[
                    [row["odds_ratio"] - row["ci95_low"]],
                    [row["ci95_high"] - row["odds_ratio"]],
                ],
                fmt="o",
                color=Q_COLORS[q],
                capsize=3,
                markersize=7,
            )
    ax_b.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.5, 2.0)
    ax_b.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ax_b.set_xticklabels(["0.50", "0.75", "1.00", "1.25", "1.50", "2.00"])
    ax_b.minorticks_off()
    ax_b.set_yticks(y_pos, QUADRANTS)
    ax_b.set_xlabel("Adjusted OR per 10 percentage-point increase")
    ax_b.set_title("B  Adjusted association with incident CVD")
    if bool(forest.loc["Q2", "sparse_quadrant_warning"]):
        ax_b.text(
            1.97,
            y_pos[1] + 0.13,
            "Q2 sparse; 95% CI extends off-scale",
            ha="right",
            va="bottom",
            fontsize=8,
            color=Q_COLORS["Q2"],
        )

    # C：OOF ROC。
    y = predictions[OUTCOME_COL].to_numpy(dtype=int)
    palette = {
        "quadrants_only": "#8e44ad",
        "clinical_baseline": "#2980b9",
        "clinical_plus_quadrants": "#c0392b",
    }
    labels = {
        "quadrants_only": "Quadrants only",
        "clinical_baseline": "Clinical baseline",
        "clinical_plus_quadrants": "Clinical + quadrants",
    }
    overall = performance.loc[performance["fold"].eq("overall_oof")].set_index("model")
    for model_name in palette:
        pred = predictions[f"pred_{model_name}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, pred)
        auc = overall.loc[model_name, "auc"]
        ax_c.plot(fpr, tpr, color=palette[model_name], lw=2, label=f"{labels[model_name]} (AUC={auc:.3f})")
    ax_c.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax_c.set_xlabel("1 - Specificity")
    ax_c.set_ylabel("Sensitivity")
    ax_c.set_title("C  Five-fold out-of-fold ROC")
    ax_c.legend(frameon=False, fontsize=9)

    # D：OOF calibration。
    for model_name in palette:
        pred = predictions[f"pred_{model_name}"].to_numpy()
        observed, predicted = calibration_curve(y, pred, n_bins=10, strategy="quantile")
        ax_d.plot(
            predicted,
            observed,
            marker="o",
            color=palette[model_name],
            lw=1.8,
            label=labels[model_name],
        )
    ax_d.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax_d.set_xlabel("Predicted probability")
    ax_d.set_ylabel("Observed event proportion")
    ax_d.set_title("D  Five-fold out-of-fold calibration")
    ax_d.legend(frameon=False, fontsize=9)

    for ax in axes.ravel():
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Q1-Q4 occupancy and incident cardiovascular risk (exploratory)",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUTPUT_DIR / "quadrant_risk_figure.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "quadrant_risk_figure.pdf", bbox_inches="tight")
    plt.close(fig)


def save_readme(
    cohort: pd.DataFrame,
    flow: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    """保存一份纯文本结果摘要，便于快速核查。"""
    overall = performance.loc[performance["fold"].eq("overall_oof")]
    lines = [
        "SHHS Q1-Q4 风险关联分析摘要",
        "=" * 42,
        f"Body: {BODY_COL}（每位受试者内部 Min-Max）",
        f"Brain: {BRAIN_COL}（每位受试者内部 Min-Max）",
        f"阈值: 0.5",
        f"主要结局: {OUTCOME_COL}",
        f"排除基线已有 CVD: {EXCLUDE_PREVALENT_CVD}",
        "",
        "队列流程：",
    ]
    for _, row in flow.iterrows():
        lines.append(f"- {row['step']}: {int(row['n']):,}")
    lines.extend(
        [
            "",
            f"最终分析队列: {len(cohort):,}",
            f"结局事件: {int(cohort[OUTCOME_COL].sum()):,} ({cohort[OUTCOME_COL].mean()*100:.2f}%)",
            "",
            "五折交叉验证（out-of-fold）：",
        ]
    )
    for _, row in overall.iterrows():
        lines.append(
            f"- {row['model']}: AUC={row['auc']:.3f} "
            f"(95% CI {row['auc_ci95_low']:.3f}-{row['auc_ci95_high']:.3f}), "
            f"Brier={row['brier']:.4f}"
        )
    lines.extend(
        [
            "",
            "注意：本结果属于回顾性探索性关联和内部验证，不能替代外部验证、",
            "前瞻性临床验证或真实临床效益评估。",
        ]
    )
    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def main() -> None:
    validate_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    q_df, failures = build_quadrant_table()
    q_df.to_csv(
        OUTPUT_DIR / "subject_quadrant_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failures.to_csv(
        OUTPUT_DIR / "failed_prediction_files.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline, outcome = load_clinical_tables()
    cohort, flow = make_analysis_cohort(q_df, baseline, outcome)
    cohort.to_csv(
        OUTPUT_DIR / "analysis_cohort.csv",
        index=False,
        encoding="utf-8-sig",
    )
    flow.to_csv(OUTPUT_DIR / "cohort_flow.csv", index=False, encoding="utf-8-sig")

    comparison = compare_outcome_groups(cohort)
    comparison.to_csv(
        OUTPUT_DIR / "quadrant_group_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage = pd.DataFrame(
        [
            {
                "quadrant": q,
                "subjects_total": len(cohort),
                "subjects_with_nonzero": int(cohort[f"{q}_prop"].gt(0).sum()),
                "subjects_with_nonzero_pct": float(
                    cohort[f"{q}_prop"].gt(0).mean() * 100.0
                ),
                "median_occupancy_pct": float(cohort[f"{q}_pct"].median()),
                "p95_occupancy_pct": float(cohort[f"{q}_pct"].quantile(0.95)),
                "max_occupancy_pct": float(cohort[f"{q}_pct"].max()),
                "sparse_warning": bool(cohort[f"{q}_prop"].gt(0).mean() < 0.20),
            }
            for q in QUADRANTS
        ]
    )
    coverage.to_csv(
        OUTPUT_DIR / "quadrant_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regression = run_logistic_models(cohort)
    regression.to_csv(
        OUTPUT_DIR / "logistic_regression_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    performance, predictions = cross_validated_performance(cohort)
    performance.to_csv(
        OUTPUT_DIR / "cross_validated_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUTPUT_DIR / "out_of_fold_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_figure(cohort, regression, performance, predictions)
    save_readme(cohort, flow, performance)

    print("\n分析完成。输出目录：")
    print(f"  {OUTPUT_DIR}")
    print(f"最终样本量：{len(cohort):,}")
    print(f"结局事件数：{int(cohort[OUTCOME_COL].sum()):,}")


if __name__ == "__main__":
    main()
