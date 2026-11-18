"""
SHHS1 Q1-Q4 与失眠（diffa10）横断面关联分析
================================================

疾病定义（由研究者指定）：
    insomnia = 1  if diffa10 == 1
    insomnia = 0  if diffa10 == 0
    diffa10 缺失者排除

Q1-Q4 定义：
    Body  = wake_depth_score
    Brain = rem_prob_rem
    每位受试者内部 Min-Max 归一化，以 0.5 为阈值。

直接运行：
    python quadrant_insomnia_analysis.py

注意：diffa10 是 SHHS1 morning survey 的“昨夜是否入睡困难”。
本分析属于横断面关联，不应表述为新发失眠预测或临床诊断验证。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

import quadrant_risk_analysis as core


# =====================================================================
# 硬编码路径：文件移动时只修改这里
# =====================================================================
PREDICTION_DIR = Path(
    r"C:\Users\SaberShuo\Desktop\SDI\SDI_all_data\csv_exports"
)
HARMONIZED_CSV = Path(
    r"C:\Users\SaberShuo\Desktop\dataset_harmonized\dataset_harmonized\harmonized\shhs-harmonized-dataset-0.20.0.csv"
)
SHHS1_DATASET_CSV = Path(
    r"C:\Users\SaberShuo\Desktop\dataset_harmonized\dataset_harmonized\harmonized\shhs1-dataset-0.20.0.csv"
)
OUTPUT_DIR = Path(
    r"C:\Users\SaberShuo\Documents\SDI\quadrant_insomnia_results"
)

BODY_COL = "wake_depth_score"
BRAIN_COL = "rem_prob_rem"
SOURCE_OUTCOME_COL = "diffa10"
OUTCOME_COL = "insomnia"

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
Q_COLORS = {
    "Q1": "#d62728",
    "Q2": "#169c9c",
    "Q3": "#55a868",
    "Q4": "#d99a00",
}


def configure_core() -> None:
    """复用已验证的 Q1-Q4、回归和交叉验证函数。"""
    core.PREDICTION_DIR = PREDICTION_DIR
    core.HARMONIZED_CSV = HARMONIZED_CSV
    core.OUTPUT_DIR = OUTPUT_DIR
    core.BODY_COL = BODY_COL
    core.BRAIN_COL = BRAIN_COL
    core.OUTCOME_COL = OUTCOME_COL
    core.N_BOOTSTRAP = 1000
    core.BASELINE_NUMERIC = ["nsrr_age", "nsrr_bmi", "log1p_ahi"]
    core.BASELINE_CATEGORICAL = [
        "nsrr_sex",
        "nsrr_race",
        "nsrr_current_smoker",
    ]
    core.ADJUSTMENT_TERMS = (
        "nsrr_age + nsrr_bmi + log1p_ahi + "
        "C(nsrr_sex) + C(nsrr_race) + C(nsrr_current_smoker)"
    )


def validate_paths() -> None:
    errors = []
    if not PREDICTION_DIR.is_dir():
        errors.append(f"预测 CSV 文件夹不存在：{PREDICTION_DIR}")
    if not HARMONIZED_CSV.is_file():
        errors.append(f"Harmonized 文件不存在：{HARMONIZED_CSV}")
    if not SHHS1_DATASET_CSV.is_file():
        errors.append(f"SHHS1 原始数据不存在：{SHHS1_DATASET_CSV}")
    if errors:
        raise FileNotFoundError("\n".join(errors))


def load_baseline_and_insomnia() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取标准化基线协变量，并按 diffa10 生成二分类失眠结局。"""
    harmonized = pd.read_csv(HARMONIZED_CSV, low_memory=False)
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
    missing_h = required_h.difference(harmonized.columns)
    if missing_h:
        raise ValueError(f"Harmonized 文件缺少字段：{sorted(missing_h)}")

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
    ].drop_duplicates("nsrrid", keep="first")

    shhs1 = pd.read_csv(
        SHHS1_DATASET_CSV,
        usecols=["nsrrid", SOURCE_OUTCOME_COL],
        low_memory=False,
    )
    shhs1[SOURCE_OUTCOME_COL] = pd.to_numeric(
        shhs1[SOURCE_OUTCOME_COL], errors="coerce"
    )
    shhs1[OUTCOME_COL] = np.where(
        shhs1[SOURCE_OUTCOME_COL].isin([0, 1]),
        shhs1[SOURCE_OUTCOME_COL],
        np.nan,
    )
    insomnia = shhs1[
        ["nsrrid", SOURCE_OUTCOME_COL, OUTCOME_COL]
    ].drop_duplicates("nsrrid", keep="first")
    return baseline, insomnia


def make_cohort(
    q_df: pd.DataFrame,
    baseline: pd.DataFrame,
    insomnia: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = q_df.merge(baseline, on="nsrrid", how="left", validate="one_to_one")
    merged = merged.merge(insomnia, on="nsrrid", how="left", validate="one_to_one")

    flow = pd.DataFrame(
        [
            {"step": "预测文件成功计算 Q1-Q4", "n": len(q_df)},
            {
                "step": "匹配 SHHS1 harmonized 基线",
                "n": int(merged["nsrr_age"].notna().sum()),
            },
            {
                "step": "diffa10 为 0 或 1",
                "n": int(merged[OUTCOME_COL].notna().sum()),
            },
        ]
    )

    cohort = merged.loc[merged[OUTCOME_COL].isin([0, 1])].copy()
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
        raise RuntimeError("失眠结局没有同时包含 0 和 1。")
    return cohort.reset_index(drop=True), flow


def make_coverage_table(cohort: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
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


def create_figure(
    cohort: pd.DataFrame,
    regression: pd.DataFrame,
    performance: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """生成白底四联图。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), dpi=160)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A：两组 Q1-Q4 平均占比和 95% CI。
    x_pos = np.arange(len(QUADRANTS))
    width = 0.36
    for offset, value, label, color in [
        (-width / 2, 0, "No insomnia", "#7f8c8d"),
        (width / 2, 1, "Insomnia (diffa10=1)", "#c0392b"),
    ]:
        group = cohort.loc[cohort[OUTCOME_COL].eq(value)]
        means = np.array([group[f"{q}_pct"].mean() for q in QUADRANTS])
        sem = np.array([group[f"{q}_pct"].sem() for q in QUADRANTS])
        ax_a.bar(
            x_pos + offset,
            means,
            width,
            yerr=1.96 * sem,
            capsize=3,
            color=color,
            alpha=0.86,
            label=label,
        )
    ax_a.set_xticks(x_pos, QUADRANTS)
    ax_a.set_ylabel("Night-time occupancy (%)")
    ax_a.set_title("A  Quadrant occupancy by insomnia status")
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
    ax_b.set_title("B  Adjusted cross-sectional association")
    if bool(forest.loc["Q2", "sparse_quadrant_warning"]):
        ax_b.text(
            1.97,
            y_pos[1] + 0.13,
            "Q2 sparse; interpret cautiously",
            ha="right",
            va="bottom",
            fontsize=8,
            color=Q_COLORS["Q2"],
        )

    # C：五折 out-of-fold ROC。
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
    overall = performance.loc[
        performance["fold"].eq("overall_oof")
    ].set_index("model")
    for model_name in palette:
        pred = predictions[f"pred_{model_name}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, pred)
        auc = overall.loc[model_name, "auc"]
        ax_c.plot(
            fpr,
            tpr,
            color=palette[model_name],
            lw=2,
            label=f"{labels[model_name]} (AUC={auc:.3f})",
        )
    ax_c.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax_c.set_xlabel("1 - Specificity")
    ax_c.set_ylabel("Sensitivity")
    ax_c.set_title("C  Five-fold out-of-fold ROC")
    ax_c.legend(frameon=False, fontsize=9)

    # D：五折 out-of-fold 校准。
    for model_name in palette:
        pred = predictions[f"pred_{model_name}"].to_numpy()
        observed, predicted = calibration_curve(
            y, pred, n_bins=10, strategy="quantile"
        )
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
    ax_d.set_ylabel("Observed insomnia proportion")
    ax_d.set_title("D  Five-fold out-of-fold calibration")
    ax_d.legend(frameon=False, fontsize=9)

    for ax in axes.ravel():
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Q1-Q4 occupancy and insomnia (diffa10=1): cross-sectional analysis",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUTPUT_DIR / "quadrant_insomnia_figure.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "quadrant_insomnia_figure.pdf", bbox_inches="tight")
    plt.close(fig)


def save_summary(
    cohort: pd.DataFrame,
    flow: pd.DataFrame,
    regression: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    adjusted = regression.loc[
        regression["model"].eq("adjusted_single_quadrant")
    ].set_index("quadrant")
    overall = performance.loc[performance["fold"].eq("overall_oof")]
    lines = [
        "SHHS1 Q1-Q4 与失眠横断面分析摘要",
        "=" * 46,
        f"失眠定义: {SOURCE_OUTCOME_COL} == 1",
        f"Body: {BODY_COL}（个体内 Min-Max）",
        f"Brain: {BRAIN_COL}（个体内 Min-Max）",
        "象限阈值: 0.5",
        "",
        "队列流程：",
    ]
    for _, row in flow.iterrows():
        lines.append(f"- {row['step']}: {int(row['n']):,}")
    lines.extend(
        [
            "",
            f"最终样本量: {len(cohort):,}",
            f"失眠人数: {int(cohort[OUTCOME_COL].sum()):,} ({cohort[OUTCOME_COL].mean()*100:.2f}%)",
            "",
            "调整后 OR（每增加10个百分点）：",
        ]
    )
    for q in QUADRANTS:
        row = adjusted.loc[q]
        lines.append(
            f"- {q}: OR={row['odds_ratio']:.3f}, "
            f"95% CI {row['ci95_low']:.3f}-{row['ci95_high']:.3f}, "
            f"p={row['p_value']:.4g}, FDR p={row['p_fdr_across_q1_q4']:.4g}"
        )
    lines.extend(["", "五折交叉验证（out-of-fold）："])
    for _, row in overall.iterrows():
        lines.append(
            f"- {row['model']}: AUC={row['auc']:.3f} "
            f"(95% CI {row['auc_ci95_low']:.3f}-{row['auc_ci95_high']:.3f}), "
            f"Brier={row['brier']:.4f}"
        )
    lines.extend(
        [
            "",
            "解释限制：diffa10 与 PSG 均来自 SHHS1，本结果为横断面关联；",
            "diffa10 表示昨夜入睡困难，不等同于经过临床诊断的慢性失眠障碍。",
        ]
    )
    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def main() -> None:
    configure_core()
    validate_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] 计算全部受试者 Q1-Q4")
    q_df, failures = core.build_quadrant_table()
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

    print("[2/6] 读取 diffa10 并构建失眠队列")
    baseline, insomnia = load_baseline_and_insomnia()
    cohort, flow = make_cohort(q_df, baseline, insomnia)
    cohort.to_csv(
        OUTPUT_DIR / "analysis_cohort.csv",
        index=False,
        encoding="utf-8-sig",
    )
    flow.to_csv(OUTPUT_DIR / "cohort_flow.csv", index=False, encoding="utf-8-sig")

    print("[3/6] 比较失眠组与对照组的 Q1-Q4 分布")
    comparison = core.compare_outcome_groups(cohort)
    comparison.to_csv(
        OUTPUT_DIR / "quadrant_group_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_coverage_table(cohort).to_csv(
        OUTPUT_DIR / "quadrant_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regression = core.run_logistic_models(cohort)
    regression.to_csv(
        OUTPUT_DIR / "logistic_regression_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    performance, predictions = core.cross_validated_performance(cohort)
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

    print("[6/6] 绘制综合图并保存摘要")
    create_figure(cohort, regression, performance, predictions)
    save_summary(cohort, flow, regression, performance)

    print("\n失眠分析完成：")
    print(f"  输出目录：{OUTPUT_DIR}")
    print(f"  最终样本量：{len(cohort):,}")
    print(
        f"  失眠人数：{int(cohort[OUTCOME_COL].sum()):,} "
        f"({cohort[OUTCOME_COL].mean()*100:.2f}%)"
    )


if __name__ == "__main__":
    main()
