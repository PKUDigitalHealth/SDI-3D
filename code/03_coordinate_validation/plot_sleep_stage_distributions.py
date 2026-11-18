
"""
睡眠特征分析：各睡眠阶段标签分布（Task1）
直接修改下方 CONFIG 中的路径后运行即可：
    python sleep_label_brian_body.py
"""

import glob
import os
import re
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==================================================
# ★ 路径配置（按需修改）
# ==================================================
CONFIG = {
    # 包含 shhs1-*_features_predictions.csv 的目录
    "features_dir":  "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/test/csv_exports_shhs/csv/csv_exports/",
    # 三个 SHHS 数据集 CSV 文件路径
    "shhs1_csv":      "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/code/data/shhs1-dataset-0.20.0.csv",
    "cvd_csv":        "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/code/data/shhs-cvd-summary-dataset-0.19.0.csv",
    "harmonized_csv": "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/code/data/shhs-harmonized-dataset-0.20.0.csv",
    # 结果输出目录
    "output_dir":     "/home/zhaoqingshuo/SDI/ECG_sleepstage/transformer_W_NW_REM_NREM/output/feature_analysis",
}

# --------------------------------------------------
# 常量定义
# --------------------------------------------------
LABEL_NAMES  = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
LABEL_ORDER  = [0, 1, 2, 3, 4]
# Colorblind-friendly (Okabe–Ito–style / Nature-friendly)
LABEL_COLORS = {
    0: "#E69F00",  # Wake
    1: "#56B4E9",  # N1
    2: "#009E73",  # N2
    3: "#9467BD",  # N3
    4: "#2CA02C",  # REM
}

FEATURES = ["body_depth_score", "brain_activity_score"]
FEATURE_LABELS = {
    "body_depth_score":     "Body sleep-depth score",
    "brain_activity_score": "Brain sleep-depth score",
}

# CSV 中旧列名 → 统一为 FEATURES 列名（读入后重命名）
_LEGACY_FEATURE_COLS = {
    "wake_depth_score": "body_depth_score",
    "rem_prob_rem":     "brain_activity_score",
}

# 图 A：直方图按分期（Wake→REM）固定纵轴上限（频数 / bin count）
HIST_COUNT_YMAX = {
    "body_depth_score":     [60_000, 20_000, 120_000, 40_000, 60_000],
    "brain_activity_score": [800_000, 80_000, 1_000_000, 500_000, 50_000],
}

DISEASE_COLS = {
    "cvd_binary":      "CVD",
    "htn_binary":      "HTN",
    "diabetes_binary": "Diabetes",
    "ahi_binary":      "OSA (AHI≥5)",
    "poor_sleep":      "Poor Sleep",
    "good_sleep":      "Good Sleep",
    "insomnia":        "Insomnia",
}

# ==================================================
# 数据加载
# ==================================================
def _standardize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将旧列名映射为 body_depth_score / brain_activity_score。"""
    out = df.copy()
    for legacy, canon in _LEGACY_FEATURE_COLS.items():
        if legacy in out.columns:
            if canon not in out.columns:
                out[canon] = out[legacy]
            out = out.drop(columns=[legacy], errors="ignore")
    return out


def load_all_feature_files(features_dir: str) -> pd.DataFrame:
    """加载目录下所有 shhs1-*_features_predictions.csv，附加 nsrrid 列。"""
    pattern = os.path.join(features_dir, "shhs1-*_features_predictions.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {features_dir} 中未找到任何特征文件，请检查路径。")
    print(f"[加载] 找到 {len(files)} 个特征文件……")

    read_ok = {"window_center", "label", *FEATURES, *_LEGACY_FEATURE_COLS.keys()}
    dfs = []
    for fp in files:
        fname = os.path.basename(fp)
        m = re.search(r"shhs1-(\d+)_features_predictions", fname)
        if not m:
            continue
        nsrrid = m.group(1)
        try:
            df = pd.read_csv(fp, usecols=lambda c: c in read_ok)
            df = _standardize_feature_columns(df)
            miss = [c for c in FEATURES if c not in df.columns]
            if miss:
                raise ValueError(f"缺少列：{miss}（需 {FEATURES} 或兼容旧列名）")
            df["nsrrid"] = nsrrid
            dfs.append(df)
        except Exception as e:
            print(f"  警告：读取 {fname} 失败：{e}")

    all_df = pd.concat(dfs, ignore_index=True)
    # 只保留有效标签
    all_df = all_df[all_df["label"].isin(LABEL_ORDER)].copy()
    print(f"[加载] 共 {all_df['nsrrid'].nunique()} 位患者，{len(all_df):,} 条记录。")
    return all_df


def normalize_per_patient(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每位患者的 body_depth_score / brain_activity_score 单独做 MinMax 归一化，
    结果写入 *_norm 列（原列保留）。
    """
    print("[归一化] 逐患者 MinMax 归一化……")
    scaler = MinMaxScaler()
    norm_cols = [f + "_norm" for f in FEATURES]
    result = all_df.copy()
    result[norm_cols] = np.nan

    for nsrrid, grp in all_df.groupby("nsrrid"):
        valid = grp[FEATURES].dropna()
        if len(valid) < 2:
            continue
        normed = scaler.fit_transform(valid)
        result.loc[valid.index, norm_cols] = normed

    return result


def load_clinical_data(shhs1_csv: str, cvd_csv: str, harmonized_csv: str) -> pd.DataFrame:
    """加载并合并 SHHS 临床数据，定义疾病二值标签。"""
    print("[临床数据] 加载并合并……")
    df_b   = pd.read_csv(shhs1_csv, low_memory=False)
    df_cvd = pd.read_csv(cvd_csv,   low_memory=False)
    df_h   = pd.read_csv(harmonized_csv, low_memory=False)

    for d in (df_b, df_cvd, df_h):
        d["nsrrid"] = d["nsrrid"].astype(str)

    df = pd.merge(df_b, df_cvd, on="nsrrid", how="inner")

    df_h_v1 = df_h[df_h["visitnumber"] == 1][["nsrrid", "nsrr_ahi_hp4u_aasm15"]].copy()
    df_h_v1["nsrrid"] = df_h_v1["nsrrid"].astype(str)
    df = pd.merge(df, df_h_v1, on="nsrrid", how="left")

    # 疾病定义（与 shhs_clustering_analysis.py 保持一致）
    df["cvd_binary"]      = df["any_cvd"].fillna(0).astype(int)
    df["htn_binary"]      = df["htnderv_s1"].fillna(0).astype(int)
    df["diabetes_binary"] = df["parrptdiab"].fillna(0).astype(int)
    df["ahi_binary"]      = (df["nsrr_ahi_hp4u_aasm15"] >= 5).astype(int)
    df["poor_sleep"]      = ((df["ltdp10"] <= 2) & (df["rest10"] <= 2)).astype(int)
    df["good_sleep"]      = ((df["ltdp10"] >= 4) & (df["rest10"] >= 4)).astype(int)
    df["insomnia"]        = df["diffa10"].fillna(0).astype(int)

    keep = ["nsrrid"] + list(DISEASE_COLS.keys())
    print(f"[临床数据] 合并后 {len(df)} 位患者。")
    return df[keep].copy()


# ==================================================
# 任务 1 出版图：图 A（直方图）、图 B（小提琴），分文件保存
# ==================================================
def _pub_style_rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.facecolor": "#f5f5f5",
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
    }


def _save_figure_png_pdf(fig, path_png: str, *, dpi: int = 200, facecolor: str = "white") -> None:
    """同时保存 PNG 与同名 PDF。"""
    kwargs = {"bbox_inches": "tight", "facecolor": facecolor}
    fig.savefig(path_png, dpi=dpi, **kwargs)
    path_pdf = os.path.splitext(path_png)[0] + ".pdf"
    fig.savefig(path_pdf, format="pdf", **kwargs)
    print(f"→ 已保存：{path_png}")
    print(f"→ 已保存：{path_pdf}")


def _plot_task1_figure_a_histograms(
    df_norm: pd.DataFrame,
    output_dir: str
):
    """
    Publication-style Figure A
    Nature / Sleep / T-BME style
    """

    label_names = [LABEL_NAMES[i] for i in LABEL_ORDER]

    n_bins = 50
    FS_STATS = 18  # 子图左上角 Median / IQR 标注字号

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(24, 13.5),
        facecolor="white",
        gridspec_kw={
            "hspace": 0.46,
            "wspace": 0.32,
        },
    )

    # =====================================
    # Panel label
    # =====================================
    # axes[0, 0].text(
    #     -0.34,
    #     1.15,
    #     "A",
    #     transform=axes[0, 0].transAxes,
    #     fontsize=32,
    #     fontweight="bold",
    #     ha="left",
    #     va="top",
    # )

    for feat_idx, feat in enumerate(FEATURES):

        col = feat + "_norm"

        for j, lid in enumerate(LABEL_ORDER):

            ax = axes[feat_idx, j]

            vals = (
                df_norm[df_norm["label"] == lid][col]
                .dropna()
                .values
            )

            color = LABEL_COLORS[lid]

            # =====================================
            # Histogram
            # =====================================
            counts, bins, _ = ax.hist(
                vals,
                bins=n_bins,
                range=(0, 1),
                color=color,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.35,
                zorder=1,
            )

            # =====================================
            # KDE
            # =====================================
            if len(vals) > 100:

                try:

                    kde = gaussian_kde(vals)

                    x = np.linspace(0, 1, 500)

                    bin_width = bins[1] - bins[0]

                    y = kde(x) * len(vals) * bin_width

                    ax.plot(
                        x,
                        y,
                        color=color,
                        linewidth=3.5,
                        zorder=10,
                    )

                except:
                    pass

            # =====================================
            # Median line
            # =====================================
            med = np.median(vals)

            ax.axvline(
                med,
                color=color,
                linestyle=(0, (4, 3)),
                linewidth=2.5,
                alpha=0.9,
                zorder=8,
            )

            # =====================================
            # Quartiles
            # =====================================
            q1 = np.quantile(vals, 0.25)
            q3 = np.quantile(vals, 0.75)

            stats_text = (
                f"Median = {med:.2f}\n"
                f"IQR = {q1:.2f}–{q3:.2f}"
            )

            # 左上角无边框标注
            ax.text(
                0.04,
                0.94,
                stats_text,
                transform=ax.transAxes,
                fontsize=FS_STATS,
                fontweight="bold",
                color="#333333",
                ha="left",
                va="top",
            )

            # =====================================
            # Stage title
            # =====================================
            if feat_idx == 0:

                ax.set_title(
                    label_names[j],
                    fontsize=28,
                    fontweight="bold",
                    color=color,
                    pad=24,
                )

            # =====================================
            # Y label
            # =====================================
            if j == 0:

                ylabel_fs = 20 if feat == "brain_activity_score" else 24
                ax.set_ylabel(
                    f"{FEATURE_LABELS[feat]}\n(epoch count)",
                    fontsize=ylabel_fs,
                    fontweight="bold",
                    labelpad=20,
                )

            # =====================================
            # X label
            # =====================================
            if feat_idx == 1:

                ax.set_xlabel(
                    "Score (0–1)",
                    fontsize=20,
                    fontweight="bold",
                )

            # =====================================
            # Axis limits
            # =====================================
            ax.set_xlim(0, 1)

            ymax = HIST_COUNT_YMAX[feat][j]

            ax.set_ylim(0, ymax)

            # =====================================
            # Tick style
            # =====================================
            ax.tick_params(
                axis="both",
                labelsize=16,
                width=1.0,
            )

            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, _: f"{int(x):,}"
                )
            )

            # =====================================
            # Grid
            # =====================================
            ax.grid(
                axis="y",
                linestyle="--",
                linewidth=0.6,
                alpha=0.25,
                color="#999999",
            )

            ax.set_axisbelow(True)

            # =====================================
            # Spine
            # =====================================
            for spine in ax.spines.values():

                spine.set_color("#666666")

                spine.set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.90,
        bottom=0.10,
    )

    save_path = os.path.join(
        output_dir,
        "task1_figure_A_stage_histograms_pubstyle.png",
    )

    _save_figure_png_pdf(fig, save_path, dpi=600, facecolor="white")
    plt.close(fig)
def _plot_task1_figure_b_violin(df_norm: pd.DataFrame, output_dir: str) -> None:
    """图 B：1×2 小提琴（归一化 0–1）。"""
    label_names = [LABEL_NAMES[i] for i in LABEL_ORDER]

    with plt.rc_context(_pub_style_rc()):
        fig, axes_v = plt.subplots(1, 2, figsize=(16.0, 7.2), facecolor="white")

        fig.suptitle(
            "Score distributions by sleep stage\n"
            "(per-patient MinMax; all epochs pooled)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        axes_v[0].text(
            -0.12,
            1.02,
            "B",
            transform=axes_v[0].transAxes,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )

        for ax_i, feat in enumerate(FEATURES):
            col = feat + "_norm"
            ax = axes_v[ax_i]
            data_by_label = [
                df_norm[df_norm["label"] == lid][col].dropna().values for lid in LABEL_ORDER
            ]

            parts = ax.violinplot(
                data_by_label,
                positions=range(5),
                showmedians=True,
                showextrema=True,
                widths=0.68,
            )
            for idx, body in enumerate(parts["bodies"]):
                body.set_facecolor(LABEL_COLORS[LABEL_ORDER[idx]])
                body.set_alpha(0.78)
                body.set_edgecolor("#333333")
                body.set_linewidth(0.55)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)
            for key in ("cmaxes", "cmins", "cbars"):
                parts[key].set_color("#555555")
                parts[key].set_linewidth(1)

            ax.boxplot(
                data_by_label,
                positions=range(5),
                widths=0.12,
                patch_artist=False,
                whiskerprops=dict(linewidth=0),
                capprops=dict(linewidth=0),
                medianprops=dict(color="white", linewidth=2.2),
                boxprops=dict(color="#222222", linewidth=1.2),
                flierprops=dict(marker="", markersize=0),
            )

            ax.set_xticks(range(5))
            ax.set_xticklabels(label_names, fontsize=13, fontweight="bold")
            ax.set_ylabel("Score (0–1)", fontsize=12)
            ax.set_xlabel("Sleep stage", fontsize=12)
            ax.set_ylim(-0.10, 1.18)
            ax.tick_params(axis="y", labelsize=10)
            ax.yaxis.grid(True, alpha=0.3, color="#b0b0b0", linestyle="-", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.set_title(FEATURE_LABELS[feat], fontsize=13, fontweight="bold", pad=14)

            y_lo, y_hi = ax.get_ylim()
            for idx, data in enumerate(data_by_label):
                med = float(np.nanmedian(data)) if len(data) else float("nan")
                n_ep = int(len(data))
                if np.isfinite(med):
                    ax.text(
                        idx,
                        min(med + 0.04, y_hi - 0.03),
                        f"{med:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                    )
                ax.text(
                    idx,
                    y_lo + 0.025 * (y_hi - y_lo),
                    f"n={n_ep:,}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#333333",
                )

        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="",
                color="none",
                label="per-patient MinMax to [0, 1]",
            )
        ]
        axes_v[1].legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=10,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            title="Normalization",
            title_fontsize=11,
        )

        fig.subplots_adjust(left=0.08, right=0.80, top=0.84, bottom=0.12, wspace=0.32)

        out_path = os.path.join(output_dir, "task1_figure_B_normalized_violin.png")
        _save_figure_png_pdf(fig, out_path, dpi=800, facecolor="white")
        plt.close(fig)


# ==================================================
# 任务 1：各睡眠阶段标签分布
# ==================================================
def task1_label_distribution(df_norm: pd.DataFrame, output_dir: str):
    """
    按 label（0-4）统计 body_depth_score_norm / brain_activity_score_norm 的分布，
    输出统计表 + 图 A（直方图 PNG）+ 图 B（小提琴 PNG）。
    """
    print("\n" + "="*60)
    print(" 任务1：各睡眠阶段归一化特征分布")
    print("="*60)

    # ---------- 1-a 描述性统计 ----------
    rows = []
    for lid in LABEL_ORDER:
        sub = df_norm[df_norm["label"] == lid]
        for feat in FEATURES:
            col  = feat + "_norm"
            vals = sub[col].dropna()
            rows.append({
                "Label":   LABEL_NAMES[lid],
                "Feature": FEATURE_LABELS[feat],
                "N_epochs": len(vals),
                "Mean":    round(vals.mean(), 4),
                "Std":     round(vals.std(),  4),
                "Median":  round(vals.median(), 4),
                "Q25":     round(vals.quantile(0.25), 4),
                "Q75":     round(vals.quantile(0.75), 4),
                "Min":     round(vals.min(), 4),
                "Max":     round(vals.max(), 4),
            })

    df_stats = pd.DataFrame(rows)
    stats_path = os.path.join(output_dir, "task1_label_distribution_stats.csv")
    df_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(df_stats.to_string(index=False))
    print(f"\n→ 统计表已保存：{stats_path}")

    # ---------- 1-b + 1-c：图 A（直方图）、图 B（小提琴），分文件 ----------
    _plot_task1_figure_a_histograms(df_norm, output_dir)
    _plot_task1_figure_b_violin(df_norm, output_dir)

    return df_stats


# ==================================================
# 任务 2：疾病组 vs 正常组差异分析
# ==================================================
def compute_patient_features(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    计算每位患者级别的特征：
    - 全局均值（所有 epoch）
    - 各睡眠阶段的均值（分阶段统计）
    """
    print("[患者特征] 聚合每位患者的归一化特征……")
    rows = []
    for nsrrid, grp in df_norm.groupby("nsrrid"):
        row = {"nsrrid": nsrrid}
        for feat in FEATURES:
            col = feat + "_norm"
            row[f"{feat}_all_mean"]    = grp[col].mean()
            row[f"{feat}_all_median"]  = grp[col].median()
            for lid in LABEL_ORDER:
                sub = grp[grp["label"] == lid][col].dropna()
                lname = LABEL_NAMES[lid]
                row[f"{feat}_{lname}_mean"]   = sub.mean()   if len(sub) > 0 else np.nan
                row[f"{feat}_{lname}_median"] = sub.median() if len(sub) > 0 else np.nan
        rows.append(row)
    df_patient = pd.DataFrame(rows)
    print(f"[患者特征] 共 {len(df_patient)} 位患者。")
    return df_patient


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "†"
    return "ns"


def task2_disease_comparison(df_norm: pd.DataFrame, df_clinical: pd.DataFrame, output_dir: str):
    """
    对每种疾病分组（disease=1 vs normal=0），
    比较归一化特征的差异：Mann-Whitney U 检验 + Cohen's d，
    并进行 Benjamini-Hochberg FDR 多重校正。
    """
    print("\n" + "="*60)
    print(" 任务2：疾病组 vs 正常组差异分析（Mann-Whitney U + FDR）")
    print("="*60)

    df_patient = compute_patient_features(df_norm)
    df_patient["nsrrid"]  = df_patient["nsrrid"].astype(str)
    df_clinical["nsrrid"] = df_clinical["nsrrid"].astype(str)

    df_merged = pd.merge(df_patient, df_clinical, on="nsrrid", how="inner")
    print(f"[合并] 有效患者数：{len(df_merged)}")

    # 待分析的特征列（全局均值 + 各阶段均值）
    analysis_cols = []
    for feat in FEATURES:
        analysis_cols.append(f"{feat}_all_mean")
        for lid in LABEL_ORDER:
            analysis_cols.append(f"{feat}_{LABEL_NAMES[lid]}_mean")

    # ---------- 逐疾病、逐特征统计 ----------
    all_results = []
    for disease_col, disease_name in DISEASE_COLS.items():
        grp_disease = df_merged[df_merged[disease_col] == 1]
        grp_normal  = df_merged[df_merged[disease_col] == 0]
        n_d, n_n = len(grp_disease), len(grp_normal)
        print(f"\n  {disease_name}：疾病组 n={n_d}，正常组 n={n_n}")

        for col in analysis_cols:
            g1 = grp_disease[col].dropna()
            g0 = grp_normal[col].dropna()
            if len(g1) < 5 or len(g0) < 5:
                continue

            mwu_stat, pval = mannwhitneyu(g1, g0, alternative="two-sided")
            pooled_std = np.sqrt(
                ((len(g1)-1)*g1.std()**2 + (len(g0)-1)*g0.std()**2)
                / (len(g1) + len(g0) - 2)
            )
            cohens_d = (g1.mean() - g0.mean()) / pooled_std if pooled_std > 0 else np.nan

            all_results.append({
                "Disease":         disease_name,
                "Disease_col":     disease_col,
                "Feature":         col,
                "Disease_N":       len(g1),
                "Normal_N":        len(g0),
                "Disease_Mean":    round(g1.mean(),   4),
                "Normal_Mean":     round(g0.mean(),   4),
                "Disease_Median":  round(g1.median(), 4),
                "Normal_Median":   round(g0.median(), 4),
                "MWU_stat":        round(mwu_stat,    2),
                "p_raw":           pval,
                "Cohens_d":        round(cohens_d,    4) if not np.isnan(cohens_d) else np.nan,
            })

    df_res = pd.DataFrame(all_results)

    # ---------- BH-FDR 校正（全局） ----------
    reject, p_fdr, _, _ = multipletests(df_res["p_raw"], method="fdr_bh")
    df_res["p_fdr"]     = p_fdr
    df_res["sig_raw"]   = df_res["p_raw"].apply(_sig_stars)
    df_res["sig_fdr"]   = df_res["p_fdr"].apply(_sig_stars)
    df_res["p_raw"]     = df_res["p_raw"].round(6)
    df_res["p_fdr"]     = df_res["p_fdr"].round(6)

    # 保存完整结果
    res_path = os.path.join(output_dir, "task2_disease_comparison.csv")
    df_res.to_csv(res_path, index=False, encoding="utf-8-sig")
    print(f"\n→ 完整统计结果已保存：{res_path}")

    # 打印显著结果（raw p<0.05）
    sig_df = df_res[df_res["p_raw"] < 0.05].sort_values("p_raw")
    if len(sig_df):
        print(f"\n显著差异（raw p<0.05，共 {len(sig_df)} 项）：")
        print(sig_df[["Disease", "Feature", "Disease_Mean", "Normal_Mean",
                       "p_raw", "sig_raw", "p_fdr", "sig_fdr", "Cohens_d"]].to_string(index=False))
    else:
        print("\n未发现 raw p<0.05 的差异。")

    # ---------- 可视化 ----------
    _plot_pvalue_heatmap(df_res, analysis_cols, output_dir)
    _plot_boxplots(df_merged, output_dir)
    _plot_effect_size_heatmap(df_res, analysis_cols, output_dir)

    return df_res, df_merged


def _plot_pvalue_heatmap(df_res: pd.DataFrame, analysis_cols: list, output_dir: str):
    """p 值（-log10）热力图：行=疾病，列=特征"""
    disease_names  = list(DISEASE_COLS.values())
    col_labels     = analysis_cols

    # 构建矩阵
    pivot_raw = df_res.pivot_table(index="Disease", columns="Feature",
                                    values="p_raw", aggfunc="first")
    pivot_raw = pivot_raw.reindex(index=disease_names,
                                   columns=[c for c in col_labels if c in pivot_raw.columns])
    pivot_fdr = df_res.pivot_table(index="Disease", columns="Feature",
                                    values="p_fdr", aggfunc="first")
    pivot_fdr = pivot_fdr.reindex(index=pivot_raw.index, columns=pivot_raw.columns)

    nrow, ncol = pivot_raw.shape
    fig, axes = plt.subplots(1, 2, figsize=(max(14, ncol * 0.9 + 4), max(5, nrow * 0.6 + 2)))

    for ax, pivot, title in zip(
        axes,
        [pivot_raw, pivot_fdr],
        ["Raw p-value (-log10)", "BH-FDR corrected p (-log10)"],
    ):
        mat = -np.log10(pivot.values.astype(float) + 1e-10)
        im  = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=4)
        plt.colorbar(im, ax=ax, label="-log10(p)")

        ax.set_xticks(range(ncol))
        ax.set_xticklabels(
            [c.replace("body_depth_score_", "BDS\n")
              .replace("brain_activity_score_", "BAS\n")
              .replace("_mean", "")
             for c in pivot.columns],
            fontsize=8, rotation=45, ha="right",
        )
        ax.set_yticks(range(nrow))
        ax.set_yticklabels(pivot.index, fontsize=11)

        for i in range(nrow):
            for j in range(ncol):
                pval = pivot.values[i, j]
                if not np.isnan(pval):
                    sig = _sig_stars(pval)
                    color = "white" if mat[i, j] > 2 else "black"
                    ax.text(j, i, f"{pval:.3f}\n{sig}",
                            ha="center", va="center", fontsize=7.5, color=color)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axhline(-0.5, color="gray", lw=0.5)

    plt.suptitle("Disease vs Normal: p-value Heatmap (Mann-Whitney U)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "task2_pvalue_heatmap.png")
    _save_figure_png_pdf(fig, path, dpi=200)
    plt.close(fig)


def _plot_effect_size_heatmap(df_res: pd.DataFrame, analysis_cols: list, output_dir: str):
    """Cohen's d 效应量热力图"""
    disease_names = list(DISEASE_COLS.values())
    pivot_d = df_res.pivot_table(index="Disease", columns="Feature",
                                  values="Cohens_d", aggfunc="first")
    pivot_d = pivot_d.reindex(
        index=disease_names,
        columns=[c for c in analysis_cols if c in pivot_d.columns],
    )
    pivot_p = df_res.pivot_table(index="Disease", columns="Feature",
                                  values="p_fdr", aggfunc="first")
    pivot_p = pivot_p.reindex(index=pivot_d.index, columns=pivot_d.columns)

    nrow, ncol = pivot_d.shape
    fig, ax = plt.subplots(figsize=(max(14, ncol * 0.9 + 4), max(5, nrow * 0.6 + 2)))

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im   = ax.imshow(pivot_d.values.astype(float), aspect="auto", cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax, label="Cohen's d")

    ax.set_xticks(range(ncol))
    ax.set_xticklabels(
        [c.replace("body_depth_score_", "BDS\n")
          .replace("brain_activity_score_", "BAS\n")
          .replace("_mean", "")
         for c in pivot_d.columns],
        fontsize=8, rotation=45, ha="right",
    )
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(pivot_d.index, fontsize=11)

    for i in range(nrow):
        for j in range(ncol):
            d_val = pivot_d.values[i, j]
            p_val = pivot_p.values[i, j] if pivot_p.values[i, j] is not None else np.nan
            if not np.isnan(d_val):
                sig   = _sig_stars(p_val) if not np.isnan(p_val) else ""
                color = "white" if abs(d_val) > 0.25 else "black"
                ax.text(j, i, f"{d_val:.2f}\n{sig}",
                        ha="center", va="center", fontsize=7.5, color=color)

    ax.set_title("Disease vs Normal: Cohen's d (with FDR significance)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "task2_cohens_d_heatmap.png")
    _save_figure_png_pdf(fig, path, dpi=200)
    plt.close(fig)


def _plot_boxplots(df_merged: pd.DataFrame, output_dir: str):
    """
    为每种疾病 × 每个特征（全局均值）绘制箱线图，
    标注 Mann-Whitney p 值。
    """
    n_diseases = len(DISEASE_COLS)
    n_feats    = len(FEATURES)

    fig, axes = plt.subplots(
        n_feats, n_diseases,
        figsize=(n_diseases * 2.5, n_feats * 3.8),
        sharey="row",
    )
    if n_feats == 1:
        axes = axes[np.newaxis, :]

    for r, feat in enumerate(FEATURES):
        col = f"{feat}_all_mean"
        for c, (disease_col, disease_name) in enumerate(DISEASE_COLS.items()):
            ax = axes[r, c]
            d1 = df_merged[df_merged[disease_col] == 1][col].dropna()
            d0 = df_merged[df_merged[disease_col] == 0][col].dropna()

            bp = ax.boxplot(
                [d0, d1], labels=["Normal", "Disease"],
                patch_artist=True, notch=False,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            )
            bp["boxes"][0].set_facecolor("#74b9ff")
            bp["boxes"][0].set_alpha(0.75)
            bp["boxes"][1].set_facecolor("#fd79a8")
            bp["boxes"][1].set_alpha(0.75)

            if len(d0) >= 5 and len(d1) >= 5:
                _, pval = mannwhitneyu(d1, d0, alternative="two-sided")
                sig     = _sig_stars(pval)
                y_max   = max(d0.quantile(0.95), d1.quantile(0.95))
                y_range = y_max - min(d0.quantile(0.05), d1.quantile(0.05))
                ax.annotate(
                    f"p={pval:.4f}\n{sig}",
                    xy=(1.5, y_max + y_range * 0.12),
                    ha="center", fontsize=8.5,
                    color="red" if pval < 0.05 else "gray",
                    fontweight="bold" if pval < 0.05 else "normal",
                )

            if r == 0:
                ax.set_title(disease_name, fontsize=9.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{FEATURE_LABELS[feat]}\n(Normalized, mean)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Disease vs Normal: Patient-Level Mean of Normalized Features\n"
        "(p-value from Mann-Whitney U test)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "task2_boxplots.png")
    _save_figure_png_pdf(fig, path, dpi=200)
    plt.close(fig)


    # ---------- 分阶段的箱线图（每种疾病一张子图矩阵） ----------
    for disease_col, disease_name in DISEASE_COLS.items():
        fig, axes = plt.subplots(
            len(FEATURES), len(LABEL_ORDER),
            figsize=(len(LABEL_ORDER) * 2.8, len(FEATURES) * 3.5),
            sharey="row",
        )
        if len(FEATURES) == 1:
            axes = axes[np.newaxis, :]

        for r, feat in enumerate(FEATURES):
            for c, lid in enumerate(LABEL_ORDER):
                ax    = axes[r, c]
                col   = f"{feat}_{LABEL_NAMES[lid]}_mean"
                d1    = df_merged[df_merged[disease_col] == 1][col].dropna()
                d0    = df_merged[df_merged[disease_col] == 0][col].dropna()

                if len(d1) < 5 or len(d0) < 5:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=11, color="gray")
                    ax.set_axis_off()
                    continue

                bp = ax.boxplot(
                    [d0, d1], labels=["N", "D"],
                    patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                )
                color = LABEL_COLORS[lid]
                bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.4)
                bp["boxes"][1].set_facecolor(color); bp["boxes"][1].set_alpha(0.85)

                _, pval = mannwhitneyu(d1, d0, alternative="two-sided")
                sig     = _sig_stars(pval)
                y_max   = max(d0.quantile(0.95), d1.quantile(0.95))
                y_range = max(y_max - min(d0.quantile(0.05), d1.quantile(0.05)), 0.01)
                ax.annotate(
                    f"{pval:.3f}\n{sig}",
                    xy=(1.5, y_max + y_range * 0.15),
                    ha="center", fontsize=8,
                    color="red" if pval < 0.05 else "gray",
                )

                if r == 0:
                    ax.set_title(LABEL_NAMES[lid], fontsize=11, fontweight="bold",
                                 color=LABEL_COLORS[lid])
                if c == 0:
                    ax.set_ylabel(f"{FEATURE_LABELS[feat]}\n(Norm. mean)", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(axis="y", alpha=0.3, linestyle="--")

        fig.suptitle(
            f"{disease_name}: Disease vs Normal by Sleep Stage",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        safe_name = disease_col.replace(" ", "_")
        path = os.path.join(output_dir, f"task2_boxplot_stage_{safe_name}.png")
        _save_figure_png_pdf(fig, path, dpi=200)
        plt.close(fig)


# ==================================================
# 主流程
# ==================================================
def main():
    features_dir = CONFIG["features_dir"]
    output_dir   = CONFIG["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录：{output_dir}\n")

    # 仅任务 1：特征分布（统计表 + 合并图 A/B）
    all_df  = load_all_feature_files(features_dir)
    df_norm = normalize_per_patient(all_df)
    task1_label_distribution(df_norm, output_dir)

    print("\n" + "="*60)
    print(" Task1 完成（统计表 + task1 图 A/B 的 PNG 与 PDF）")
    print("="*60)


if __name__ == "__main__":
    main()
