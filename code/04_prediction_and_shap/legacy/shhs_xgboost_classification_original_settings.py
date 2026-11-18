"""
SHHS XGBoost 疾病分类分析
对每种疾病做三组对比分类：
  A. ECG-SDI 特征（shhs_sdi_features.csv）
  B. SDI 3D 聚类特征（SDI_Full_Layer_Features.csv，5 个特征）
  C. A + B 合并特征

使用 TreeSHAP 进行特征解释，输出特征重要性与 SHAP 分析论文插图（PNG + PDF，800 DPI）及 CSV。
全量训练后的 XGBoost 模型（含 StandardScaler）保存至 output/models/，可用于单患者推理。

直接修改下方 CONFIG 路径后运行：
    python shhs_xgboost_classification.py
    python shhs_xgboost_classification.py --predict 202037
"""

import os
import pickle
import re
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==================================================
# ★ 路径配置（按需修改）
# ==================================================
CONFIG = {
    # ECG-SDI 特征文件
    "ecg_sdi_csv": r"D:\Project\Python_Project\SDI\ECG-sdi\ECG-sdi\data\shhs_sdi_features.csv",
    # SDI 3D 全特征文件
    "sdi_full_csv": r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\SDI_Full_Layer_Features.csv",
    # 临床数据
    "shhs1_csv":      r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs1-dataset-0.20.0.csv",
    "cvd_csv":        r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs-cvd-summary-dataset-0.19.0.csv",
    "harmonized_csv": r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs-harmonized-dataset-0.20.0.csv",
    # 输出目录
    "output_dir": r"C:\Users\SaberShuo\Documents\SDI\analysis\importance_reproduction_20260824\outputs",
}

# ==================================================
# 常量
# ==================================================
# SDI 3D 聚类特征（来自 shhs_clustering_analysis.py）
SDI_CLUSTER_FEATURES = [
    "Conf_Weighted_AP",
    "Directional_Entropy",
    "Path_Length_2D",
    "Transition_Rate",
    "BB_Antagonistic_Ratio",
]

# 5 个 SDI-3D 特征的学术缩写与全称
FEATURE_ABBR = {
    "Transition_Rate":       "SSTR",   # Sleep State Transition Rate
    "Conf_Weighted_AP":      "CWSD",   # Confidence-Weighted Sleep Depth
    "Path_Length_2D":        "TDTL",   # Two-Dimensional Trajectory Length
    "BB_Antagonistic_Ratio": "BRAI",   # Brain-Body Antagonism Index
    "Directional_Entropy":   "STDE",   # Sleep Trajectory Directional Entropy
}
FEATURE_FULLNAME = {
    "Transition_Rate":       "Sleep State Transition Rate",
    "Conf_Weighted_AP":      "Confidence-Weighted Sleep Depth",
    "Path_Length_2D":        "Two-Dimensional Trajectory Length",
    "BB_Antagonistic_Ratio": "Brain-Body Antagonism Index",
    "Directional_Entropy":   "Sleep Trajectory Directional Entropy",
}

# ECG-SDI 特征列
ECG_SDI_FEATURES = ["AP", "CV", "RB", "RB03", "SK", "APPe", "DETRf", "MDR", "PR"]

# 疾病定义（与 shhs_clustering_analysis.py 一致）
DISEASE_COLS = {
    "cvd_binary":      "CVD",
    "htn_binary":      "HTN",
    "diabetes_binary": "Diabetes",
    "ahi_binary":      "OSA (AHI≥5)",
    "insomnia":        "Insomnia",
}

# 三组分类器
GROUP_NAMES = {
    "A": "ECG-SDI",
    "B": "SDI-3D (5 features)",
    "C": "Combined",
}
GROUP_COLORS = {"A": "#2196F3", "B": "#FF5722", "C": "#4CAF50"}

CV_FOLDS = 5  # 交叉验证折数
SHAP_DPI = 800  # 栅格图（PNG）输出分辨率
PUB_SAVE_FORMATS = ("png", "pdf")  # 特征重要性 / SHAP 论文插图输出格式

# 论文大图：热图 / 五病 SHAP 合一（画布与字号）
HEATMAP_FIG = {"w_per_col": 1.35, "w_base": 6.75, "h_per_row": 0.55, "h_base": 2.8}
HEATMAP_FONT = {
    "cell": 18, "tick": 20, "label": 22, "cbar_tick": 18,
    "legend": 19, "footnote": 16, "star_ms": 22,
}
# Keep the physical width moderate: Overleaf scales the whole figure, including text.
SHAP_FIG = {"w_per_panel": 3.45, "h_per_feat": 0.45, "h_base": 2.2, "wspace": 0.48}
SHAP_FONT = {
    "title": 24, "suptitle": 26, "ytick": 23, "xtick": 18,
    "cbar": 18, "xlabel": 19, "footnote": 18,
}
# 五病合一 Gain 条形图（与 SHAP 图字号对齐）
GAIN_BAR_FIG = {"w_per_panel": 3.2, "h_per_feat": 0.45, "h_base": 2.2}
GAIN_BAR_FONT = {
    "title": 24, "suptitle": 26, "ytick": 23, "xtick": 18,
    "xlabel": 19, "legend": 18,
}
SHAP_CACHE_NAME = "shap_figure_cache.pkl"
MODELS_DIR_NAME = "models"
MODELS_MANIFEST = "models_manifest.csv"

# 科研出版风格 matplotlib 参数
RC_PUB = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   9,
    "axes.linewidth":    0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.unicode_minus": False,
    "pdf.fonttype":      42,  # PDF 中保留 TrueType 字体（便于后期编辑）
    "ps.fonttype":       42,
}

# SHAP 五病合一图专用（覆盖 RC_PUB 中偏小的默认字号）
RC_PUB_SHAP = {
    **RC_PUB,
    "axes.labelsize":    SHAP_FONT["xlabel"],
    "axes.titlesize":    SHAP_FONT["title"],
    "xtick.labelsize":   SHAP_FONT["xtick"],
    "ytick.labelsize":   SHAP_FONT["ytick"],
    "legend.fontsize":   GAIN_BAR_FONT["legend"],
}

RC_PUB_GAIN_BAR = {
    **RC_PUB,
    "axes.labelsize":    GAIN_BAR_FONT["xlabel"],
    "axes.titlesize":    GAIN_BAR_FONT["title"],
    "xtick.labelsize":   GAIN_BAR_FONT["xtick"],
    "ytick.labelsize":   GAIN_BAR_FONT["ytick"],
    "legend.fontsize":   GAIN_BAR_FONT["legend"],
}


# ==================================================
# 数据加载
# ==================================================
def extract_nsrrid(s: str) -> str | None:
    """从文件名中提取 nsrrid，支持两种格式：
    shhs1-202916_features.csv  →  202916
    shhs1-202037_features_predictions.csv  →  202037
    """
    if pd.isna(s):
        return None
    m = re.search(r"shhs1-(\d+)", str(s))
    return m.group(1) if m else None


def load_ecg_sdi(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["nsrrid"] = df["filename"].apply(extract_nsrrid)
    df = df.dropna(subset=["nsrrid"])
    df["nsrrid"] = df["nsrrid"].astype(str)
    print(f"[ECG-SDI]  {len(df)} 行，列：{ECG_SDI_FEATURES}")
    return df[["nsrrid"] + ECG_SDI_FEATURES].copy()


def load_sdi_full(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["nsrrid"] = df["NSRRID"].apply(extract_nsrrid)
    df = df.dropna(subset=["nsrrid"])
    df["nsrrid"] = df["nsrrid"].astype(str)
    available = [c for c in SDI_CLUSTER_FEATURES if c in df.columns]
    print(f"[SDI-3D]   {len(df)} 行，使用特征：{available}")
    return df[["nsrrid"] + available].copy()


def _read_csv_any_encoding(path: str, **kwargs) -> pd.DataFrame:
    """依次尝试常见编码，避免 UnicodeDecodeError。"""
    for enc in ("utf-8", "latin-1", "gbk", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法以任何常见编码读取文件：{path}")


def load_clinical(shhs1_csv: str, cvd_csv: str, harmonized_csv: str) -> pd.DataFrame:
    df_b   = _read_csv_any_encoding(shhs1_csv,      low_memory=False)
    df_cvd = _read_csv_any_encoding(cvd_csv,        low_memory=False)
    df_h   = _read_csv_any_encoding(harmonized_csv, low_memory=False)

    for d in (df_b, df_cvd, df_h):
        d["nsrrid"] = d["nsrrid"].astype(str)

    df = pd.merge(df_b, df_cvd, on="nsrrid", how="inner")
    df_h_v1 = (df_h[df_h["visitnumber"] == 1]
               [["nsrrid", "nsrr_ahi_hp4u_aasm15"]].copy())
    df_h_v1["nsrrid"] = df_h_v1["nsrrid"].astype(str)
    df = pd.merge(df, df_h_v1, on="nsrrid", how="left")

    df["cvd_binary"]      = df["any_cvd"].fillna(0).astype(int)
    df["htn_binary"]      = df["htnderv_s1"].fillna(0).astype(int)
    df["diabetes_binary"] = df["parrptdiab"].fillna(0).astype(int)
    df["ahi_binary"]      = (df["nsrr_ahi_hp4u_aasm15"] >= 5).astype(int)
    df["poor_sleep"]      = ((df["ltdp10"] <= 2) & (df["rest10"] <= 2)).astype(int)
    df["good_sleep"]      = ((df["ltdp10"] >= 4) & (df["rest10"] >= 4)).astype(int)
    df["insomnia"]        = df["diffa10"].fillna(0).astype(int)

    keep = ["nsrrid"] + list(DISEASE_COLS.keys())
    print(f"[Clinical] {len(df)} 行")
    return df[keep].copy()


# ==================================================
# DeLong 检验（FastDeLong, Sun & Xu 2014）
# ==================================================
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N)
    T2[J] = T + 1
    return T2


def _fast_delong(pred_T: np.ndarray, m: int):
    """
    pred_T : shape [k, m+n]，前 m 列为正例预测概率，后 n 列为负例。
    返回 (aucs, cov_matrix)。
    """
    n = pred_T.shape[1] - m
    pos = pred_T[:, :m]
    neg = pred_T[:, m:]
    k   = pred_T.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(pred_T[r])

    aucs = (tz[:, :m].sum(1) - tx.sum(1)) / (m * n)
    v01  = (tz[:, :m] - tx) / n          # shape [k, m]
    v10  = 1.0 - (tz[:, m:] - ty) / m   # shape [k, n]

    # 当 k==2（比较两个分类器），np.cov 返回 2×2 矩阵
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01[0])]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10[0])]])
    cov = sx / m + sy / n
    return aucs, cov


def delong_roc_test(y_true: np.ndarray,
                    y_prob1: np.ndarray,
                    y_prob2: np.ndarray) -> tuple[float, float]:
    """
    DeLong 检验：比较两个分类器在同一批样本上的 AUC 差异。
    返回 (z_stat, p_value)，双侧检验。
    y_prob1 对应"备择假说更好"的分类器（通常是 Combined）。
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    if m == 0 or n == 0:
        return 0.0, 1.0

    predictions = np.vstack([y_prob1, y_prob2])
    ordered     = np.hstack([predictions[:, pos_idx], predictions[:, neg_idx]])

    aucs, cov = _fast_delong(ordered, m)
    diff = aucs[0] - aucs[1]
    var  = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return 0.0, 1.0

    z = diff / np.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), p


def _sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "†"
    return "ns"


# ==================================================
# XGBoost 交叉验证分类
# ==================================================
def run_cv_classification(X: np.ndarray, y: np.ndarray,
                          n_splits: int = CV_FOLDS) -> dict:
    """
    StratifiedKFold 交叉验证。
    返回指标统计、ROC 曲线数据，以及 **OOF 预测**（用于 DeLong 检验）。
    OOF 预测使用与 DeLong 配对所需的固定 CV 分割（random_state=42）。
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_aucs, fold_f1s, fold_precs, fold_recs, fold_aps = [], [], [], [], []
    all_fpr = np.linspace(0, 1, 200)
    tprs    = []

    oof_y_true = np.empty(len(y), dtype=int)
    oof_y_prob = np.empty(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        oof_y_true[test_idx] = y_te
        oof_y_prob[test_idx] = y_prob

        fold_aucs.append(roc_auc_score(y_te, y_prob))
        fold_f1s.append(f1_score(y_te, y_pred, zero_division=0))
        fold_precs.append(precision_score(y_te, y_pred, zero_division=0))
        fold_recs.append(recall_score(y_te, y_pred, zero_division=0))
        fold_aps.append(average_precision_score(y_te, y_prob))

        fpr, tpr, _ = roc_curve(y_te, y_prob)
        tprs.append(np.interp(all_fpr, fpr, tpr))

    mean_tpr      = np.mean(tprs, axis=0)
    mean_tpr[-1]  = 1.0
    std_tpr       = np.std(tprs, axis=0)

    return {
        "auc":        (np.mean(fold_aucs),  np.std(fold_aucs)),
        "f1":         (np.mean(fold_f1s),   np.std(fold_f1s)),
        "precision":  (np.mean(fold_precs), np.std(fold_precs)),
        "recall":     (np.mean(fold_recs),  np.std(fold_recs)),
        "ap":         (np.mean(fold_aps),   np.std(fold_aps)),
        "roc_fpr":    all_fpr,
        "roc_tpr":    mean_tpr,
        "roc_std":    std_tpr,
        "oof_y_true": oof_y_true,   # 用于 DeLong 配对检验
        "oof_y_prob": oof_y_prob,
    }


# ==================================================
# 绘图
# ==================================================
def plot_roc_for_disease(results: dict, disease_name: str,
                         n_pos: int, n_neg: int, output_dir: str,
                         delong_p: float | None = None,
                         delong_z: float | None = None):
    """
    为单个疾病绘制三组 ROC 曲线对比图。
    若传入 delong_p，则在图上标注 Combined vs ECG-SDI 的 DeLong 检验结果。
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for gk, gname in GROUP_NAMES.items():
        if gk not in results:
            continue
        res   = results[gk]
        fpr   = res["roc_fpr"]
        tpr   = res["roc_tpr"]
        std   = res["roc_std"]
        mean_auc, std_auc = res["auc"]

        ax.plot(fpr, tpr, color=GROUP_COLORS[gk], lw=2,
                label=f"{gname}  AUC={mean_auc:.3f}±{std_auc:.3f}")
        ax.fill_between(fpr, tpr - std, tpr + std,
                        color=GROUP_COLORS[gk], alpha=0.15)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"{disease_name}  (pos={n_pos}, neg={n_neg})\n"
        f"XGBoost {CV_FOLDS}-Fold Stratified CV ROC",
        fontsize=12, fontweight="bold",
    )

    # DeLong 检验标注
    if delong_p is not None:
        sig  = _sig_label(delong_p)
        auc_c = results["C"]["auc"][0] if "C" in results else 0
        auc_a = results["A"]["auc"][0] if "A" in results else 0
        delta = auc_c - auc_a
        sign  = "+" if delta >= 0 else ""
        txt   = (f"DeLong test (Combined vs ECG-SDI)\n"
                 f"z={delong_z:+.3f},  p={delong_p:.4f}  {sig}\n"
                 f"ΔAUC={sign}{delta:.4f}")
        ax.text(0.38, 0.08, txt, transform=ax.transAxes, fontsize=8.5,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fname = disease_name.replace(" ", "_").replace("(", "").replace(")", "").replace("≥", "ge")
    path  = os.path.join(output_dir, f"roc_{fname}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  → ROC 图已保存：{path}")


def plot_metric_summary(df_metrics: pd.DataFrame, output_dir: str):
    """
    汇总柱状图：只展示 ECG-SDI (A) 和 Combined (C)，
    Combined 柱上方标注 DeLong p 值。
    """
    diseases    = list(df_metrics["Disease"].unique())
    show_groups = ["A", "C"]   # 只展示这两组
    n_d         = len(diseases)
    x           = np.arange(n_d)
    width       = 0.32

    fig, ax = plt.subplots(figsize=(max(10, n_d * 1.8), 6))
    bar_tops = {d: 0.0 for d in diseases}

    for i, gk in enumerate(show_groups):
        sub  = df_metrics[df_metrics["Group"] == gk].set_index("Disease")
        aucs = [float(sub.loc[d, "AUC_mean"]) if d in sub.index else 0.0 for d in diseases]
        stds = [float(sub.loc[d, "AUC_std"])  if d in sub.index else 0.0 for d in diseases]
        bars = ax.bar(x + i * width, aucs, width, yerr=stds, capsize=4,
                      color=GROUP_COLORS[gk], alpha=0.82,
                      label=GROUP_NAMES[gk], error_kw={"elinewidth": 1.2})
        for bar, v, std, d in zip(bars, aucs, stds, diseases):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
            if gk == "C":
                bar_tops[d] = bar.get_height() + std + 0.028

    # Combined 柱上方标注 DeLong p 值（对 ECG-SDI）
    sub_c_p = df_metrics[df_metrics["Group"] == "C"].set_index("Disease")
    for di, d in enumerate(diseases):
        if d not in sub_c_p.index:
            continue
        pval = sub_c_p.loc[d, "DeLong_CvsA_p"]
        if pd.isna(pval):
            continue
        pval  = float(pval)
        sig   = _sig_label(pval)
        color = "red" if pval < 0.05 else "gray"
        plabel = f"p={pval:.4f}" if pval >= 0.0001 else "p<0.0001"
        cx = x[di] + 1 * width   # Combined 柱中心
        ax.text(cx, bar_tops[d], f"{plabel}\n{sig}", ha="center", va="bottom",
                fontsize=8, color=color,
                fontweight="bold" if pval < 0.05 else "normal")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(diseases, fontsize=11)
    ax.set_ylim(0.4, 1.15)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("AUC (mean ± std)", fontsize=11)
    ax.set_title(
        f"XGBoost Classification AUC — {CV_FOLDS}-Fold CV\n"
        f"(p-value: DeLong test, Combined vs ECG-SDI)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_auc_barplot.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n→ AUC 汇总柱状图已保存：{path}")


def plot_metric_heatmap(df_metrics: pd.DataFrame, output_dir: str):
    """AUC 热力图：行=疾病，列=指标×组别。"""
    metrics  = ["AUC", "F1", "Precision", "Recall", "AP"]
    groups   = list(GROUP_NAMES.keys())
    diseases = df_metrics["Disease"].unique().tolist()

    cols   = [f"{m}_{gk}" for m in metrics for gk in groups]
    mat    = pd.DataFrame(index=diseases, columns=cols, dtype=float)

    for _, row in df_metrics.iterrows():
        d  = row["Disease"]
        gk = row["Group"]
        for m in metrics:
            col = f"{m}_{gk}"
            mat.loc[d, col] = row[f"{m}_mean"]

    fig, ax = plt.subplots(figsize=(len(cols) * 0.9 + 2, len(diseases) * 0.6 + 2))
    im = ax.imshow(mat.values.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in cols],
        fontsize=7, rotation=45, ha="right",
    )
    ax.set_yticks(range(len(diseases)))
    ax.set_yticklabels(diseases, fontsize=10)

    for i, d in enumerate(diseases):
        for j, c in enumerate(cols):
            v = mat.loc[d, c]
            if not pd.isna(v):
                color = "white" if float(v) > 0.8 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=6.5, color=color)

    ax.set_title("XGBoost Metrics Heatmap (mean over CV folds)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_metrics_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"→ 指标热力图已保存：{path}")


# ==================================================
# 特征重要性与 SHAP 分析（论文插图）
# ==================================================
def _disease_filename(disease_name: str) -> str:
    return disease_name.replace(" ", "_").replace("(", "").replace(")", "").replace("≥", "ge")


def _train_xgb_full(X: np.ndarray, y: np.ndarray):
    """在全量数据上训练 XGBoost（与 CV 使用相同超参）。"""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    model  = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_sc, y)
    return model, scaler, X_sc


def _model_path(output_dir: str, disease_name: str, group_key: str) -> str:
    fname = f"model_{_disease_filename(disease_name)}_{group_key}.pkl"
    return os.path.join(output_dir, MODELS_DIR_NAME, fname)


def save_trained_model(output_dir: str, disease_name: str, group_key: str,
                       model, scaler: StandardScaler,
                       feature_names: list, n_pos: int, n_neg: int) -> str:
    """保存全量训练的 XGBoost 模型与 StandardScaler。"""
    models_dir = os.path.join(output_dir, MODELS_DIR_NAME)
    os.makedirs(models_dir, exist_ok=True)
    path = _model_path(output_dir, disease_name, group_key)
    payload = {
        "model":          model,
        "scaler":         scaler,
        "feature_names":  feature_names,
        "disease":        disease_name,
        "group":          group_key,
        "group_name":     GROUP_NAMES[group_key],
        "n_pos":          n_pos,
        "n_neg":          n_neg,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_trained_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_with_model(payload: dict, features: dict,
                       threshold: float = 0.5) -> dict:
    """用已保存模型对单个患者的特征字典做推理。"""
    feature_names = payload["feature_names"]
    missing = [c for c in feature_names if c not in features]
    if missing:
        raise ValueError(f"缺少特征：{missing}")
    X = np.array([[features[c] for c in feature_names]], dtype=float)
    X_sc = payload["scaler"].transform(X)
    prob = float(payload["model"].predict_proba(X_sc)[0, 1])
    return {
        "disease":    payload["disease"],
        "group":      payload["group"],
        "group_name": payload["group_name"],
        "prob":       prob,
        "pred":       int(prob >= threshold),
        "threshold":  threshold,
    }


def _patient_feature_dict(row: pd.Series, feature_names: list) -> dict:
    return {c: float(row[c]) for c in feature_names}


def predict_patient(nsrrid: str, output_dir: str | None = None,
                    threshold: float = 0.5) -> pd.DataFrame:
    """加载已保存模型，对指定 nsrrid 做全部疾病/特征组推理。"""
    output_dir = output_dir or CONFIG["output_dir"]
    manifest_path = os.path.join(output_dir, MODELS_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"未找到模型清单：{manifest_path}，请先运行 python shhs_xgboost_classification.py"
        )

    df_all, _ = _load_merged_feature_frames()
    nsrrid = str(nsrrid)
    patient = df_all[df_all["nsrrid"] == nsrrid]
    if patient.empty:
        raise ValueError(f"nsrrid={nsrrid} 不在合并特征表中")
    row = patient.iloc[0]

    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, m in manifest.iterrows():
        payload = load_trained_model(m["model_path"])
        feat_dict = _patient_feature_dict(row, payload["feature_names"])
        pred = predict_with_model(payload, feat_dict, threshold=threshold)
        rows.append({
            "nsrrid":     nsrrid,
            "disease":    pred["disease"],
            "group":      pred["group"],
            "group_name": pred["group_name"],
            "prob":       round(pred["prob"], 4),
            "pred":       pred["pred"],
            "threshold":  threshold,
        })
    return pd.DataFrame(rows)


def collect_feature_importance(X: np.ndarray, y: np.ndarray,
                                feature_names: list) -> dict:
    """在全量数据上训练 XGBoost，返回 gain 归一化特征重要性。"""
    model, _, _ = _train_xgb_full(X, y)
    return _gain_importance_from_model(model, feature_names)


def _gain_importance_from_model(model, feature_names: list) -> dict:
    scores = model.get_booster().get_score(importance_type="gain")
    imp = {}
    for fname, fscore in scores.items():
        idx = int(fname[1:])
        if idx < len(feature_names):
            imp[feature_names[idx]] = fscore
    total = sum(imp.values()) if imp else 1.0
    return {k: v / total for k, v in imp.items()}


def collect_shap_analysis(X: np.ndarray, y: np.ndarray,
                          feature_names: list) -> dict:
    """在全量数据上训练 XGBoost 并计算 TreeSHAP。"""
    model, _, X_sc = _train_xgb_full(X, y)
    return _shap_from_model(model, X_sc, feature_names)


def _shap_from_model(model, X_sc: np.ndarray, feature_names: list) -> dict:
    explainer   = shap.TreeExplainer(model)
    shap_values = _positive_class_shap(explainer.shap_values(X_sc))
    mean_abs = {
        fname: float(np.mean(np.abs(shap_values[:, i])))
        for i, fname in enumerate(feature_names)
    }
    return {
        "mean_abs_shap": mean_abs,
        "shap_values":   shap_values,
        "X_scaled":      X_sc,
        "feature_names": feature_names,
    }


def collect_model_explanations(X: np.ndarray, y: np.ndarray,
                               feature_names: list) -> tuple[dict, dict, object, StandardScaler]:
    """一次训练，同时返回 gain 特征重要性与 SHAP 分析结果，以及 model/scaler。"""
    model, scaler, X_sc = _train_xgb_full(X, y)
    gain = _gain_importance_from_model(model, feature_names)
    shap_entry = _shap_from_model(model, X_sc, feature_names)
    return gain, shap_entry, model, scaler


def _positive_class_shap(shap_values) -> np.ndarray:
    """统一为 (n_samples, n_features) 的正类 SHAP 矩阵。"""
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def _feature_display_label(name: str) -> str:
    """SDI-3D 特征显示缩写，其余保持原名。"""
    return FEATURE_ABBR.get(name, name)


def _heatmap_feature_label(name: str) -> str:
    """热图 Y 轴：统一用简称（SDI-3D 为学术缩写，ECG-SDI 保持原列名）。"""
    return _feature_display_label(name)


def _sdi_abbrev_footnote_text(sep: str = "  |  ") -> str:
    """SDI-3D 五特征：缩写 → 全称，用于图下方注释。"""
    return sep.join(
        f"{FEATURE_ABBR[f]}: {FEATURE_FULLNAME[f]}" for f in SDI_CLUSTER_FEATURES
    )


# Group C 热图：前 5 行 SDI-3D，其后 ECG-SDI（与 Combined 模型特征集一致）
HEATMAP_FEATURE_ORDER = list(SDI_CLUSTER_FEATURES) + list(ECG_SDI_FEATURES)

HEATMAP_DISEASE_ORDER = [
    "CVD",
    "Diabetes",
    "HTN",
    "Insomnia",
    "OSA (AHI≥5)",
]


def _save_pub_figure(
    fig, path_base: str, dpi: int = SHAP_DPI,
    bbox_inches: str | None = None, pad_inches: float = 0.2,
    formats: tuple[str, ...] = PUB_SAVE_FORMATS,
):
    """保存特征重要性 / SHAP 论文插图（默认 PNG + PDF）。bbox_inches=None 保留完整画布。"""
    for ext in formats:
        path = f"{path_base}.{ext}"
        kwargs = dict(
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor=fig.get_facecolor(),
        )
        if ext == "pdf":
            kwargs["format"] = "pdf"
        else:
            kwargs["dpi"] = dpi
            kwargs["format"] = ext
        fig.savefig(path, **kwargs)
        tag = "PDF（矢量）" if ext == "pdf" else f"{ext.upper()}, {dpi} DPI"
        print(f"→ 论文插图已保存（{tag}）：{path}")


def _pub_legend_handles():
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=GROUP_COLORS["A"], edgecolor="black", linewidth=0.45,
              alpha=0.88, label="ECG-SDI features"),
        Patch(facecolor=GROUP_COLORS["B"], edgecolor="black", linewidth=0.45,
              alpha=0.88, label="SDI-3D features"),
    ]


def _feat_order_by_importance(
    feat_cols: list,
    importance_by_feat: dict[str, float],
    sdi_on_top: bool = True,
) -> list:
    """特征排序；sdi_on_top 时 SDI-3D 五条置顶，组内按重要性降序。"""
    def _sort_group(cols: list) -> list:
        return sorted(
            cols,
            key=lambda f: importance_by_feat.get(f, 0.0),
            reverse=True,
        )

    if not sdi_on_top:
        return _sort_group(feat_cols)

    sdi_feats = [f for f in SDI_CLUSTER_FEATURES if f in feat_cols]
    ecg_feats = [f for f in feat_cols if f not in sdi_feats]
    return _sort_group(sdi_feats) + _sort_group(ecg_feats)


def _combined_feat_order(importance_data: dict, diseases: list,
                         feat_cols: list, gk: str = "C") -> list:
    """Combined 组特征顺序：SDI-3D 置顶，组内按跨疾病平均 Gain 降序。"""
    avg = {
        f: float(np.mean([
            importance_data.get(d, {}).get(gk, {}).get(f, 0.0)
            for d in diseases
        ]))
        for f in feat_cols
    }
    return _feat_order_by_importance(feat_cols, avg, sdi_on_top=True)


def save_feature_importance_csv(importance_data: dict, feature_sets: dict,
                                output_dir: str) -> str:
    """保存各疾病 gain 特征重要性 CSV。"""
    rows = []
    for disease in importance_data:
        for gk, gname in GROUP_NAMES.items():
            feat_cols = feature_sets.get(gk, [])
            imp = importance_data.get(disease, {}).get(gk, {})
            for f in feat_cols:
                rows.append({
                    "Disease": disease,
                    "Group": gk,
                    "Group_Name": gname,
                    "Feature": f,
                    "Feature_Label": _feature_display_label(f),
                    "Normalized_Gain": float(imp.get(f, 0.0)),
                })

    path_detail = os.path.join(output_dir, "feature_importance_by_disease.csv")
    if not rows:
        pd.DataFrame(columns=[
            "Disease", "Group", "Group_Name", "Feature", "Feature_Label",
            "Normalized_Gain", "Rank",
        ]).to_csv(path_detail, index=False, encoding="utf-8-sig")
        return path_detail

    df = pd.DataFrame(rows)
    df["Rank"] = (
        df.groupby(["Disease", "Group"])["Normalized_Gain"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    df = df.sort_values(["Disease", "Group", "Rank", "Feature"], ignore_index=True)
    df.to_csv(path_detail, index=False, encoding="utf-8-sig")
    print(f"→ 特征重要性 CSV 已保存：{path_detail}")
    return path_detail


def save_shap_csv(shap_data: dict, feature_sets: dict, output_dir: str) -> str:
    """保存各疾病 mean |SHAP| CSV。"""
    rows = []
    for disease in shap_data:
        for gk, gname in GROUP_NAMES.items():
            feat_cols = feature_sets.get(gk, [])
            imp = shap_data.get(disease, {}).get(gk, {}).get("mean_abs_shap", {})
            for f in feat_cols:
                rows.append({
                    "Disease": disease,
                    "Group": gk,
                    "Group_Name": gname,
                    "Feature": f,
                    "Feature_Label": _feature_display_label(f),
                    "Mean_Abs_SHAP": float(imp.get(f, 0.0)),
                })

    path_detail = os.path.join(output_dir, "shap_by_disease.csv")
    if not rows:
        pd.DataFrame(columns=[
            "Disease", "Group", "Group_Name", "Feature", "Feature_Label",
            "Mean_Abs_SHAP", "Rank",
        ]).to_csv(path_detail, index=False, encoding="utf-8-sig")
        return path_detail

    df = pd.DataFrame(rows)
    df["Rank"] = (
        df.groupby(["Disease", "Group"])["Mean_Abs_SHAP"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    df = df.sort_values(["Disease", "Group", "Rank", "Feature"], ignore_index=True)
    df.to_csv(path_detail, index=False, encoding="utf-8-sig")
    print(f"→ SHAP CSV 已保存：{path_detail}")
    return path_detail


def plot_fig_feature_importance_all_diseases(
    importance_data: dict, feature_sets: dict, output_dir: str,
    gk: str = "C", dpi: int = SHAP_DPI,
):
    """
    论文插图 Fig.1：五病合一特征重要性（Combined 组，归一化 Gain）。
    1×N 多面板水平条形图，统一特征排序与 x 轴。
    """
    diseases  = list(importance_data.keys())
    feat_cols = feature_sets.get(gk, [])
    if not diseases or not feat_cols:
        return

    feat_order = _combined_feat_order(importance_data, diseases, feat_cols, gk)
    n_d, n_feat = len(diseases), len(feat_cols)

    global_max = max(
        importance_data.get(d, {}).get(gk, {}).get(f, 0.0)
        for d in diseases for f in feat_cols
    )
    x_lim = global_max * 1.18 if global_max > 0 else 1.0
    gf = GAIN_BAR_FIG
    gft = GAIN_BAR_FONT

    with plt.rc_context(RC_PUB_GAIN_BAR):
        fig, axes = plt.subplots(
            1, n_d,
            figsize=(n_d * gf["w_per_panel"], n_feat * gf["h_per_feat"] + gf["h_base"]),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for di, (disease, ax) in enumerate(zip(diseases, axes)):
            imp = importance_data.get(disease, {}).get(gk, {})
            sv  = [imp.get(f, 0.0) for f in feat_order]
            clr = [
                GROUP_COLORS["A"] if f in feature_sets["A"] else GROUP_COLORS["B"]
                for f in feat_order
            ]
            labels = [_feature_display_label(f) for f in feat_order]

            ax.barh(
                range(n_feat), sv,
                color=clr, edgecolor="black", linewidth=0.45,
                height=0.72, alpha=0.88,
            )
            ax.set_xlim(0, x_lim)
            ax.set_yticks(range(n_feat))
            if di == 0:
                ax.set_yticklabels(labels, fontsize=gft["ytick"], fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.set_title(f"({chr(97 + di)}) {disease}",
                         fontsize=gft["title"], fontweight="bold", pad=8)
            ax.tick_params(axis="x", labelsize=gft["xtick"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", alpha=0.35, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=3)
            ax.tick_params(axis="y", length=0, pad=2)

        axes[-1].set_xlabel(
            "Normalized Gain Importance",
            fontsize=gft["xlabel"], fontweight="bold", labelpad=6,
        )
        fig.legend(
            handles=_pub_legend_handles(),
            loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False,
            fontsize=gft["legend"],
        )
        fig.suptitle(
            "Feature Importance across Five Diseases (Combined Model)",
            fontsize=gft["suptitle"], fontweight="bold", y=1.08,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        _save_pub_figure(fig, os.path.join(output_dir, "Fig_feature_importance_all_diseases"), dpi)
        plt.close(fig)


def _gain_matrix_from_importance(importance_data: dict, gk: str = "C") -> pd.DataFrame:
    """构建 (feature × disease) 归一化 Gain 矩阵。"""
    rows = []
    for disease in HEATMAP_DISEASE_ORDER:
        imp = importance_data.get(disease, {}).get(gk, {})
        for feat in HEATMAP_FEATURE_ORDER:
            rows.append({
                "Disease": disease,
                "Feature": feat,
                "Normalized_Gain": float(imp.get(feat, np.nan)),
            })
    return pd.DataFrame(rows)


def _gain_matrix_from_csv(csv_path: str, gk: str = "C") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    sub = df[(df["Group"] == gk) & df["Disease"].isin(HEATMAP_DISEASE_ORDER)]
    return sub[["Disease", "Feature", "Normalized_Gain"]].copy()


def plot_group_c_gain_heatmap(
    output_dir: str,
    importance_data: dict | None = None,
    csv_path: str | None = None,
    gk: str = "C",
    dpi: int = SHAP_DPI,
):
    """
    Group C 归一化 Gain 特征重要性热图（五病 × 14 特征）。
    每列标注 Top 3 星标，800 DPI 输出 PNG/PDF。
    """
    if importance_data is not None:
        long_df = _gain_matrix_from_importance(importance_data, gk=gk)
    elif csv_path and os.path.isfile(csv_path):
        long_df = _gain_matrix_from_csv(csv_path, gk=gk)
    else:
        print("→ 跳过热图：无 importance_data 且 CSV 不存在")
        return

    pivot = (
        long_df.pivot(index="Feature", columns="Disease", values="Normalized_Gain")
        .reindex(index=HEATMAP_FEATURE_ORDER, columns=HEATMAP_DISEASE_ORDER)
    )
    data = pivot.values.astype(float)
    n_feat, n_dis = data.shape

    vmin, vmax = 0.04, 0.13
    top3_mask = np.zeros_like(data, dtype=bool)
    for j in range(n_dis):
        col = data[:, j]
        valid = np.isfinite(col)
        if valid.sum() < 3:
            continue
        idx = np.argsort(col)[::-1][:3]
        top3_mask[idx, j] = True

    y_labels = [_heatmap_feature_label(f) for f in HEATMAP_FEATURE_ORDER]

    hf = HEATMAP_FIG
    ft = HEATMAP_FONT
    with plt.rc_context(RC_PUB):
        fig_w = hf["w_per_col"] * n_dis + hf["w_base"]
        fig_h = hf["h_per_row"] * n_feat + hf["h_base"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(
            data,
            cmap="RdYlBu_r",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )

        for i in range(n_feat):
            for j in range(n_dis):
                val = data[i, j]
                if not np.isfinite(val):
                    continue
                txt_color = "white" if val >= 0.105 else "black"
                ax.text(
                    j, i, f"{val:.4f}",
                    ha="center", va="center",
                    fontsize=ft["cell"], color=txt_color, fontweight="medium",
                )
                if top3_mask[i, j]:
                    ax.plot(
                        j + 0.32, i - 0.30, marker="*",
                        markersize=ft["star_ms"], color="white",
                        markeredgecolor="black", markeredgewidth=0.7,
                        clip_on=False, zorder=5,
                    )

        ax.set_xticks(range(n_dis))
        ax.set_xticklabels(HEATMAP_DISEASE_ORDER, fontsize=ft["tick"], fontweight="bold")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Disease", fontsize=ft["label"], labelpad=10, fontweight="bold")

        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(y_labels, fontsize=ft["tick"], fontweight="bold")
        ax.set_ylabel("Features", fontsize=ft["label"], labelpad=10, fontweight="bold")

        ax.set_xticks(np.arange(-0.5, n_dis, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_feat, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Importance index", fontsize=ft["label"], labelpad=12, fontweight="bold")
        cbar.set_ticks([0.04, 0.06, 0.08, 0.10, 0.13])
        cbar.ax.tick_params(labelsize=ft["cbar_tick"])
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_fontweight("bold")

        star_handle = Line2D(
            [], [], linestyle="None", marker="*",
            markersize=ft["star_ms"], markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=0.9,
            label="Top 3 features in each disease",
        )
        leg = fig.legend(
            handles=[star_handle],
            loc="lower right",
            bbox_to_anchor=(0.99, 0.08),
            frameon=False,
            fontsize=ft["legend"],
            handletextpad=0.5,
        )
        for txt in leg.get_texts():
            txt.set_fontweight("bold")
        fig.text(
            0.5, 0.028,
            "*Values represent normalized gain importance. "
            "Higher values indicate greater contribution to disease prediction.",
            ha="center", va="bottom",
            fontsize=ft["footnote"], fontweight="bold",
            style="italic", color="#333333",
        )

        fig.subplots_adjust(
            left=0.21, right=0.84, bottom=0.15, top=0.88,
        )
        out_base = os.path.join(output_dir, "Fig_gain_importance_heatmap_group_C")
        _save_pub_figure(fig, out_base, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)


def _add_shap_feature_colorbar(fig, anchor_ax, label: str = "Feature value",
                               sf: dict = SHAP_FONT):
    """在最后一列子图旁添加 Feature value 色条（避免 shap 自带 colorbar 撑出空白条）。"""
    import matplotlib.cm as cm

    m = cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array([0.0, 1.0])
    cbar = fig.colorbar(
        m, ax=anchor_ax, ticks=[0, 1],
        fraction=0.046, pad=0.02, aspect=28,
    )
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label(label, fontsize=sf["cbar"], labelpad=6, fontweight="bold")
    cbar.ax.tick_params(labelsize=sf["cbar"], length=0)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    cbar.outline.set_visible(False)
    return cbar


def _enlarge_shap_panel_fonts(fig, panel_axes, sf: dict = SHAP_FONT):
    """放大 SHAP beeswarm 面板与坐标轴字号（覆盖 shap 默认偏小字体）。"""
    panel_ids = {id(a) for a in np.atleast_1d(panel_axes) if a.get_visible()}
    for ax in np.atleast_1d(panel_axes):
        if not ax.get_visible():
            continue
        ax.title.set_fontsize(sf["title"])
        ax.title.set_fontweight("bold")
        ax.tick_params(axis="y", labelsize=sf["ytick"], length=3, width=0.8)
        ax.tick_params(axis="x", labelsize=sf["xtick"])
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(sf["ytick"])
            lbl.set_fontweight("bold")
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(sf["xtick"])
        xlab = ax.xaxis.get_label()
        if xlab.get_text():
            xlab.set_fontsize(sf["xlabel"])
            xlab.set_fontweight("bold")
    # 共享 x 轴标签（shap 常在首/末 panel 写入）
    for ax in fig.axes:
        if id(ax) in panel_ids:
            continue
        xlab = ax.xaxis.get_label()
        if xlab.get_text() and "SHAP" in xlab.get_text():
            xlab.set_fontsize(sf["xlabel"])


def plot_fig_shap_all_diseases(
    shap_data: dict, feature_sets: dict, output_dir: str,
    gk: str = "C", dpi: int = SHAP_DPI,
):
    """
    论文插图 Fig.2：五病合一 SHAP beeswarm 分析（Combined 组）。
    1×N 多面板，统一特征排序与 x 轴。
    """
    diseases  = list(shap_data.keys())
    feat_cols = feature_sets.get(gk, [])
    if not diseases or not feat_cols:
        return

    avg_imp = {
        f: float(np.mean([
            shap_data.get(d, {}).get(gk, {}).get("mean_abs_shap", {}).get(f, 0.0)
            for d in diseases
        ]))
        for f in feat_cols
    }
    # SDI-3D 五条固定置顶（与热图行序一致），ECG-SDI 按跨疾病平均 |SHAP| 降序
    sdi_part = [f for f in SDI_CLUSTER_FEATURES if f in feat_cols]
    ecg_part = sorted(
        [f for f in feat_cols if f not in sdi_part],
        key=lambda f: avg_imp.get(f, 0.0),
        reverse=True,
    )
    feat_order = sdi_part + ecg_part

    ref_entry = next(
        (shap_data.get(d, {}).get(gk) for d in diseases if shap_data.get(d, {}).get(gk)),
        None,
    )
    if ref_entry is None:
        return
    order_idx = [ref_entry["feature_names"].index(f) for f in feat_order]

    n_d, n_feat = len(diseases), len(feat_cols)

    sf = SHAP_FIG
    fig_w = n_d * sf["w_per_panel"] + 0.8   # 右侧留给 colorbar
    fig_h = max(8.5, n_feat * sf["h_per_feat"] + sf["h_base"])

    with plt.rc_context(RC_PUB_SHAP):
        fig, axes = plt.subplots(
            1, n_d,
            figsize=(fig_w, fig_h),
            sharex=False,
            facecolor="white",
        )
        axes = np.atleast_1d(axes)
        panel_axes = []

        for i, (disease, ax) in enumerate(zip(diseases, axes)):
            entry = shap_data.get(disease, {}).get(gk)
            if not entry:
                ax.set_visible(False)
                continue

            panel_axes.append(ax)
            display_names = [_feature_display_label(f) for f in entry["feature_names"]]
            expl = shap.Explanation(
                values=entry["shap_values"],
                data=entry["X_scaled"],
                feature_names=display_names,
            )
            shap.plots.beeswarm(
                expl,
                max_display=n_feat,
                order=order_idx,
                show=False,
                ax=ax,
                color_bar=False,
                plot_size=None,
                s=28,
            )
            ax.set_title(f"({chr(97 + i)}) {disease}",
                         fontsize=SHAP_FONT["title"], fontweight="bold", pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if panel_axes:
            _add_shap_feature_colorbar(fig, panel_axes[-1], sf=SHAP_FONT)
            # 统一 x 轴标签字号（仅最底行显示）
            for ax in panel_axes:
                ax.set_xlabel("")

        _enlarge_shap_panel_fonts(fig, np.array(panel_axes))

        fig.suptitle(
            "SHAP Value Analysis across Five Diseases",
            fontsize=SHAP_FONT["suptitle"], fontweight="bold", y=1.01,
        )
        fig.supxlabel(
            "SHAP value (impact on model output)",
            fontsize=SHAP_FONT["xlabel"], fontweight="bold", y=0.015,
        )
        fig.subplots_adjust(
            left=0.08, right=0.97, bottom=0.12, top=0.88,
            wspace=sf.get("wspace", 0.32),
        )
        _save_pub_figure(
            fig,
            os.path.join(output_dir, "Fig_shap_analysis_all_diseases"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.25,
        )
        plt.close(fig)


def save_shap_figure_cache(shap_data: dict, feature_sets: dict, output_dir: str):
    """缓存 SHAP 数据，便于仅重绘 Fig_shap 而无需重跑 XGBoost。"""
    path = os.path.join(output_dir, SHAP_CACHE_NAME)
    with open(path, "wb") as f:
        pickle.dump({"shap_data": shap_data, "feature_sets": feature_sets}, f)
    print(f"→ SHAP 图缓存已保存：{path}")


def load_shap_figure_cache(output_dir: str) -> tuple[dict, dict] | None:
    path = os.path.join(output_dir, SHAP_CACHE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["shap_data"], obj["feature_sets"]


def plot_publication_figures(
    importance_data: dict, shap_data: dict, feature_sets: dict,
    output_dir: str, dpi: int = SHAP_DPI,
):
    """生成两张论文插图（特征重要性 + SHAP 分析）及配套 CSV。"""
    save_feature_importance_csv(importance_data, feature_sets, output_dir)
    save_shap_csv(shap_data, feature_sets, output_dir)
    plot_fig_feature_importance_all_diseases(
        importance_data, feature_sets, output_dir, gk="C", dpi=dpi,
    )
    plot_fig_shap_all_diseases(
        shap_data, feature_sets, output_dir, gk="C", dpi=dpi,
    )
    csv_path = os.path.join(output_dir, "feature_importance_by_disease.csv")
    plot_group_c_gain_heatmap(
        output_dir,
        importance_data=importance_data,
        csv_path=csv_path,
        gk="C",
        dpi=dpi,
    )


# ==================================================
# 主流程
# ==================================================
def main():
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录：{output_dir}\n")

    # ---------- 加载特征 ----------
    df_ecg  = load_ecg_sdi(CONFIG["ecg_sdi_csv"])
    df_sdi  = load_sdi_full(CONFIG["sdi_full_csv"])
    df_clin = load_clinical(CONFIG["shhs1_csv"], CONFIG["cvd_csv"], CONFIG["harmonized_csv"])

    # 三路合并（以 inner join 确保每个患者在所有数据源都有记录）
    df_all = (df_clin
              .merge(df_ecg, on="nsrrid", how="inner", suffixes=("", "_ecg"))
              .merge(df_sdi, on="nsrrid", how="inner", suffixes=("", "_sdi")))
    print(f"\n[合并后] {len(df_all)} 位患者可用于分析\n")

    # 三组特征列（处理列名冲突）
    ecg_cols = [c for c in df_all.columns if c in ECG_SDI_FEATURES
                or c.rstrip("_ecg") in ECG_SDI_FEATURES and c.endswith("_ecg")]
    ecg_cols = ECG_SDI_FEATURES  # 以原始名为准（已做 inner merge 保留原列）
    sdi_cols = [c for c in df_all.columns
                if c in SDI_CLUSTER_FEATURES
                or (c.endswith("_sdi") and c[:-4] in SDI_CLUSTER_FEATURES)]
    sdi_cols = [c for c in SDI_CLUSTER_FEATURES if c in df_all.columns]

    feature_sets = {
        "A": ecg_cols,
        "B": sdi_cols,
        "C": ecg_cols + sdi_cols,
    }
    print("特征组：")
    for gk, cols in feature_sets.items():
        print(f"  {GROUP_NAMES[gk]}: {cols}")

    # ---------- 逐疾病分类 ----------
    all_metric_rows  = []
    shap_data        = {}   # {disease_name: {gk: collect_shap_analysis 返回值}}
    importance_data  = {}   # {disease_name: {gk: {feature: norm_gain}}}
    model_manifest   = []

    for disease_col, disease_name in DISEASE_COLS.items():
        print(f"\n{'='*60}")
        print(f"  疾病：{disease_name}")
        print(f"{'='*60}")

        sub = df_all[["nsrrid", disease_col] + ecg_cols + sdi_cols].dropna()
        y   = sub[disease_col].values.astype(int)
        n_pos, n_neg = y.sum(), (y == 0).sum()
        print(f"  正例(疾病)={n_pos}, 负例(正常)={n_neg}, 患病率={n_pos/len(y)*100:.1f}%")

        if n_pos < 20 or n_neg < 20:
            print("  样本不足，跳过。")
            continue

        disease_results = {}
        shap_data[disease_name]        = {}
        importance_data[disease_name]  = {}
        for gk, feat_cols in feature_sets.items():
            available = [c for c in feat_cols if c in sub.columns]
            if not available:
                continue
            X = sub[available].values
            print(f"  [{GROUP_NAMES[gk]}] 训练中…", end="  ", flush=True)
            res = run_cv_classification(X, y)
            disease_results[gk] = res
            # 特征重要性（Gain）与 SHAP 分析（单次训练），并保存全量模型
            gain, shap_entry, model, scaler = collect_model_explanations(X, y, available)
            importance_data[disease_name][gk] = gain
            shap_data[disease_name][gk]       = shap_entry
            model_path = save_trained_model(
                output_dir, disease_name, gk, model, scaler, available, n_pos, n_neg,
            )
            model_manifest.append({
                "disease":        disease_name,
                "group":          gk,
                "group_name":     GROUP_NAMES[gk],
                "model_path":     model_path,
                "n_features":     len(available),
                "feature_names":  "|".join(available),
                "n_pos":          n_pos,
                "n_neg":          n_neg,
            })
            print(f"AUC={res['auc'][0]:.3f}±{res['auc'][1]:.3f}  "
                  f"F1={res['f1'][0]:.3f}±{res['f1'][1]:.3f}  "
                  f"[模型已保存]")

            all_metric_rows.append({
                "Disease":        disease_name,
                "Group":          gk,
                "Group_Name":     GROUP_NAMES[gk],
                "N_pos":          n_pos,
                "N_neg":          n_neg,
                "AUC_mean":       round(res["auc"][0],       4),
                "AUC_std":        round(res["auc"][1],       4),
                "F1_mean":        round(res["f1"][0],        4),
                "F1_std":         round(res["f1"][1],        4),
                "Precision_mean": round(res["precision"][0], 4),
                "Precision_std":  round(res["precision"][1], 4),
                "Recall_mean":    round(res["recall"][0],    4),
                "Recall_std":     round(res["recall"][1],    4),
                "AP_mean":        round(res["ap"][0],        4),
                "AP_std":         round(res["ap"][1],        4),
                "DeLong_CvsA_z":  None,   # 下方填入
                "DeLong_CvsA_p":  None,
                "DeLong_CvsA_sig": None,
            })

        # ---------- DeLong 检验：Combined vs ECG-SDI ----------
        delong_z, delong_p = None, None
        if "C" in disease_results and "A" in disease_results:
            delong_z, delong_p = delong_roc_test(
                disease_results["C"]["oof_y_true"],   # 两组 OOF 标签相同
                disease_results["C"]["oof_y_prob"],
                disease_results["A"]["oof_y_prob"],
            )
            sig = _sig_label(delong_p)
            print(f"  [DeLong C vs A] z={delong_z:+.3f},  p={delong_p:.4f}  {sig}")

            # 回填到 all_metric_rows 中 C 和 A 对应的行
            for row in all_metric_rows:
                if row["Disease"] == disease_name and row["Group"] in ("A", "C"):
                    row["DeLong_CvsA_z"]   = round(delong_z, 4)
                    row["DeLong_CvsA_p"]   = round(delong_p, 6)
                    row["DeLong_CvsA_sig"] = sig

        # ---------- 挑选 A 错 C 对 的患者编号 ----------
        if "A" in disease_results and "C" in disease_results:
            nsrrids   = sub["nsrrid"].values
            y_true    = disease_results["A"]["oof_y_true"]
            a_prob    = disease_results["A"]["oof_y_prob"]
            c_prob    = disease_results["C"]["oof_y_prob"]
            a_pred    = (a_prob >= 0.5).astype(int)
            c_pred    = (c_prob >= 0.5).astype(int)

            a_correct = (a_pred == y_true)
            c_correct = (c_pred == y_true)
            mask_rescued   = (~a_correct) & c_correct   # A错 C对
            mask_both_wrong = (~a_correct) & (~c_correct)
            mask_a_only    = a_correct & (~c_correct)   # A对 C错（退步）
            mask_both_ok   = a_correct & c_correct

            df_pred = pd.DataFrame({
                "nsrrid":      nsrrids,
                "y_true":      y_true,
                "ECG_SDI_prob": a_prob.round(4),
                "Combined_prob": c_prob.round(4),
                "ECG_SDI_pred": a_pred,
                "Combined_pred": c_pred,
                "category": np.select(
                    [mask_rescued, mask_both_wrong, mask_a_only, mask_both_ok],
                    ["rescued_by_combined", "both_wrong", "combined_regressed", "both_correct"],
                    default="unknown",
                ),
            })

            # 打印统计
            print(f"\n  [预测对比] A错C对（rescued）: {mask_rescued.sum()}  "
                  f"两者都错: {mask_both_wrong.sum()}  "
                  f"A对C错（退步）: {mask_a_only.sum()}  "
                  f"两者都对: {mask_both_ok.sum()}")

            # 保存全部预测结果
            fname    = disease_name.replace(" ", "_").replace("(", "").replace(")", "").replace("≥", "ge")
            pred_path = os.path.join(output_dir, f"casestudy_{fname}_predictions.csv")
            df_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
            print(f"  → 全患者预测结果已保存：{pred_path}")

            # 单独保存 rescued 患者编号
            df_rescued = df_pred[df_pred["category"] == "rescued_by_combined"].copy()
            rescued_path = os.path.join(output_dir, f"casestudy_{fname}_rescued_ids.csv")
            df_rescued.to_csv(rescued_path, index=False, encoding="utf-8-sig")
            print(f"  → Rescued 患者编号已保存（{len(df_rescued)} 人）：{rescued_path}")
            if len(df_rescued):
                print(f"     前10个 nsrrid：{df_rescued['nsrrid'].head(10).tolist()}")

        # 绘制该疾病的 ROC 对比图
        plot_roc_for_disease(disease_results, disease_name, n_pos, n_neg,
                             output_dir, delong_p=delong_p, delong_z=delong_z)

    # ---------- 汇总输出 ----------
    df_metrics = pd.DataFrame(all_metric_rows)
    csv_path   = os.path.join(output_dir, "xgboost_metrics_summary.csv")
    df_metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n→ 指标汇总表已保存：{csv_path}")

    if model_manifest:
        manifest_path = os.path.join(output_dir, MODELS_MANIFEST)
        pd.DataFrame(model_manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"→ 模型清单已保存（{len(model_manifest)} 个）：{manifest_path}")
        print(f"  模型目录：{os.path.join(output_dir, MODELS_DIR_NAME)}")

    print("\n指标汇总（AUC）：")
    pivot = df_metrics.pivot_table(
        index="Disease", columns="Group_Name", values="AUC_mean"
    )
    print(pivot.to_string())

    plot_metric_summary(df_metrics, output_dir)
    plot_metric_heatmap(df_metrics, output_dir)
    save_shap_figure_cache(shap_data, feature_sets, output_dir)
    plot_publication_figures(
        importance_data, shap_data, feature_sets, output_dir, dpi=SHAP_DPI,
    )

    print("\n" + "="*60)
    print(" 所有任务完成！")
    print("="*60)


def _load_merged_feature_frames():
    """加载并合并 ECG-SDI + SDI-3D + 临床表（与 main 相同）。"""
    df_ecg  = load_ecg_sdi(CONFIG["ecg_sdi_csv"])
    df_sdi  = load_sdi_full(CONFIG["sdi_full_csv"])
    df_clin = load_clinical(CONFIG["shhs1_csv"], CONFIG["cvd_csv"], CONFIG["harmonized_csv"])
    df_all = (df_clin
              .merge(df_ecg, on="nsrrid", how="inner", suffixes=("", "_ecg"))
              .merge(df_sdi, on="nsrrid", how="inner", suffixes=("", "_sdi")))
    ecg_cols = ECG_SDI_FEATURES
    sdi_cols = [c for c in SDI_CLUSTER_FEATURES if c in df_all.columns]
    feature_sets = {"A": ecg_cols, "B": sdi_cols, "C": ecg_cols + sdi_cols}
    return df_all, feature_sets


def _collect_combined_shap_only() -> tuple[dict, dict]:
    """仅对五病 Combined 组训练并计算 SHAP（用于快速重绘 Fig_shap）。"""
    df_all, feature_sets = _load_merged_feature_frames()
    ecg_cols = feature_sets["A"]
    sdi_cols = feature_sets["B"]
    combined = feature_sets["C"]
    shap_data = {}

    print(f"[合并后] {len(df_all)} 位患者\n")
    for disease_col, disease_name in DISEASE_COLS.items():
        print(f"  SHAP：{disease_name} …", flush=True)
        sub = df_all[["nsrrid", disease_col] + combined].dropna()
        y = sub[disease_col].values.astype(int)
        if y.sum() < 20 or (y == 0).sum() < 20:
            print("    样本不足，跳过")
            continue
        available = [c for c in combined if c in sub.columns]
        X = sub[available].values
        _, shap_entry, _, _ = collect_model_explanations(X, y, available)
        shap_data[disease_name] = {"C": shap_entry}

    return shap_data, feature_sets


def main_heatmap_only():
    """仅从已有 CSV 绘制 Group C Gain 热图（无需重跑 XGBoost）。"""
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "feature_importance_by_disease.csv")
    print(f"从 CSV 绘制热图：{csv_path}\n")
    plot_group_c_gain_heatmap(output_dir, csv_path=csv_path, dpi=SHAP_DPI)


def main_shap_figure_only():
    """重绘 Fig_shap：优先读缓存，否则仅重算 Combined SHAP。"""
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cached = load_shap_figure_cache(output_dir)
    if cached:
        shap_data, feature_sets = cached
        print(f"从缓存加载 SHAP：{os.path.join(output_dir, SHAP_CACHE_NAME)}\n")
    else:
        print("未找到 SHAP 缓存，仅计算 Combined 组（比完整 main 快）…\n")
        shap_data, feature_sets = _collect_combined_shap_only()
        save_shap_figure_cache(shap_data, feature_sets, output_dir)

    plot_fig_shap_all_diseases(
        shap_data, feature_sets, output_dir, gk="C", dpi=SHAP_DPI,
    )


def main_predict(nsrrid: str, threshold: float = 0.5):
    """对单个患者加载已保存模型并输出全部疾病预测。"""
    output_dir = CONFIG["output_dir"]
    print(f"患者 nsrrid={nsrrid}，阈值={threshold}\n")
    df_pred = predict_patient(nsrrid, output_dir=output_dir, threshold=threshold)
    print(df_pred.to_string(index=False))

    fname = f"patient_{nsrrid}_predictions.csv"
    out_path = os.path.join(output_dir, fname)
    df_pred.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n→ 已保存：{out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--heatmap", "--heatmap-only"):
        main_heatmap_only()
    elif len(sys.argv) > 1 and sys.argv[1] in ("--shap", "--shap-only"):
        main_shap_figure_only()
    elif len(sys.argv) > 2 and sys.argv[1] == "--predict":
        main_predict(sys.argv[2])
    else:
        main()
