"""
SHHS disease classification using the current 14-feature SDI panel.

This is a single-model analysis: all 14 features are used together for each
disease.  The XGBoost hyperparameters, fold construction, scaling, decision
threshold, and full-sample Gain/TreeSHAP analysis follow the original
shhs_xgboost_classification.py implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_CSV = Path(
    r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14\shhs_SDI_14_Features.csv"
)
SHHS1_CSV = Path(
    r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs1-dataset-0.20.0.csv"
)
CVD_CSV = Path(
    r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs-cvd-summary-dataset-0.19.0.csv"
)
HARMONIZED_CSV = Path(
    r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs-harmonized-dataset-0.20.0.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

ECG_SDI_FEATURES = ["AP", "RB", "MDR", "PR", "CV", "SK"]
TRAJECTORY_FEATURES = [
    "SSTR", "CWSD", "TDTL", "BRAI", "STDE", "TOS", "TCI", "SRI"
]
FEATURES = ECG_SDI_FEATURES + TRAJECTORY_FEATURES
DISPLAY_LABELS = {"BRAI": "BBAI"}

DISEASE_COLS = {
    "cvd_binary": "CVD",
    "htn_binary": "HTN",
    "diabetes_binary": "Diabetes",
    "ahi_binary": "OSA (AHI≥5)",
    "insomnia": "Insomnia",
}
DISEASE_ORDER = list(DISEASE_COLS.values())

CV_FOLDS = 5
RANDOM_STATE = 42
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
DECISION_THRESHOLD = 0.5
PUB_DPI = 800

BLOCK_COLORS = {
    "ECG-SDI (6 features)": "#2196F3",
    "Trajectory (8 features)": "#FF5722",
}

RC_PUB = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------
def extract_nsrrid(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.search(r"shhs1-(\d+)", str(value))
    return match.group(1) if match else None


def read_csv_any_encoding(path: Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "gbk", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode CSV: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_table() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(FEATURE_CSV)
    expected = ["NSRRID"] + FEATURES
    missing_columns = [column for column in expected if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in expected]
    if missing_columns:
        raise ValueError(f"Feature CSV is missing columns: {missing_columns}")
    if extra_columns:
        raise ValueError(f"Unexpected columns in feature CSV: {extra_columns}")

    df["nsrrid"] = df["NSRRID"].apply(extract_nsrrid)
    if df["nsrrid"].isna().any():
        raise ValueError("Some NSRRID values could not be parsed")
    if df["nsrrid"].duplicated().any():
        duplicated = df.loc[df["nsrrid"].duplicated(), "nsrrid"].tolist()[:10]
        raise ValueError(f"Duplicate patient IDs detected: {duplicated}")

    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    missing_counts = numeric.isna().sum()
    nonfinite_counts = (~np.isfinite(numeric.to_numpy())).sum(axis=0)
    if missing_counts.sum() or nonfinite_counts.sum():
        raise ValueError(
            "Invalid feature values detected: "
            f"missing={missing_counts.to_dict()}, "
            f"nonfinite={dict(zip(FEATURES, nonfinite_counts.tolist()))}"
        )
    df[FEATURES] = numeric

    audit = {
        "source_file": str(FEATURE_CSV),
        "source_sha256": sha256_file(FEATURE_CSV),
        "rows": int(len(df)),
        "unique_patient_ids": int(df["nsrrid"].nunique()),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "missing_by_feature": {key: int(value) for key, value in missing_counts.items()},
        "nonfinite_by_feature": dict(
            zip(FEATURES, [int(value) for value in nonfinite_counts])
        ),
    }
    return df[["nsrrid"] + FEATURES].copy(), audit


def load_clinical_table() -> pd.DataFrame:
    baseline = read_csv_any_encoding(SHHS1_CSV, low_memory=False)
    cvd = read_csv_any_encoding(CVD_CSV, low_memory=False)
    harmonized = read_csv_any_encoding(HARMONIZED_CSV, low_memory=False)

    for frame in (baseline, cvd, harmonized):
        frame["nsrrid"] = frame["nsrrid"].astype(str)

    clinical = baseline.merge(cvd, on="nsrrid", how="inner")
    ahi_visit1 = harmonized.loc[
        harmonized["visitnumber"].eq(1),
        ["nsrrid", "nsrr_ahi_hp4u_aasm15"],
    ].copy()
    clinical = clinical.merge(ahi_visit1, on="nsrrid", how="left")

    # Disease definitions are intentionally identical to the original code.
    clinical["cvd_binary"] = clinical["any_cvd"].fillna(0).astype(int)
    clinical["htn_binary"] = clinical["htnderv_s1"].fillna(0).astype(int)
    clinical["diabetes_binary"] = clinical["parrptdiab"].fillna(0).astype(int)
    clinical["ahi_binary"] = clinical["nsrr_ahi_hp4u_aasm15"].ge(5).astype(int)
    clinical["insomnia"] = clinical["diffa10"].fillna(0).astype(int)
    return clinical[["nsrrid"] + list(DISEASE_COLS)].copy()


def load_analysis_table() -> tuple[pd.DataFrame, dict]:
    features, audit = load_feature_table()
    clinical = load_clinical_table()
    merged = clinical.merge(features, on="nsrrid", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No patients remained after merging features and clinical labels")
    audit["clinical_rows"] = int(len(clinical))
    audit["merged_rows"] = int(len(merged))
    audit["unmatched_feature_rows"] = int(len(features) - len(merged))
    return merged, audit


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
def make_model() -> XGBClassifier:
    return XGBClassifier(**MODEL_PARAMS)


def run_cv_classification(X: np.ndarray, y: np.ndarray) -> dict:
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_metrics = []
    all_fpr = np.linspace(0, 1, 200)
    interpolated_tprs = []
    oof_probability = np.empty(len(y), dtype=float)
    oof_fold = np.empty(len(y), dtype=int)

    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])

        model = make_model()
        model.fit(X_train, y[train_index])
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= DECISION_THRESHOLD).astype(int)

        oof_probability[test_index] = probability
        oof_fold[test_index] = fold
        fpr, tpr, _ = roc_curve(y[test_index], probability)
        interpolated_tprs.append(np.interp(all_fpr, fpr, tpr))
        fold_metrics.append(
            {
                "fold": fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "n_pos_test": int(y[test_index].sum()),
                "n_neg_test": int((y[test_index] == 0).sum()),
                "auc": float(roc_auc_score(y[test_index], probability)),
                "f1": float(f1_score(y[test_index], prediction, zero_division=0)),
                "precision": float(
                    precision_score(y[test_index], prediction, zero_division=0)
                ),
                "recall": float(
                    recall_score(y[test_index], prediction, zero_division=0)
                ),
                "ap": float(average_precision_score(y[test_index], probability)),
            }
        )

    fold_frame = pd.DataFrame(fold_metrics)
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0
    return {
        "fold_metrics": fold_frame,
        "auc": (float(fold_frame["auc"].mean()), float(fold_frame["auc"].std(ddof=0))),
        "f1": (float(fold_frame["f1"].mean()), float(fold_frame["f1"].std(ddof=0))),
        "precision": (
            float(fold_frame["precision"].mean()),
            float(fold_frame["precision"].std(ddof=0)),
        ),
        "recall": (
            float(fold_frame["recall"].mean()),
            float(fold_frame["recall"].std(ddof=0)),
        ),
        "ap": (float(fold_frame["ap"].mean()), float(fold_frame["ap"].std(ddof=0))),
        "roc_fpr": all_fpr,
        "roc_tpr": mean_tpr,
        "roc_std": np.std(interpolated_tprs, axis=0),
        "oof_probability": oof_probability,
        "oof_fold": oof_fold,
    }


def train_full_model(X: np.ndarray, y: np.ndarray) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = make_model()
    model.fit(X_scaled, y)
    return model, scaler, X_scaled


def explain_full_model(model, X_scaled: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    raw_gain = model.get_booster().get_score(importance_type="gain")
    gain = {feature: 0.0 for feature in FEATURES}
    for key, value in raw_gain.items():
        index = int(key[1:])
        if index < len(FEATURES):
            gain[FEATURES[index]] = float(value)
    total_gain = sum(gain.values()) or 1.0
    gain = {feature: value / total_gain for feature, value in gain.items()}

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    mean_abs_shap = {
        feature: float(np.mean(np.abs(shap_values[:, index])))
        for index, feature in enumerate(FEATURES)
    }
    return gain, mean_abs_shap, shap_values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_figure(fig, base_path: Path, dpi: int = PUB_DPI) -> None:
    fig.savefig(
        base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        base_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"
    )


def feature_color(feature: str) -> str:
    if feature in ECG_SDI_FEATURES:
        return BLOCK_COLORS["ECG-SDI (6 features)"]
    return BLOCK_COLORS["Trajectory (8 features)"]


def plot_single_roc(result: dict, disease: str, n_pos: int, n_neg: int) -> None:
    with plt.rc_context(RC_PUB):
        fig, ax = plt.subplots(figsize=(7, 6))
        mean_auc, std_auc = result["auc"]
        ax.plot(
            result["roc_fpr"], result["roc_tpr"], color="#4CAF50", lw=2.2,
            label=f"All 14 features  AUC={mean_auc:.3f}±{std_auc:.3f}",
        )
        ax.fill_between(
            result["roc_fpr"],
            np.clip(result["roc_tpr"] - result["roc_std"], 0, 1),
            np.clip(result["roc_tpr"] + result["roc_std"], 0, 1),
            color="#4CAF50", alpha=0.18,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.set_title(
            f"{disease} (pos={n_pos}, neg={n_neg})\n"
            f"XGBoost {CV_FOLDS}-Fold Stratified CV ROC",
            fontweight="bold",
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", disease).strip("_")
        fig.savefig(OUTPUT_DIR / f"roc_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_all_roc(results: dict) -> None:
    with plt.rc_context(RC_PUB):
        fig, axes = plt.subplots(1, len(DISEASE_ORDER), figsize=(19, 3.8))
        for index, (disease, ax) in enumerate(zip(DISEASE_ORDER, axes)):
            result = results[disease]
            mean_auc, std_auc = result["auc"]
            ax.plot(result["roc_fpr"], result["roc_tpr"], color="#4CAF50", lw=2)
            ax.fill_between(
                result["roc_fpr"],
                np.clip(result["roc_tpr"] - result["roc_std"], 0, 1),
                np.clip(result["roc_tpr"] + result["roc_std"], 0, 1),
                color="#4CAF50", alpha=0.18,
            )
            ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.45)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"({chr(97 + index)}) {disease}\nAUC={mean_auc:.3f}±{std_auc:.3f}", fontweight="bold")
            ax.set_xlabel("False Positive Rate")
            if index == 0:
                ax.set_ylabel("True Positive Rate")
            ax.grid(alpha=0.25)
        fig.suptitle("Disease Classification with the 14-Feature SDI Panel", fontsize=15, fontweight="bold", y=1.04)
        fig.tight_layout()
        save_figure(fig, OUTPUT_DIR / "Fig_roc_all_diseases")
        plt.close(fig)


def mean_order(values_by_disease: dict) -> list[str]:
    return sorted(
        FEATURES,
        key=lambda feature: np.mean(
            [values_by_disease[disease][feature] for disease in DISEASE_ORDER]
        ),
        reverse=True,
    )


def plot_gain_panels(gain_by_disease: dict) -> None:
    order = mean_order(gain_by_disease)
    global_max = max(
        gain_by_disease[disease][feature]
        for disease in DISEASE_ORDER for feature in FEATURES
    )
    with plt.rc_context(RC_PUB):
        fig, axes = plt.subplots(1, len(DISEASE_ORDER), figsize=(20, 8.5), sharex=True)
        for index, (disease, ax) in enumerate(zip(DISEASE_ORDER, axes)):
            values = [gain_by_disease[disease][feature] for feature in order]
            ax.barh(
                np.arange(len(order)), values,
                color=[feature_color(feature) for feature in order],
                edgecolor="black", linewidth=0.4, alpha=0.88,
            )
            ax.set_xlim(0, global_max * 1.15)
            ax.set_yticks(np.arange(len(order)))
            if index == 0:
                ax.set_yticklabels(order, fontsize=11, fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.set_title(f"({chr(97 + index)}) {disease}", fontsize=14, fontweight="bold")
            ax.grid(axis="x", alpha=0.3, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("Normalized Gain")
        handles = [
            Patch(facecolor=color, edgecolor="black", label=label)
            for label, color in BLOCK_COLORS.items()
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2, frameon=False)
        fig.suptitle("Feature Importance across Five Diseases (14-Feature Model)", fontsize=16, fontweight="bold", y=1.045)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_figure(fig, OUTPUT_DIR / "Fig_feature_importance_all_diseases")
        plt.close(fig)


def plot_gain_heatmap(gain_by_disease: dict) -> None:
    order = mean_order(gain_by_disease)
    matrix = np.array(
        [[gain_by_disease[disease][feature] for disease in DISEASE_ORDER] for feature in order]
    )
    top3 = np.zeros_like(matrix, dtype=bool)
    for column in range(matrix.shape[1]):
        top3[np.argsort(matrix[:, column])[::-1][:3], column] = True

    with plt.rc_context(RC_PUB):
        fig, ax = plt.subplots(figsize=(11.5, 9.5))
        image = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto")
        midpoint = float((np.nanmin(matrix) + np.nanmax(matrix)) / 2)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(
                    column, row, f"{value:.4f}", ha="center", va="center",
                    fontsize=10, color="white" if value > midpoint else "black",
                )
                if top3[row, column]:
                    ax.plot(column + 0.34, row - 0.30, marker="*", markersize=12,
                            color="white", markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xticks(np.arange(len(DISEASE_ORDER)), labels=DISEASE_ORDER, fontweight="bold")
        ax.set_yticks(np.arange(len(order)), labels=order, fontweight="bold")
        ax.xaxis.tick_top()
        ax.set_xlabel("Disease", fontweight="bold", labelpad=10)
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Features", fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(DISEASE_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Normalized Gain", fontweight="bold")
        fig.suptitle("Normalized Gain Importance for the 14-Feature Model", fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        save_figure(fig, OUTPUT_DIR / "Fig_gain_heatmap")
        plt.close(fig)


def plot_shap_panels(shap_entries: dict, mean_shap_by_disease: dict) -> None:
    order = mean_order(mean_shap_by_disease)
    order_indices = [FEATURES.index(feature) for feature in order]
    with plt.rc_context(RC_PUB):
        # The figure is ultimately scaled to journal two-column width.  Use a
        # moderately compact canvas and deliberately large source fonts so
        # feature labels remain readable after that reduction.
        fig, axes = plt.subplots(1, len(DISEASE_ORDER), figsize=(18.5, 10.2))
        for index, (disease, ax) in enumerate(zip(DISEASE_ORDER, axes)):
            entry = shap_entries[disease]
            explanation = shap.Explanation(
                values=entry["shap_values"],
                data=entry["X_scaled"],
                feature_names=[DISPLAY_LABELS.get(feature, feature) for feature in FEATURES],
            )
            shap.plots.beeswarm(
                explanation,
                max_display=len(FEATURES),
                order=order_indices,
                color_bar=False,
                show=False,
                ax=ax,
                plot_size=None,
                s=18,
            )
            ax.set_title(
                f"({chr(97 + index)}) {disease}",
                fontsize=22,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("SHAP value", fontsize=18, fontweight="bold", labelpad=7)
            ax.tick_params(
                axis="y",
                labelsize=20,
                labelcolor="#000000",
                colors="#000000",
                length=3,
                width=1.0,
                pad=3,
            )
            ax.tick_params(axis="x", labelsize=15, length=3, width=0.9)
            for label in ax.get_yticklabels():
                # SHAP's default y-label style is light gray.  Force fully
                # opaque black, heavy labels for journal-size reproduction.
                label.set_color("#000000")
                label.set_alpha(1.0)
                label.set_fontsize(20)
                label.set_fontweight("black")
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            "TreeSHAP Analysis across Five Diseases",
            fontsize=25,
            fontweight="bold",
            y=0.995,
        )
        fig.subplots_adjust(left=0.085, right=0.925, bottom=0.11, top=0.90, wspace=0.46)
        scalar = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="coolwarm")
        scalar.set_array([])
        colorbar_axis = fig.add_axes([0.947, 0.29, 0.014, 0.40])
        colorbar = fig.colorbar(scalar, cax=colorbar_axis)
        colorbar.set_ticks([0, 1], labels=["Low", "High"])
        colorbar.ax.tick_params(labelsize=15)
        colorbar.set_label("Feature value", fontsize=18, fontweight="bold", labelpad=8)
        save_figure(fig, OUTPUT_DIR / "Fig_shap_analysis_all_diseases")
        plt.close(fig)


def plot_auc_summary(metrics: pd.DataFrame) -> None:
    with plt.rc_context(RC_PUB):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(metrics))
        bars = ax.bar(
            x, metrics["AUC_mean"], yerr=metrics["AUC_std"], capsize=5,
            color="#4CAF50", edgecolor="black", alpha=0.86,
        )
        for bar, mean, std in zip(bars, metrics["AUC_mean"], metrics["AUC_std"]):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.008,
                    f"{mean:.3f}", ha="center", va="bottom")
        ax.set_xticks(x, metrics["Disease"])
        ax.set_ylim(0.45, max(0.85, float((metrics["AUC_mean"] + metrics["AUC_std"]).max()) + 0.07))
        ax.set_ylabel("Mean AUC (5-fold CV)")
        ax.set_title("Disease Classification Performance: All 14 Features", fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_figure(fig, OUTPUT_DIR / "Fig_auc_summary")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Output and orchestration
# ---------------------------------------------------------------------------
def save_model(model, scaler, disease: str, n_pos: int, n_neg: int) -> Path:
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", disease).strip("_")
    path = model_dir / f"xgb_{safe}_all14.pkl"
    payload = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURES,
        "disease": disease,
        "group": "ALL14",
        "group_name": "All 14 features",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "decision_threshold": DECISION_THRESHOLD,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def save_run_manifest(audit: dict) -> None:
    manifest = {
        "analysis": "SHHS disease classification using all 14 SDI features",
        "feature_csv": str(FEATURE_CSV),
        "clinical_files": [str(SHHS1_CSV), str(CVD_CSV), str(HARMONIZED_CSV)],
        "output_dir": str(OUTPUT_DIR),
        "features": FEATURES,
        "disease_definitions": DISEASE_COLS,
        "cv_folds": CV_FOLDS,
        "cv_shuffle": True,
        "random_state": RANDOM_STATE,
        "standardization": "StandardScaler fitted within each CV training fold",
        "decision_threshold": DECISION_THRESHOLD,
        "model_parameters": MODEL_PARAMS,
        "explanation_model": "full-sample model with normalized Gain and TreeSHAP",
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
            "scipy": scipy.__version__,
        },
        "input_audit": audit,
    }
    with (OUTPUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, audit = load_analysis_table()
    save_run_manifest(audit)
    print(f"Input feature rows: {audit['rows']}")
    print(f"Merged analysis rows: {audit['merged_rows']}")
    print(f"Features ({len(FEATURES)}): {FEATURES}")

    metric_rows = []
    fold_frames = []
    oof_frames = []
    class_rows = []
    gain_rows = []
    shap_rows = []
    model_rows = []
    cv_results = {}
    gain_by_disease = {}
    mean_shap_by_disease = {}
    shap_entries = {}

    X = data[FEATURES].to_numpy(dtype=float)

    for disease_column, disease in DISEASE_COLS.items():
        print(f"\n[{disease}]", flush=True)
        y = data[disease_column].to_numpy(dtype=int)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        if n_pos < 20 or n_neg < 20:
            raise ValueError(f"Insufficient class count for {disease}: pos={n_pos}, neg={n_neg}")
        class_rows.append(
            {
                "Disease": disease,
                "N_total": int(len(y)),
                "N_pos": n_pos,
                "N_neg": n_neg,
                "Prevalence": n_pos / len(y),
            }
        )

        cv_result = run_cv_classification(X, y)
        cv_results[disease] = cv_result
        fold_frame = cv_result["fold_metrics"].copy()
        fold_frame.insert(0, "Disease", disease)
        fold_frames.append(fold_frame)
        oof_frames.append(
            pd.DataFrame(
                {
                    "nsrrid": data["nsrrid"],
                    "Disease": disease,
                    "y_true": y,
                    "fold": cv_result["oof_fold"],
                    "y_probability": cv_result["oof_probability"],
                    "y_pred": (cv_result["oof_probability"] >= DECISION_THRESHOLD).astype(int),
                }
            )
        )

        metric_rows.append(
            {
                "Disease": disease,
                "N_total": int(len(y)),
                "N_pos": n_pos,
                "N_neg": n_neg,
                "Prevalence": n_pos / len(y),
                "N_features": len(FEATURES),
                "AUC_mean": cv_result["auc"][0],
                "AUC_std": cv_result["auc"][1],
                "F1_mean": cv_result["f1"][0],
                "F1_std": cv_result["f1"][1],
                "Precision_mean": cv_result["precision"][0],
                "Precision_std": cv_result["precision"][1],
                "Recall_mean": cv_result["recall"][0],
                "Recall_std": cv_result["recall"][1],
                "AP_mean": cv_result["ap"][0],
                "AP_std": cv_result["ap"][1],
            }
        )

        model, scaler, X_scaled = train_full_model(X, y)
        gain, mean_abs_shap, shap_values = explain_full_model(model, X_scaled)
        gain_by_disease[disease] = gain
        mean_shap_by_disease[disease] = mean_abs_shap
        shap_entries[disease] = {
            "shap_values": shap_values,
            "X_scaled": X_scaled,
        }
        for feature in FEATURES:
            gain_rows.append(
                {
                    "Disease": disease,
                    "Feature": feature,
                    "Feature_Block": "ECG-SDI" if feature in ECG_SDI_FEATURES else "Trajectory",
                    "Normalized_Gain": gain[feature],
                }
            )
            shap_rows.append(
                {
                    "Disease": disease,
                    "Feature": feature,
                    "Feature_Block": "ECG-SDI" if feature in ECG_SDI_FEATURES else "Trajectory",
                    "Mean_Abs_SHAP": mean_abs_shap[feature],
                }
            )

        model_path = save_model(model, scaler, disease, n_pos, n_neg)
        model_rows.append(
            {
                "Disease": disease,
                "Model_Path": str(model_path),
                "N_features": len(FEATURES),
                "Feature_Names": "|".join(FEATURES),
                "N_pos": n_pos,
                "N_neg": n_neg,
            }
        )
        plot_single_roc(cv_result, disease, n_pos, n_neg)
        print(
            f"  AUC={cv_result['auc'][0]:.3f}±{cv_result['auc'][1]:.3f}; "
            f"F1={cv_result['f1'][0]:.3f}±{cv_result['f1'][1]:.3f}"
        )

    metrics = pd.DataFrame(metric_rows)
    folds = pd.concat(fold_frames, ignore_index=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    classes = pd.DataFrame(class_rows)
    gains = pd.DataFrame(gain_rows)
    shap_summary = pd.DataFrame(shap_rows)
    models = pd.DataFrame(model_rows)

    gains["Rank"] = gains.groupby("Disease")["Normalized_Gain"].rank(
        ascending=False, method="dense"
    ).astype(int)
    shap_summary["Rank"] = shap_summary.groupby("Disease")["Mean_Abs_SHAP"].rank(
        ascending=False, method="dense"
    ).astype(int)

    metrics.to_csv(OUTPUT_DIR / "xgboost_metrics_summary.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUTPUT_DIR / "xgboost_fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUTPUT_DIR / "oof_predictions_all_diseases.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(OUTPUT_DIR / "disease_class_counts.csv", index=False, encoding="utf-8-sig")
    gains.sort_values(["Disease", "Rank", "Feature"]).to_csv(
        OUTPUT_DIR / "feature_importance_by_disease.csv", index=False, encoding="utf-8-sig"
    )
    shap_summary.sort_values(["Disease", "Rank", "Feature"]).to_csv(
        OUTPUT_DIR / "shap_by_disease.csv", index=False, encoding="utf-8-sig"
    )
    models.to_csv(OUTPUT_DIR / "models_manifest.csv", index=False, encoding="utf-8-sig")

    with (OUTPUT_DIR / "shap_figure_cache.pkl").open("wb") as handle:
        pickle.dump(
            {
                "feature_names": FEATURES,
                "disease_order": DISEASE_ORDER,
                "entries": shap_entries,
            },
            handle,
        )

    plot_all_roc(cv_results)
    plot_auc_summary(metrics)
    plot_gain_panels(gain_by_disease)
    plot_gain_heatmap(gain_by_disease)
    plot_shap_panels(shap_entries, mean_shap_by_disease)

    print("\nSummary:")
    print(metrics[["Disease", "AUC_mean", "AUC_std", "F1_mean", "AP_mean"]].to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
