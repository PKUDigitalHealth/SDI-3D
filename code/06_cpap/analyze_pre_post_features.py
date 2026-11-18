from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


BASE = Path(r"C:\Users\SaberShuo\Desktop")
PRE_PATH = BASE / "\u6cbb\u7597\u524d.csv"
POST_PATH = BASE / "\u6cbb\u7597\u540e.csv"
OUT_DIR = Path(r"C:\Users\SaberShuo\Documents\SDI\analysis")

FEATURES = [
    "Body_CV", "Body_SK", "Body_RB", "Body_AP", "Body_APEn", "Body_DFA",
    "Body_MDR", "REM_PR",
    "REM_CV", "REM_SK", "REM_RB", "REM_AP", "REM_APEn",
    "REM_High_Activation_Ratio",
    "REM_High_Activation_Ratio_During_REM", "REM_NREM_Contrast",
    "Conf_Weighted_AP", "BB_Antagonistic_Ratio", "Path_Length_2D",
    "Transition_Rate_per_hour", "Directional_Entropy", "TOS", "TCI", "SRI",
]

# Only high-confidence spelling variants are used for sensitivity analysis.
TYPO_PAIRS = {
    "\u4e8e\u91d1\u57ce_predictions.csv": "\u4e8e\u91d1\u6210_predictions.csv",
    "\u4ed8\u5065\u5c97_predictions.csv": "\u4ed8\u5efa\u5c97_predictions.csv",
    "\u5b8b\u670b\u98de_predictions.csv": "\u5b8b\u9e4f\u98de_predictions.csv",
    "\u6731\u5408\u5168_predictions.csv": "\u6731\u548c\u5168_predictions.csv",
    "\u77f3\u6c5f\u9f99_predictions.csv": "\u77f3\u9f99\u6c5f_predictions.csv",
    "\u8bb8\u56fd\u5e86_predictions.csv": "\u5f90\u56fd\u5e86_predictions.csv",
}


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(float)
    valid = np.isfinite(p)
    out = np.full(len(p), np.nan)
    if not valid.any():
        return pd.Series(out, index=pvalues.index)
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    out[np.flatnonzero(valid)] = restored
    return pd.Series(out, index=pvalues.index)


def rank_biserial(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if diff.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff), method="average")
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def summarize_pair(pre_values: pd.Series, post_values: pd.Series) -> dict:
    x = pd.to_numeric(pre_values, errors="coerce").to_numpy(float)
    y = pd.to_numeric(post_values, errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    d = y - x
    n = len(d)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else np.nan
    se = sd_d / np.sqrt(n) if n > 1 else np.nan
    crit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    if n > 1 and np.any(d != 0):
        t_p = float(stats.ttest_rel(y, x, nan_policy="omit").pvalue)
        w_p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                                   method="approx").pvalue)
    else:
        t_p = 1.0
        w_p = 1.0
    pre_mean = float(np.mean(x))
    return {
        "n_pairs": n,
        "pre_mean": pre_mean,
        "pre_sd": float(np.std(x, ddof=1)),
        "pre_median": float(np.median(x)),
        "pre_q1": float(np.quantile(x, 0.25)),
        "pre_q3": float(np.quantile(x, 0.75)),
        "post_mean": float(np.mean(y)),
        "post_sd": float(np.std(y, ddof=1)),
        "post_median": float(np.median(y)),
        "post_q1": float(np.quantile(y, 0.25)),
        "post_q3": float(np.quantile(y, 0.75)),
        "mean_change": mean_d,
        "mean_change_ci_low": mean_d - crit * se,
        "mean_change_ci_high": mean_d + crit * se,
        "median_change": float(np.median(d)),
        "relative_mean_change_pct": (100 * mean_d / abs(pre_mean)) if pre_mean != 0 else np.nan,
        "cohen_dz": mean_d / sd_d if sd_d > 0 else 0.0,
        "rank_biserial": rank_biserial(d),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def make_pairs(pre: pd.DataFrame, post: pd.DataFrame, include_typos: bool) -> pd.DataFrame:
    exact = pre.merge(post, on="source_file", suffixes=("_pre", "_post"), how="inner")
    if not include_typos:
        return exact
    pre_indexed = pre.set_index("source_file")
    post_indexed = post.set_index("source_file")
    rows = []
    for pre_name, post_name in TYPO_PAIRS.items():
        if pre_name not in pre_indexed.index or post_name not in post_indexed.index:
            continue
        row = {"source_file": f"{pre_name} -> {post_name}"}
        row.update({f"{c}_pre": pre_indexed.at[pre_name, c] for c in pre.columns if c != "source_file"})
        row.update({f"{c}_post": post_indexed.at[post_name, c] for c in post.columns if c != "source_file"})
        rows.append(row)
    return pd.concat([exact, pd.DataFrame(rows)], ignore_index=True, sort=False)


def duration_adjusted_change(
    pre_values: pd.Series,
    post_values: pd.Series,
    pre_duration: pd.Series,
    post_duration: pd.Series,
) -> dict:
    x = pd.to_numeric(pre_values, errors="coerce").to_numpy(float)
    y = pd.to_numeric(post_values, errors="coerce").to_numpy(float)
    d0 = pd.to_numeric(pre_duration, errors="coerce").to_numpy(float)
    d1 = pd.to_numeric(post_duration, errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(d0) & np.isfinite(d1)
    feature_change = y[keep] - x[keep]
    duration_change = d1[keep] - d0[keep]
    design = sm.add_constant(duration_change, has_constant="add")
    fit = sm.OLS(feature_change, design).fit(cov_type="HC3")
    ci = fit.conf_int(alpha=0.05)[0]
    return {
        "duration_adjusted_change": float(fit.params[0]),
        "duration_adjusted_ci_low": float(ci[0]),
        "duration_adjusted_ci_high": float(ci[1]),
        "duration_adjusted_p": float(fit.pvalues[0]),
        "change_duration_slope": float(fit.params[1]),
        "change_duration_slope_p": float(fit.pvalues[1]),
    }


def analyze(pairs: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        row = {
            "feature": feature,
            **summarize_pair(pairs[f"{feature}_pre"], pairs[f"{feature}_post"]),
            **duration_adjusted_change(
                pairs[f"{feature}_pre"], pairs[f"{feature}_post"],
                pairs["duration_hours_pre"], pairs["duration_hours_post"],
            ),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result["wilcoxon_q_bh"] = bh_fdr(result["wilcoxon_p"])
    result["paired_t_q_bh"] = bh_fdr(result["paired_t_p"])
    result["duration_adjusted_q_bh"] = bh_fdr(result["duration_adjusted_p"])
    result["direction"] = np.where(result["median_change"] > 0, "increase",
                            np.where(result["median_change"] < 0, "decrease", "no_change"))
    result["fdr_significant"] = result["wilcoxon_q_bh"] < 0.05
    result["analysis"] = label
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired pre-post feature analysis")
    parser.add_argument("--pre", type=Path, default=PRE_PATH)
    parser.add_argument("--post", type=Path, default=POST_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    arguments = parser.parse_args()
    pre = pd.read_csv(arguments.pre, encoding="utf-8-sig")
    post = pd.read_csv(arguments.post, encoding="utf-8-sig")
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    exact_pairs = make_pairs(pre, post, include_typos=False)
    sensitivity_pairs = make_pairs(pre, post, include_typos=True)
    exact = analyze(exact_pairs, "exact_name_pairs")
    sensitivity = analyze(sensitivity_pairs, "exact_plus_6_high_confidence_name_variants")

    exact.to_csv(arguments.out_dir / "pre_post_feature_comparison_exact_pairs.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(arguments.out_dir / "pre_post_feature_comparison_sensitivity.csv", index=False, encoding="utf-8-sig")

    qc_rows = []
    for feature in ["n_epochs", "duration_hours"]:
        qc_rows.append({"feature": feature, **summarize_pair(
            exact_pairs[f"{feature}_pre"], exact_pairs[f"{feature}_post"]
        )})
    qc = pd.DataFrame(qc_rows)
    qc["wilcoxon_q_bh"] = bh_fdr(qc["wilcoxon_p"])
    qc.to_csv(arguments.out_dir / "pre_post_recording_qc.csv", index=False, encoding="utf-8-sig")

    pre_only = sorted(set(pre["source_file"]) - set(post["source_file"]))
    post_only = sorted(set(post["source_file"]) - set(pre["source_file"]))
    pd.DataFrame({"pre_only": pd.Series(pre_only), "post_only": pd.Series(post_only)}).to_csv(
        arguments.out_dir / "pre_post_unmatched_names.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Exact pairs: {len(exact_pairs)}; sensitivity pairs: {len(sensitivity_pairs)}")
    print(exact[["feature", "pre_median", "post_median", "median_change", "rank_biserial",
                 "wilcoxon_p", "wilcoxon_q_bh", "fdr_significant"]].to_string(index=False))


if __name__ == "__main__":
    main()
