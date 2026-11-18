from pathlib import Path
import re

import numpy as np
import pandas as pd


FEATURE_ROOT = Path(r"C:\Users\SaberShuo\Desktop\ECG-3D\feature_14")
CLINICAL = {
    "cfs": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\cfs\cfs-visit5-dataset-0.7.0.csv"),
    "mros": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\mros\mros-visit1-dataset-0.6.0.csv"),
    "shhs": Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs1-dataset-0.20.0.csv"),
}
SHHS_CVD = Path(r"D:\Project\Python_Project\SDI\W_NW_REM_NREM\data\shhs\shhs-cvd-summary-dataset-0.19.0.csv")


def normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().str.replace(r"\.0$", "", regex=True)


def feature_id(series: pd.Series, cohort: str) -> pd.Series:
    patterns = {
        "cfs": r"cfs-visit5-([^_\.]+)",
        "mros": r"mros-visit1-([^_\.]+)",
        "shhs": r"shhs1-([^_\.]+)",
    }
    return normalize(series).str.extract(patterns[cohort], expand=False)


def strict_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.isin([0, 1])).astype(float)


def load_labels(cohort: str) -> pd.DataFrame:
    if cohort == "cfs":
        cols = ["nsrrid", "cvd", "htn", "diadiag", "ahi_a0h4", "easlp", "difbak"]
        d = pd.read_csv(CLINICAL[cohort], usecols=cols, low_memory=False)
        out = pd.DataFrame({"subject_id": normalize(d["nsrrid"])})
        out["cvd"] = strict_binary(d["cvd"])
        out["hypertension"] = strict_binary(d["htn"])
        out["diabetes"] = strict_binary(d["diadiag"])
        ahi = pd.to_numeric(d["ahi_a0h4"], errors="coerce")
        out["osa"] = np.where(ahi.notna(), (ahi >= 5).astype(float), np.nan)
        easy = pd.to_numeric(d["easlp"], errors="coerce")
        return_difficulty = pd.to_numeric(d["difbak"], errors="coerce")
        out["insomnia"] = np.nan
        out.loc[(easy <= 3) & (return_difficulty == 0), "insomnia"] = 0.0
        out.loc[(easy > 4) & (return_difficulty == 1), "insomnia"] = 1.0
        return out
    if cohort == "mros":
        cols = ["nsrrid", "mhbp", "mhdiab", "poahi3", "poxfall"]
        d = pd.read_csv(CLINICAL[cohort], usecols=cols, low_memory=False)
        out = pd.DataFrame({"subject_id": normalize(d["nsrrid"])})
        out["hypertension"] = strict_binary(d["mhbp"])
        out["diabetes"] = strict_binary(d["mhdiab"])
        ahi = pd.to_numeric(d["poahi3"], errors="coerce")
        out["osa"] = np.where(ahi.notna(), (ahi >= 5).astype(float), np.nan)
        latency = pd.to_numeric(d["poxfall"], errors="coerce")
        out["insomnia"] = np.where(latency.notna(), (latency > 40).astype(float), np.nan)
        return out
    cols = ["nsrrid", "htnderv_s1", "parrptdiab", "ahi_a0h4", "diffa10"]
    d = pd.read_csv(CLINICAL[cohort], usecols=cols, low_memory=False)
    cvd = pd.read_csv(SHHS_CVD, usecols=["nsrrid", "any_cvd"], low_memory=False)
    d["subject_id"] = normalize(d["nsrrid"])
    cvd["subject_id"] = normalize(cvd["nsrrid"])
    d = d.merge(cvd[["subject_id", "any_cvd"]], on="subject_id", how="left", validate="one_to_one")
    out = pd.DataFrame({"subject_id": d["subject_id"]})
    out["cvd"] = strict_binary(d["any_cvd"])
    out["hypertension"] = strict_binary(d["htnderv_s1"])
    out["diabetes"] = strict_binary(d["parrptdiab"])
    ahi = pd.to_numeric(d["ahi_a0h4"], errors="coerce")
    out["osa"] = np.where(ahi.notna(), (ahi >= 5).astype(float), np.nan)
    out["insomnia"] = strict_binary(d["diffa10"])
    return out


for cohort, sdi_name, emb_name in [
    ("cfs", "cfs_SDI_14_Features.csv", "cfs_embeddings_8d.csv"),
    ("mros", "mros_SDI_14_Features.csv", "mros_embeddings_8d.csv"),
    ("shhs", "shhs_SDI_14_Features.csv", "shhs_external_embeddings_8d.csv"),
]:
    sdi = pd.read_csv(FEATURE_ROOT / sdi_name)
    emb = pd.read_csv(FEATURE_ROOT / emb_name)
    sdi["subject_id"] = feature_id(sdi["NSRRID"], cohort)
    emb["subject_id"] = feature_id(emb["night_id"], cohort)
    labels = load_labels(cohort)
    merged = sdi.merge(emb, on="subject_id", validate="one_to_one").merge(labels, on="subject_id", validate="one_to_one")
    print(cohort.upper(), {"sdi": len(sdi), "embedding": len(emb), "clinical": len(labels), "merged": len(merged)})
    print("duplicate_ids", sdi["subject_id"].duplicated().sum(), emb["subject_id"].duplicated().sum(), labels["subject_id"].duplicated().sum())
    for outcome in ["osa", "insomnia", "hypertension", "diabetes", "cvd"]:
        if outcome not in merged:
            continue
        valid = merged[outcome].dropna()
        print(outcome, {"n": len(valid), "events": int(valid.sum()), "rate": float(valid.mean())})
    print()
