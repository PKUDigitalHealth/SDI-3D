from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class NightRecord:
    subject_id: str
    visit: str
    source_file: str
    time: np.ndarray
    depth: np.ndarray
    rem: np.ndarray
    stage: np.ndarray | None

    @property
    def n_epochs(self) -> int:
        return int(len(self.time))


def _normalise_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def _read_table(
    path: str | Path,
    usecols: list[str] | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=usecols)
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(
        path,
        sep=separator,
        usecols=usecols,
        low_memory=False,
        encoding=encoding or "utf-8",
    )


def discover_trajectory_files(config: dict[str, Any]) -> list[Path]:
    folder = Path(config["paths"]["trajectory_dir"])
    if not folder.exists():
        raise FileNotFoundError(f"Trajectory directory does not exist: {folder}")
    pattern = config["data"]["file_pattern"]
    iterator = folder.rglob(pattern) if config["data"]["recursive"] else folder.glob(pattern)
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise FileNotFoundError(f"No trajectory CSV files matched {folder / pattern}")
    return files


def read_night(path: Path, data_config: dict[str, Any]) -> NightRecord:
    columns = [
        data_config["time_col"],
        data_config["depth_col"],
        data_config["rem_col"],
    ]
    stage_col = data_config.get("stage_col")
    header = pd.read_csv(path, nrows=0).columns
    if stage_col and stage_col in header:
        columns.append(stage_col)
    missing = sorted(set(columns[:3]).difference(header))
    if missing:
        raise KeyError(f"missing required columns: {', '.join(missing)}")
    for identity_column in [data_config["id_col"], data_config["visit_col"]]:
        if identity_column in header:
            columns.append(identity_column)

    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if data_config["id_col"] in frame and data_config["visit_col"] in frame:
        subject_values = (
            frame[data_config["id_col"]]
            .dropna()
            .map(_normalise_identifier)
            .unique()
        )
        visit_values = (
            frame[data_config["visit_col"]]
            .dropna()
            .map(_normalise_identifier)
            .unique()
        )
        if len(subject_values) != 1 or len(visit_values) != 1:
            raise ValueError("Each trajectory CSV must contain exactly one subject and visit.")
        subject_id = subject_values[0]
        visit = visit_values[0]
    else:
        filename_regex = data_config.get("filename_regex")
        if not filename_regex:
            raise KeyError(
                "ID/visit columns are absent and data.filename_regex is not configured."
            )
        match = re.match(filename_regex, path.name)
        if not match or not {"subject_id", "visit"}.issubset(match.groupdict()):
            raise ValueError(
                f"Filename does not match configured named groups: {path.name}"
            )
        subject_id = _normalise_identifier(match.group("subject_id"))
        visit = _normalise_identifier(match.group("visit"))

    numeric = frame[
        [data_config["time_col"], data_config["depth_col"], data_config["rem_col"]]
    ].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    numeric = numeric.loc[valid]
    order = np.argsort(numeric[data_config["time_col"]].to_numpy(float), kind="stable")
    numeric = numeric.iloc[order]
    sorted_time = numeric[data_config["time_col"]].to_numpy(float)
    if len(sorted_time) > 1 and np.any(np.diff(sorted_time) <= 0):
        raise ValueError("window centers must be unique and strictly increasing")
    stage = None
    if stage_col and stage_col in frame.columns:
        stage = pd.to_numeric(frame.loc[valid, stage_col], errors="coerce").to_numpy(float)[order]

    return NightRecord(
        subject_id=subject_id,
        visit=visit,
        source_file=str(path),
        time=numeric[data_config["time_col"]].to_numpy(np.float32),
        depth=numeric[data_config["depth_col"]].to_numpy(np.float32),
        rem=numeric[data_config["rem_col"]].to_numpy(np.float32),
        stage=stage,
    )


def load_nights(config: dict[str, Any]) -> tuple[list[NightRecord], pd.DataFrame]:
    records: list[NightRecord] = []
    audit_rows: list[dict[str, Any]] = []
    required_visit = config["data"].get("required_visit")
    minimum_epochs = int(config["data"]["minimum_epochs"])
    base = Path(config["paths"]["trajectory_dir"])

    for path in discover_trajectory_files(config):
        row: dict[str, Any] = {"source_file": str(path.relative_to(base))}
        try:
            record = read_night(path, config["data"])
            row.update(
                {
                    "subject_id": record.subject_id,
                    "visit": record.visit,
                    "n_epochs": record.n_epochs,
                    "duration_hours": (
                        record.n_epochs * float(config["data"]["epoch_seconds"]) / 3600.0
                    ),
                }
            )
            if required_visit is not None and record.visit != _normalise_identifier(required_visit):
                row.update({"status": "excluded_visit", "reason": "visit mismatch"})
            elif record.n_epochs < minimum_epochs:
                row.update({"status": "excluded_short", "reason": "too few finite epochs"})
            else:
                row.update({"status": "included", "reason": ""})
                records.append(record)
        except Exception as error:
            row.update(
                {
                    "status": "read_error",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
        audit_rows.append(row)

    if not records:
        raise RuntimeError("No eligible trajectory records remained after quality control.")
    record_keys = [(record.subject_id, record.visit) for record in records]
    duplicated_keys = pd.Series(record_keys).duplicated(keep=False)
    if duplicated_keys.any():
        examples = sorted(
            {
                record_keys[index]
                for index, duplicated in enumerate(duplicated_keys)
                if duplicated
            }
        )[:5]
        raise ValueError(
            "Duplicate eligible subject-visit trajectory files detected; examples: "
            f"{examples}"
        )
    return records, pd.DataFrame(audit_rows)


def _binary_indicator(series: pd.Series, positive_values: list[Any]) -> pd.Series:
    positive_text = {_normalise_identifier(value) for value in positive_values}
    result = series.map(_normalise_identifier).isin(positive_text).astype(float)
    result[series.isna()] = np.nan
    return result


def build_analysis_table(
    records: list[NightRecord], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_config = config["metadata"]
    outcome_config = config["outcome"]
    harmonized_columns = [
        meta_config["harmonized_id_col"],
        *meta_config["continuous_covariates"],
        *meta_config["categorical_covariates"],
        *meta_config.get("audit_columns", []),
    ]
    harmonized_visit_col = meta_config.get("harmonized_visit_col")
    if harmonized_visit_col:
        harmonized_columns.append(harmonized_visit_col)
    outcome_columns = [
        meta_config["outcomes_id_col"],
        outcome_config["event_col"],
    ]
    if outcome_config.get("followup_time_col"):
        outcome_columns.append(outcome_config["followup_time_col"])
    if outcome_config.get("censor_time_col"):
        outcome_columns.append(outcome_config["censor_time_col"])
    outcome_columns.extend(outcome_config.get("event_time_cols", []))
    if outcome_config.get("prevalent_col"):
        outcome_columns.append(outcome_config["prevalent_col"])
    outcome_columns.extend(outcome_config.get("prevalent_cols", []))

    harmonized = _read_table(
        config["paths"]["harmonized_file"],
        list(dict.fromkeys(harmonized_columns)),
        meta_config.get("harmonized_encoding"),
    )
    outcomes = _read_table(
        config["paths"]["outcomes_file"],
        list(dict.fromkeys(outcome_columns)),
        meta_config.get("outcomes_encoding"),
    )
    if harmonized_visit_col:
        visit_value = meta_config.get(
            "harmonized_visit_value", config["data"].get("required_visit")
        )
        harmonized = harmonized.loc[
            harmonized[harmonized_visit_col].map(_normalise_identifier)
            == _normalise_identifier(visit_value)
        ].copy()
    harmonized["subject_id"] = harmonized[meta_config["harmonized_id_col"]].map(
        _normalise_identifier
    )
    outcomes["subject_id"] = outcomes[meta_config["outcomes_id_col"]].map(
        _normalise_identifier
    )
    if harmonized["subject_id"].duplicated().any():
        raise ValueError("Harmonized table has duplicate subject IDs after normalization.")
    if outcomes["subject_id"].duplicated().any():
        raise ValueError("Outcome table has duplicate subject IDs after normalization.")

    manifest = pd.DataFrame(
        {
            "record_index": np.arange(len(records), dtype=int),
            "subject_id": [record.subject_id for record in records],
            "visit": [record.visit for record in records],
            "source_file": [record.source_file for record in records],
            "n_epochs": [record.n_epochs for record in records],
        }
    )
    table = manifest.merge(
        harmonized.drop(columns=[meta_config["harmonized_id_col"]]),
        on="subject_id",
        how="left",
        validate="many_to_one",
    ).merge(
        outcomes.drop(columns=[meta_config["outcomes_id_col"]]),
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    event = _binary_indicator(
        table[outcome_config["event_col"]],
        outcome_config.get("event_positive_values", [1]),
    )
    eligible = event.notna()
    exclusion_reason = pd.Series("", index=table.index, dtype=object)

    prevalent_cols = list(outcome_config.get("prevalent_cols", []))
    prevalent_col = outcome_config.get("prevalent_col")
    if prevalent_cols:
        prevalent_values = table[prevalent_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        prevalent = prevalent_values.gt(
            float(outcome_config.get("prevalent_greater_than", 0))
        ).any(axis=1)
        prevalent_unknown = prevalent_values.isna().all(axis=1)
        table["derived_prevalent_disease"] = prevalent.astype(float)
        table.loc[prevalent_unknown, "derived_prevalent_disease"] = np.nan
        prevalent_mask = prevalent
        eligible &= ~prevalent_mask
        exclusion_reason.loc[prevalent_mask] = "prevalent disease"
        if outcome_config.get("require_prevalent_data", False):
            eligible &= ~prevalent_unknown
            exclusion_reason.loc[prevalent_unknown] = "prevalent history unavailable"
    elif prevalent_col:
        prevalent = _binary_indicator(
            table[prevalent_col],
            outcome_config.get("prevalent_positive_values", [1]),
        )
        prevalent_mask = prevalent == 1
        eligible &= ~prevalent_mask
        exclusion_reason.loc[prevalent_mask] = "prevalent disease"

    followup_col = outcome_config.get("followup_time_col")
    event_time_cols = list(outcome_config.get("event_time_cols", []))
    censor_time_col = outcome_config.get("censor_time_col")
    time_scale = float(outcome_config.get("time_scale_to_years", 1.0))
    horizon = outcome_config.get("horizon_years")
    if horizon is not None and (followup_col or event_time_cols):
        horizon = float(horizon)
        if event_time_cols:
            event_time = (
                table[event_time_cols]
                .apply(pd.to_numeric, errors="coerce")
                .min(axis=1, skipna=True)
                * time_scale
            )
            if not censor_time_col:
                raise KeyError(
                    "outcome.censor_time_col is required with event_time_cols."
                )
            censor_time = (
                pd.to_numeric(table[censor_time_col], errors="coerce")
                * time_scale
            )
            table["derived_event_time_years"] = event_time
            table["derived_censor_time_years"] = censor_time
            case = (event == 1) & event_time.notna() & (event_time <= horizon)
            control = (
                ((event == 1) & event_time.notna() & (event_time > horizon))
                | ((event == 0) & (censor_time >= horizon))
            )
            missing_case_time = (event == 1) & event_time.isna()
            exclusion_reason.loc[eligible & missing_case_time] = (
                "event reported without event date"
            )
        else:
            followup = (
                pd.to_numeric(table[followup_col], errors="coerce")
                * time_scale
            )
            case = (event == 1) & (followup <= horizon)
            control = ((event == 0) & (followup >= horizon)) | (
                (event == 1) & (followup > horizon)
            )
        horizon_eligible = case | control
        no_reason = exclusion_reason.eq("")
        exclusion_reason.loc[eligible & ~horizon_eligible & no_reason] = (
            "insufficient follow-up"
        )
        eligible &= horizon_eligible
        table["outcome"] = case.astype(int)
    else:
        table["outcome"] = event

    covariates = [
        *meta_config["continuous_covariates"],
        *meta_config["categorical_covariates"],
    ]
    missing_all_covariates = table[covariates].isna().all(axis=1) if covariates else False
    if isinstance(missing_all_covariates, pd.Series):
        exclusion_reason.loc[eligible & missing_all_covariates] = "all covariates missing"
        eligible &= ~missing_all_covariates

    audit = table.copy()
    audit["analysis_eligible"] = eligible
    audit["exclusion_reason"] = exclusion_reason
    analysis = table.loc[eligible].reset_index(drop=True)
    analysis["outcome"] = analysis["outcome"].astype(int)
    if analysis["outcome"].nunique() != 2:
        counts = analysis["outcome"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Binary outcome requires both classes; observed {json.dumps(counts)}")
    return analysis, audit
