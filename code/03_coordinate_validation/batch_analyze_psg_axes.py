"""Batch physiological validation of ECG-derived Body and REM axes.

The script matches one prediction CSV to one EDF by participant-night ID,
aligns each CSV to the EDF with a symmetric epoch offset, extracts independent
EEG/EOG/EMG features for every 30-second epoch, and writes:

* ``epoch_features/<subject>_epoch_psg_features.csv`` - aligned epoch data.
* ``participant_summary.csv`` - one row per successfully processed record.
* ``pca_ica_summary.csv`` - participant-level PCA/ICA decomposition, stability,
  nonlinear dependence, and PSG-marker associations.
* ``all_subject_correlations.csv`` - long-format Pearson/Spearman results.
* ``file_manifest.csv`` - input matching, alignment, channels, and status.
* ``processing_errors.csv`` - unmatched files and processing failures.

Only NumPy and pandas are required. PCA and FastICA are implemented directly so
that no scikit-learn dependency is needed. EDF files are read directly without
MNE or pyEDFlib. Input CSV and EDF files are never modified.
"""

from __future__ import annotations

import argparse
import math
import re
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPOCH_SECONDS = 30.0
REQUIRED_COLUMNS = {"wake_depth_score", "rem_logit_rem"}
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def normalize_channel_name(value: str) -> str:
    """Normalize an EDF channel label for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def participant_id_from_csv(path: Path) -> str:
    """Return the participant-night ID represented by a prediction CSV."""
    stem = path.stem
    for suffix in (
        "_features_predictions",
        "-features-predictions",
        "_feature_predictions",
        "_predictions",
        "-predictions",
    ):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def minmax(values: pd.Series) -> tuple[np.ndarray, float, float]:
    """Return per-record min-max values plus the original endpoints."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    low = float(np.nanmin(numeric))
    high = float(np.nanmax(numeric))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(numeric), low, high
    return (numeric - low) / (high - low), low, high


def safe_pearson(first: Iterable[float], second: Iterable[float]) -> float:
    """Compute Pearson correlation after paired finite-value filtering."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    if (
        np.sum(valid) < 3
        or np.std(first_array[valid]) <= 0
        or np.std(second_array[valid]) <= 0
    ):
        return float("nan")
    return float(np.corrcoef(first_array[valid], second_array[valid])[0, 1])


def safe_spearman(first: Iterable[float], second: Iterable[float]) -> float:
    """Compute Spearman correlation without SciPy."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    if np.sum(valid) < 3:
        return float("nan")
    first_ranks = pd.Series(first_array[valid]).rank(method="average").to_numpy()
    second_ranks = pd.Series(second_array[valid]).rank(method="average").to_numpy()
    return safe_pearson(first_ranks, second_ranks)


def symmetric_decorrelation(matrix: np.ndarray) -> np.ndarray:
    """Return a row-orthonormalized square matrix for symmetric FastICA."""
    covariance = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    inverse_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    return inverse_root @ matrix


def quantile_mutual_information(
    first: Iterable[float], second: Iterable[float], bins: int = 10
) -> tuple[float, float]:
    """Estimate mutual information in nats using marginal quantile bins."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    if np.sum(valid) < max(20, bins * 2):
        return float("nan"), float("nan")
    first_codes = pd.qcut(
        first_array[valid], q=bins, labels=False, duplicates="drop"
    )
    second_codes = pd.qcut(
        second_array[valid], q=bins, labels=False, duplicates="drop"
    )
    first_codes = np.asarray(first_codes, dtype=int)
    second_codes = np.asarray(second_codes, dtype=int)
    first_bins = int(np.max(first_codes)) + 1
    second_bins = int(np.max(second_codes)) + 1
    if first_bins < 2 or second_bins < 2:
        return float("nan"), float("nan")
    counts = np.zeros((first_bins, second_bins), dtype=float)
    np.add.at(counts, (first_codes, second_codes), 1)
    joint = counts / np.sum(counts)
    first_probability = np.sum(joint, axis=1)
    second_probability = np.sum(joint, axis=0)
    expected = first_probability[:, None] * second_probability[None, :]
    positive = joint > 0
    mutual_information = float(
        np.sum(joint[positive] * np.log(joint[positive] / expected[positive]))
    )
    first_entropy = -float(
        np.sum(first_probability[first_probability > 0]
               * np.log(first_probability[first_probability > 0]))
    )
    second_entropy = -float(
        np.sum(second_probability[second_probability > 0]
               * np.log(second_probability[second_probability > 0]))
    )
    denominator = min(first_entropy, second_entropy)
    normalized = (
        mutual_information / denominator
        if denominator > 0
        else float("nan")
    )
    return mutual_information, float(normalized)


def two_axis_pca(
    body: Iterable[float], rem: Iterable[float], *, standardize: bool
) -> dict[str, float]:
    """Return deterministic two-variable PCA metrics after finite filtering."""
    body_array = np.asarray(body, dtype=float)
    rem_array = np.asarray(rem, dtype=float)
    valid = np.isfinite(body_array) & np.isfinite(rem_array)
    values = np.column_stack([body_array[valid], rem_array[valid]])
    if len(values) < 20 or np.any(np.std(values, axis=0) <= 0):
        return {}
    centered = values - np.mean(values, axis=0)
    scale = np.std(centered, axis=0) if standardize else np.ones(2)
    transformed = centered / scale
    covariance = transformed.T @ transformed / len(transformed)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvectors[1, 0] < 0:
        eigenvectors[:, 0] *= -1
    if eigenvectors[0, 1] < 0:
        eigenvectors[:, 1] *= -1
    ratios = eigenvalues / np.sum(eigenvalues)
    return {
        "n_epochs": int(len(values)),
        "body_rem_pearson": safe_pearson(values[:, 0], values[:, 1]),
        "pc1_eigenvalue": float(eigenvalues[0]),
        "pc2_eigenvalue": float(eigenvalues[1]),
        "pc1_explained_ratio": float(ratios[0]),
        "pc2_explained_ratio": float(ratios[1]),
        "pc1_body_coefficient": float(eigenvectors[0, 0]),
        "pc1_rem_coefficient": float(eigenvectors[1, 0]),
        "pc2_body_coefficient": float(eigenvectors[0, 1]),
        "pc2_rem_coefficient": float(eigenvectors[1, 1]),
        "components_for_80pct": 1 if ratios[0] >= 0.80 else 2,
        "components_for_90pct": 1 if ratios[0] >= 0.90 else 2,
        "components_for_95pct": 1 if ratios[0] >= 0.95 else 2,
    }


def fastica_once(
    whitened: np.ndarray,
    whitening: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    tolerance: float,
) -> dict[str, object]:
    """Fit one deterministic two-component symmetric FastICA restart."""
    rng = np.random.default_rng(seed)
    unmixing_white = symmetric_decorrelation(rng.normal(size=(2, 2)))
    converged = False
    iterations = max_iter
    for iteration in range(1, max_iter + 1):
        projection = whitened @ unmixing_white.T
        transformed = np.tanh(projection)
        derivative_mean = np.mean(1.0 - transformed**2, axis=0)
        updated = (
            transformed.T @ whitened / len(whitened)
            - derivative_mean[:, None] * unmixing_white
        )
        updated = symmetric_decorrelation(updated)
        change = float(
            np.max(
                np.abs(
                    np.abs(np.diag(updated @ unmixing_white.T)) - 1.0
                )
            )
        )
        unmixing_white = updated
        if change < tolerance:
            converged = True
            iterations = iteration
            break
    scores = whitened @ unmixing_white.T
    unmixing_standardized = unmixing_white @ whitening.T
    kurtosis = np.mean(scores**4, axis=0) - 3.0
    objective = float(np.sum(np.abs(kurtosis)))
    return {
        "converged": converged,
        "iterations": iterations,
        "scores": scores,
        "unmixing": unmixing_standardized,
        "objective": objective,
        "kurtosis": kurtosis,
    }


def align_ica_components(
    result: dict[str, object], standardized: np.ndarray
) -> dict[str, object]:
    """Align ICA order/sign to Body-like and REM-like components."""
    scores = np.asarray(result["scores"], dtype=float).copy()
    unmixing = np.asarray(result["unmixing"], dtype=float).copy()
    correlations = np.asarray(
        [
            [safe_pearson(scores[:, component], standardized[:, variable])
             for variable in range(2)]
            for component in range(2)
        ]
    )
    diagonal = abs(correlations[0, 0]) + abs(correlations[1, 1])
    swapped = abs(correlations[1, 0]) + abs(correlations[0, 1])
    order = [0, 1] if diagonal >= swapped else [1, 0]
    scores = scores[:, order]
    unmixing = unmixing[order]
    for component, target_variable in enumerate((0, 1)):
        if safe_pearson(scores[:, component], standardized[:, target_variable]) < 0:
            scores[:, component] *= -1
            unmixing[component] *= -1
    return {
        **result,
        "scores": scores,
        "unmixing": unmixing,
        "axis_correlations": np.asarray(
            [
                [safe_pearson(scores[:, component], standardized[:, variable])
                 for variable in range(2)]
                for component in range(2)
            ]
        ),
    }


def two_axis_ica(
    body: Iterable[float],
    rem: Iterable[float],
    *,
    restarts: int,
    max_iter: int,
    tolerance: float,
    seed: int,
) -> dict[str, object]:
    """Fit restart-stabilized ICA and return aligned scores and metrics."""
    body_array = np.asarray(body, dtype=float)
    rem_array = np.asarray(rem, dtype=float)
    valid = np.isfinite(body_array) & np.isfinite(rem_array)
    values = np.column_stack([body_array[valid], rem_array[valid]])
    if len(values) < 50 or np.any(np.std(values, axis=0) <= 0):
        return {"valid": valid, "metrics": {}}
    standardized = (values - np.mean(values, axis=0)) / np.std(values, axis=0)
    covariance = standardized.T @ standardized / len(standardized)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) <= 1e-8:
        return {"valid": valid, "metrics": {}}
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
    whitened = standardized @ whitening
    runs = []
    for restart in range(restarts):
        fitted = fastica_once(
            whitened,
            whitening,
            seed=seed + restart,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        runs.append(align_ica_components(fitted, standardized))
    converged_runs = [run for run in runs if bool(run["converged"])]
    candidates = converged_runs if converged_runs else runs
    best = max(candidates, key=lambda run: float(run["objective"]))
    stability_values = []
    for first_index in range(len(converged_runs)):
        for second_index in range(first_index + 1, len(converged_runs)):
            first_scores = np.asarray(converged_runs[first_index]["scores"])
            second_scores = np.asarray(converged_runs[second_index]["scores"])
            stability_values.append(
                np.mean(
                    [
                        abs(safe_pearson(
                            first_scores[:, component],
                            second_scores[:, component],
                        ))
                        for component in range(2)
                    ]
                )
            )
    original_mi, original_nmi = quantile_mutual_information(
        standardized[:, 0], standardized[:, 1]
    )
    component_mi, component_nmi = quantile_mutual_information(
        np.asarray(best["scores"])[:, 0],
        np.asarray(best["scores"])[:, 1],
    )
    axis_correlations = np.asarray(best["axis_correlations"])
    unmixing = np.asarray(best["unmixing"])
    kurtosis = np.mean(np.asarray(best["scores"])**4, axis=0) - 3.0
    metrics: dict[str, object] = {
        "n_epochs": int(len(values)),
        "converged": bool(best["converged"]),
        "iterations": int(best["iterations"]),
        "converged_restarts": int(len(converged_runs)),
        "total_restarts": int(restarts),
        "restart_stability": (
            float(np.median(stability_values))
            if stability_values
            else float("nan")
        ),
        "body_component_body_corr": float(axis_correlations[0, 0]),
        "body_component_rem_corr": float(axis_correlations[0, 1]),
        "rem_component_body_corr": float(axis_correlations[1, 0]),
        "rem_component_rem_corr": float(axis_correlations[1, 1]),
        "body_component_body_coefficient": float(unmixing[0, 0]),
        "body_component_rem_coefficient": float(unmixing[0, 1]),
        "rem_component_body_coefficient": float(unmixing[1, 0]),
        "rem_component_rem_coefficient": float(unmixing[1, 1]),
        "body_component_kurtosis": float(kurtosis[0]),
        "rem_component_kurtosis": float(kurtosis[1]),
        "component_pearson": safe_pearson(
            np.asarray(best["scores"])[:, 0],
            np.asarray(best["scores"])[:, 1],
        ),
        "original_mutual_information_nats": original_mi,
        "component_mutual_information_nats": component_mi,
        "original_normalized_mutual_information": original_nmi,
        "component_normalized_mutual_information": component_nmi,
        "normalized_mutual_information_reduction": (
            original_nmi - component_nmi
            if np.isfinite(original_nmi) and np.isfinite(component_nmi)
            else float("nan")
        ),
    }
    return {
        "valid": valid,
        "scores": np.asarray(best["scores"]),
        "metrics": metrics,
    }


def pca_ica_summary(
    subject_id: str,
    frame: pd.DataFrame,
    *,
    ica_restarts: int,
    ica_max_iter: int,
    ica_tolerance: float,
    ica_seed: int,
) -> dict[str, object]:
    """Build one wide PCA/ICA result row from aligned epoch features."""
    masks = subset_masks(frame)
    result: dict[str, object] = {"subject_id": subject_id}
    for subset_name in ("all", "sleep", "NREM"):
        selected = frame.loc[masks[subset_name], ["body_norm", "rem_axis_norm"]]
        for scale_name, standardize in (("zscaled", True), ("unscaled", False)):
            metrics = two_axis_pca(
                selected["body_norm"],
                selected["rem_axis_norm"],
                standardize=standardize,
            )
            result.update(
                {
                    f"pca_{subset_name}_{scale_name}_{key}": value
                    for key, value in metrics.items()
                }
            )

    sleep_indices = np.flatnonzero(masks["sleep"])
    sleep = frame.iloc[sleep_indices]
    stable_subject_seed = (
        int(ica_seed) + zlib.crc32(subject_id.encode("utf-8"))
    ) % (2**32 - 1)
    decomposition = two_axis_ica(
        sleep["body_norm"],
        sleep["rem_axis_norm"],
        restarts=ica_restarts,
        max_iter=ica_max_iter,
        tolerance=ica_tolerance,
        seed=stable_subject_seed,
    )
    result.update(
        {
            f"ica_sleep_{key}": value
            for key, value in dict(decomposition["metrics"]).items()
        }
    )
    if "scores" not in decomposition:
        return result

    body_scores = np.full(len(frame), np.nan)
    rem_scores = np.full(len(frame), np.nan)
    valid_sleep = np.asarray(decomposition["valid"], dtype=bool)
    valid_indices = sleep_indices[valid_sleep]
    aligned_scores = np.asarray(decomposition["scores"], dtype=float)
    body_scores[valid_indices] = aligned_scores[:, 0]
    rem_scores[valid_indices] = aligned_scores[:, 1]
    marker_pairs = [
        ("body_component_eeg_slope_primary_pearson_nrem", body_scores,
         "eeg_slope_primary", masks["NREM"]),
        ("body_component_eeg_delta_primary_pearson_nrem", body_scores,
         "eeg_delta_rel_primary", masks["NREM"]),
        ("rem_component_eog_low_pearson_rem", rem_scores,
         "eog_low_log", masks["REM"]),
        ("rem_component_eog_rms_pearson_rem", rem_scores,
         "eog_rms", masks["REM"]),
        ("rem_component_emg_high_pearson_rem", rem_scores,
         "emg_high_log", masks["REM"]),
    ]
    for name, scores, marker, mask in marker_pairs:
        result[f"ica_sleep_{name}"] = (
            safe_pearson(scores[mask], frame.loc[mask, marker])
            if marker in frame.columns
            else float("nan")
        )
    return result


@dataclass
class EDFChannel:
    values: np.ndarray
    sampling_rate: float
    label: str


class EDFReader:
    """Minimal EDF reader supporting standard little-endian EDF files."""

    FIELD_WIDTHS = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    FIELD_NAMES = [
        "label",
        "transducer",
        "unit",
        "phys_min",
        "phys_max",
        "dig_min",
        "dig_max",
        "prefilter",
        "samples_per_record",
        "reserved",
    ]

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as file:
            fixed = file.read(256)
        if len(fixed) != 256:
            raise ValueError("EDF fixed header is incomplete")

        def text(start: int, end: int) -> str:
            return fixed[start:end].decode("latin-1").strip()

        self.header_bytes = int(text(184, 192))
        self.n_records = int(text(236, 244))
        self.record_duration = float(text(244, 252))
        self.n_signals = int(text(252, 256))
        if self.n_records <= 0 or self.record_duration <= 0 or self.n_signals <= 0:
            raise ValueError(
                "Unsupported EDF header: records, record duration, and signals "
                "must all be positive"
            )

        with path.open("rb") as file:
            file.seek(256)
            signal_header = file.read(self.header_bytes - 256)
        expected_header = self.n_signals * sum(self.FIELD_WIDTHS)
        if len(signal_header) != expected_header:
            raise ValueError(
                f"EDF signal header has {len(signal_header)} bytes; "
                f"expected {expected_header}"
            )

        self.columns: dict[str, list[str]] = {}
        position = 0
        for name, width in zip(self.FIELD_NAMES, self.FIELD_WIDTHS):
            self.columns[name] = [
                signal_header[
                    position + index * width : position + (index + 1) * width
                ]
                .decode("latin-1")
                .strip()
                for index in range(self.n_signals)
            ]
            position += self.n_signals * width

        self.samples_per_record = np.asarray(
            self.columns["samples_per_record"], dtype=int
        )
        self.channel_starts = np.concatenate(
            [[0], np.cumsum(self.samples_per_record)[:-1]]
        )
        samples_per_data_record = int(np.sum(self.samples_per_record))
        expected_data_bytes = (
            self.header_bytes + self.n_records * samples_per_data_record * 2
        )
        if path.stat().st_size < expected_data_bytes:
            raise ValueError(
                f"EDF file is truncated: {path.stat().st_size} bytes; "
                f"expected at least {expected_data_bytes}"
            )
        self.data = np.memmap(
            path,
            mode="r",
            dtype="<i2",
            offset=self.header_bytes,
            shape=(self.n_records, samples_per_data_record),
        )
        self.normalized_labels = [
            normalize_channel_name(label) for label in self.columns["label"]
        ]

    @property
    def duration_seconds(self) -> float:
        return self.n_records * self.record_duration

    @property
    def n_epochs(self) -> int:
        return int(math.floor(self.duration_seconds / EPOCH_SECONDS))

    @property
    def labels(self) -> list[str]:
        return list(self.columns["label"])

    def find_channel(
        self,
        aliases: Iterable[str],
        *,
        contains: str | None = None,
        exclude: Iterable[int] = (),
    ) -> int | None:
        excluded = set(exclude)
        normalized_aliases = [normalize_channel_name(alias) for alias in aliases]
        for alias in normalized_aliases:
            for index, label in enumerate(self.normalized_labels):
                if index not in excluded and label == alias:
                    return index
        if contains:
            needle = normalize_channel_name(contains)
            for index, label in enumerate(self.normalized_labels):
                if index not in excluded and needle in label:
                    return index
        return None

    def channel(self, index: int) -> EDFChannel:
        start = int(self.channel_starts[index])
        count = int(self.samples_per_record[index])
        digital = np.asarray(
            self.data[:, start : start + count], dtype=float
        ).reshape(-1)
        digital_min = float(self.columns["dig_min"][index])
        digital_max = float(self.columns["dig_max"][index])
        physical_min = float(self.columns["phys_min"][index])
        physical_max = float(self.columns["phys_max"][index])
        if digital_max == digital_min:
            raise ValueError(
                f"Channel {self.columns['label'][index]!r} has zero digital range"
            )
        physical = (
            (digital - digital_min)
            * (physical_max - physical_min)
            / (digital_max - digital_min)
            + physical_min
        )
        return EDFChannel(
            values=physical,
            sampling_rate=count / self.record_duration,
            label=self.columns["label"][index],
        )


def resolve_channels(reader: EDFReader) -> dict[str, int | None]:
    """Resolve common PSG channel names with SHHS-compatible fallbacks."""
    primary_eeg = reader.find_channel(
        ["EEG", "C4-A1", "C4-M1", "C3-A2", "C3-M2"],
        contains="eeg",
    )
    secondary_eeg = reader.find_channel(
        ["EEG(sec)", "EEG2", "C3-A2", "C3-M2", "C4-A1", "C4-M1"],
        contains="eeg",
        exclude=[] if primary_eeg is None else [primary_eeg],
    )
    left_eog = reader.find_channel(
        ["EOG(L)", "LOC", "E1", "LOC-A2", "E1-M2"],
        contains="eog",
    )
    right_eog = reader.find_channel(
        ["EOG(R)", "ROC", "E2", "ROC-A1", "E2-M1"],
        contains="eog",
        exclude=[] if left_eog is None else [left_eog],
    )
    emg = reader.find_channel(
        ["EMG", "Chin EMG", "Chin1-Chin2", "EMG1-EMG2"],
        contains="emg",
    )
    return {
        "eeg_primary": primary_eeg,
        "eeg_secondary": secondary_eeg,
        "eog_left": left_eog,
        "eog_right": right_eog,
        "emg": emg,
    }


def epoch_spectral_features(
    signal: np.ndarray,
    sampling_rate: float,
    epoch_indices: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Extract simple independent PSG features for aligned 30-second epochs."""
    epoch_samples = int(round(EPOCH_SECONDS * sampling_rate))
    if epoch_samples <= 0:
        raise ValueError(f"Invalid sampling rate: {sampling_rate}")
    frequencies = np.fft.rfftfreq(epoch_samples, d=1 / sampling_rate)
    window = np.hanning(epoch_samples)
    window_norm = float(np.sum(window**2))
    rows = []
    for epoch_index in epoch_indices:
        start = int(round(epoch_index * EPOCH_SECONDS * sampling_rate))
        stop = start + epoch_samples
        if start < 0 or stop > len(signal):
            raise IndexError(
                f"Epoch {epoch_index} exceeds channel duration "
                f"({len(signal) / sampling_rate:.1f} seconds)"
            )
        segment = np.asarray(signal[start:stop], dtype=float)
        segment -= np.mean(segment)
        spectrum = np.abs(np.fft.rfft(segment * window)) ** 2 / window_norm
        rms = float(np.sqrt(np.mean(segment**2)))

        if kind == "eeg":
            total = (frequencies >= 0.5) & (frequencies <= 30)
            delta = (frequencies >= 0.5) & (frequencies < 4)
            theta = (frequencies >= 4) & (frequencies < 8)
            slope_mask = (
                (frequencies >= 2)
                & (frequencies <= 30)
                & (spectrum > 0)
            )
            slope = np.polyfit(
                np.log10(frequencies[slope_mask]),
                np.log10(spectrum[slope_mask]),
                1,
            )[0]
            total_power = float(np.sum(spectrum[total])) + 1e-12
            rows.append(
                [
                    np.log10(float(np.sum(spectrum[delta])) + 1e-12),
                    float(np.sum(spectrum[delta])) / total_power,
                    float(np.sum(spectrum[theta])) / total_power,
                    slope,
                    rms,
                ]
            )
        elif kind == "eog":
            total = (frequencies >= 0.3) & (frequencies <= 10)
            low = (frequencies >= 0.3) & (frequencies <= 2)
            rows.append(
                [
                    np.log10(float(np.sum(spectrum[low])) + 1e-12),
                    float(np.sum(spectrum[low]))
                    / (float(np.sum(spectrum[total])) + 1e-12),
                    rms,
                ]
            )
        elif kind == "emg":
            upper = min(40, sampling_rate / 2 - 0.1)
            high = (frequencies >= 10) & (frequencies <= upper)
            rows.append(
                [
                    np.log10(float(np.sum(spectrum[high])) + 1e-12),
                    rms,
                ]
            )
        else:
            raise ValueError(f"Unknown feature kind: {kind}")
    return np.asarray(rows, dtype=float)


def add_psg_features(
    frame: pd.DataFrame,
    reader: EDFReader,
    channels: dict[str, int | None],
    epoch_indices: np.ndarray,
) -> pd.DataFrame:
    """Add aligned EEG/EOG/EMG features to a prediction frame."""
    output = frame.copy()
    output["edf_epoch_index"] = epoch_indices
    output["edf_start_seconds"] = epoch_indices * EPOCH_SECONDS

    body, body_min, body_max = minmax(output["wake_depth_score"])
    rem_axis, rem_min, rem_max = minmax(output["rem_logit_rem"])
    output["body_norm"] = body
    output["rem_axis_norm"] = rem_axis
    output["body_raw_min_record"] = body_min
    output["body_raw_max_record"] = body_max
    output["rem_logit_min_record"] = rem_min
    output["rem_logit_max_record"] = rem_max
    if "rem_prob_rem" in output.columns:
        probability = np.clip(
            pd.to_numeric(output["rem_prob_rem"], errors="coerce").to_numpy(float),
            1e-6,
            1 - 1e-6,
        )
        output["rem_log_odds"] = np.log(probability / (1 - probability))

    if channels["eeg_primary"] is not None:
        channel = reader.channel(int(channels["eeg_primary"]))
        values = epoch_spectral_features(
            channel.values, channel.sampling_rate, epoch_indices, "eeg"
        )
        output["eeg_delta_log_primary"] = values[:, 0]
        output["eeg_delta_rel_primary"] = values[:, 1]
        output["eeg_theta_rel_primary"] = values[:, 2]
        output["eeg_slope_primary"] = values[:, 3]
        output["eeg_rms_primary"] = values[:, 4]

    if channels["eeg_secondary"] is not None:
        channel = reader.channel(int(channels["eeg_secondary"]))
        values = epoch_spectral_features(
            channel.values, channel.sampling_rate, epoch_indices, "eeg"
        )
        output["eeg_delta_log_secondary"] = values[:, 0]
        output["eeg_delta_rel_secondary"] = values[:, 1]
        output["eeg_theta_rel_secondary"] = values[:, 2]
        output["eeg_slope_secondary"] = values[:, 3]
        output["eeg_rms_secondary"] = values[:, 4]

    left_index = channels["eog_left"]
    right_index = channels["eog_right"]
    if left_index is not None:
        left = reader.channel(int(left_index))
        if right_index is not None:
            right = reader.channel(int(right_index))
            if not math.isclose(
                left.sampling_rate, right.sampling_rate, rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"EOG sampling rates differ: {left.sampling_rate} and "
                    f"{right.sampling_rate}"
                )
            eog_signal = left.values - right.values
        else:
            eog_signal = left.values
        values = epoch_spectral_features(
            eog_signal, left.sampling_rate, epoch_indices, "eog"
        )
        output["eog_low_log"] = values[:, 0]
        output["eog_low_rel"] = values[:, 1]
        output["eog_rms"] = values[:, 2]

    if channels["emg"] is not None:
        channel = reader.channel(int(channels["emg"]))
        values = epoch_spectral_features(
            channel.values, channel.sampling_rate, epoch_indices, "emg"
        )
        output["emg_high_log"] = values[:, 0]
        output["emg_rms"] = values[:, 1]
    return output


def subset_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return clinically interpretable epoch subsets."""
    if "label" not in frame.columns:
        return {"all": np.ones(len(frame), dtype=bool)}
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(float)
    return {
        "all": np.isfinite(labels),
        "sleep": np.isin(labels, [1, 2, 3, 4]),
        "NREM": np.isin(labels, [1, 2, 3]),
        "N2_N3": np.isin(labels, [2, 3]),
        "REM": labels == 4,
        "Wake": labels == 0,
        "N1": labels == 1,
        "N2": labels == 2,
        "N3": labels == 3,
    }


def correlation_rows(subject_id: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return long-format correlations for all available axis-marker pairs."""
    pairs = [
        ("body_norm", "rem_axis_norm"),
        ("body_norm", "eeg_delta_rel_primary"),
        ("body_norm", "eeg_slope_primary"),
        ("body_norm", "eeg_delta_rel_secondary"),
        ("body_norm", "eeg_slope_secondary"),
        ("rem_axis_norm", "eog_low_log"),
        ("rem_axis_norm", "eog_rms"),
        ("rem_axis_norm", "emg_high_log"),
        ("rem_log_odds", "eog_low_log"),
    ]
    available_pairs = [
        pair for pair in pairs if pair[0] in frame.columns and pair[1] in frame.columns
    ]
    rows: list[dict[str, object]] = []
    for subset_name, mask in subset_masks(frame).items():
        for first, second in available_pairs:
            subset = frame.loc[mask, [first, second]]
            valid = np.isfinite(subset[first]) & np.isfinite(subset[second])
            rows.append(
                {
                    "subject_id": subject_id,
                    "subset": subset_name,
                    "n": int(np.sum(valid)),
                    "first": first,
                    "second": second,
                    "pearson": safe_pearson(
                        subset.loc[valid, first], subset.loc[valid, second]
                    ),
                    "spearman": safe_spearman(
                        subset.loc[valid, first], subset.loc[valid, second]
                    ),
                }
            )
    return rows


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(design), axis=1) & np.isfinite(values)
    output = np.full(len(values), np.nan)
    if np.sum(valid) <= design.shape[1]:
        return output
    coefficients = np.linalg.lstsq(
        design[valid], values[valid], rcond=None
    )[0]
    output[valid] = values[valid] - design[valid] @ coefficients
    return output


def joint_axis_result(
    frame: pd.DataFrame,
    mask: np.ndarray,
    outcome: str,
    include_stage: bool,
) -> dict[str, float]:
    """Assess the incremental variance of Body and REM axes for one marker."""
    required = {"body_norm", "rem_axis_norm", outcome}
    if not required.issubset(frame.columns):
        return {}
    subset = frame.loc[mask].copy()
    if len(subset) < 20:
        return {}
    time = np.linspace(-1, 1, len(frame))[mask]
    base_columns = [np.ones(len(subset)), time, time**2, time**3]
    if include_stage and "label" in subset.columns:
        stages = pd.get_dummies(
            subset["label"].astype(str), drop_first=True, dtype=float
        ).to_numpy()
        if stages.size:
            base_columns.extend(stages.T)
    base = np.column_stack(base_columns)
    body = subset["body_norm"].to_numpy(float)
    rem = subset["rem_axis_norm"].to_numpy(float)
    target = subset[outcome].to_numpy(float)
    valid = (
        np.all(np.isfinite(base), axis=1)
        & np.isfinite(body)
        & np.isfinite(rem)
        & np.isfinite(target)
    )
    base, body, rem, target = base[valid], body[valid], rem[valid], target[valid]
    if len(target) <= base.shape[1] + 2:
        return {}

    def sse(additions: list[np.ndarray]) -> float:
        design = np.column_stack([base] + additions)
        residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
        return float(np.sum(residual**2))

    sse_base = sse([])
    sse_body = sse([body])
    sse_rem = sse([rem])
    sse_both = sse([body, rem])
    if sse_base <= 0:
        return {}
    return {
        f"{outcome}_body_partial_r2": 1 - sse_body / sse_base,
        f"{outcome}_rem_partial_r2": 1 - sse_rem / sse_base,
        f"{outcome}_both_partial_r2": 1 - sse_both / sse_base,
        f"{outcome}_body_increment_after_rem": (
            1 - sse_both / sse_rem if sse_rem > 0 else float("nan")
        ),
        f"{outcome}_rem_increment_after_body": (
            1 - sse_both / sse_body if sse_body > 0 else float("nan")
        ),
    }


def participant_summary(
    subject_id: str,
    frame: pd.DataFrame,
    reader: EDFReader,
    offset: int,
    channels: dict[str, int | None],
) -> dict[str, object]:
    """Build one wide participant-night summary row."""
    masks = subset_masks(frame)

    def corr(
        subset: str, first: str, second: str, method: str = "pearson"
    ) -> float:
        if first not in frame.columns or second not in frame.columns:
            return float("nan")
        selected = frame.loc[masks[subset]]
        function = safe_pearson if method == "pearson" else safe_spearman
        return function(selected[first], selected[second])

    result: dict[str, object] = {
        "subject_id": subject_id,
        "n_csv_epochs": len(frame),
        "edf_epochs": reader.n_epochs,
        "epoch_offset": offset,
        "duration_hours_analyzed": len(frame) * EPOCH_SECONDS / 3600,
        "body_raw_min": float(frame["wake_depth_score"].min()),
        "body_raw_max": float(frame["wake_depth_score"].max()),
        "rem_logit_min": float(frame["rem_logit_rem"].min()),
        "rem_logit_max": float(frame["rem_logit_rem"].max()),
        "body_rem_pearson_sleep": corr(
            "sleep", "body_norm", "rem_axis_norm"
        ),
        "body_rem_spearman_sleep": corr(
            "sleep", "body_norm", "rem_axis_norm", "spearman"
        ),
        "body_rem_pearson_nrem": corr("NREM", "body_norm", "rem_axis_norm"),
        "body_rem_spearman_nrem": corr(
            "NREM", "body_norm", "rem_axis_norm", "spearman"
        ),
        "body_eeg_slope_primary_pearson_nrem": corr(
            "NREM", "body_norm", "eeg_slope_primary"
        ),
        "body_eeg_slope_primary_spearman_nrem": corr(
            "NREM", "body_norm", "eeg_slope_primary", "spearman"
        ),
        "body_eeg_slope_secondary_pearson_nrem": corr(
            "NREM", "body_norm", "eeg_slope_secondary"
        ),
        "body_eeg_delta_primary_pearson_nrem": corr(
            "NREM", "body_norm", "eeg_delta_rel_primary"
        ),
        "rem_eog_low_pearson_rem": corr(
            "REM", "rem_axis_norm", "eog_low_log"
        ),
        "rem_eog_low_spearman_rem": corr(
            "REM", "rem_axis_norm", "eog_low_log", "spearman"
        ),
        "rem_eog_rms_pearson_rem": corr(
            "REM", "rem_axis_norm", "eog_rms"
        ),
        "rem_emg_high_pearson_rem": corr(
            "REM", "rem_axis_norm", "emg_high_log"
        ),
    }
    for key, index in channels.items():
        result[f"{key}_channel"] = (
            reader.labels[index] if index is not None else ""
        )
    if "label" in frame.columns:
        numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
        for stage, name in STAGE_NAMES.items():
            stage_mask = numeric_labels == stage
            result[f"n_{name}"] = int(np.sum(stage_mask))
            result[f"body_median_{name}"] = (
                float(frame.loc[stage_mask, "body_norm"].median())
                if np.any(stage_mask)
                else float("nan")
            )
            result[f"rem_median_{name}"] = (
                float(frame.loc[stage_mask, "rem_axis_norm"].median())
                if np.any(stage_mask)
                else float("nan")
            )
    result.update(
        joint_axis_result(
            frame,
            masks["NREM"],
            "eeg_slope_primary",
            include_stage=True,
        )
    )
    result.update(
        joint_axis_result(
            frame,
            masks["NREM"],
            "eeg_slope_secondary",
            include_stage=True,
        )
    )
    result.update(
        joint_axis_result(
            frame,
            masks["REM"],
            "eog_low_log",
            include_stage=False,
        )
    )
    result.update(
        joint_axis_result(
            frame,
            masks["REM"],
            "eog_rms",
            include_stage=False,
        )
    )
    return result


def resolve_epoch_offset(
    requested: str, reader: EDFReader, csv_epochs: int
) -> int:
    """Resolve a fixed integer offset or infer symmetric trimming."""
    if requested.lower() != "auto":
        offset = int(requested)
        if offset < 0:
            raise ValueError("epoch offset must be non-negative")
        return offset
    difference = reader.n_epochs - csv_epochs
    if difference < 0 or difference % 2:
        raise ValueError(
            "Cannot infer symmetric epoch offset: "
            f"EDF has {reader.n_epochs} epochs and CSV has {csv_epochs}"
        )
    return difference // 2


def process_one(task: dict[str, object]) -> dict[str, object]:
    """Process one matched CSV/EDF pair; safe for ProcessPoolExecutor."""
    subject_id = str(task["subject_id"])
    csv_path = Path(str(task["csv_path"]))
    edf_path = Path(str(task["edf_path"]))
    output_dir = Path(str(task["output_dir"]))
    try:
        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise KeyError(
                f"Prediction CSV is missing columns: {', '.join(sorted(missing))}"
            )
        reader = EDFReader(edf_path)
        offset = resolve_epoch_offset(
            str(task["epoch_offset"]), reader, len(frame)
        )
        last_exclusive = offset + len(frame)
        if last_exclusive > reader.n_epochs:
            raise ValueError(
                f"Aligned CSV ends at EDF epoch {last_exclusive}; "
                f"EDF has only {reader.n_epochs} epochs"
            )
        channels = resolve_channels(reader)
        if channels["eeg_primary"] is None:
            raise KeyError(
                f"No EEG channel found. EDF channels: {', '.join(reader.labels)}"
            )
        epoch_indices = np.arange(len(frame), dtype=int) + offset
        enhanced = add_psg_features(frame, reader, channels, epoch_indices)
        epoch_path = output_dir / "epoch_features" / (
            f"{subject_id}_epoch_psg_features.csv"
        )
        if bool(task["write_epoch_files"]):
            epoch_path.parent.mkdir(parents=True, exist_ok=True)
            enhanced.to_csv(epoch_path, index=False)
        summary = participant_summary(
            subject_id, enhanced, reader, offset, channels
        )
        decomposition_summary = pca_ica_summary(
            subject_id,
            enhanced,
            ica_restarts=int(task.get("ica_restarts", 10)),
            ica_max_iter=int(task.get("ica_max_iter", 1000)),
            ica_tolerance=float(task.get("ica_tolerance", 1e-6)),
            ica_seed=int(task.get("ica_seed", 20260822)),
        )
        summary.update(
            {
                "csv_path": str(csv_path),
                "edf_path": str(edf_path),
                "epoch_output_path": (
                    str(epoch_path) if bool(task["write_epoch_files"]) else ""
                ),
            }
        )
        manifest = {
            "subject_id": subject_id,
            "csv_path": str(csv_path),
            "edf_path": str(edf_path),
            "status": "processed",
            "csv_epochs": len(frame),
            "edf_epochs": reader.n_epochs,
            "epoch_offset": offset,
            "first_edf_epoch": offset,
            "last_edf_epoch": last_exclusive - 1,
            "channels": "|".join(reader.labels),
        }
        return {
            "ok": True,
            "summary": summary,
            "pca_ica": decomposition_summary,
            "correlations": correlation_rows(subject_id, enhanced),
            "manifest": manifest,
        }
    except Exception as error:
        return {
            "ok": False,
            "manifest": {
                "subject_id": subject_id,
                "csv_path": str(csv_path),
                "edf_path": str(edf_path),
                "status": "error",
            },
            "error": {
                "subject_id": subject_id,
                "csv_path": str(csv_path),
                "edf_path": str(edf_path),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=5),
            },
        }


def collect_files(folder: Path, suffix: str, recursive: bool) -> list[Path]:
    pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match prediction CSV files to EDF files and batch-extract "
            "EEG/EOG/EMG physiological validation features."
        )
    )
    parser.add_argument("csv_dir", type=Path, help="Folder containing prediction CSVs")
    parser.add_argument("edf_dir", type=Path, help="Folder containing EDF files")
    parser.add_argument("output_dir", type=Path, help="New analysis output folder")
    parser.add_argument(
        "--epoch-offset",
        default="67",
        help=(
            "EDF epoch index corresponding to CSV row 0. Use an integer "
            "(default: 67) or 'auto' for symmetric inference."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search CSV and EDF folders recursively",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--skip-epoch-files",
        action="store_true",
        help="Write only participant summaries, not enhanced epoch CSVs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many CSVs; useful for a test run",
    )
    parser.add_argument(
        "--ica-restarts",
        type=int,
        default=10,
        help="FastICA random restarts per participant (default: 10)",
    )
    parser.add_argument(
        "--ica-max-iter",
        type=int,
        default=1000,
        help="Maximum FastICA iterations per restart (default: 1000)",
    )
    parser.add_argument(
        "--ica-tolerance",
        type=float,
        default=1e-6,
        help="FastICA convergence tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--ica-seed",
        type=int,
        default=20260822,
        help="Base random seed for reproducible FastICA (default: 20260822)",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    csv_dir = arguments.csv_dir.resolve()
    edf_dir = arguments.edf_dir.resolve()
    output_dir = arguments.output_dir.resolve()
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"CSV folder does not exist: {csv_dir}")
    if not edf_dir.is_dir():
        raise NotADirectoryError(f"EDF folder does not exist: {edf_dir}")
    if arguments.workers < 1:
        raise ValueError("--workers must be at least 1")
    if arguments.ica_restarts < 1:
        raise ValueError("--ica-restarts must be at least 1")
    if arguments.ica_max_iter < 1:
        raise ValueError("--ica-max-iter must be at least 1")
    if arguments.ica_tolerance <= 0:
        raise ValueError("--ica-tolerance must be positive")

    csv_files = collect_files(csv_dir, ".csv", arguments.recursive)
    csv_files = [
        path
        for path in csv_files
        if path.name not in {
            "participant_summary.csv",
            "pca_ica_summary.csv",
            "all_subject_correlations.csv",
            "file_manifest.csv",
            "processing_errors.csv",
        }
    ]
    if arguments.limit is not None:
        csv_files = csv_files[: arguments.limit]
    edf_files = collect_files(edf_dir, ".edf", arguments.recursive)
    edf_by_id: dict[str, list[Path]] = {}
    for path in edf_files:
        edf_by_id.setdefault(path.stem.lower(), []).append(path)

    tasks = []
    manifests: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for csv_path in csv_files:
        subject_id = participant_id_from_csv(csv_path)
        matches = edf_by_id.get(subject_id.lower(), [])
        if len(matches) != 1:
            status = "edf_not_found" if not matches else "duplicate_edf"
            manifest = {
                "subject_id": subject_id,
                "csv_path": str(csv_path),
                "edf_path": "|".join(str(path) for path in matches),
                "status": status,
            }
            manifests.append(manifest)
            errors.append(
                {
                    **manifest,
                    "error_type": status,
                    "error": (
                        "No matching EDF"
                        if not matches
                        else f"{len(matches)} matching EDF files"
                    ),
                    "traceback": "",
                }
            )
            continue
        tasks.append(
            {
                "subject_id": subject_id,
                "csv_path": str(csv_path),
                "edf_path": str(matches[0]),
                "output_dir": str(output_dir),
                "epoch_offset": arguments.epoch_offset,
                "write_epoch_files": not arguments.skip_epoch_files,
                "ica_restarts": arguments.ica_restarts,
                "ica_max_iter": arguments.ica_max_iter,
                "ica_tolerance": arguments.ica_tolerance,
                "ica_seed": arguments.ica_seed,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    pca_ica_summaries: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []

    def record_result(result: dict[str, object]) -> None:
        manifests.append(dict(result["manifest"]))
        if bool(result["ok"]):
            summaries.append(dict(result["summary"]))
            pca_ica_summaries.append(dict(result["pca_ica"]))
            correlations.extend(list(result["correlations"]))
            print(
                f"[OK] {result['manifest']['subject_id']} "
                f"({len(summaries)}/{len(tasks)})"
            )
        else:
            errors.append(dict(result["error"]))
            print(
                f"[ERROR] {result['manifest']['subject_id']}: "
                f"{result['error']['error']}"
            )

    if arguments.workers == 1:
        for task in tasks:
            record_result(process_one(task))
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            future_map = {
                executor.submit(process_one, task): task for task in tasks
            }
            for future in as_completed(future_map):
                record_result(future.result())

    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values("subject_id")
    summary_frame.to_csv(output_dir / "participant_summary.csv", index=False)

    pca_ica_frame = pd.DataFrame(pca_ica_summaries)
    if not pca_ica_frame.empty:
        pca_ica_frame = pca_ica_frame.sort_values("subject_id")
    pca_ica_frame.to_csv(output_dir / "pca_ica_summary.csv", index=False)

    correlation_frame = pd.DataFrame(correlations)
    if not correlation_frame.empty:
        correlation_frame = correlation_frame.sort_values(
            ["subject_id", "subset", "first", "second"]
        )
    correlation_frame.to_csv(
        output_dir / "all_subject_correlations.csv", index=False
    )

    manifest_frame = pd.DataFrame(manifests)
    if not manifest_frame.empty:
        manifest_frame = manifest_frame.sort_values(["status", "subject_id"])
    manifest_frame.to_csv(output_dir / "file_manifest.csv", index=False)

    error_frame = pd.DataFrame(errors)
    error_frame.to_csv(output_dir / "processing_errors.csv", index=False)

    print(
        f"Finished: {len(summaries)} processed, {len(errors)} skipped/failed. "
        f"PCA/ICA summary: {output_dir / 'pca_ica_summary.csv'}. "
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
