from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .data import NightRecord


def _minmax(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


@dataclass
class TrajectoryScaler:
    mode: str
    lower_quantile: float
    upper_quantile: float
    rem_transform: str
    depth_low: float | None = None
    depth_high: float | None = None
    rem_low: float | None = None
    rem_high: float | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrajectoryScaler":
        return cls(
            mode=config["mode"],
            lower_quantile=float(config["lower_quantile"]),
            upper_quantile=float(config["upper_quantile"]),
            rem_transform=config.get("rem_transform", "sigmoid"),
        )

    def fit(self, records: Iterable[NightRecord]) -> "TrajectoryScaler":
        if self.mode == "per_record_minmax":
            return self
        if self.mode != "train_quantile":
            raise ValueError(f"Unknown scaling mode: {self.mode}")
        records = list(records)
        depth = np.concatenate([record.depth for record in records]).astype(float)
        self.depth_low, self.depth_high = np.quantile(
            depth, [self.lower_quantile, self.upper_quantile]
        )
        if self.rem_transform == "quantile_minmax":
            rem = np.concatenate([record.rem for record in records]).astype(float)
            self.rem_low, self.rem_high = np.quantile(
                rem, [self.lower_quantile, self.upper_quantile]
            )
        elif self.rem_transform != "sigmoid":
            raise ValueError(f"Unknown REM transform: {self.rem_transform}")
        return self

    def transform(self, record: NightRecord) -> np.ndarray:
        time = record.time.astype(float)
        time_scaled = _minmax(time, float(np.min(time)), float(np.max(time)))
        if self.mode == "per_record_minmax":
            depth = _minmax(
                record.depth, float(np.min(record.depth)), float(np.max(record.depth))
            )
            rem = _minmax(record.rem, float(np.min(record.rem)), float(np.max(record.rem)))
        else:
            if self.depth_low is None or self.depth_high is None:
                raise RuntimeError("TrajectoryScaler must be fitted before transform.")
            depth = _minmax(record.depth, self.depth_low, self.depth_high)
            if self.rem_transform == "sigmoid":
                clipped = np.clip(record.rem.astype(float), -30.0, 30.0)
                rem = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
            else:
                if self.rem_low is None or self.rem_high is None:
                    raise RuntimeError("REM quantile scaler was not fitted.")
                rem = _minmax(record.rem, self.rem_low, self.rem_high)
        return np.column_stack([time_scaled, depth, rem]).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "rem_transform": self.rem_transform,
            "depth_low": self.depth_low,
            "depth_high": self.depth_high,
            "rem_low": self.rem_low,
            "rem_high": self.rem_high,
        }


def patchify(sequence: np.ndarray, patch_size: int) -> np.ndarray:
    """Summarize 30-second points into tokens: time, axis means, axis SDs."""
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    tokens = []
    for start in range(0, len(sequence), patch_size):
        block = sequence[start : start + patch_size]
        if len(block) == 0:
            continue
        tokens.append(
            [
                float(np.mean(block[:, 0])),
                float(np.mean(block[:, 1])),
                float(np.mean(block[:, 2])),
                float(np.std(block[:, 1], ddof=0)),
                float(np.std(block[:, 2], ddof=0)),
            ]
        )
    return np.asarray(tokens, dtype=np.float32)


def _safe_moment(values: np.ndarray, order: int) -> float:
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1e-12:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean(centered**order) / standard_deviation**order)


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= lag:
        return float("nan")
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spectral_features(values: np.ndarray, epoch_seconds: float) -> tuple[float, float]:
    if len(values) < 8 or np.std(values) <= 1e-12:
        return 0.0, 0.0
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequency = np.fft.rfftfreq(len(centered), d=epoch_seconds)
    power = power[1:]
    frequency = frequency[1:]
    if power.size == 0 or np.sum(power) <= 0:
        return 0.0, 0.0
    probability = power / np.sum(power)
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, None)))
    entropy /= math.log(len(probability)) if len(probability) > 1 else 1.0
    dominant_cycles_per_hour = float(frequency[np.argmax(power)] * 3600.0)
    return dominant_cycles_per_hour, float(entropy)


def _run_lengths(states: np.ndarray, state: int) -> np.ndarray:
    match = states == state
    if not np.any(match):
        return np.asarray([], dtype=float)
    boundaries = np.diff(np.r_[False, match, False].astype(int))
    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)
    return (ends - starts).astype(float)


def _directional_entropy(dx: np.ndarray, dy: np.ndarray, bins: int) -> float:
    moving = np.hypot(dx, dy) > 1e-12
    if not np.any(moving):
        return 0.0
    counts, _ = np.histogram(
        np.arctan2(dy[moving], dx[moving]), bins=bins, range=(-np.pi, np.pi)
    )
    probability = counts[counts > 0] / np.sum(counts)
    entropy = -np.sum(probability * np.log(probability))
    return float(entropy / math.log(bins))


def _axis_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    result = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_skew": _safe_moment(values, 3),
        f"{prefix}_kurtosis": _safe_moment(values, 4) - 3.0,
    }
    for quantile in [0.05, 0.25, 0.50, 0.75, 0.95]:
        result[f"{prefix}_q{int(quantile * 100):02d}"] = float(
            np.quantile(values, quantile)
        )
    return result


def extract_handcrafted_features(
    sequence: np.ndarray,
    stage: np.ndarray | None,
    epoch_seconds: float,
    config: dict[str, Any],
) -> dict[str, float]:
    """Extract interpretable distributional, geometric and temporal features."""
    if sequence.ndim != 2 or sequence.shape[1] != 3:
        raise ValueError("Expected a T x 3 normalized trajectory.")
    time, depth, rem = sequence.T
    result: dict[str, float] = {
        "n_epochs": float(len(sequence)),
        "duration_hours": float(len(sequence) * epoch_seconds / 3600.0),
    }
    result.update(_axis_summary(depth, "depth"))
    result.update(_axis_summary(rem, "rem"))
    result["depth_rem_correlation"] = (
        float(np.corrcoef(depth, rem)[0, 1])
        if np.std(depth) > 1e-12 and np.std(rem) > 1e-12
        else 0.0
    )

    threshold = float(config["quadrant_threshold"])
    quadrant = (depth >= threshold).astype(int) * 2 + (rem >= threshold).astype(int)
    for state in range(4):
        result[f"quadrant_{state}_proportion"] = float(np.mean(quadrant == state))
        runs = _run_lengths(quadrant, state)
        result[f"quadrant_{state}_mean_dwell_minutes"] = (
            float(np.mean(runs) * epoch_seconds / 60.0) if len(runs) else 0.0
        )
        result[f"quadrant_{state}_max_dwell_minutes"] = (
            float(np.max(runs) * epoch_seconds / 60.0) if len(runs) else 0.0
        )
    transition_counts = np.zeros((4, 4), dtype=float)
    np.add.at(transition_counts, (quadrant[:-1], quadrant[1:]), 1)
    transition_probabilities = transition_counts / np.clip(
        transition_counts.sum(axis=1, keepdims=True), 1.0, None
    )
    for source in range(4):
        for target in range(4):
            result[f"quadrant_transition_{source}_{target}"] = float(
                transition_probabilities[source, target]
            )
    result["quadrant_transition_rate_per_hour"] = float(
        np.count_nonzero(quadrant[1:] != quadrant[:-1])
        / max(result["duration_hours"], 1e-12)
    )

    d_depth = np.diff(depth)
    d_rem = np.diff(rem)
    step_2d = np.hypot(d_depth, d_rem)
    step_3d = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    result.update(
        {
            "path_length_2d_per_hour": float(
                np.sum(step_2d) / max(result["duration_hours"], 1e-12)
            ),
            "path_length_3d": float(np.sum(step_3d)),
            "mean_speed_2d_per_hour": float(np.mean(step_2d) * 3600.0 / epoch_seconds),
            "speed_2d_q95_per_hour": float(
                np.quantile(step_2d, 0.95) * 3600.0 / epoch_seconds
            ),
            "mean_acceleration": float(np.mean(np.abs(np.diff(step_2d))))
            if len(step_2d) > 1
            else 0.0,
            "antagonistic_change_proportion": float(np.mean(d_depth * d_rem < 0)),
            "directional_entropy": _directional_entropy(
                d_depth, d_rem, int(config["direction_bins"])
            ),
        }
    )
    if len(d_depth) > 1:
        vectors = np.column_stack([d_depth, d_rem])
        norm = np.linalg.norm(vectors, axis=1)
        valid = (norm[:-1] > 1e-12) & (norm[1:] > 1e-12)
        if np.any(valid):
            cosine = np.sum(vectors[:-1][valid] * vectors[1:][valid], axis=1)
            cosine /= norm[:-1][valid] * norm[1:][valid]
            angles = np.arccos(np.clip(cosine, -1.0, 1.0))
            result["mean_turning_angle"] = float(np.mean(angles))
        else:
            result["mean_turning_angle"] = 0.0

    for lag in config["autocorrelation_lags"]:
        result[f"depth_autocorrelation_lag_{lag}"] = _autocorrelation(depth, int(lag))
        result[f"rem_autocorrelation_lag_{lag}"] = _autocorrelation(rem, int(lag))
    for values, prefix in [(depth, "depth"), (rem, "rem")]:
        dominant, entropy = _spectral_features(values, epoch_seconds)
        result[f"{prefix}_dominant_cycles_per_hour"] = dominant
        result[f"{prefix}_spectral_entropy"] = entropy

    number_of_segments = int(config["temporal_segments"])
    for segment_index, indices in enumerate(np.array_split(np.arange(len(sequence)), number_of_segments)):
        if len(indices) == 0:
            continue
        result[f"segment_{segment_index}_depth_mean"] = float(np.mean(depth[indices]))
        result[f"segment_{segment_index}_rem_mean"] = float(np.mean(rem[indices]))
        for state in range(4):
            result[f"segment_{segment_index}_quadrant_{state}_proportion"] = float(
                np.mean(quadrant[indices] == state)
            )

    if stage is not None and len(stage) == len(sequence):
        finite_stage = stage[np.isfinite(stage)].astype(int)
        for state in range(5):
            result[f"stage_{state}_proportion"] = (
                float(np.mean(finite_stage == state)) if len(finite_stage) else float("nan")
            )
        if len(finite_stage) > 1:
            result["stage_transition_rate_per_hour"] = float(
                np.count_nonzero(finite_stage[1:] != finite_stage[:-1])
                / max(result["duration_hours"], 1e-12)
            )
    return result
