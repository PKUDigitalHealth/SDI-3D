"""Extract Body/SDI, REM-axis, and dual-axis trajectory features.

Each input CSV represents one participant-night. Required columns:

* ``wake_depth_score``: continuous Body/SDI score.
* ``rem_logit_rem``: continuous REM-axis score.

Optional columns are used to identify REM and sleep epochs, in this order:

* REM mask: ``rem_true`` -> ``label == 4`` -> ``rem_prob_rem >= 0.5``
  -> normalized REM-axis threshold.
* Sleep mask for REM proportion: ``wake_true == 0`` -> ``label != 0``
  -> all finite epochs.

The default scaling is per-record min-max scaling, matching the current study
pipeline. The final analysis panel contains 24 features: REM_DFA and REM_MRA
are intentionally omitted, while TOS, TCI, and SRI are included.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, cKDTree


REQUIRED_COLUMNS = {"wake_depth_score", "rem_logit_rem"}


def minmax(values: np.ndarray) -> np.ndarray:
    """Return per-record min-max scaled values in [0, 1]."""
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if math.isclose(mean, 0.0, abs_tol=1e-12):
        return float("nan")
    return float(np.std(values, ddof=0) / abs(mean))


def skewness(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    standard_deviation = float(np.std(values, ddof=0))
    if math.isclose(standard_deviation, 0.0, abs_tol=1e-12):
        return 0.0
    return float(np.mean(centered**3) / standard_deviation**3)


def approximate_entropy(
    values: np.ndarray,
    embedding_dimension: int = 2,
    tolerance_ratio: float = 0.2,
) -> float:
    """Compute approximate entropy using Chebyshev distance.

    The tolerance is ``tolerance_ratio * SD``. Self-matches are included, as
    required by the standard approximate-entropy definition.
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= embedding_dimension + 1:
        return float("nan")
    tolerance = tolerance_ratio * float(np.std(values, ddof=0))
    if tolerance <= 0:
        return 0.0

    def phi(dimension: int) -> float:
        vectors = np.lib.stride_tricks.sliding_window_view(values, dimension)
        counts = cKDTree(vectors).query_ball_point(
            vectors, r=tolerance, p=np.inf, return_length=True
        )
        proportions = counts / len(vectors)
        return float(np.mean(np.log(np.clip(proportions, 1e-12, None))))

    return phi(embedding_dimension) - phi(embedding_dimension + 1)


def dfa_alpha(values: np.ndarray, number_of_scales: int = 12) -> float:
    """Estimate the detrended fluctuation analysis scaling exponent."""
    values = np.asarray(values, dtype=float)
    if len(values) < 32 or np.std(values) <= 0:
        return float("nan")

    integrated = np.cumsum(values - np.mean(values))
    maximum_scale = len(values) // 4
    scales = np.unique(
        np.floor(
            np.geomspace(4, max(5, maximum_scale), num=number_of_scales)
        ).astype(int)
    )
    valid_scales = []
    fluctuations = []
    for scale in scales:
        number_of_segments = len(integrated) // scale
        if number_of_segments < 2:
            continue
        segment_rms = []
        time = np.arange(scale, dtype=float)
        for segment_index in range(number_of_segments):
            segment = integrated[
                segment_index * scale : (segment_index + 1) * scale
            ]
            trend = np.polyval(np.polyfit(time, segment, 1), time)
            segment_rms.append(np.mean((segment - trend) ** 2))
        fluctuation = math.sqrt(float(np.mean(segment_rms)))
        if fluctuation > 0:
            valid_scales.append(scale)
            fluctuations.append(fluctuation)

    if len(valid_scales) < 3:
        return float("nan")
    return float(
        np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)[0]
    )


def shannon_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def axis_features(
    values: np.ndarray,
    below_threshold: float,
    prefix: str,
    include_dfa: bool = True,
) -> dict[str, float]:
    """Return SDI-style whole-night features for one normalized axis."""
    features = {
        f"{prefix}_CV": coefficient_of_variation(values),
        f"{prefix}_SK": skewness(values),
        f"{prefix}_RB": float(np.mean(values < below_threshold)),
        f"{prefix}_AP": float(np.mean(values)),
        f"{prefix}_APEn": approximate_entropy(values),
    }
    if include_dfa:
        features[f"{prefix}_DFA"] = dfa_alpha(values)
    return features


def trajectory_occupancy_surface(body: np.ndarray, rem: np.ndarray) -> float:
    """TOS: convex-hull area occupied by the normalized 2D trajectory."""
    points = np.column_stack((body, rem))
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def trajectory_curvature_index(body: np.ndarray, rem: np.ndarray) -> float:
    """TCI: mean turning angle between consecutive 2D steps, in degrees."""
    vectors = np.column_stack((np.diff(body), np.diff(rem)))
    if len(vectors) < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1)
    valid = (norms[:-1] > 1e-12) & (norms[1:] > 1e-12)
    if not np.any(valid):
        return 0.0
    cosine = np.sum(vectors[:-1][valid] * vectors[1:][valid], axis=1)
    cosine /= norms[:-1][valid] * norms[1:][valid]
    turning_angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.mean(turning_angles)))


def state_recurrence_index(
    body: np.ndarray,
    rem: np.ndarray,
    radius: float = 0.05,
    theiler_window: int = 20,
) -> float:
    """SRI: fraction of recurrent 2D states outside a Theiler window.

    A pair is recurrent when its Euclidean distance is smaller than ``radius``.
    Pairs separated by at most ``theiler_window`` epochs are excluded to avoid
    counting trivial short-lag continuity as physiological recurrence.
    """
    points = np.column_stack((body, rem))
    n_points = len(points)
    if n_points <= theiler_window + 1:
        return float("nan")
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    recurrent_pairs = (
        int(np.count_nonzero((pairs[:, 1] - pairs[:, 0]) > theiler_window))
        if pairs.size
        else 0
    )
    comparable_pairs = (
        (n_points - theiler_window - 1) * (n_points - theiler_window) // 2
    )
    return float(recurrent_pairs / comparable_pairs)


def resolve_masks(
    frame: pd.DataFrame,
    finite_mask: np.ndarray,
    normalized_rem: np.ndarray,
    rem_threshold: float,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Resolve REM and sleep masks while documenting their provenance."""
    filtered = frame.loc[finite_mask].reset_index(drop=True)

    if "rem_true" in filtered.columns:
        rem_mask = (
            pd.to_numeric(filtered["rem_true"], errors="coerce") >= 0.5
        ).fillna(False).to_numpy(bool)
        rem_source = "rem_true"
    elif "label" in filtered.columns:
        rem_mask = (
            pd.to_numeric(filtered["label"], errors="coerce") == 4
        ).fillna(False).to_numpy(bool)
        rem_source = "label_eq_4"
    elif "rem_prob_rem" in filtered.columns:
        rem_mask = (
            pd.to_numeric(filtered["rem_prob_rem"], errors="coerce") >= 0.5
        ).fillna(False).to_numpy(bool)
        rem_source = "rem_prob_ge_0.5"
    else:
        rem_mask = normalized_rem >= rem_threshold
        rem_source = "normalized_rem_axis_threshold"

    if "wake_true" in filtered.columns:
        sleep_mask = (
            pd.to_numeric(filtered["wake_true"], errors="coerce") < 0.5
        ).fillna(False).to_numpy(bool)
        sleep_source = "wake_true_eq_0"
    elif "label" in filtered.columns:
        sleep_mask = (
            pd.to_numeric(filtered["label"], errors="coerce") != 0
        ).fillna(False).to_numpy(bool)
        sleep_source = "label_ne_0"
    else:
        sleep_mask = np.ones(len(filtered), dtype=bool)
        sleep_source = "all_finite_epochs"

    return rem_mask, sleep_mask, rem_source, sleep_source


def confidence_weights(rem: np.ndarray, method: str) -> np.ndarray:
    """Return REM-axis certainty weights in [0, 1]."""
    probability_like = np.clip(rem, 1e-7, 1 - 1e-7)
    if method == "distance":
        return 2 * np.abs(probability_like - 0.5)
    if method == "entropy":
        binary_entropy = -(
            probability_like * np.log2(probability_like)
            + (1 - probability_like) * np.log2(1 - probability_like)
        )
        return 1 - binary_entropy
    raise ValueError(f"unknown confidence method: {method}")


def core_features(
    frame: pd.DataFrame,
    filename: str,
    scale: str,
    body_threshold: float,
    rem_threshold: float,
    epoch_seconds: float,
    confidence_method: str,
    direction_bins: int,
    minimum_epochs: int,
    sri_radius: float,
    sri_theiler_window: int,
) -> dict[str, object]:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise KeyError(f"missing required columns: {', '.join(sorted(missing))}")

    raw_body = pd.to_numeric(
        frame["wake_depth_score"], errors="coerce"
    ).to_numpy(float)
    raw_rem = pd.to_numeric(
        frame["rem_logit_rem"], errors="coerce"
    ).to_numpy(float)
    finite_mask = np.isfinite(raw_body) & np.isfinite(raw_rem)
    body = raw_body[finite_mask]
    rem = raw_rem[finite_mask]
    if len(body) < minimum_epochs:
        raise ValueError(
            f"fewer than {minimum_epochs} finite paired epochs: {len(body)}"
        )

    if scale == "per_record":
        body = minmax(body)
        rem = minmax(rem)

    rem_mask, sleep_mask, rem_source, sleep_source = resolve_masks(
        frame, finite_mask, minmax(rem), rem_threshold
    )
    duration_hours = len(body) * epoch_seconds / 3600
    sleep_epochs = int(np.sum(sleep_mask))
    rem_sleep_mask = rem_mask & sleep_mask
    rem_proportion = (
        float(np.sum(rem_sleep_mask) / sleep_epochs)
        if sleep_epochs > 0
        else float("nan")
    )

    result: dict[str, object] = {
        "source_file": filename,
        "n_epochs": len(body),
        "duration_hours": duration_hours,
        "scale_mode": scale,
        "body_threshold": body_threshold,
        "rem_threshold": rem_threshold,
        "rem_mask_source": rem_source,
        "sleep_mask_source": sleep_source,
    }
    result.update(axis_features(body, body_threshold, "Body", include_dfa=True))
    result.update(axis_features(rem, rem_threshold, "REM", include_dfa=False))

    result["Body_MDR"] = (
        float(np.mean(body[rem_sleep_mask]))
        if np.any(rem_sleep_mask)
        else float("nan")
    )
    result["REM_PR"] = rem_proportion
    result["REM_High_Activation_Ratio"] = float(
        np.mean(rem >= rem_threshold)
    )
    result["REM_High_Activation_Ratio_During_REM"] = (
        float(np.mean(rem[rem_sleep_mask] >= rem_threshold))
        if np.any(rem_sleep_mask)
        else float("nan")
    )
    result["REM_NREM_Contrast"] = (
        float(np.mean(rem[rem_sleep_mask]) - np.mean(rem[sleep_mask & ~rem_mask]))
        if np.any(rem_sleep_mask) and np.any(sleep_mask & ~rem_mask)
        else float("nan")
    )

    body_change = np.diff(body)
    rem_change = np.diff(rem)
    transition_count = int(
        np.count_nonzero(
            (body[1:] >= body_threshold)
            != (body[:-1] >= body_threshold)
        )
    )
    path_step = np.hypot(body_change, rem_change)
    moving = path_step > 1e-12
    angles = np.arctan2(rem_change[moving], body_change[moving])
    direction_counts, _ = np.histogram(
        angles, bins=direction_bins, range=(-np.pi, np.pi)
    )
    weights = confidence_weights(rem, confidence_method)
    weight_sum = float(np.sum(weights))

    result.update(
        {
            "Conf_Weighted_AP": (
                float(np.sum(body * weights) / weight_sum)
                if weight_sum > 0
                else float(np.mean(body))
            ),
            "BB_Antagonistic_Ratio": float(
                np.mean((body_change * rem_change) < 0)
            ),
            "Path_Length_2D": float(np.sum(path_step) / duration_hours),
            "Transition_Rate_per_hour": float(
                transition_count / duration_hours
            ),
            "Directional_Entropy": shannon_entropy(direction_counts),
            "TOS": trajectory_occupancy_surface(body, rem),
            "TCI": trajectory_curvature_index(body, rem),
            "SRI": state_recurrence_index(
                body, rem, sri_radius, sri_theiler_window
            ),
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Body/SDI, REM-axis, and dual-axis features from "
            "whole-night CSV files"
        )
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--scale", choices=["per_record", "none"], default="per_record"
    )
    parser.add_argument("--body-threshold", type=float, default=0.1406)
    parser.add_argument("--rem-threshold", type=float, default=0.5)
    parser.add_argument("--epoch-seconds", type=float, default=30.0)
    parser.add_argument(
        "--confidence-method",
        choices=["distance", "entropy"],
        default="distance",
    )
    parser.add_argument("--direction-bins", type=int, default=12)
    parser.add_argument("--minimum-epochs", type=int, default=60)
    parser.add_argument("--sri-radius", type=float, default=0.05)
    parser.add_argument("--sri-theiler-window", type=int, default=20)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    input_folder = arguments.input_dir.resolve()
    output_csv = arguments.output_csv.resolve()
    pattern = "**/*.csv" if arguments.recursive else "*.csv"
    files = sorted(
        path
        for path in input_folder.glob(pattern)
        if path.resolve() != output_csv
        and path.name != "body_rem_feature_errors.csv"
    )

    rows = []
    errors = []
    for path in files:
        source_name = str(path.relative_to(input_folder))
        try:
            rows.append(
                core_features(
                    pd.read_csv(path),
                    source_name,
                    arguments.scale,
                    arguments.body_threshold,
                    arguments.rem_threshold,
                    arguments.epoch_seconds,
                    arguments.confidence_method,
                    arguments.direction_bins,
                    arguments.minimum_epochs,
                    arguments.sri_radius,
                    arguments.sri_theiler_window,
                )
            )
        except Exception as error:  # Continue batch extraction and audit failures.
            errors.append(
                {
                    "source_file": source_name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    error_path = output_csv.with_name("body_rem_feature_errors.csv")
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    elif error_path.exists():
        error_path.unlink()

    print(
        f"Wrote {len(rows)} participant-night rows to {output_csv}; "
        f"skipped {len(errors)} files."
    )


if __name__ == "__main__":
    main()
