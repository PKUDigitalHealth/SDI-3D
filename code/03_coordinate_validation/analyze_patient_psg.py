from pathlib import Path

import numpy as np
import pandas as pd


EDF_PATH = Path(r"C:\Users\SaberShuo\Desktop\edf_files\shhs1-200005.edf")
CSV_PATH = Path(
    r"C:\Users\SaberShuo\Desktop\edf_files\shhs1-200005_features_predictions.csv"
)
EPOCH_OFFSET = 67


def parse_edf(path: Path):
    with path.open("rb") as file:
        fixed = file.read(256)

    def text(start, end):
        return fixed[start:end].decode("latin-1").strip()

    n_records = int(text(236, 244))
    record_duration = float(text(244, 252))
    n_signals = int(text(252, 256))
    header_bytes = int(text(184, 192))

    with path.open("rb") as file:
        file.seek(256)
        signal_header = file.read(header_bytes - 256)

    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    names = [
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
    columns = {}
    position = 0
    for name, width in zip(names, widths):
        columns[name] = [
            signal_header[position + i * width : position + (i + 1) * width]
            .decode("latin-1")
            .strip()
            for i in range(n_signals)
        ]
        position += n_signals * width

    samples_per_record = np.array(columns["samples_per_record"], dtype=int)
    starts = np.concatenate([[0], np.cumsum(samples_per_record)[:-1]])
    data = np.memmap(
        path,
        mode="r",
        dtype="<i2",
        offset=header_bytes,
        shape=(n_records, int(samples_per_record.sum())),
    )

    def get_channel(label):
        index = columns["label"].index(label)
        digital = np.asarray(
            data[:, starts[index] : starts[index] + samples_per_record[index]],
            dtype=float,
        ).reshape(-1)
        dmin = float(columns["dig_min"][index])
        dmax = float(columns["dig_max"][index])
        pmin = float(columns["phys_min"][index])
        pmax = float(columns["phys_max"][index])
        physical = (digital - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
        sampling_rate = samples_per_record[index] / record_duration
        return physical, sampling_rate

    metadata = {
        "n_records": n_records,
        "record_duration": record_duration,
        "duration_seconds": n_records * record_duration,
        "labels": columns["label"],
    }
    return get_channel, metadata


def epoch_psd_features(signal, sampling_rate, epoch_starts, kind):
    epoch_samples = int(round(30 * sampling_rate))
    frequencies = np.fft.rfftfreq(epoch_samples, d=1 / sampling_rate)
    window = np.hanning(epoch_samples)
    window_norm = np.sum(window**2)
    outputs = []
    for epoch in epoch_starts:
        start = int(round(epoch * 30 * sampling_rate))
        segment = np.asarray(signal[start : start + epoch_samples], dtype=float)
        segment = segment - np.mean(segment)
        spectrum = np.abs(np.fft.rfft(segment * window)) ** 2 / window_norm
        total_mask = (frequencies >= 0.5) & (frequencies <= 30)
        if kind == "eeg":
            delta_mask = (frequencies >= 0.5) & (frequencies < 4)
            theta_mask = (frequencies >= 4) & (frequencies < 8)
            slope_mask = (frequencies >= 2) & (frequencies <= 30)
            valid = slope_mask & (spectrum > 0)
            slope = np.polyfit(
                np.log10(frequencies[valid]), np.log10(spectrum[valid]), 1
            )[0]
            outputs.append(
                (
                    np.log10(spectrum[delta_mask].sum() + 1e-12),
                    spectrum[delta_mask].sum() / (spectrum[total_mask].sum() + 1e-12),
                    spectrum[theta_mask].sum() / (spectrum[total_mask].sum() + 1e-12),
                    slope,
                    np.sqrt(np.mean(segment**2)),
                )
            )
        elif kind == "eog":
            eye_mask = (frequencies >= 0.3) & (frequencies <= 2)
            outputs.append(
                (
                    np.log10(spectrum[eye_mask].sum() + 1e-12),
                    spectrum[eye_mask].sum() / (spectrum[(frequencies >= 0.3) & (frequencies <= 10)].sum() + 1e-12),
                    np.sqrt(np.mean(segment**2)),
                )
            )
        elif kind == "emg":
            emg_mask = (frequencies >= 10) & (frequencies <= min(40, sampling_rate / 2 - 0.1))
            outputs.append(
                (
                    np.log10(spectrum[emg_mask].sum() + 1e-12),
                    np.sqrt(np.mean(segment**2)),
                )
            )
    return np.asarray(outputs)


def rank_correlation(first, second):
    return np.corrcoef(
        pd.Series(first).rank(method="average"),
        pd.Series(second).rank(method="average"),
    )[0, 1]


def correlation_table(data, group_name, mask, variables):
    rows = []
    subset = data.loc[mask]
    for axis, marker in variables:
        rows.append(
            {
                "subset": group_name,
                "n": len(subset),
                "axis": axis,
                "marker": marker,
                "pearson": subset[axis].corr(subset[marker]),
                "spearman": rank_correlation(subset[axis], subset[marker]),
            }
        )
    return rows


def residualize(values, design):
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def subset_design(data, mask, include_stage=True):
    subset = data.loc[mask]
    time = np.linspace(-1, 1, len(data))[mask]
    columns = [np.ones(len(subset)), time, time**2, time**3]
    if include_stage:
        stages = pd.get_dummies(
            subset["label"].astype(str), drop_first=True, dtype=float
        ).to_numpy()
        if stages.size:
            columns.extend(stages.T)
    return np.column_stack(columns)


def joint_axis_model(data, mask, outcome, include_stage=True):
    subset = data.loc[mask]
    base = subset_design(data, mask, include_stage=include_stage)
    body = subset["body"].to_numpy()
    rem = subset["rem_axis"].to_numpy()
    target = subset[outcome].to_numpy()

    def fit(additions):
        design = np.column_stack([base] + additions)
        residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
        return np.sum(residual**2)

    sse_base = fit([])
    sse_body = fit([body])
    sse_rem = fit([rem])
    sse_both = fit([body, rem])

    target_residual = residualize(target, base)
    body_residual = residualize(body, base)
    rem_residual = residualize(rem, base)
    standardized = np.column_stack(
        [
            (body_residual - body_residual.mean()) / body_residual.std(),
            (rem_residual - rem_residual.mean()) / rem_residual.std(),
        ]
    )
    standardized_target = (
        target_residual - target_residual.mean()
    ) / target_residual.std()
    betas = np.linalg.lstsq(standardized, standardized_target, rcond=None)[0]
    return {
        "subset_n": len(subset),
        "outcome": outcome,
        "body_beta": betas[0],
        "rem_beta": betas[1],
        "body_partial_r2_alone": 1 - sse_body / sse_base,
        "rem_partial_r2_alone": 1 - sse_rem / sse_base,
        "both_partial_r2": 1 - sse_both / sse_base,
        "body_increment_after_rem": 1 - sse_both / sse_rem,
        "rem_increment_after_body": 1 - sse_both / sse_body,
    }


def moving_block_ci(data, mask, first, second, block_length=20, repetitions=3000):
    rng = np.random.default_rng(20260722)
    n = len(data)
    values = []
    mask_array = np.asarray(mask)
    first_values = data[first].to_numpy()
    second_values = data[second].to_numpy()
    blocks_needed = int(np.ceil(n / block_length))
    for _ in range(repetitions):
        starts = rng.integers(0, n - block_length + 1, blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block_length) for start in starts]
        )[:n]
        selected = indices[mask_array[indices]]
        if len(selected) >= 20:
            values.append(np.corrcoef(first_values[selected], second_values[selected])[0, 1])
    return np.quantile(values, [0.025, 0.5, 0.975])


def main():
    predictions = pd.read_csv(CSV_PATH)
    get_channel, metadata = parse_edf(EDF_PATH)
    epoch_starts = np.arange(len(predictions)) + EPOCH_OFFSET

    eeg_primary, eeg_rate = get_channel("EEG")
    eeg_secondary, eeg_secondary_rate = get_channel("EEG(sec)")
    eog_left, eog_rate = get_channel("EOG(L)")
    eog_right, eog_right_rate = get_channel("EOG(R)")
    emg, emg_rate = get_channel("EMG")

    eeg1 = epoch_psd_features(eeg_primary, eeg_rate, epoch_starts, "eeg")
    eeg2 = epoch_psd_features(eeg_secondary, eeg_secondary_rate, epoch_starts, "eeg")
    eog_difference = epoch_psd_features(
        eog_left - eog_right, eog_rate, epoch_starts, "eog"
    )
    emg_features = epoch_psd_features(emg, emg_rate, epoch_starts, "emg")

    data = predictions.copy()
    data["body"] = (data["wake_depth_score"] - data["wake_depth_score"].min()) / (
        data["wake_depth_score"].max() - data["wake_depth_score"].min()
    )
    data["rem_axis"] = (data["rem_logit_rem"] - data["rem_logit_rem"].min()) / (
        data["rem_logit_rem"].max() - data["rem_logit_rem"].min()
    )
    data["rem_log_odds"] = np.log(
        np.clip(data["rem_prob_rem"], 1e-6, 1 - 1e-6)
        / np.clip(1 - data["rem_prob_rem"], 1e-6, 1)
    )
    data["eeg_delta_log_primary"] = eeg1[:, 0]
    data["eeg_delta_rel_primary"] = eeg1[:, 1]
    data["eeg_theta_rel_primary"] = eeg1[:, 2]
    data["eeg_slope_primary"] = eeg1[:, 3]
    data["eeg_rms_primary"] = eeg1[:, 4]
    data["eeg_delta_rel_secondary"] = eeg2[:, 1]
    data["eeg_slope_secondary"] = eeg2[:, 3]
    data["eog_low_log"] = eog_difference[:, 0]
    data["eog_low_rel"] = eog_difference[:, 1]
    data["eog_rms"] = eog_difference[:, 2]
    data["emg_high_log"] = emg_features[:, 0]
    data["emg_rms"] = emg_features[:, 1]

    variables = [
        ("body", "eeg_delta_rel_primary"),
        ("body", "eeg_slope_primary"),
        ("body", "eeg_delta_rel_secondary"),
        ("body", "eeg_slope_secondary"),
        ("rem_axis", "eog_low_log"),
        ("rem_axis", "eog_rms"),
        ("rem_axis", "emg_high_log"),
        ("rem_log_odds", "eog_low_log"),
    ]
    rows = []
    rows += correlation_table(data, "all", np.ones(len(data), dtype=bool), variables)
    rows += correlation_table(data, "sleep", data["label"] != 0, variables)
    rows += correlation_table(data, "NREM", data["label"].isin([1, 2, 3]), variables)
    rows += correlation_table(data, "N2_N3", data["label"].isin([2, 3]), variables)
    rows += correlation_table(data, "REM", data["label"] == 4, variables)
    correlations = pd.DataFrame(rows)

    time = np.linspace(-1, 1, len(data))
    stage = pd.get_dummies(data["label"].astype(str), drop_first=True, dtype=float).to_numpy()
    design = np.column_stack(
        [np.ones(len(data)), time, time**2, time**3, stage]
    )
    conditional_rows = []
    for axis, marker in variables:
        axis_residual = residualize(data[axis].to_numpy(), design)
        marker_residual = residualize(data[marker].to_numpy(), design)
        conditional_rows.append(
            {
                "axis": axis,
                "marker": marker,
                "partial_pearson": np.corrcoef(axis_residual, marker_residual)[0, 1],
                "partial_spearman": rank_correlation(axis_residual, marker_residual),
            }
        )
    conditional = pd.DataFrame(conditional_rows)

    stage_rows = []
    for label in range(5):
        stage_mask = data["label"] == label
        for axis, marker in [
            ("body", "eeg_slope_primary"),
            ("body", "eeg_slope_secondary"),
            ("rem_axis", "eog_low_log"),
            ("rem_axis", "eog_rms"),
            ("rem_axis", "emg_high_log"),
        ]:
            subset = data.loc[stage_mask]
            stage_rows.append(
                {
                    "label": label,
                    "n": len(subset),
                    "axis": axis,
                    "marker": marker,
                    "pearson": subset[axis].corr(subset[marker]),
                    "spearman": rank_correlation(subset[axis], subset[marker]),
                }
            )
    stage_correlations = pd.DataFrame(stage_rows)

    joint_models = pd.DataFrame(
        [
            {
                "subset": "NREM",
                **joint_axis_model(
                    data,
                    data["label"].isin([1, 2, 3]),
                    "eeg_slope_primary",
                    include_stage=True,
                ),
            },
            {
                "subset": "NREM",
                **joint_axis_model(
                    data,
                    data["label"].isin([1, 2, 3]),
                    "eeg_slope_secondary",
                    include_stage=True,
                ),
            },
            {
                "subset": "REM",
                **joint_axis_model(
                    data,
                    data["label"] == 4,
                    "eog_low_log",
                    include_stage=False,
                ),
            },
            {
                "subset": "REM",
                **joint_axis_model(
                    data,
                    data["label"] == 4,
                    "eog_rms",
                    include_stage=False,
                ),
            },
            {
                "subset": "REM",
                **joint_axis_model(
                    data,
                    data["label"] == 4,
                    "emg_high_log",
                    include_stage=False,
                ),
            },
        ]
    )

    key_cis = []
    for subset_name, mask, axis, marker in [
        ("NREM", data["label"].isin([1, 2, 3]), "body", "eeg_slope_primary"),
        ("NREM", data["label"].isin([1, 2, 3]), "body", "eeg_slope_secondary"),
        ("REM", data["label"] == 4, "rem_axis", "eog_low_log"),
        ("REM", data["label"] == 4, "rem_axis", "eog_rms"),
        ("REM", data["label"] == 4, "rem_axis", "emg_high_log"),
    ]:
        ci = moving_block_ci(data, mask, axis, marker)
        key_cis.append(
            {
                "subset": subset_name,
                "axis": axis,
                "marker": marker,
                "ci_low": ci[0],
                "bootstrap_median": ci[1],
                "ci_high": ci[2],
            }
        )

    stage_summary = data.groupby("label")[
        [
            "body",
            "rem_axis",
            "rem_prob_rem",
            "eeg_delta_rel_primary",
            "eeg_slope_primary",
            "eog_low_log",
            "emg_high_log",
        ]
    ].agg(["median", "mean", "std"])

    print("EDF_METADATA", metadata)
    print("ALIGNMENT first_epoch", int(epoch_starts[0]), "last_epoch", int(epoch_starts[-1]))
    print("\nSTAGE_SUMMARY\n", stage_summary.round(4).to_string())
    print("\nCORRELATIONS\n", correlations.round(4).to_string(index=False))
    print("\nCONDITIONAL\n", conditional.round(4).to_string(index=False))
    print("\nSTAGE_CORRELATIONS\n", stage_correlations.round(4).to_string(index=False))
    print("\nJOINT_MODELS\n", joint_models.round(4).to_string(index=False))
    print("\nKEY_BLOCK_BOOTSTRAP_CI\n", pd.DataFrame(key_cis).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
