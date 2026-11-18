"""End-to-end adapter: HeartVoice BIN3_16 -> SDI-3D predictions.

Default usage (after configuring the two checkpoint paths below once):

    python run_wenxin_sdi3d.py /path/to/recording.BIN3_16

The pipeline decodes one channel from the three-channel 200-Hz source,
reproduces the published 100-Hz preprocessing and 39 handcrafted features,
extracts the 1,152-dimensional Stage-1 embedding, runs the 15-epoch Stage-2
model, and writes CSV/NPZ outputs. Use ``--prepare-only`` to generate and
inspect model inputs without requiring checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# Configure these paths once. After that, only the raw input path is required.
# ---------------------------------------------------------------------------
STAGE1_CHECKPOINT = Path(
    "/home/zhaoqingshuo/SDI/ECG_sleepstage/step1_model/all_42_final.pth"
)
STAGE2_CHECKPOINT = Path(
    "/data/0shared/zhaoqingshuo/SDI/model_W_NW_REM_NREM_all/best.pth"
)

# The two model definitions may live in different directories. Configure each
# directory once; do not point these at the input/output data directory.
STAGE1_CODE_DIR = Path(
    r"C:\Users\SaberShuo\Desktop\ECG-3D\code\train_feature_extract"
)
STAGE2_CODE_DIR = Path(
    r"C:\Users\SaberShuo\Desktop\ECG-3D\code\train_feature_extract"
)

OUTPUT_ROOT = Path("output/real_world_inference")
CHANNEL_INDEX = 0  # zero-based; verify which device channel matches training ECG.
SOURCE_ENDIAN_DTYPE = ">i2"
TARGET_FS = 100
EPOCH_SECONDS = 30
TRIM_EPOCHS_EACH_SIDE = 60
STAGE1_BATCH_SIZE = 128
WINDOW_SIZE = 15


SPECTRAL_FEATURE_NAMES = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "total_power",
    "peak_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "spectral_entropy",
]


def _wavelet_feature_names() -> list[str]:
    levels = ["cA4", "cD4", "cD3", "cD2", "cD1"]
    names = [f"{level}_energy" for level in levels]
    for level in levels:
        names.extend(
            [
                f"{level}_mean",
                f"{level}_std",
                f"{level}_variance",
                f"{level}_maximum",
                f"{level}_minimum",
            ]
        )
    return names


FEATURE_NAMES = SPECTRAL_FEATURE_NAMES + _wavelet_feature_names()
assert len(FEATURE_NAMES) == 39


def resolve_input(path: Path) -> tuple[Path, Path]:
    """Return (BIN3_16 file, companion directory)."""
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.BIN3_16"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one *.BIN3_16 in {path}, found {len(candidates)}"
            )
        return candidates[0], path
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.upper() != ".BIN3_16":
        raise ValueError(f"Expected a *.BIN3_16 or prepared *.npz file, got: {path}")
    return path, path.parent


def load_source(bin_path: Path, companion_dir: Path, channel: int) -> tuple[np.ndarray, float, dict[str, Any]]:
    if channel not in (0, 1, 2):
        raise ValueError("channel must be 0, 1, or 2")

    info_path = companion_dir / "info.bin.npy"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing sampling-rate metadata: {info_path}")
    info = np.load(info_path, allow_pickle=False)
    source_fs = float(np.ravel(info)[0])
    if not np.isfinite(source_fs) or source_fs <= 0:
        raise ValueError(f"Invalid source sampling rate: {source_fs}")

    raw = np.memmap(bin_path, dtype=SOURCE_ENDIAN_DTYPE, mode="r")
    if raw.size % 3:
        raise ValueError(
            f"Raw int16 count {raw.size} is not divisible by three channels"
        )
    n_samples = raw.size // 3
    selected = np.asarray(raw.reshape(n_samples, 3)[:, channel], dtype=np.float64)

    metadata: dict[str, Any] = {
        "source_file": str(bin_path),
        "source_dtype": SOURCE_ENDIAN_DTYPE,
        "source_channels": 3,
        "selected_channel_zero_based": channel,
        "source_fs_hz": source_fs,
        "source_samples_per_channel": int(n_samples),
        "source_duration_seconds": n_samples / source_fs,
    }
    return selected, source_fs, metadata


def robust_zscore(x: np.ndarray) -> np.ndarray:
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("Signal is constant after preprocessing")
    return (x - median) / scale


def scale_minus_one_to_one(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Cannot min-max scale a constant/non-finite signal")
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def preprocess_recording(x: np.ndarray, source_fs: float) -> np.ndarray:
    """Published preprocessing: 100 Hz, 0.5-40 Hz, detrend, robust Z, [-1,1]."""
    if not np.all(np.isfinite(x)):
        raise ValueError("Source ECG contains non-finite values")

    source_fs_int = int(round(source_fs))
    if not math.isclose(source_fs, source_fs_int, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Non-integer source sampling rate is unsupported: {source_fs}")
    divisor = math.gcd(source_fs_int, TARGET_FS)
    up = TARGET_FS // divisor
    down = source_fs_int // divisor
    x = scipy_signal.resample_poly(x, up=up, down=down)

    sos = scipy_signal.butter(
        4,
        [0.5, 40.0],
        btype="bandpass",
        fs=TARGET_FS,
        output="sos",
    )
    x = scipy_signal.sosfiltfilt(sos, x)
    x = scipy_signal.detrend(x, type="linear")
    x = robust_zscore(x)
    x = scale_minus_one_to_one(x)
    return np.asarray(x, dtype=np.float32)


def split_epochs(x: np.ndarray, trim_each_side: int) -> tuple[np.ndarray, np.ndarray, int]:
    epoch_samples = TARGET_FS * EPOCH_SECONDS
    n_complete = len(x) // epoch_samples
    remainder = len(x) - n_complete * epoch_samples
    if n_complete <= 2 * trim_each_side + WINDOW_SIZE:
        raise ValueError(
            f"Only {n_complete} complete epochs; not enough after trimming "
            f"{trim_each_side} from each side"
        )
    epochs = x[: n_complete * epoch_samples].reshape(n_complete, epoch_samples)
    original_epoch_indices = np.arange(n_complete, dtype=np.int32)
    if trim_each_side:
        epochs = epochs[trim_each_side:-trim_each_side]
        original_epoch_indices = original_epoch_indices[trim_each_side:-trim_each_side]
    return epochs[:, None, :], original_epoch_indices, remainder


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if np.sum(mask) < 2:
        return np.zeros(psd.shape[0], dtype=np.float64)
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def extract_39_features(epochs: np.ndarray) -> np.ndarray:
    """Nine Welch spectral + thirty db4 wavelet descriptors per epoch.

    Wavelet order follows the manuscript table: five energies first, followed
    by mean/std/variance/max/min for cA4, cD4, cD3, cD2, and cD1.
    """
    values = np.asarray(epochs[:, 0, :], dtype=np.float64)
    freqs, psd = scipy_signal.welch(
        values,
        fs=TARGET_FS,
        axis=1,
        nperseg=256,
        noverlap=128,
        detrend="constant",
        scaling="density",
    )
    eps = np.finfo(np.float64).tiny
    delta = _band_power(freqs, psd, 0.5, 4.0)
    theta = _band_power(freqs, psd, 4.0, 8.0)
    alpha = _band_power(freqs, psd, 8.0, 13.0)
    spectral_mask = (freqs >= 0.5) & (freqs <= 40.0)
    sf = freqs[spectral_mask]
    sp = psd[:, spectral_mask]
    total = np.trapz(sp, sf, axis=1)
    peak = sf[np.argmax(sp, axis=1)]
    weights = sp / np.maximum(np.sum(sp, axis=1, keepdims=True), eps)
    centroid = np.sum(weights * sf[None, :], axis=1)
    bandwidth = np.sqrt(
        np.sum(weights * (sf[None, :] - centroid[:, None]) ** 2, axis=1)
    )
    flatness = np.exp(np.mean(np.log(np.maximum(sp, eps)), axis=1)) / np.maximum(
        np.mean(sp, axis=1), eps
    )
    entropy = -np.sum(weights * np.log(np.maximum(weights, eps)), axis=1)
    entropy /= np.log(weights.shape[1])

    spectral = np.column_stack(
        [delta, theta, alpha, total, peak, centroid, bandwidth, flatness, entropy]
    )

    wavelet_rows: list[list[float]] = []
    for epoch in values:
        coeffs = pywt.wavedec(epoch, wavelet="db4", level=4, mode="symmetric")
        if len(coeffs) != 5:
            raise RuntimeError(f"Expected five db4 coefficient arrays, got {len(coeffs)}")
        row = [float(np.sum(c * c)) for c in coeffs]
        for c in coeffs:
            row.extend(
                [
                    float(np.mean(c)),
                    float(np.std(c)),
                    float(np.var(c)),
                    float(np.max(c)),
                    float(np.min(c)),
                ]
            )
        wavelet_rows.append(row)
    wavelet = np.asarray(wavelet_rows, dtype=np.float64)
    features = np.concatenate([spectral, wavelet], axis=1)
    if features.shape != (len(epochs), 39):
        raise RuntimeError(f"Unexpected handcrafted feature shape: {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("Handcrafted features contain non-finite values")
    return features[:, None, :].astype(np.float32)


def _require_model_file(code_dir: Path, filename: str, setting_name: str) -> Path:
    code_dir = code_dir.expanduser().resolve()
    model_file = code_dir / filename
    if not model_file.is_file():
        raise FileNotFoundError(
            f"{filename} was not found at {model_file}. "
            f"Set {setting_name} to the directory that directly contains {filename}; "
            "do not use the data directory."
        )
    return model_file


def _import_models(stage1_code_dir: Path, stage2_code_dir: Path):
    net1d_file = _require_model_file(
        stage1_code_dir, "net1d.py", "STAGE1_CODE_DIR/--stage1-code-dir"
    )
    transformer_file = _require_model_file(
        stage2_code_dir,
        "model_transformer.py",
        "STAGE2_CODE_DIR/--stage2-code-dir",
    )
    # Keep each file's own directory importable in case it has local helpers.
    for code_dir in (net1d_file.parent, transformer_file.parent):
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
    try:
        from net1d import Net1DWithAttention
        from model_transformer import TransformerSleepModel, ThreeHeadSleepModel
    except Exception as exc:
        raise ImportError(
            "Cannot import model definitions. "
            f"net1d.py={net1d_file}; model_transformer.py={transformer_file}; "
            f"original error: {exc}"
        ) from exc
    return Net1DWithAttention, TransformerSleepModel, ThreeHeadSleepModel


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch cannot be imported. Install/repair torch and typing_extensions "
            "before model inference; --prepare-only does not require PyTorch."
        ) from exc
    return torch


def _state_dict(checkpoint_path: Path, device: Any) -> dict[str, Any]:
    torch = _load_torch()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Edit the checkpoint constant once."
        )
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Unsupported checkpoint structure: {checkpoint_path}")
    return {
        (str(k)[7:] if str(k).startswith("module.") else str(k)): v
        for k, v in payload.items()
    }


def build_stage1(Net1DWithAttention, device: Any):
    model = Net1DWithAttention(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 3, 3, 4, 4, 5, 5],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=True,
        use_do=True,
        n_classes=5,
    ).to(device)
    state = _state_dict(STAGE1_CHECKPOINT, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_stage1_embeddings(
    epochs: np.ndarray,
    handcrafted: np.ndarray,
    model,
    device: Any,
) -> np.ndarray:
    torch = _load_torch()
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.from_numpy(epochs), torch.from_numpy(handcrafted))
    loader = DataLoader(
        dataset,
        batch_size=STAGE1_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for x, freq in loader:
            x = x.to(device, non_blocking=True).float()
            freq = freq.to(device, non_blocking=True).float()
            out = model.first_conv(x)
            out = model.first_bn(out)
            out = model.first_activation(out)
            for stage in model.stage_list:
                out = stage(out)
            out = out.mean(-1)
            out = model.time_attention(out)
            out = model.time_fc(out)
            freq_out = model.freq_fc(freq)
            freq_out = model.freq_attention(freq_out)
            combined = torch.cat([out, freq_out], dim=1)
            if combined.shape[1] != 1152:
                raise RuntimeError(f"Unexpected Stage-1 output shape: {combined.shape}")
            chunks.append(combined.cpu().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float32)
    if features.shape != (len(epochs), 1152):
        raise RuntimeError(f"Unexpected Stage-1 feature shape: {features.shape}")
    return features


def build_stage2(TransformerSleepModel, ThreeHeadSleepModel, device: Any):
    backbone = TransformerSleepModel(
        input_dim=WINDOW_SIZE * 1152,
        hidden_dim=512,
        n_heads=8,
        num_layers=3,
        num_classes=2,
        dropout=0.1,
    )
    model = ThreeHeadSleepModel(
        feature_extractor=backbone,
        classes=2,
        hidden_dim=512,
    ).to(device)
    state = _state_dict(STAGE2_CHECKPOINT, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def percentile_normalize(x: np.ndarray, low: float = 5.0, high: float = 95.0) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def run_stage2(
    features: np.ndarray,
    original_epoch_indices: np.ndarray,
    model,
    device: Any,
) -> pd.DataFrame:
    torch = _load_torch()
    half = WINDOW_SIZE // 2
    if len(features) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} feature epochs, got {len(features)}")
    windows = np.stack(
        [features[i - half : i + half + 1] for i in range(half, len(features) - half)],
        axis=0,
    ).astype(np.float32)
    centers = original_epoch_indices[half : len(features) - half]
    inputs = torch.from_numpy(windows).unsqueeze(0).to(device)
    with torch.inference_mode():
        rem_logits, wake_logits, body_raw = model(inputs)
        rem_prob = torch.softmax(rem_logits, dim=-1)[0, :, 1].cpu().numpy()
        wake_prob = torch.softmax(wake_logits, dim=-1)[0, :, 1].cpu().numpy()
        rem_logit = rem_logits[0, :, 1].cpu().numpy()
        wake_logit = wake_logits[0, :, 1].cpu().numpy()
        body_raw = body_raw[0, :, 0].cpu().numpy()

    body_norm = percentile_normalize(body_raw)
    rem_norm = percentile_normalize(rem_prob)
    return pd.DataFrame(
        {
            "original_epoch_index": centers,
            "elapsed_seconds": centers.astype(np.float64) * EPOCH_SECONDS,
            "wake_probability": wake_prob,
            "rem_probability": rem_prob,
            "wake_logit": wake_logit,
            "rem_logit": rem_logit,
            "body_score_raw": body_raw,
            "body_score_normalized": body_norm,
            "rem_score_normalized": rem_norm,
        }
    )


def load_prepared_npz(npz_path: Path):
    """Load an already segmented Stage-1 input without preprocessing it again."""
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [key for key in ("x", "freq_features") if key not in data.files]
        if missing:
            raise KeyError(
                f"Prepared NPZ is missing {missing}; available keys: {data.files}"
            )
        original_x_dtype = str(data["x"].dtype)
        original_freq_dtype = str(data["freq_features"].dtype)
        epochs = np.asarray(data["x"], dtype=np.float32)
        handcrafted = np.asarray(data["freq_features"], dtype=np.float32)

        # Accept convenient 2-D forms but standardize them for Net1D.
        if epochs.ndim == 2 and epochs.shape[1] == TARGET_FS * EPOCH_SECONDS:
            epochs = epochs[:, None, :]
        if handcrafted.ndim == 2 and handcrafted.shape[1] == 39:
            handcrafted = handcrafted[:, None, :]
        if epochs.ndim != 3 or epochs.shape[1:] != (1, TARGET_FS * EPOCH_SECONDS):
            raise ValueError(
                f"Expected x shape (n,1,{TARGET_FS * EPOCH_SECONDS}), got {epochs.shape}"
            )
        if handcrafted.ndim != 3 or handcrafted.shape[1:] != (1, 39):
            raise ValueError(
                f"Expected freq_features shape (n,1,39), got {handcrafted.shape}"
            )
        if len(epochs) != len(handcrafted):
            raise ValueError(
                f"x and freq_features have different epoch counts: "
                f"{len(epochs)} vs {len(handcrafted)}"
            )
        if len(epochs) < WINDOW_SIZE:
            raise ValueError(
                f"Prepared NPZ needs at least {WINDOW_SIZE} epochs, got {len(epochs)}"
            )
        if not np.isfinite(epochs).all() or not np.isfinite(handcrafted).all():
            raise ValueError("Prepared NPZ contains NaN or Inf")

        if "original_epoch_indices" in data.files:
            original_indices = np.asarray(
                data["original_epoch_indices"], dtype=np.int32
            ).reshape(-1)
        else:
            original_indices = np.arange(len(epochs), dtype=np.int32)
        if len(original_indices) != len(epochs):
            raise ValueError("original_epoch_indices length does not match x")

        label_key = "y" if "y" in data.files else "labels" if "labels" in data.files else None
        if label_key is None:
            labels = np.full(len(epochs), -1, dtype=np.int16)
            label_source = "generated -1 (unlabeled inference)"
        else:
            labels = np.asarray(data[label_key]).reshape(-1)
            if len(labels) != len(epochs):
                raise ValueError(f"{label_key} length does not match x")
            labels = labels.astype(np.int16, copy=False)
            label_source = label_key

    metadata: dict[str, Any] = {
        "source_file": str(npz_path),
        "input_mode": "prepared_npz",
        "input_keys": ["x", "freq_features"],
        "input_x_dtype": original_x_dtype,
        "input_freq_features_dtype": original_freq_dtype,
        "model_dtype": "float32",
        "target_fs_hz": TARGET_FS,
        "epoch_seconds": EPOCH_SECONDS,
        "epoch_samples": TARGET_FS * EPOCH_SECONDS,
        "retained_epochs": int(len(epochs)),
        "label_source": label_source,
        "trim_epochs_each_side": 0,
        "preprocessing": [
            "supplied x and freq_features used as-is",
            "no resampling/filtering/detrending/feature recomputation applied",
            "converted to float32 for model inference",
        ],
    }
    return npz_path, epochs, handcrafted, original_indices, labels, metadata


def prepare_inputs(input_path: Path, channel: int, trim_epochs: int):
    input_path = input_path.expanduser().resolve()
    if input_path.is_file() and input_path.suffix.lower() == ".npz":
        return load_prepared_npz(input_path)

    bin_path, companion_dir = resolve_input(input_path)
    source, source_fs, metadata = load_source(bin_path, companion_dir, channel)
    preprocessed = preprocess_recording(source, source_fs)
    epochs, original_indices, remainder = split_epochs(preprocessed, trim_epochs)
    handcrafted = extract_39_features(epochs)
    metadata.update(
        {
            "target_fs_hz": TARGET_FS,
            "epoch_seconds": EPOCH_SECONDS,
            "epoch_samples": TARGET_FS * EPOCH_SECONDS,
            "trim_epochs_each_side": trim_epochs,
            "retained_epochs": int(len(epochs)),
            "discarded_tail_samples_at_100hz": int(remainder),
            "frequency_feature_order": FEATURE_NAMES,
            "preprocessing": [
                "polyphase resampling to 100 Hz",
                "4th-order Butterworth 0.5-40 Hz zero-phase bandpass",
                "linear detrending",
                "record-level robust Z-score using 1.4826*MAD",
                "record-level linear scaling to [-1,1]",
            ],
        }
    )
    labels = np.full(len(epochs), -1, dtype=np.int16)
    metadata["input_mode"] = "raw_bin3_16"
    metadata["label_source"] = "generated -1 (unlabeled inference)"
    return bin_path, epochs, handcrafted, original_indices, labels, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a HeartVoice BIN3_16 recording or prepared x/freq_features NPZ "
            "through the two-stage SDI-3D model."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="BIN3_16 file, its containing directory, or prepared NPZ",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--channel", type=int, choices=(0, 1, 2), default=CHANNEL_INDEX)
    parser.add_argument("--trim-epochs", type=int, default=TRIM_EPOCHS_EACH_SIDE)
    parser.add_argument(
        "--stage1-code-dir",
        type=Path,
        default=STAGE1_CODE_DIR,
        help="Directory containing net1d.py",
    )
    parser.add_argument(
        "--stage2-code-dir",
        type=Path,
        default=STAGE2_CODE_DIR,
        help="Directory containing model_transformer.py",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write preprocessed x/freq_features NPZ without loading model checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path, epochs, handcrafted, original_indices, labels, metadata = prepare_inputs(
        args.input, args.channel, args.trim_epochs
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else OUTPUT_ROOT / source_path.stem
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_path = output_dir / "stage1_input.npz"
    np.savez_compressed(
        prepared_path,
        x=epochs,
        freq_features=handcrafted,
        y=labels,
        original_epoch_indices=original_indices,
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared Stage-1 input: {prepared_path}")
    print(f"x={epochs.shape}, freq_features={handcrafted.shape}, y={labels.shape}")

    if args.prepare_only:
        print("Preparation-only mode complete; no model inference was run.")
        return

    torch = _load_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    Net1DWithAttention, TransformerSleepModel, ThreeHeadSleepModel = _import_models(
        args.stage1_code_dir, args.stage2_code_dir
    )
    stage1 = build_stage1(Net1DWithAttention, device)
    features = extract_stage1_embeddings(epochs, handcrafted, stage1, device)
    feature_path = output_dir / "stage1_features.npz"
    np.savez_compressed(
        feature_path,
        features=features,
        labels=labels,
        original_epoch_indices=original_indices,
    )
    del stage1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stage2 = build_stage2(TransformerSleepModel, ThreeHeadSleepModel, device)
    predictions = run_stage2(features, original_indices, stage2, device)
    csv_path = output_dir / "sdi3d_predictions.csv"
    predictions.to_csv(csv_path, index=False)

    metadata.update(
        {
            "stage1_checkpoint": str(STAGE1_CHECKPOINT),
            "stage2_checkpoint": str(STAGE2_CHECKPOINT),
            "stage1_feature_shape": list(features.shape),
            "stage2_prediction_rows": int(len(predictions)),
            "normalization_note": (
                "Label-free deployment normalization uses recording-wide 5th/95th "
                "percentiles; it is not the label-conditioned evaluation normalization."
            ),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Stage-1 features: {feature_path} | shape={features.shape}")
    print(f"Predictions: {csv_path} | rows={len(predictions)}")


if __name__ == "__main__":
    main()
