"""Eight-dimensional sleep autoencoder with external SHHS validation.

Edit the USER CONFIGURATION section below, then run this file directly.
CFS and MROS form the internal development cohort. SHHS is never used to fit
normalization or model weights and is evaluated only as an external cohort.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


RAW_COLUMNS = ("wake_depth_score", "rem_logit_rem")
EMBEDDING_DIM = 8


# ============================================================================
# USER CONFIGURATION: edit values in this block before running the script.
# ============================================================================
class Settings:
    # One overnight CSV per participant/night in each folder.
    cfs_dir = Path(r"D:\path\to\CFS")
    mros_dir = Path(r"D:\path\to\MROS")
    shhs_dir = Path(r"D:\path\to\SHHS")
    output_dir = Path(r"D:\path\to\sleep_ae_output_8d")

    # Change recursive to True when CSV files are stored in subfolders.
    file_pattern = "*_features_predictions.csv"
    recursive = False

    # Training and model settings.
    epochs = 80
    batch_size = 16
    hidden = 96
    mask_ratio = 0.20
    learning_rate = 1e-3
    val_fraction = 0.10
    test_fraction = 0.10
    global_only = False
    cpu = False
    seed = 42


SETTINGS = Settings()


def robust_location_scale(values):
    """Return median and IQR; protect constant signals from division by zero."""
    median = np.median(values, axis=0)
    iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    return median, np.maximum(iqr, 1e-6)


class SleepNights(Dataset):
    def __init__(
        self,
        paths,
        dataset_names,
        include_within_night=True,
        global_median=None,
        global_iqr=None,
    ):
        paths = [Path(path) for path in paths]
        if not paths:
            raise ValueError("Dataset split has no CSV files")
        if len(paths) != len(dataset_names):
            raise ValueError("Each CSV path must have one dataset name")
        self.dataset_names = list(dataset_names)
        raw, self.ids = [], []
        for path in paths:
            frame = pd.read_csv(path)
            missing = set(RAW_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(
                    f"{path.name} is missing columns: {sorted(missing)}"
                )
            values = (
                frame.loc[:, RAW_COLUMNS]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float32)
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{path.name} contains missing/non-numeric feature values"
                )
            if len(values) < 2:
                raise ValueError(
                    f"{path.name} has fewer than two 30-second windows"
                )
            raw.append(values)
            self.ids.append(path.stem)

        # Fit the global scaler from training nights only. This retains
        # between-night absolute-level differences without test leakage.
        if global_median is None:
            all_windows = np.concatenate(raw, axis=0)
            self.global_median, self.global_iqr = robust_location_scale(
                all_windows
            )
        else:
            self.global_median = np.asarray(global_median, dtype=np.float32)
            self.global_iqr = np.asarray(global_iqr, dtype=np.float32)

        self.x = []
        for one_night in raw:
            global_x = (
                one_night - self.global_median
            ) / self.global_iqr
            if include_within_night:
                median, iqr = robust_location_scale(one_night)
                local_x = (one_night - median) / iqr
                # Four channels: global depth/REM + night-relative depth/REM.
                values = np.concatenate([global_x, local_x], axis=1)
            else:
                values = global_x
            self.x.append(values.astype(np.float32))
        self.n_input = self.x[0].shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.ids[index], self.dataset_names[index]


def collate(batch):
    sequences, ids, dataset_names = zip(*batch)
    lengths = torch.tensor(
        [len(sequence) for sequence in sequences], dtype=torch.long
    )
    max_length = int(lengths.max())
    feature_count = sequences[0].shape[1]
    values = torch.zeros(len(sequences), max_length, feature_count)
    valid_mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        values[index, :length] = torch.from_numpy(sequence)
        valid_mask[index, :length] = True
    return values, valid_mask, lengths, list(ids), list(dataset_names)


class SleepAutoencoder(nn.Module):
    """BiGRU encoder, attention pooling and an eight-dimensional bottleneck."""

    def __init__(self, n_input, hidden=96):
        super().__init__()
        self.encoder = nn.GRU(
            n_input, hidden, batch_first=True, bidirectional=True
        )
        self.score = nn.Linear(2 * hidden, 1)
        self.to_embedding = nn.Sequential(
            nn.Linear(2 * hidden, EMBEDDING_DIM), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden + EMBEDDING_DIM, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, n_input),
        )

    def forward(self, corrupted_x, mask, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            corrupted_x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=corrupted_x.size(1),
        )
        weights = self.score(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        embedding = self.to_embedding(pooled)
        if embedding.shape[1] != EMBEDDING_DIM:
            raise RuntimeError("Encoder did not produce eight features")
        repeated = embedding[:, None].expand(-1, hidden.size(1), -1)
        decoded = self.decoder(torch.cat([hidden, repeated], dim=-1))
        return decoded, embedding, weights


def mask_segments(values, valid_mask, ratio=0.20, max_span=20):
    """Hide contiguous 2.5-10 minute segments for masked reconstruction."""
    corrupted = values.clone()
    hidden = torch.zeros_like(valid_mask)
    for batch_index, length in enumerate(
        valid_mask.sum(dim=1).tolist()
    ):
        target = max(1, round(length * ratio))
        count = 0
        while count < target:
            span = min(random.randint(5, max_span), target - count)
            start = random.randint(0, max(0, length - span))
            hidden[batch_index, start : start + span] = True
            count += span
    corrupted[hidden] = 0.0
    return corrupted, hidden & valid_mask


def discover_paths(data_dir, pattern, recursive):
    folder = Path(data_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {folder}")
    iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
    paths = sorted(path for path in iterator if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matched {folder / pattern}")
    return paths


def split_internal_paths(
    data_dir,
    dataset_name,
    pattern,
    recursive,
    val_fraction,
    test_fraction,
    seed,
):
    """Split one internal cohort so both CFS and MROS occur in every split."""
    paths = discover_paths(data_dir, pattern, recursive)
    if len(paths) < 3:
        raise ValueError(
            f"{dataset_name} needs at least three nights for train/val/test"
        )
    if (
        not 0 < val_fraction < 1
        or not 0 < test_fraction < 1
        or val_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "val_fraction and test_fraction must be positive and sum to less than 1"
        )
    rng = random.Random(seed)
    rng.shuffle(paths)
    count = len(paths)
    n_test = max(1, round(count * test_fraction))
    n_validation = max(1, round(count * val_fraction))
    if n_test + n_validation >= count:
        n_test = 1
        n_validation = 1
    return (
        paths[n_test + n_validation :],
        paths[n_test : n_test + n_validation],
        paths[:n_test],
    )


def masked_mse(model, loader, device, mask_ratio):
    model.eval()
    total = 0.0
    points = 0
    with torch.no_grad():
        for values, valid, lengths, _, _ in loader:
            values = values.to(device)
            valid = valid.to(device)
            lengths = lengths.to(device)
            corrupted, hidden = mask_segments(values, valid, mask_ratio)
            prediction, _, _ = model(corrupted, valid, lengths)
            total += (
                (prediction - values).square().mean(dim=-1)[hidden]
            ).sum().item()
            points += hidden.sum().item()
    return total / points


def export_embeddings(model, dataset, split, device, batch_size):
    """Return one row per night with exactly feature_1 through feature_8."""
    model.eval()
    rows = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    with torch.no_grad():
        for values, valid, lengths, ids, dataset_names in loader:
            _, embedding, _ = model(
                values.to(device), valid.to(device), lengths.to(device)
            )
            embedding = embedding.cpu().numpy()
            if embedding.shape[1] != EMBEDDING_DIM:
                raise RuntimeError("Expected exactly eight exported features")
            for night_id, dataset_name, vector in zip(
                ids, dataset_names, embedding
            ):
                row = {
                    "dataset": dataset_name,
                    "night_id": night_id,
                    "split": split,
                }
                row.update(
                    {
                        f"feature_{index + 1}": float(value)
                        for index, value in enumerate(vector)
                    }
                )
                rows.append(row)
    return rows


def train(settings):
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not settings.cpu else "cpu"
    )
    cfs_split = split_internal_paths(
        settings.cfs_dir,
        "CFS",
        settings.file_pattern,
        settings.recursive,
        settings.val_fraction,
        settings.test_fraction,
        settings.seed,
    )
    mros_split = split_internal_paths(
        settings.mros_dir,
        "MROS",
        settings.file_pattern,
        settings.recursive,
        settings.val_fraction,
        settings.test_fraction,
        settings.seed + 1,
    )
    external_paths = discover_paths(
        settings.shhs_dir, settings.file_pattern, settings.recursive
    )
    train_paths = cfs_split[0] + mros_split[0]
    validation_paths = cfs_split[1] + mros_split[1]
    test_paths = cfs_split[2] + mros_split[2]
    train_names = ["CFS"] * len(cfs_split[0]) + ["MROS"] * len(mros_split[0])
    validation_names = (
        ["CFS"] * len(cfs_split[1]) + ["MROS"] * len(mros_split[1])
    )
    test_names = ["CFS"] * len(cfs_split[2]) + ["MROS"] * len(mros_split[2])
    external_names = ["SHHS"] * len(external_paths)

    train_set = SleepNights(
        train_paths,
        train_names,
        include_within_night=not settings.global_only,
    )
    common = {
        "include_within_night": not settings.global_only,
        "global_median": train_set.global_median,
        "global_iqr": train_set.global_iqr,
    }
    validation_set = SleepNights(
        validation_paths, validation_names, **common
    )
    test_set = SleepNights(test_paths, test_names, **common)
    external_set = SleepNights(
        external_paths, external_names, **common
    )
    loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=settings.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=settings.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    external_loader = DataLoader(
        external_set,
        batch_size=settings.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = SleepAutoencoder(train_set.n_input, settings.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=1e-4
    )

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total = 0.0
        points = 0
        for values, valid, lengths, _, _ in loader:
            values = values.to(device)
            valid = valid.to(device)
            lengths = lengths.to(device)
            corrupted, hidden = mask_segments(
                values, valid, settings.mask_ratio
            )
            prediction, _, _ = model(corrupted, valid, lengths)
            loss = (
                (prediction - values).square().mean(dim=-1)[hidden]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * hidden.sum().item()
            points += hidden.sum().item()
        if epoch == 1 or epoch % 10 == 0 or epoch == settings.epochs:
            validation_loss = masked_mse(
                model, validation_loader, device, settings.mask_ratio
            )
            print(
                f"epoch {epoch:03d} | train masked-MSE={total / points:.6f} "
                f"| val masked-MSE={validation_loss:.6f}"
            )

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_loss = masked_mse(model, test_loader, device, settings.mask_ratio)
    external_loss = masked_mse(
        model, external_loader, device, settings.mask_ratio
    )
    print(f"internal test masked-MSE={test_loss:.6f}")
    print(f"SHHS external masked-MSE={external_loss:.6f}")
    rows = (
        export_embeddings(
            model, train_set, "train", device, settings.batch_size
        )
        + export_embeddings(
            model,
            validation_set,
            "validation",
            device,
            settings.batch_size,
        )
        + export_embeddings(
            model, test_set, "test", device, settings.batch_size
        )
        + export_embeddings(
            model,
            external_set,
            "external_test",
            device,
            settings.batch_size,
        )
    )
    embedding_frame = pd.DataFrame(rows)
    expected_columns = [
        "dataset",
        "night_id",
        "split",
        *[f"feature_{index}" for index in range(1, EMBEDDING_DIM + 1)],
    ]
    embedding_frame = embedding_frame.loc[:, expected_columns]
    embedding_frame.to_csv(
        output_dir / "all_datasets_embeddings_8d.csv", index=False
    )
    output_names = {
        "CFS": "cfs_embeddings_8d.csv",
        "MROS": "mros_embeddings_8d.csv",
        "SHHS": "shhs_external_embeddings_8d.csv",
    }
    for dataset_name, filename in output_names.items():
        embedding_frame.loc[
            embedding_frame["dataset"] == dataset_name
        ].to_csv(output_dir / filename, index=False)

    # Fixed bug in the pasted code: dataset.n_input did not exist.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_input": train_set.n_input,
            "hidden": settings.hidden,
            "embedding_dim": EMBEDDING_DIM,
            "global_median": train_set.global_median,
            "global_iqr": train_set.global_iqr,
            "raw_columns": RAW_COLUMNS,
            "internal_datasets": ["CFS", "MROS"],
            "external_dataset": "SHHS",
        },
        output_dir / "sleep_autoencoder_8d.pt",
    )
    (output_dir / "normalization.json").write_text(
        json.dumps(
            {
                "columns": RAW_COLUMNS,
                "global_median": train_set.global_median.tolist(),
                "global_iqr": train_set.global_iqr.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    split_rows = []
    split_specifications = [
        (train_paths, train_names, "train"),
        (validation_paths, validation_names, "validation"),
        (test_paths, test_names, "test"),
        (external_paths, external_names, "external_test"),
    ]
    for paths, names, split in split_specifications:
        split_rows.extend(
            {
                "dataset": name,
                "file": path.name,
                "source_path": str(path),
                "split": split,
            }
            for path, name in zip(paths, names)
        )
    pd.DataFrame(split_rows).to_csv(
        output_dir / "data_split.csv", index=False
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test_masked_mse": test_loss,
                "shhs_external_masked_mse": external_loss,
                "embedding_dim": EMBEDDING_DIM,
                "seed": settings.seed,
                "n_train": len(train_set),
                "n_validation": len(validation_set),
                "n_test": len(test_set),
                "n_shhs_external": len(external_set),
                "n_cfs_total": sum(len(part) for part in cfs_split),
                "n_mros_total": sum(len(part) for part in mros_split),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Saved CFS, MROS and SHHS eight-dimensional embeddings to: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    train(SETTINGS)
