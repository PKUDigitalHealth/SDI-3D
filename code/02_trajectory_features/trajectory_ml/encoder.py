from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import patchify


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], patch_size: int):
        self.tokens = [patchify(sequence, patch_size) for sequence in sequences]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.tokens[index]


@dataclass
class MaskedBatchCollator:
    mask_ratio: float
    training: bool = True

    def __call__(
        self, items: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        maximum_length = max(len(item) for item in items)
        batch_size = len(items)
        values = np.zeros((batch_size, maximum_length, 5), dtype=np.float32)
        padding = np.ones((batch_size, maximum_length), dtype=bool)
        masked = np.zeros((batch_size, maximum_length), dtype=bool)
        for row, item in enumerate(items):
            length = len(item)
            values[row, :length] = item
            padding[row, :length] = False
            if self.training:
                candidate = np.random.random(length) < self.mask_ratio
                if not np.any(candidate):
                    candidate[np.random.randint(0, length)] = True
                masked[row, :length] = candidate

        model_input = values.copy()
        model_input[masked, 1:] = 0.0
        mask_indicator = masked.astype(np.float32)[..., None]
        model_input = np.concatenate([model_input, mask_indicator], axis=2)
        targets = values[..., 1:]
        return (
            torch.from_numpy(model_input),
            torch.from_numpy(targets),
            torch.from_numpy(padding),
            torch.from_numpy(masked),
        )


class SinusoidalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, maximum_length: int = 2048):
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(maximum_length, hidden_dim)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]]


class MaskedTrajectoryEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        embedding_dim = int(config["embedding_dim"])
        self.architecture = config.get("architecture", "transformer")
        self.input_projection = nn.Linear(6, hidden_dim)
        self.position = SinusoidalEncoding(hidden_dim)
        if self.architecture == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(config["num_heads"]),
                dim_feedforward=hidden_dim * 2,
                dropout=float(config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=int(config["num_layers"]))
        elif self.architecture == "lstm":
            if hidden_dim % 2:
                raise ValueError("LSTM hidden_dim must be even for a bidirectional model.")
            self.backbone = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=int(config["num_layers"]),
                dropout=float(config["dropout"]) if int(config["num_layers"]) > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown encoder architecture: {self.architecture}")
        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )
        self.reconstruction_head = nn.Linear(hidden_dim, 4)

    def encode_tokens(
        self, model_input: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.position(self.input_projection(model_input))
        if self.architecture == "transformer":
            hidden = self.backbone(hidden, src_key_padding_mask=padding_mask)
        else:
            hidden, _ = self.backbone(hidden)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        embedding = self.embedding_projection(pooled)
        return hidden, embedding

    def forward(
        self, model_input: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, embedding = self.encode_tokens(model_input, padding_mask)
        return self.reconstruction_head(hidden), embedding


def _masked_loss(
    prediction: torch.Tensor, target: torch.Tensor, masked: torch.Tensor
) -> torch.Tensor:
    squared = (prediction - target) ** 2
    weights = masked.unsqueeze(-1).expand_as(squared)
    return squared[weights].mean()


def _run_epoch(
    model: MaskedTrajectoryEncoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    for model_input, target, padding, masked in loader:
        model_input = model_input.to(device)
        target = target.to(device)
        padding = padding.to(device)
        masked = masked.to(device)
        with torch.set_grad_enabled(training):
            prediction, _ = model(model_input, padding)
            loss = _masked_loss(prediction, target, masked)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_masked_encoder(
    sequences: list[np.ndarray],
    groups: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[MaskedTrajectoryEncoder, list[dict[str, float]], str]:
    set_random_seed(seed)
    if len(sequences) < 4:
        raise ValueError("At least four nights are required to train the encoder.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        raise ValueError("At least four independent subjects are required.")

    validation_fraction = float(config["validation_fraction"])
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    train_index, validation_index = next(
        splitter.split(np.arange(len(sequences)), groups=groups)
    )
    patch_size = int(config["patch_size_epochs"])
    train_dataset = TokenDataset([sequences[index] for index in train_index], patch_size)
    validation_dataset = TokenDataset(
        [sequences[index] for index in validation_index], patch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=MaskedBatchCollator(float(config["mask_ratio"]), training=True),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=MaskedBatchCollator(float(config["mask_ratio"]), training=True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedTrajectoryEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = _run_epoch(model, train_loader, device, optimizer)
        validation_loss = _run_epoch(model, validation_loader, device, None)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, str(device)


@torch.no_grad()
def extract_embeddings(
    model: MaskedTrajectoryEncoder,
    sequences: list[np.ndarray],
    config: dict[str, Any],
    device_name: str | None = None,
) -> np.ndarray:
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)
    dataset = TokenDataset(sequences, int(config["patch_size_epochs"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=MaskedBatchCollator(0.0, training=False),
    )
    rows = []
    for model_input, _, padding, _ in loader:
        _, embedding = model.encode_tokens(model_input.to(device), padding.to(device))
        rows.append(embedding.cpu().numpy())
    return np.vstack(rows)
