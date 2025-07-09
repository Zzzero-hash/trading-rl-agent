"""Supervised learning model and training utilities for trend prediction.

This module implements ``TrendPredictor`` - a CNN + LSTM architecture used to
predict market trends from sequences of engineered features.  A simple training
routine ``train_supervised`` is provided which can train the model on numpy or
pandas inputs and optionally runs on GPU.  Example usage:

>>> from trading_rl_agent.supervised_model import ModelConfig, TrainingConfig, train_supervised
>>> features, targets = load_some_data()
>>> model_cfg = ModelConfig(task='classification')
>>> train_cfg = TrainingConfig(epochs=5)
>>> model, history = train_supervised(features, targets, model_cfg, train_cfg)

The training function returns the model and a history dictionary of losses and
metrics.  Hyperparameters are configurable via ``ModelConfig`` and
``TrainingConfig`` and can easily be extended for Ray Tune hyperparameter
search (see the ``tune_example`` function at the bottom of this file).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Tuple

from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import ray
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _to_tensor(data: Any) -> torch.Tensor:
    """Convert numpy array or pandas DataFrame to float tensor."""
    if torch.is_tensor(data):
        # If already a tensor, just ensure it's float32
        return data.to(dtype=torch.float32)
    elif hasattr(data, "values"):
        # Handle pandas DataFrame/Series
        return torch.tensor(data.values, dtype=torch.float32)
    else:
        # Handle numpy arrays and other array-like data
        return torch.tensor(data, dtype=torch.float32)


@dataclass
class ModelConfig:
    """Configuration for :class:`TrendPredictor`."""

    cnn_filters: Iterable[int] = field(default_factory=lambda: [16, 32])
    cnn_kernel_sizes: Iterable[int] = field(default_factory=lambda: [3, 3])
    lstm_units: int = 32
    dropout: float = 0.1
    output_size: int = 1
    task: str = "classification"  # or 'regression'
    classification_threshold: float = 0.5  # Threshold for classification tasks


@dataclass
class TrainingConfig:
    """Configuration for ``train_supervised``."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    val_split: float = 0.2


class TrendPredictor(nn.Module):
    """CNN + LSTM model for trend prediction."""

    def __init__(self, input_dim: int, config: ModelConfig | None = None):
        super().__init__()
        cfg = config or ModelConfig()
        self.task = cfg.task
        self.config = cfg
        self.input_dim = input_dim
        filters = list(cfg.cnn_filters)
        kernels = list(cfg.cnn_kernel_sizes)
        if len(filters) != len(kernels):
            raise ValueError("cnn_filters and cnn_kernel_sizes must have same length")

        layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch, k in zip(filters, kernels):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=in_ch, hidden_size=cfg.lstm_units, batch_first=True
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.lstm_units, cfg.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, features)``.
        """
        x = x.transpose(1, 2)  # -> (batch, channels, seq)
        x = self.conv(x)
        x = x.transpose(1, 2)  # -> (batch, seq, channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        if self.task == "classification":
            out = torch.sigmoid(out)
        return out


def _split_data(
    x: torch.Tensor, y: torch.Tensor, val_split: float
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Split tensors into train and validation sets chronologically."""
    n = x.shape[0]
    split = int(n * (1 - val_split))
    return (x[:split], y[:split]), (x[split:], y[split:])


@ray.remote(num_gpus=1, num_cpus=4)
def train_supervised(
    features: Any,
    targets: Any,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainingConfig | None = None,
) -> tuple[TrendPredictor, dict[str, list[float]]]:
    """Train ``TrendPredictor`` on provided data.

    Parameters
    ----------
    features : array-like
        Input data of shape ``(samples, seq_len, features)``.
    targets : array-like
        Target values of shape ``(samples, output_size)`` or ``(samples,)``.
    model_cfg : ModelConfig, optional
        Configuration for the model.
    train_cfg : TrainingConfig, optional
        Training hyperparameters.

    Returns
    -------
    model : TrendPredictor
        The trained model.
    history : dict
        Dictionary with lists of ``train_loss`` and ``val_loss`` (and ``val_acc`` if classification).
    """
    m_cfg = model_cfg or ModelConfig()
    t_cfg = train_cfg or TrainingConfig()

    # Input validation
    if features is None or targets is None:
        raise ValueError("Input data cannot be empty")
    x = _to_tensor(features)
    y = _to_tensor(targets)
    if x.numel() == 0 or y.numel() == 0:
        raise ValueError("Input data cannot be empty")
    if len(x) != len(y):
        raise ValueError("Input and target dimensions must match")
    if t_cfg.epochs <= 0:
        raise ValueError("Number of epochs must be greater than zero")
    if t_cfg.batch_size > len(x):
        raise ValueError("Batch size cannot be larger than the dataset")

    # Smart device selection for Ray workers
    gpu_ids = ray.get_gpu_ids()
    if gpu_ids and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f"cuda:{int(gpu_ids[0])}")
    else:
        device = torch.device("cpu")
    logger.info("Using device %s", device)

    y = y.reshape(len(features), -1)
    (x_train, y_train), (x_val, y_val) = _split_data(x, y, t_cfg.val_split)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=t_cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=t_cfg.batch_size)

    model = TrendPredictor(input_dim=x.shape[2], config=m_cfg).to(device)

    if m_cfg.task == "classification":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(t_cfg.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(x_train)
        history["train_loss"].append(avg_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
                if m_cfg.task == "classification":
                    predicted = (pred > 0.5).float()
                    correct += (predicted == yb).all(dim=1).sum().item()
                    total_samples += xb.size(0)
        if len(y_val) == 0:
            logger.warning(
                "Validation set is empty. Skipping validation for this epoch."
            )
            continue
        val_loss /= len(y_val)
        history["val_loss"].append(val_loss)
        if m_cfg.task == "classification":
            acc = correct / max(1, total_samples)
            history["val_acc"].append(acc)
            logger.info(
                "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                epoch + 1,
                avg_loss,
                val_loss,
                acc,
            )
        else:
            logger.info(
                "Epoch %d: train_loss=%.4f val_loss=%.4f", epoch + 1, avg_loss, val_loss
            )

    # Move model to CPU before returning to avoid CUDA serialization issues
    model = model.cpu()
    return model, history


def train_supervised_local(
    features: Any,
    targets: Any,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainingConfig | None = None,
) -> tuple[TrendPredictor, dict[str, list[float]]]:
    raise NotImplementedError(
        "train_supervised_local has been removed. Use train_supervised (Ray) instead."
    )


def tune_example():
    """Illustrative example of wrapping training for Ray Tune."""

    # from ray import tune    # from ray import train
    # def train_fn(config):
    #     model_cfg = ModelConfig(**config.get("model", {}))
    #     train_cfg = TrainingConfig(**config.get("train", {}))
    #     _, history = train_supervised(features, targets, model_cfg, train_cfg)
    #     train.report(loss=history["val_loss"][-1])  # Updated for Ray 2.0+
    pass


def save_model(model: TrendPredictor, path: str) -> None:
    """Save ``model`` to ``path`` using :func:`torch.save`."""
    try:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(model.config),
                "input_dim": model.input_dim,
            },
            path,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")


def load_model(path: str, device: str | torch.device | None = None) -> TrendPredictor:
    """Load a :class:`TrendPredictor` from ``path``."""
    device = torch.device(device or "cpu")
    path_str = str(path)
    if not path_str.endswith(".pt"):
        raise ValueError("Unsupported file format")
    try:
        checkpoint = torch.load(path, map_location=device)
        cfg = ModelConfig(**checkpoint["config"])
        model = TrendPredictor(checkpoint["input_dim"], cfg).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def evaluate_model(
    model_or_path: TrendPredictor | str,
    features: Any,
    targets: Any,
) -> dict[str, float]:
    """Evaluate a trained model on ``features`` and ``targets``."""
    if features is None or len(features) == 0:
        raise ValueError("Features cannot be empty")
    if targets is None or len(targets) == 0:
        raise ValueError("Targets cannot be empty")
    if isinstance(model_or_path, (str, bytes)):
        model = load_model(model_or_path)
    else:
        model = model_or_path
    device = next(model.parameters()).device
    x = _to_tensor(features).to(device)
    y = _to_tensor(targets).reshape(len(features), -1).to(device)
    if y.shape[0] != x.shape[0] or y.shape[1] != model.config.output_size:
        raise ValueError("Target dimensions must match model output")
    model.eval()
    with torch.no_grad():
        pred = model(x)
    if model.task == "classification":
        predicted = (pred > 0.5).float()
        y_true = y.cpu().numpy().reshape(-1)
        y_pred = predicted.cpu().numpy().reshape(-1)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        return {"accuracy": acc, "precision": precision, "recall": recall}
    else:
        mse = torch.mean((pred - y) ** 2).item()
        mae = torch.mean(torch.abs(pred - y)).item()
        return {"mse": mse, "mae": mae}


def predict_features(
    model_or_path: TrendPredictor | str,
    recent_data: Any,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return model prediction for ``recent_data``."""
    if callable(recent_data):
        raise ValueError(
            "recent_data must be an array or tensor, not a function or method"
        )
    if isinstance(model_or_path, (str, bytes)):
        model = load_model(model_or_path, device)
    else:
        model = model_or_path
        if device is not None:
            model = model.to(device)
    device = next(model.parameters()).device
    x = _to_tensor(recent_data)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError("Input to predict_features must be 2D or 3D array/tensor")
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
    # Only squeeze the batch dimension if present, preserve output dimension
    if out.dim() > 1 and out.shape[0] == 1:
        out = out.squeeze(0)
    # Ensure output is always at least 1D to prevent scalar outputs
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return out.cpu()


def select_best_model(log_dir: str) -> str:
    """Return path to the best model checkpoint inside ``log_dir``."""
    import json
    import os

    best_path = ""
    best_loss = float("inf")
    for root, _, files in os.walk(log_dir):
        if "metrics.json" in files:
            mpath = os.path.join(root, "metrics.json")
            with open(mpath) as f:
                metrics = json.load(f)
            loss = metrics.get("val_loss")
            if loss is not None and loss < best_loss:
                best_loss = loss
                ck_rel = metrics.get("checkpoint", "model.pt")
                best_path = os.path.join(root, ck_rel)
    if not best_path:
        raise FileNotFoundError("No metrics.json found in log_dir")
    return best_path


__all__ = [
    "TrendPredictor",
    "ModelConfig",
    "TrainingConfig",
    "train_supervised",
    "save_model",
    "load_model",
    "evaluate_model",
    "predict_features",
    "select_best_model",
]
