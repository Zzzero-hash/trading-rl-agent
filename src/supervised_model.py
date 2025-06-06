"""Supervised learning model and training utilities for trend prediction.

This module implements ``TrendPredictor`` - a CNN + LSTM architecture used to
predict market trends from sequences of engineered features.  A simple training
routine ``train_supervised`` is provided which can train the model on numpy or
pandas inputs and optionally runs on GPU.  Example usage:

>>> from src.supervised_model import ModelConfig, TrainingConfig, train_supervised
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

import logging
from dataclasses import dataclass, field
from typing import Iterable, Tuple, Dict, List, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _to_tensor(data: Any) -> torch.Tensor:
    """Convert numpy array or pandas DataFrame to float tensor."""
    if hasattr(data, "values"):
        data = data.values
    return torch.as_tensor(data, dtype=torch.float32)


@dataclass
class ModelConfig:
    """Configuration for :class:`TrendPredictor`."""

    cnn_filters: Iterable[int] = field(default_factory=lambda: [16, 32])
    cnn_kernel_sizes: Iterable[int] = field(default_factory=lambda: [3, 3])
    lstm_units: int = 32
    dropout: float = 0.1
    output_size: int = 1
    task: str = "classification"  # or 'regression'


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

        layers: List[nn.Module] = []
        in_ch = input_dim
        for out_ch, k in zip(filters, kernels):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=cfg.lstm_units, batch_first=True)
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


def _split_data(x: torch.Tensor, y: torch.Tensor, val_split: float) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
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
) -> Tuple[TrendPredictor, Dict[str, List[float]]]:
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

    gpu_ids = ray.get_gpu_ids()
    if gpu_ids:
        device = torch.device(f"cuda:{int(gpu_ids[0])}")
    else:
        device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    x = _to_tensor(features)
    y = _to_tensor(targets).reshape(len(features), -1)

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
        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
                if m_cfg.task == "classification":
                    predicted = (pred > 0.5).float()
                    correct += (predicted == yb).all(dim=1).sum().item()
                    count += xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))
        history["val_loss"].append(val_loss)
        if m_cfg.task == "classification":
            acc = correct / max(1, count)
            history["val_acc"].append(acc)
            logger.info("Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.3f", epoch + 1, avg_loss, val_loss, acc)
        else:
            logger.info("Epoch %d: train_loss=%.4f val_loss=%.4f", epoch + 1, avg_loss, val_loss)

    return model, history


def tune_example():
    """Illustrative example of wrapping training for Ray Tune."""

    # from ray import tune
    # def train_fn(config):
    #     model_cfg = ModelConfig(**config.get("model", {}))
    #     train_cfg = TrainingConfig(**config.get("train", {}))
    #     _, history = train_supervised(features, targets, model_cfg, train_cfg)
    #     tune.report(loss=history["val_loss"][-1])
    pass


def save_model(model: TrendPredictor, path: str) -> None:
    """Save ``model`` to ``path`` using :func:`torch.save`."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "input_dim": model.input_dim,
        },
        path,
    )


def load_model(path: str, device: str | torch.device | None = None) -> TrendPredictor:
    """Load a :class:`TrendPredictor` from ``path``."""
    device = torch.device(device or "cpu")
    checkpoint = torch.load(path, map_location=device)
    cfg = ModelConfig(**checkpoint["config"])
    model = TrendPredictor(checkpoint["input_dim"], cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def evaluate_model(
    model_or_path: TrendPredictor | str,
    features: Any,
    targets: Any,
) -> Dict[str, float]:
    """Evaluate a trained model on ``features`` and ``targets``."""
    if isinstance(model_or_path, (str, bytes)):
        model = load_model(model_or_path)
    else:
        model = model_or_path

    device = next(model.parameters()).device
    x = _to_tensor(features).to(device)
    y = _to_tensor(targets).reshape(len(features), -1).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x)

    if model.task == "classification":
        predicted = (pred > 0.5).float()
        correct = (predicted == y).all(dim=1).float().mean().item()
        tp = ((predicted == 1) & (y == 1)).sum().item()
        fp = ((predicted == 1) & (y == 0)).sum().item()
        fn = ((predicted == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return {"accuracy": correct, "precision": precision, "recall": recall}
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
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
    return out.squeeze().cpu()


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
__all__ = ["TrendPredictor", "ModelConfig", "TrainingConfig", "train_supervised"]