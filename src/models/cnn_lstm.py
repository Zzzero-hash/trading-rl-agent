import yaml
import torch
from torch import nn
from pathlib import Path
from typing import Sequence, Optional, Union, Dict


def _load_config(config: Optional[Union[str, Dict]]) -> Dict:
    if config is None:
        default_path = Path(__file__).resolve().parent.parent / "configs" / "model" / "cnn_lstm.yaml"
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}
    elif isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f) or {}
    return config


class CNNLSTMModel(nn.Module):
    """Simple CNN + LSTM model with optional attention."""

    def __init__(self, input_dim: int, output_size: int = 1,
                 config: Optional[Union[str, Dict]] = None,
                 use_attention: bool = False):
        super().__init__()
        cfg = _load_config(config)
        filters: Sequence[int] = cfg.get("cnn_filters", [32, 64])
        kernels: Sequence[int] = cfg.get("cnn_kernel_sizes", [3, 3])
        lstm_units: int = cfg.get("lstm_units", 128)
        dropout: float = cfg.get("dropout", 0.5)

        layers = []
        in_channels = input_dim
        for out_c, k in zip(filters, kernels):
            layers.append(nn.Conv1d(in_channels, out_c, kernel_size=k))
            layers.append(nn.ReLU())
            in_channels = out_c
        self.conv = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_units,
                            batch_first=True)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(lstm_units, num_heads=1,
                                                   batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq, features)
        Returns:
            Tensor of shape (batch, output_size)
        """
        # Convolution expects (batch, channels, seq)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Back to (batch, seq, channels)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        if self.use_attention:
            out, _ = self.attention(out, out, out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

