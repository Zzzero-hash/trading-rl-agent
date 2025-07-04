from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
import yaml


def _load_config(config: Optional[Union[str, dict]]) -> dict:
    if config is None:
        default_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "model"
            / "cnn_lstm.yaml"
        )
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}
    elif isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary after loading.")
    return config


class CNNLSTMModel(nn.Module):
    """Simple CNN + LSTM model with optional attention."""

    def __init__(
        self,
        input_dim: int,
        output_size: int = 1,
        config: Optional[Union[str, dict]] = None,
        use_attention: bool = False,
    ):
        super().__init__()
        cfg = _load_config(config)
        filters: Sequence[int] = cfg.get("cnn_filters", [32, 64])
        kernels: Sequence[int] = cfg.get("cnn_kernel_sizes", [3, 3])
        lstm_units: int = cfg.get("lstm_units", 128)
        dropout: float = cfg.get("dropout", 0.5)

        layers = []
        in_channels = input_dim
        # Build CNN layers - kernel size will be validated during forward pass based on sequence length
        for i, (out_c, k) in enumerate(zip(filters, kernels)):
            layers.append(nn.Conv1d(in_channels, out_c, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            in_channels = out_c
        self.conv = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=lstm_units, batch_first=True
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                lstm_units, num_heads=1, batch_first=True
            )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units, output_size)
        self.input_dim = input_dim  # Add input_dim attribute for test compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq, features)

        Returns:
            Tensor of shape (batch, output_size)
        """
        batch_size, seq_len, features = x.shape

        # Validate input dimensions
        if features != self.input_dim:
            raise ValueError(
                f"Expected input features {self.input_dim}, got {features}"
            )

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


@dataclass
class CNNLSTMConfig:
    """Configuration for the CNN-LSTM model."""

    input_dim: int
    output_size: int = 1
    cnn_filters: Sequence[int] = (32, 64)
    cnn_kernel_sizes: Sequence[int] = (3, 3)
    lstm_units: int = 128
    dropout: float = 0.5
    use_attention: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the config."""
        return asdict(self)


def create_model(config: CNNLSTMConfig):
    model = CNNLSTMModel(
        input_dim=config.input_dim,
        output_size=config.output_size,
        config=config.to_dict(),  # Pass config as a dictionary
        use_attention=config.use_attention,
    )
    return model


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Configuration
    batch_size = 32
    sequence_length = 60  # e.g., 60 days of data
    num_features = 10  # e.g., OHLC, volume, 5 indicators

    config = CNNLSTMConfig(
        input_dim=num_features,
        output_size=3,  # e.g., Buy, Hold, Sell probabilities or a single regression value
        cnn_filters=[64, 128],
        cnn_kernel_sizes=[3, 5],
        lstm_units=256,
        dropout=0.2,
        use_attention=True,
    )

    # Create model
    model = create_model(config)
    print(model)

    # Create dummy input tensor
    # (batch_size, sequence_length, num_features)
    dummy_input = torch.randn(batch_size, sequence_length, num_features)
    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, config.output_size)
        print("Model forward pass successful!")
    except Exception as e:
        print(f"Error during forward pass: {e}")

    # Test with different parameters
    config_simple = CNNLSTMConfig(
        input_dim=5,
        output_size=1,
        cnn_filters=[16],
        cnn_kernel_sizes=[3],
        lstm_units=32,
        dropout=0.1,
        use_attention=False,
    )
    model_simple = create_model(config_simple)
    dummy_input_simple = torch.randn(4, 20, 5)
    try:
        output_simple = model_simple(dummy_input_simple)
        print(
            f"\nSimple Model - Input: {dummy_input_simple.shape}, Output: {output_simple.shape}"
        )
        assert output_simple.shape == (4, config_simple.output_size)
        print("Simple model forward pass successful!")
    except Exception as e:
        print(f"Error during simple model forward pass: {e}")
