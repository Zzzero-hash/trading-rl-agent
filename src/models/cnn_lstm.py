"""
Hybrid CNN+LSTM Model for Trading RL Agent

This module implements a hybrid neural network architecture that combines
Convolutional Neural Networks (CNNs) for cross-sectional feature extraction
with Long Short-Term Memory (LSTM) networks for temporal sequence modeling.
The model is designed to process multi-asset financial time series data for
reinforcement learning-based trading agents.
"""


import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    A hybrid CNN+LSTM neural network for processing financial time series data.

    The architecture combines:
    1. CNN layers for extracting spatial features from cross-sectional market data
    2. LSTM layers for capturing temporal dependencies across time steps
    3. Dense layers for producing final output features compatible with RL agents

    Attributes:
        cnn_layers: Sequential CNN layers for feature extraction
        lstm_layers: LSTM layers for temporal processing
        dense_layers: Dense layers for final output
    """

    def __init__(self,
                 time_steps: int = 30,
                 assets: int = 100,
                 features: int = 15,
                 cnn_filters: list[int] = [64, 32, 16],
                 lstm_units: list[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 output_dim: int = 20):
        """
        Initialize the CNNLSTM model.

        Args:
            time_steps (int): Number of time steps in the input sequence (default: 30)
            assets (int): Number of assets in the input (default: 100)
            features (int): Number of features per asset (default: 15)
            cnn_filters (list): Number of filters for each CNN layer (default: [64, 32, 16])
            lstm_units (list): Number of units for each LSTM layer (default: [128, 64])
            dropout_rate (float): Dropout rate for regularization (default: 0.2)
            output_dim (int): Dimension of the output layer (default: 20)
        """
        super().__init__()

        # Set default values for cnn_filters and lstm_units if not provided
        if cnn_filters is None:
            cnn_filters = [64, 32, 16]
        if lstm_units is None:
            lstm_units = [128, 64]

        self.time_steps = time_steps
        self.assets = assets
        self.features = features
        self.output_dim = output_dim

        # CNN layers for spatial feature extraction
        # Reshape input from (batch, time_steps, assets, features) to (batch*time_steps, assets, features)
        # Then apply 1D convolutions along the asset dimension
        self.cnn_layers = nn.Sequential(
            # First CNN layer
            nn.Conv1d(in_channels=features, out_channels=cnn_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),

            # Second CNN layer
            nn.Conv1d(in_channels=cnn_filters[0], out_channels=cnn_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),

            # Third CNN layer
            nn.Conv1d(in_channels=cnn_filters[1], out_channels=cnn_filters[2], kernel_size=3, padding=1),
            nn.ReLU(),

            # Global average pooling to reduce dimensionality
            nn.AdaptiveAvgPool1d(1)
        )

        # LSTM layers for temporal processing
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(cnn_filters[2], lstm_units[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            self.lstm_layers.append(nn.LSTM(lstm_units[i-1], lstm_units[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Dense layers for final output
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, assets, features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # Reshape input for CNN processing
        # From (batch, time_steps, assets, features) to (batch*time_steps, features, assets)
        x_reshaped = x.view(-1, self.assets, self.features).transpose(1, 2)

        # Apply CNN layers
        cnn_output = self.cnn_layers(x_reshaped)

        # Remove the last dimension (since we used AdaptiveAvgPool1d(1))
        cnn_output = cnn_output.squeeze(-1)

        # Reshape for LSTM processing
        # From (batch*time_steps, cnn_filters[2]) to (batch, time_steps, cnn_filters[2])
        lstm_input = cnn_output.view(batch_size, self.time_steps, -1)

        # Apply LSTM layers
        lstm_output = lstm_input
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            lstm_output, _ = lstm_layer(lstm_output)
            lstm_output = dropout_layer(lstm_output)

        # Take the last output from the LSTM sequence
        # lstm_output shape: (batch, time_steps, lstm_units[-1])
        final_lstm_output = lstm_output[:, -1, :]

        # Apply dense layers
        output = self.dense_layers(final_lstm_output)

        return output


# Example usage
if __name__ == "__main__":
    # Create a model instance
    model = CNNLSTM(time_steps=30, assets=100, features=15, output_dim=20)

    # Create a sample input tensor
    batch_size = 32
    sample_input = torch.randn(batch_size, 30, 100, 15)

    # Forward pass
    output = model(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")

    # Print model summary
    print("\nModel architecture:")
    print(model)
