from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConcatModel(nn.Module):
    """Simple model that processes market features and model prediction separately
    then concatenates them before the policy and value heads."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__()

        market_shape = int(np.prod(obs_space["market_features"].shape))
        pred_shape = int(np.prod(obs_space["model_pred"].shape))

        self.market_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(market_shape, 64),
            nn.ReLU(),
        )
        self.pred_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pred_shape, 16),
            nn.ReLU(),
        )

        hidden = 64 + 16
        self.policy_net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )
        self.value_net = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        # Correctly access the keys directly
        market = self.market_net(input_dict["market_features"])
        pred = self.pred_net(input_dict["model_pred"])
        concat = torch.cat([market, pred], dim=-1)
        # Store the combined features so value_function() can reuse them
        self._features = concat
        logits = self.policy_net(concat)
        value = self.value_net(concat)
        return logits, value

    def value_function(self):
        assert self._features is not None, "must call forward first"
        value = self.value_net(self._features)
        return torch.reshape(value, [-1])


__all__ = ["ConcatModel"]
