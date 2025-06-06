from __future__ import annotations

import numpy as np
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_ops import FLOAT_MIN
from ray.rllib.models import ModelCatalog


class ConcatModel(TorchModelV2, nn.Module):
    """Simple model that processes market features and model prediction separately
    then concatenates them before the policy and value heads."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

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
        self.policy = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )
        self.value_branch = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        market = self.market_net(input_dict["obs"]["market_features"])
        pred = self.pred_net(input_dict["obs"]["model_pred"])
        features = torch.cat([market, pred], dim=1)
        self._features = features
        logits = self.policy(features)
        return logits, state

    def value_function(self):
        assert self._features is not None, "must call forward first"
        value = self.value_branch(self._features)
        return torch.reshape(value, [-1])


__all__ = ["ConcatModel", "ModelCatalog"]
