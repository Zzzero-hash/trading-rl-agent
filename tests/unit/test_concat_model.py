import torch
from gymnasium import spaces

from trading_rl_agent.models.concat_model import ConcatModel


def test_concat_model_output_shape():
    obs_space = spaces.Dict(
        {
            "market_features": spaces.Box(-1.0, 1.0, shape=(5, 3)),
            "model_pred": spaces.Box(-1.0, 1.0, shape=(1,)),
        },
    )
    action_space = spaces.Discrete(2)
    model = ConcatModel(
        obs_space,
        action_space,
        num_outputs=action_space.n,
        model_config={},
        name="concat",
    )
    sample = {
        "market_features": torch.zeros(1, 5, 3),
        "model_pred": torch.zeros(1, 1),
    }
    logits, _ = model(sample, [], None)
    assert logits.shape == (1, 2)
    value = model.value_function()
    assert value.shape == (1,)
