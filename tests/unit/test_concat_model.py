import torch

from trade_agent.models.concat_model import ConcatModel


def test_concat_model_output_shape():
    """Test the output shape of the concatenation model."""
    model = ConcatModel(
        dim1=10,
        dim2=5,
        hidden_dim=32,
        output_dim=1,
    )
    input1 = torch.randn(32, 10)
    input2 = torch.randn(32, 5)
    output = model(input1, input2)
    assert output.shape == (32, 1)
