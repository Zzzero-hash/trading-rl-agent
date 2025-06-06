import asyncio

from src.serve_deployment import PredictorDeployment, PolicyDeployment


def test_predictor_returns_value():
    predictor = PredictorDeployment()
    result = asyncio.get_event_loop().run_until_complete(
        predictor({"features": [1.0, 2.0, 3.0]})
    )
    assert "prediction" in result


def test_policy_returns_action():
    predictor = PredictorDeployment()
    policy = PolicyDeployment(predictor)
    result = asyncio.get_event_loop().run_until_complete(
        policy({"observation": [0.1, -0.2]})
    )
    assert "action" in result
    assert result["action"] in (0, 1)
