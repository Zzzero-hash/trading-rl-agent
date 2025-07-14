"""
Ray Serve Deployment for SAC Agent

This module provides Ray Serve deployment capabilities for the custom
SAC agent used in this repository.
"""

import asyncio
from typing import Any, cast

import numpy as np

# Ray imports with fallback
try:
    import ray
    from ray import serve

    RAY_AVAILABLE = True
except ImportError:  # pragma: no cover - Ray optional
    RAY_AVAILABLE = False
    serve = None

from agents.sac_agent import SACAgent
from configs.hyperparameters import get_agent_config

if RAY_AVAILABLE and serve:

    @serve.deployment
    class SACServeDeployment:
        """Ray Serve deployment for SAC agent."""

        def __init__(
            self,
            model_path: str | None = None,
            config: dict[str, Any] | None = None,
        ):
            self.config = config or get_agent_config("enhanced_sac")
            self.state_dim = self.config.get("state_dim", 10)
            self.action_dim = self.config.get("action_dim", 3)

            self.agent = SACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=self.config,
                device="cpu",
            )

            if model_path:
                try:
                    self.agent.load(model_path)
                    print(f"✅ Loaded SAC model from {model_path}")
                except Exception as e:  # pragma: no cover - serve runtime
                    print(f"⚠️ Failed to load model: {e}")

            print(
                f"✅ SAC Serve deployment initialized with state_dim={self.state_dim}, action_dim={self.action_dim}",
            )

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            try:
                observation = request.get("observation", [])
                if isinstance(observation, list):
                    observation = np.array(observation, dtype=np.float32)
                elif not isinstance(observation, np.ndarray):
                    observation = np.array([observation], dtype=np.float32)

                if observation.shape[0] != self.state_dim:
                    observation = observation.flatten()[: self.state_dim]
                    if len(observation) < self.state_dim:
                        padded = np.zeros(self.state_dim)
                        padded[: len(observation)] = observation
                        observation = padded

                action = self.agent.select_action(observation, evaluate=True)
                return {"action": action.tolist(), "status": "success", "agent": "sac"}
            except Exception as e:  # pragma: no cover - serve runtime
                return {
                    "action": np.zeros(self.action_dim).tolist(),
                    "status": "error",
                    "error": str(e),
                }

    def create_sac_deployment_graph(
        model_path: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        """Create SAC deployment graph for Ray Serve."""
        return cast(Any, SACServeDeployment).bind(model_path=model_path, config=config)

    def deploy_sac_model(
        model_path: str | None = None,
        config: dict[str, Any] | None = None,
        deployment_name: str = "sac-model",
    ) -> Any:
        deployment = (
            cast(Any, SACServeDeployment)
            .options(
                name=deployment_name,
                num_replicas=2,
            )
            .bind(model_path=model_path, config=config)
        )
        serve.run(deployment)
        print(f"✅ SAC model deployed as '{deployment_name}'")
        return deployment

else:

    class SACServeDeployment:  # type: ignore[no-redef]
        """Fallback SAC deployment when Ray is unavailable."""

        def __init__(
            self,
            model_path: str | None = None,
            config: dict[str, Any] | None = None,
        ):
            print("⚠️ Ray not available - using fallback SAC deployment")
            self.agent = SACAgent(state_dim=10, action_dim=3, config=None)

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            observation = np.array(request.get("observation", [0] * 10))
            action = self.agent.select_action(observation, evaluate=True)
            return {"action": action.tolist(), "status": "fallback"}

    def create_sac_deployment_graph(
        model_path: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        return SACServeDeployment(model_path, config)

    def deploy_sac_model(
        model_path: str | None = None,
        config: dict[str, Any] | None = None,
        deployment_name: str = "sac-model",
    ) -> Any:
        print("⚠️ Ray not available - skipping SAC model deployment")
        return SACServeDeployment(model_path, config)


def start_sac_serve_cluster(
    model_path: str,
    config: dict[str, Any] | None = None,
    cluster_config: dict[str, Any] | None = None,
) -> serve.Deployment | None:
    """Start Ray cluster and deploy SAC model."""
    if not RAY_AVAILABLE:
        print("⚠️ Ray not available - cannot start serve cluster")
        return None

    if not ray.is_initialized():
        if cluster_config:
            ray.init(**cluster_config)
        else:
            ray.init()

    if not serve.status().applications:
        serve.start()

    deployment = deploy_sac_model(model_path, config)
    print("✅ SAC Ray Serve cluster started successfully")
    return deployment


def get_sac_serve_handle(deployment_name: str = "sac-model") -> serve.Deployment | None:
    """Get handle to deployed SAC model."""
    if not RAY_AVAILABLE:
        return None
    try:
        return serve.get_deployment(deployment_name).get_handle()
    except Exception as e:  # pragma: no cover - serve runtime
        print(f"⚠️ Failed to get SAC deployment handle: {e}")
        return None


async def predict_with_sac_serve(
    observation: np.ndarray,
    deployment_name: str = "sac-model",
) -> np.ndarray | None:
    """Make prediction using deployed SAC model."""
    handle = get_sac_serve_handle(deployment_name)
    if handle is None:
        return None
    try:
        request = {"observation": observation.tolist()}
        response = await handle.remote(request)
        if response.get("status") == "success":
            return np.array(response["action"])
        print(f"⚠️ SAC prediction failed: {response.get('error', 'Unknown error')}")
        return None
    except Exception as e:  # pragma: no cover - serve runtime
        print(f"⚠️ SAC serve prediction error: {e}")
        return None


if __name__ == "__main__":
    print("=== SAC Ray Deployment Test ===")
    if RAY_AVAILABLE:
        print("✅ Ray available - testing SAC deployment")
        deployment = create_sac_deployment_graph(model_path="test_model.pth")
        print("✅ Created SAC deployment graph")
        sac_deploy = SACServeDeployment()
        test_obs = np.random.randn(10)
        test_request = {"observation": test_obs.tolist()}

        async def test_prediction() -> dict[str, Any]:
            result = await sac_deploy(test_request)
            print(f"✅ Test prediction: {result}")
            return result

        result = asyncio.run(test_prediction())
    else:
        print("⚠️ Ray not available - testing fallback implementations")
        deployment = create_sac_deployment_graph(model_path="test_model.pth")
        print("✅ Created fallback SAC deployment")
        test_obs = np.random.randn(10)
        test_request = {"observation": test_obs.tolist()}

        async def test_fallback() -> dict[str, Any]:
            result: dict[str, Any] = await deployment(test_request)
            print(f"✅ Fallback prediction: {result}")
            return result

        result = asyncio.run(test_fallback())

    print("🎯 SAC Ray deployment test complete!")
