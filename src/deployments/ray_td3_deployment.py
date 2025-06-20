"""
Ray Serve Deployment for TD3 Agent

This module provides Ray Serve deployment capabilities for the TD3 agent,
making it a fully-fledged Ray deployment like SAC.
"""

import asyncio
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Ray imports with fallback
try:
    import ray
    from ray import serve

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    serve = None

from agents.td3_agent import TD3Agent
from configs.hyperparameters import get_agent_config

if RAY_AVAILABLE and serve:

    @serve.deployment
    class TD3ServeDeployment:
        """Ray Serve deployment for TD3 agent."""

        def __init__(
            self, model_path: Optional[str] = None, config: Optional[dict] = None
        ):
            """Initialize TD3 deployment.

            Args:
                model_path: Path to saved TD3 model
                config: TD3 configuration dictionary
            """
            self.config = config or get_agent_config("enhanced_td3")

            # Extract dimensions from config or use defaults
            self.state_dim = self.config.get("state_dim", 10)
            self.action_dim = self.config.get("action_dim", 3)

            # Initialize agent
            self.agent = TD3Agent(
                config=self.config,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device="cpu",  # Use CPU for serving
            )

            # Load model if path provided
            if model_path:
                try:
                    self.agent.load(model_path)
                    print(f"✅ Loaded TD3 model from {model_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load model: {e}")

            print(
                f"✅ TD3 Serve deployment initialized with state_dim={self.state_dim}, action_dim={self.action_dim}"
            )

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            """Handle prediction requests.

            Args:
                request: Request with 'observation' key containing state

            Returns:
                Dictionary with 'action' key containing predicted action
            """
            try:
                # Extract observation
                observation = request.get("observation", [])
                if isinstance(observation, list):
                    observation = np.array(observation, dtype=np.float32)
                elif not isinstance(observation, np.ndarray):
                    observation = np.array([observation], dtype=np.float32)

                # Ensure correct shape
                if observation.shape[0] != self.state_dim:
                    observation = observation.flatten()[: self.state_dim]
                    if len(observation) < self.state_dim:
                        # Pad with zeros if needed
                        padded = np.zeros(self.state_dim)
                        padded[: len(observation)] = observation
                        observation = padded

                # Get action from agent (deterministic for serving)
                action = self.agent.select_action(observation, add_noise=False)

                return {"action": action.tolist(), "status": "success", "agent": "td3"}

            except Exception as e:
                return {
                    "action": np.zeros(self.action_dim).tolist(),
                    "status": "error",
                    "error": str(e),
                }

    def create_td3_deployment_graph(
        model_path: Optional[str] = None, config: Optional[dict] = None
    ):
        """Create TD3 deployment graph for Ray Serve.

        Args:
            model_path: Path to saved TD3 model
            config: TD3 configuration

        Returns:
            TD3 deployment instance bound for Ray Serve
        """
        return TD3ServeDeployment.bind(model_path, config)

    def deploy_td3_model(
        model_path: Optional[str] = None,
        config: Optional[dict] = None,
        deployment_name: str = "td3-model",
    ):
        """Deploy TD3 model to Ray Serve.

        Args:
            model_path: Path to saved TD3 model
            config: TD3 configuration
            deployment_name: Name for the deployment

        Returns:
            Deployment handle
        """
        # Create deployment
        deployment = TD3ServeDeployment.options(
            name=deployment_name, num_replicas=1, ray_actor_options={"num_cpus": 1}
        ).bind(model_path, config)

        # Deploy
        serve.run(deployment, name=deployment_name)

        print(f"✅ TD3 model deployed as '{deployment_name}'")
        return deployment

else:
    # Fallback implementations when Ray is not available

    class TD3ServeDeployment:
        """Fallback TD3 deployment."""

        def __init__(
            self, model_path: Optional[str] = None, config: Optional[dict] = None
        ):
            print("⚠️ Ray not available - using fallback TD3 deployment")
            self.agent = TD3Agent(config=None, state_dim=10, action_dim=3)

        async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
            observation = np.array(request.get("observation", [0] * 10))
            action = self.agent.select_action(observation, add_noise=False)
            return {"action": action.tolist(), "status": "fallback"}

    def create_td3_deployment_graph(
        model_path: Optional[str] = None, config: Optional[dict] = None
    ):
        """Fallback deployment graph."""
        return TD3ServeDeployment(model_path, config)

    def deploy_td3_model(
        model_path: Optional[str] = None,
        config: Optional[dict] = None,
        deployment_name: str = "td3-model",
    ):
        """Fallback deployment."""
        print("⚠️ Ray not available - skipping TD3 model deployment")
        return TD3ServeDeployment(model_path, config)


# Integration functions for easy usage
def start_td3_serve_cluster(
    model_path: Optional[str] = None,
    config: Optional[dict] = None,
    cluster_config: Optional[dict] = None,
):
    """Start Ray cluster and deploy TD3 model.

    Args:
        model_path: Path to saved TD3 model
        config: TD3 configuration
        cluster_config: Ray cluster configuration

    Returns:
        Deployment handle or None if Ray unavailable
    """
    if not RAY_AVAILABLE:
        print("⚠️ Ray not available - cannot start serve cluster")
        return None

    # Initialize Ray
    if not ray.is_initialized():
        if cluster_config:
            ray.init(**cluster_config)
        else:
            ray.init()

    # Initialize Serve
    if not serve.status().applications:
        serve.start()

    # Deploy TD3 model
    deployment = deploy_td3_model(model_path, config)

    print("✅ TD3 Ray Serve cluster started successfully")
    return deployment


def get_td3_serve_handle(deployment_name: str = "td3-model"):
    """Get handle to deployed TD3 model.

    Args:
        deployment_name: Name of the deployment

    Returns:
        Deployment handle or None if not available
    """
    if not RAY_AVAILABLE:
        return None

    try:
        handle = serve.get_deployment(deployment_name).get_handle()
        return handle
    except Exception as e:
        print(f"⚠️ Failed to get TD3 deployment handle: {e}")
        return None


async def predict_with_td3_serve(
    observation: np.ndarray, deployment_name: str = "td3-model"
) -> Optional[np.ndarray]:
    """Make prediction using deployed TD3 model.

    Args:
        observation: State observation
        deployment_name: Name of the deployment

    Returns:
        Predicted action or None if failed
    """
    handle = get_td3_serve_handle(deployment_name)
    if handle is None:
        return None

    try:
        request = {"observation": observation.tolist()}
        response = await handle.remote(request)

        if response.get("status") == "success":
            return np.array(response["action"])
        else:
            print(f"⚠️ TD3 prediction failed: {response.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"⚠️ TD3 serve prediction error: {e}")
        return None


if __name__ == "__main__":
    # Test TD3 deployment
    print("=== TD3 Ray Deployment Test ===")

    if RAY_AVAILABLE:
        # Test local deployment
        print("✅ Ray available - testing TD3 deployment")

        # Create deployment (without actually deploying)
        deployment = create_td3_deployment_graph()
        print("✅ Created TD3 deployment graph")

        # Test the deployment class directly
        td3_deploy = TD3ServeDeployment()

        # Test prediction
        test_obs = np.random.randn(10)
        test_request = {"observation": test_obs.tolist()}

        # Test async call
        async def test_prediction():
            result = await td3_deploy(test_request)
            print(f"✅ Test prediction: {result}")
            return result

        # Run async test
        result = asyncio.run(test_prediction())

    else:
        print("⚠️ Ray not available - testing fallback implementations")

        # Test fallback deployment
        deployment = create_td3_deployment_graph()
        print("✅ Created fallback TD3 deployment")

        # Test fallback prediction
        test_obs = np.random.randn(10)
        test_request = {"observation": test_obs.tolist()}

        async def test_fallback():
            result = await deployment(test_request)
            print(f"✅ Fallback prediction: {result}")
            return result

        result = asyncio.run(test_fallback())

    print("🎯 TD3 Ray deployment test complete!")
