"""Ray Serve deployment stubs for predictor and policy models.

This module defines basic `PredictorDeployment` and `PolicyDeployment` classes
using Ray Serve. They illustrate how the trained supervised model and RL policy
could be exposed as independent services and composed together. The logic is
simplified so the code can run without heavy dependencies, but the structure is
ready for future production use.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ray import serve
except Exception:  # pragma: no cover - ray might not be installed
    serve = None

from trading_rl_agent.supervised_model import load_model, predict_features

if serve:

    @serve.deployment
    class PredictorDeployment:
        """Load a trained ``TrendPredictor`` and return predictions."""

        def __init__(self, model_path: str | None = None):
            self.model_path = model_path
            self.model = None
            if model_path:
                try:
                    self.model = load_model(model_path)
                except Exception:
                    self.model = None

        async def __call__(self, request: Any) -> dict:
            if isinstance(request, dict):
                payload = request
            else:  # assume HTTP request
                payload = await request.json()
            features = np.asarray(payload.get("features", []), dtype=np.float32)
            if self.model is not None:
                pred = predict_features(self.model, features).cpu().numpy()
                pred = pred.tolist()
            else:
                pred = float(features.mean()) if features.size > 0 else 0.0
            return {"prediction": pred}

    @serve.deployment
    class PolicyDeployment:
        """RL policy that optionally queries ``PredictorDeployment``."""

        def __init__(self, predictor_handle: Any | None = None):
            self.predictor = predictor_handle
            self.action_space = 3  # hold/buy/sell

        async def __call__(self, request: Any) -> dict:
            if isinstance(request, dict):
                payload = request
            else:
                payload = await request.json()
            obs = np.asarray(payload.get("observation", []), dtype=np.float32)
            pred = 0.0
            if self.predictor is not None:
                res = await self.predictor.__call__({"features": obs.tolist()})
                pred = float(res.get("prediction", 0.0))
            # Dummy policy: buy if mean(obs)+pred > 0
            action = 1 if (obs.mean() + pred) > 0 else 0
            return {"action": int(action)}

    def deployment_graph(model_path: str | None = None) -> dict[str, Any]:
        """Return a Serve deployment graph for the predictor and policy."""
        predictor = PredictorDeployment.bind(model_path)  # type: ignore[attr-defined]
        policy = PolicyDeployment.bind(predictor)  # type: ignore[attr-defined]
        return {"predictor": predictor, "policy": policy}

else:  # pragma: no cover - serve not available

    class PredictorDeployment:  # type: ignore
        """Fallback predictor when Ray Serve is unavailable."""

        def __init__(self, model_path: str | None = None):
            self.model_path = model_path
            self.model = load_model(model_path) if model_path else None

        async def __call__(self, request: Any) -> dict:
            features = np.asarray(request.get("features", []), dtype=np.float32)
            pred = (
                predict_features(self.model, features).cpu().numpy().tolist()
                if self.model is not None
                else float(features.mean())
                if features.size > 0
                else 0.0
            )
            return {"prediction": pred}

    class PolicyDeployment:  # type: ignore
        """Fallback policy implementation."""

        def __init__(self, predictor_handle: Any | None = None):
            self.predictor = predictor_handle

        async def __call__(self, request: Any) -> dict:
            obs = np.asarray(request.get("observation", []), dtype=np.float32)
            pred = 0.0
            if self.predictor is not None:
                res = await self.predictor.__call__({"features": obs.tolist()})
                pred = float(res.get("prediction", 0.0))
            action = 1 if (obs.mean() + pred) > 0 else 0
            return {"action": int(action)}

    def deployment_graph(model_path: str | None = None) -> dict[str, Any]:
        predictor = PredictorDeployment(model_path)
        policy = PolicyDeployment(predictor)
        return {"predictor": predictor, "policy": policy}


__all__ = ["PolicyDeployment", "PredictorDeployment", "deployment_graph"]
