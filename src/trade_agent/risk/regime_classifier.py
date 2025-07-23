from typing import Any


class RegimeClassifier:
    def __init__(self, model_type: str = "HMM", **kwargs: Any) -> None:
        self.model_type = model_type
        self.model: Any = None
        self.initialize_model(**kwargs)

    def initialize_model(self, **kwargs: Any) -> Any:
        if self.model_type == "HMM":
            try:
                from hmmlearn import hmm
                self.model = hmm.GaussianHMM(**kwargs)
            except ImportError:
                # Fallback if hmmlearn is not available
                self.model = {"type": "HMM", "params": kwargs}
        elif self.model_type == "GMM":
            try:
                from sklearn.mixture import GaussianMixture
                self.model = GaussianMixture(**kwargs)
            except ImportError:
                # Fallback if sklearn is not available
                self.model = {"type": "GMM", "params": kwargs}
        else:
            raise ValueError("Unsupported model type. Use 'HMM' or 'GMM'.")
        return self.model

    def classify_regime(self, features: Any) -> Any:
        if self.model is not None and hasattr(self.model, "predict"):
            return self.model.predict(features)
        return "unknown"

    def fit(self, features: Any) -> None:
        if self.model is not None and hasattr(self.model, "fit"):
            self.model.fit(features)

    def get_regime_probabilities(self, features: Any) -> Any:
        if self.model is not None and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)
        return None
