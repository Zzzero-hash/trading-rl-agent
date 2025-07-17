"""
Tests for the supervised model module.
"""

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.supervised_model import ModelEvaluator, PricePredictor, SignalClassifier


class TestBaseSupervisedModel:
    """Test suite for BaseSupervisedModel."""

    def test_initialization(self):
        """Test BaseSupervisedModel initialization."""
        # This is an abstract class, so we'll test through a concrete implementation
        model = PricePredictor()

        assert model.model_name == "price_predictor_random_forest"
        assert model.is_trained is False
        assert model.feature_names is None
        assert isinstance(model.model_params, dict)

    def test_get_model_params(self):
        """Test getting model parameters."""
        model = PricePredictor()

        params = model.get_model_params()
        assert isinstance(params, dict)

    def test_set_model_params(self):
        """Test setting model parameters."""
        model = PricePredictor()

        new_params = {"test_param": 42}
        model.set_model_params(new_params)

        params = model.get_model_params()
        assert "test_param" in params
        assert params["test_param"] == 42


class TestPricePredictor:
    """Test suite for PricePredictor."""

    def test_initialization_random_forest(self):
        """Test PricePredictor initialization with random forest."""
        model = PricePredictor(model_type="random_forest")

        assert model.model_type == "random_forest"
        assert model.model is not None
        assert model.is_trained is False

    def test_initialization_linear(self):
        """Test PricePredictor initialization with linear regression."""
        model = PricePredictor(model_type="linear")

        assert model.model_type == "linear"
        assert model.model is not None
        assert model.is_trained is False

    def test_initialization_invalid_type(self):
        """Test PricePredictor initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            PricePredictor(model_type="invalid_type")

    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        model = PricePredictor(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Fit the model
        model.fit(X, y)

        assert model.is_trained is True

        # Make predictions
        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10

    def test_fit_with_dataframe(self):
        """Test model fitting with pandas DataFrame."""
        model = PricePredictor(model_type="random_forest")

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
        y = pd.Series(np.random.randn(100))

        # Fit the model
        model.fit(X, y)

        assert model.is_trained is True
        assert model.feature_names == ["f1", "f2", "f3", "f4", "f5"]

    def test_predict_without_training(self):
        """Test prediction without training."""
        model = PricePredictor()

        X_test = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model.predict(X_test)

    def test_get_feature_importance_random_forest(self):
        """Test feature importance for random forest."""
        model = PricePredictor(model_type="random_forest")

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(np.random.randn(100))

        # Fit the model
        model.fit(X, y)

        # Get feature importance
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert "f1" in importance
        assert "f2" in importance
        assert "f3" in importance
        assert all(isinstance(v, float) for v in importance.values())

    def test_get_feature_importance_linear(self):
        """Test feature importance for linear regression."""
        model = PricePredictor(model_type="linear")

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(np.random.randn(100))

        # Fit the model
        model.fit(X, y)

        # Get feature importance
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert "f1" in importance
        assert "f2" in importance
        assert "f3" in importance

    def test_evaluate(self):
        """Test model evaluation."""
        model = PricePredictor(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Fit the model
        model.fit(X, y)

        # Evaluate
        metrics = model.evaluate(X, y)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestSignalClassifier:
    """Test suite for SignalClassifier."""

    def test_initialization_random_forest(self):
        """Test SignalClassifier initialization with random forest."""
        model = SignalClassifier(model_type="random_forest")

        assert model.model_type == "random_forest"
        assert model.model is not None
        assert model.is_trained is False
        assert model.classes_ is None

    def test_initialization_logistic(self):
        """Test SignalClassifier initialization with logistic regression."""
        model = SignalClassifier(model_type="logistic")

        assert model.model_type == "logistic"
        assert model.model is not None
        assert model.is_trained is False

    def test_initialization_invalid_type(self):
        """Test SignalClassifier initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            SignalClassifier(model_type="invalid_type")

    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        model = SignalClassifier(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.choice(["buy", "sell", "hold"], 100)

        # Fit the model
        model.fit(X, y)

        assert model.is_trained is True
        assert model.classes_ is not None

        # Make predictions
        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        assert all(pred in ["buy", "sell", "hold"] for pred in predictions)

    def test_predict_proba(self):
        """Test probability predictions."""
        model = SignalClassifier(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.choice(["buy", "sell", "hold"], 100)

        # Fit the model
        model.fit(X, y)

        # Get probability predictions
        X_test = np.random.randn(10, 5)
        proba = model.predict_proba(X_test)

        assert isinstance(proba, np.ndarray)
        assert proba.shape[0] == 10
        assert proba.shape[1] == 3  # 3 classes
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_get_feature_importance_random_forest(self):
        """Test feature importance for random forest classifier."""
        model = SignalClassifier(model_type="random_forest")

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(np.random.choice(["buy", "sell", "hold"], 100))

        # Fit the model
        model.fit(X, y)

        # Get feature importance
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert "f1" in importance
        assert "f2" in importance
        assert "f3" in importance

    def test_get_feature_importance_logistic(self):
        """Test feature importance for logistic regression."""
        model = SignalClassifier(model_type="logistic")

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(np.random.choice(["buy", "sell", "hold"], 100))

        # Fit the model
        model.fit(X, y)

        # Get feature importance
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert "f1" in importance
        assert "f2" in importance
        assert "f3" in importance

    def test_evaluate(self):
        """Test model evaluation."""
        model = SignalClassifier(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.choice(["buy", "sell", "hold"], 100)

        # Fit the model
        model.fit(X, y)

        # Evaluate
        metrics = model.evaluate(X, y)

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "precision_weighted" in metrics
        assert "recall_weighted" in metrics
        assert "f1_weighted" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_get_classification_report(self):
        """Test classification report generation."""
        model = SignalClassifier(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.choice(["buy", "sell", "hold"], 100)

        # Fit the model
        model.fit(X, y)

        # Get classification report
        report = model.get_classification_report(X, y)

        assert isinstance(report, str)
        assert "precision" in report.lower()
        assert "recall" in report.lower()
        assert "f1-score" in report.lower()

    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        model = SignalClassifier()

        # Create sample data
        y = pd.Series(["buy", "sell", "hold", "buy", "sell"])

        distribution = model.get_class_distribution(y)

        assert isinstance(distribution, dict)
        assert distribution["buy"] == 2
        assert distribution["sell"] == 2
        assert distribution["hold"] == 1


class TestModelEvaluator:
    """Test suite for ModelEvaluator."""

    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()

        assert evaluator is not None

    def test_cross_validate_classification(self):
        """Test cross-validation for classification model."""
        evaluator = ModelEvaluator()
        model = SignalClassifier(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.choice(["buy", "sell", "hold"], 100)

        # Perform cross-validation
        results = evaluator.cross_validate(model, X, y, cv=3)

        assert "scores" in results
        assert "mean" in results
        assert "std" in results
        assert "min" in results
        assert "max" in results
        assert len(results["scores"]) == 3
        assert isinstance(results["mean"], float)

    def test_cross_validate_regression(self):
        """Test cross-validation for regression model."""
        evaluator = ModelEvaluator()
        model = PricePredictor(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Perform cross-validation
        results = evaluator.cross_validate(model, X, y, cv=3)

        assert "scores" in results
        assert "mean" in results
        assert "std" in results
        assert "min" in results
        assert "max" in results
        assert len(results["scores"]) == 3

    def test_train_test_evaluate_classification(self):
        """Test train-test evaluation for classification model."""
        evaluator = ModelEvaluator()
        model = SignalClassifier(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.choice(["buy", "sell", "hold"], 100)

        # Perform train-test evaluation
        metrics = evaluator.train_test_evaluate(model, X, y, test_size=0.2)

        assert "accuracy" in metrics
        assert "test_size" in metrics
        assert "train_size" in metrics
        assert metrics["test_size"] == 20
        assert metrics["train_size"] == 80

    def test_train_test_evaluate_regression(self):
        """Test train-test evaluation for regression model."""
        evaluator = ModelEvaluator()
        model = PricePredictor(model_type="random_forest")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Perform train-test evaluation
        metrics = evaluator.train_test_evaluate(model, X, y, test_size=0.2)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "test_size" in metrics
        assert "train_size" in metrics

    def test_compare_models(self):
        """Test model comparison."""
        evaluator = ModelEvaluator()

        # Create different models
        model1 = PricePredictor(model_type="random_forest")
        model2 = PricePredictor(model_type="linear")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Compare models
        comparison = evaluator.compare_models([model1, model2], X, y, cv=3)

        assert "price_predictor_random_forest" in comparison
        assert "price_predictor_linear" in comparison
        assert "mean" in comparison["price_predictor_random_forest"]
        assert "mean" in comparison["price_predictor_linear"]

    def test_get_best_model(self):
        """Test getting the best model."""
        evaluator = ModelEvaluator()

        # Create different models
        model1 = PricePredictor(model_type="random_forest")
        model2 = PricePredictor(model_type="linear")

        # Create sample data
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        # Get best model
        best_model = evaluator.get_best_model([model1, model2], X, y, cv=3)

        assert best_model in [model1, model2]

    def test_generate_report(self):
        """Test report generation."""
        evaluator = ModelEvaluator()
        model = PricePredictor(model_type="random_forest")

        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
        y = pd.Series(np.random.randn(100))

        # Generate report
        report = evaluator.generate_report(model, X, y)

        assert isinstance(report, str)
        assert "Model Evaluation Report" in report
        assert "Cross-Validation Results" in report
        assert "Train-Test Split Results" in report
        assert "Feature Importance" in report
