import numpy as np
import pandas as pd  # type: ignore  # pandas-stubs missing in some environments
import pytest

# import torch  # unused
from trading_rl_agent.supervised_model import (  # type: ignore  # may show as unresolved in some editors
    ModelEvaluator,
    PricePredictor,
    SignalClassifier,
)


def test_base_supervised_model_initialization():
    """Test BaseSupervisedModel initialization and basic functionality via a concrete subclass."""
    model = PricePredictor()
    # Test basic properties
    assert hasattr(model, "model")
    assert hasattr(model, "is_trained")
    assert model.is_trained is False
    # Test get_model_params
    params = model.get_model_params()
    assert isinstance(params, dict)
    # Test set_model_params
    model.set_model_params({"test_param": "test_value"})
    params = model.get_model_params()
    assert params.get("test_param") == "test_value"


def test_price_predictor_initialization():
    """Test PricePredictor initialization with different model types."""
    # Test Random Forest initialization
    rf_predictor = PricePredictor(model_type="random_forest")
    assert rf_predictor.model_type == "random_forest"
    assert rf_predictor.is_trained is False
    # Test Linear initialization
    linear_predictor = PricePredictor(model_type="linear")
    assert linear_predictor.model_type == "linear"
    assert linear_predictor.is_trained is False
    # Test invalid model type
    with pytest.raises(ValueError):
        PricePredictor(model_type="invalid_type")


def test_signal_classifier_initialization():
    """Test SignalClassifier initialization with different model types."""
    # Test Random Forest initialization
    rf_classifier = SignalClassifier(model_type="random_forest")
    assert rf_classifier.model_type == "random_forest"
    assert rf_classifier.is_trained is False
    # Test Logistic initialization
    logistic_classifier = SignalClassifier(model_type="logistic")
    assert logistic_classifier.model_type == "logistic"
    assert logistic_classifier.is_trained is False
    # Test invalid model type
    with pytest.raises(ValueError):
        SignalClassifier(model_type="invalid_type")


def test_model_evaluator_initialization():
    """Test ModelEvaluator initialization and basic functionality."""
    evaluator = ModelEvaluator()
    assert isinstance(evaluator, ModelEvaluator)


def test_price_predictor_fit_and_predict():
    """Test PricePredictor fit and predict functionality."""
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    rf_predictor = PricePredictor(model_type="random_forest")
    rf_predictor.fit(X, y)
    assert rf_predictor.is_trained is True
    X_test = np.random.randn(10, 5)
    predictions = rf_predictor.predict(X_test)
    assert len(predictions) == 10
    assert isinstance(predictions, np.ndarray)


def test_signal_classifier_fit_and_predict():
    """Test SignalClassifier fit and predict functionality."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    rf_classifier.fit(X, y)
    assert rf_classifier.is_trained is True
    X_test = np.random.randn(10, 5)
    predictions = rf_classifier.predict(X_test)
    assert len(predictions) == 10
    assert isinstance(predictions, np.ndarray)
    proba = rf_classifier.predict_proba(X_test)
    assert proba.shape[0] == 10
    assert proba.shape[1] == 3


def test_model_evaluator_cross_validation():
    """Test ModelEvaluator cross-validation functionality."""
    evaluator = ModelEvaluator()
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    results = evaluator.cross_validate(rf_classifier, X, y, cv=3)
    assert "scores" in results
    assert "mean" in results
    assert "std" in results
    assert "min" in results
    assert "max" in results
    assert all(isinstance(v, (int, float, list)) for v in results.values())


def test_model_evaluator_train_test_evaluate():
    """Test ModelEvaluator train-test evaluation functionality."""
    evaluator = ModelEvaluator()
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    results = evaluator.train_test_evaluate(rf_classifier, X, y, test_size=0.2)
    assert "accuracy" in results
    assert "test_size" in results
    assert "train_size" in results
    assert all(isinstance(v, (int, float)) for v in results.values())


def test_feature_importance():
    """Test feature importance functionality for Random Forest models."""
    X_df = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    rf_classifier.fit(X_df, y)
    importance = rf_classifier.get_feature_importance()
    assert len(importance) == 5
    assert all(isinstance(v, float) for v in importance.values())
    assert all(v >= 0 for v in importance.values())


def test_model_evaluation_with_dataframe():
    """Test model evaluation with pandas DataFrame input."""
    X_df = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    rf_classifier.fit(X_df, y)
    assert rf_classifier.is_trained is True
    X_test_df = pd.DataFrame(np.random.randn(10, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    predictions = rf_classifier.predict(X_test_df)
    assert len(predictions) == 10


def test_error_handling():
    """Test error handling for various edge cases."""
    rf_classifier = SignalClassifier(model_type="random_forest")
    X_test = np.random.randn(10, 5)
    with pytest.raises(ValueError):
        rf_classifier.predict(X_test)
    rf_classifier.fit(np.random.randn(100, 5), np.random.randint(0, 3, 100))
    with pytest.raises(ValueError):
        rf_classifier.predict(np.random.randn(10, 3))


def test_price_predictor_evaluation():
    """Test PricePredictor evaluation functionality."""
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    rf_predictor = PricePredictor(model_type="random_forest")
    rf_predictor.fit(X, y)
    X_test = np.random.randn(20, 5)
    y_test = np.random.randn(20)
    metrics = rf_predictor.evaluate(X_test, y_test)
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "mape" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_model_comparison():
    """Test model comparison functionality."""
    evaluator = ModelEvaluator()
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    logistic_classifier = SignalClassifier(model_type="logistic")
    models = [rf_classifier, logistic_classifier]
    comparison = evaluator.compare_models(models, X, y, cv=3)
    assert len(comparison) == 2
    assert all(model.model_name in comparison for model in models)
    best_model = evaluator.get_best_model(models, X, y, cv=3)
    assert best_model in models


def test_model_report_generation():
    """Test model report generation."""
    evaluator = ModelEvaluator()
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    report = evaluator.generate_report(rf_classifier, X, y)
    assert isinstance(report, str)
    assert len(report) > 0
    assert "Model Evaluation Report" in report
    assert rf_classifier.model_name in report


def test_feature_names_preservation():
    """Test that feature names are preserved when using DataFrames."""
    X_df = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = np.random.randint(0, 3, 100)
    rf_classifier = SignalClassifier(model_type="random_forest")
    rf_classifier.fit(X_df, y)
    assert rf_classifier.feature_names == ["f1", "f2", "f3", "f4", "f5"]
    importance = rf_classifier.get_feature_importance()
    assert all(feature in importance for feature in ["f1", "f2", "f3", "f4", "f5"])


def test_model_persistence():
    """Test model saving and loading (basic structure)."""
    rf_predictor = PricePredictor(model_type="random_forest")
    with pytest.raises(NotImplementedError):
        rf_predictor.save_model("test_model.pkl")
    with pytest.raises(NotImplementedError):
        rf_predictor.load_model("test_model.pkl")


def test_linear_model_coefficients():
    """Test linear model coefficient extraction."""
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    linear_predictor = PricePredictor(model_type="linear")
    linear_predictor.fit(X, y)
    importance = linear_predictor.get_feature_importance()
    assert importance is None  # No feature names stored for numpy array input
    X_df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4", "f5"])
    linear_predictor_df = PricePredictor(model_type="linear")
    linear_predictor_df.fit(X_df, y)
    importance_df = linear_predictor_df.get_feature_importance()
    assert importance_df is not None
    assert all(feature in importance_df for feature in ["f1", "f2", "f3", "f4", "f5"])
