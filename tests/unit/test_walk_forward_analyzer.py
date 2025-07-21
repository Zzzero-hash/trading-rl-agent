"""
Unit tests for WalkForwardAnalyzer.

Tests the walk-forward analysis functionality for robust model evaluation.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from trade_agent.eval.walk_forward_analyzer import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WindowResult,
)


class TestWalkForwardConfig:
    """Test WalkForwardConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()

        assert config.train_window_size == 252
        assert config.validation_window_size == 63
        assert config.test_window_size == 63
        assert config.step_size == 21
        assert config.model_type == "cnn_lstm"
        assert config.confidence_level == 0.95
        assert config.generate_plots is True
        assert config.save_results is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WalkForwardConfig(
            train_window_size=100,
            validation_window_size=20,
            test_window_size=20,
            step_size=10,
            model_type="rl",
            confidence_level=0.99,
            generate_plots=False,
            save_results=False,
        )

        assert config.train_window_size == 100
        assert config.validation_window_size == 20
        assert config.test_window_size == 20
        assert config.step_size == 10
        assert config.model_type == "rl"
        assert config.confidence_level == 0.99
        assert config.generate_plots is False
        assert config.save_results is False


class TestWindowResult:
    """Test WindowResult dataclass."""

    def test_window_result_creation(self):
        """Test WindowResult creation with all fields."""
        timestamp = pd.Timestamp("2023-01-01")

        result = WindowResult(
            window_id=0,
            train_start=timestamp,
            train_end=timestamp,
            validation_start=timestamp,
            validation_end=timestamp,
            test_start=timestamp,
            test_end=timestamp,
            train_metrics={"mae": 0.1},
            validation_metrics={"mae": 0.15},
            test_metrics={"mae": 0.2},
            test_predictions=np.array([0.1, 0.2, 0.3]),
            test_actuals=np.array([0.1, 0.2, 0.3]),
            test_returns=np.array([0.01, 0.02, 0.03]),
            model_path="/path/to/model.pth",
            training_time=10.0,
            inference_time=1.0,
        )

        assert result.window_id == 0
        assert result.train_metrics["mae"] == 0.1
        assert result.validation_metrics["mae"] == 0.15
        assert result.test_metrics["mae"] == 0.2
        assert len(result.test_predictions) == 3
        assert result.training_time == 10.0
        assert result.inference_time == 1.0


class TestWalkForwardAnalyzer:
    """Test WalkForwardAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")

        return pd.DataFrame(
            {
                "returns": np.random.normal(0, 0.02, 1000),
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.normal(0, 1, 1000),
            },
            index=dates,
        )

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WalkForwardConfig(
            train_window_size=100,
            validation_window_size=20,
            test_window_size=20,
            step_size=10,
            model_type="cnn_lstm",
            generate_plots=False,
            save_results=False,
        )

    @pytest.fixture
    def analyzer(self, config):
        """Create WalkForwardAnalyzer instance."""
        return WalkForwardAnalyzer(config)

    def test_analyzer_initialization(self, config):
        """Test analyzer initialization."""
        analyzer = WalkForwardAnalyzer(config)

        assert analyzer.config == config
        assert len(analyzer.window_results) == 0
        assert analyzer.overall_metrics == {}
        assert analyzer.stability_analysis == {}

    def test_create_time_windows(self, analyzer, sample_data):
        """Test time window creation."""
        windows = analyzer._create_time_windows(sample_data)

        assert len(windows) > 0

        # Check window structure
        for train_idx, val_idx, test_idx in windows:
            assert len(train_idx) == analyzer.config.train_window_size
            assert len(val_idx) == analyzer.config.validation_window_size
            assert len(test_idx) == analyzer.config.test_window_size

            # Check no overlap between train and validation
            assert len(set(train_idx) & set(val_idx)) == 0
            # Check no overlap between validation and test
            assert len(set(val_idx) & set(test_idx)) == 0

    def test_calculate_returns(self, analyzer):
        """Test returns calculation."""
        predictions = np.array([0.1, -0.2, 0.3, -0.1])
        actuals = np.array([0.05, -0.15, 0.25, -0.05])

        returns = analyzer._calculate_returns(predictions, actuals)

        expected_returns = np.array([0.05, 0.15, 0.25, 0.05])
        np.testing.assert_array_almost_equal(returns, expected_returns)

    def test_calculate_average_metrics(self, analyzer):
        """Test average metrics calculation."""
        metrics_list = [
            {"mae": 0.1, "rmse": 0.2, "r2_score": 0.8},
            {"mae": 0.15, "rmse": 0.25, "r2_score": 0.7},
            {"mae": 0.12, "rmse": 0.22, "r2_score": 0.75},
        ]

        avg_metrics = analyzer._calculate_average_metrics(metrics_list)

        assert avg_metrics["mae"] == pytest.approx(0.1233, rel=1e-3)
        assert avg_metrics["rmse"] == pytest.approx(0.2233, rel=1e-3)
        assert avg_metrics["r2_score"] == pytest.approx(0.75, rel=1e-3)

    def test_calculate_std_metrics(self, analyzer):
        """Test standard deviation metrics calculation."""
        metrics_list = [
            {"mae": 0.1, "rmse": 0.2},
            {"mae": 0.15, "rmse": 0.25},
            {"mae": 0.12, "rmse": 0.22},
        ]

        std_metrics = analyzer._calculate_std_metrics(metrics_list)

        assert std_metrics["mae"] == pytest.approx(0.0258, rel=1e-3)
        assert std_metrics["rmse"] == pytest.approx(0.0258, rel=1e-3)

    def test_calculate_confidence_intervals(self, analyzer):
        """Test confidence interval calculation."""
        metrics_list = [
            {"mae": 0.1, "rmse": 0.2},
            {"mae": 0.15, "rmse": 0.25},
            {"mae": 0.12, "rmse": 0.22},
        ]

        confidence_intervals = analyzer._calculate_confidence_intervals(metrics_list)

        assert "mae" in confidence_intervals
        assert "rmse" in confidence_intervals

        mae_ci = confidence_intervals["mae"]
        assert len(mae_ci) == 2
        assert mae_ci[0] < mae_ci[1]  # Lower bound < upper bound

    @patch("trading_rl_agent.eval.walk_forward_analyzer.plt")
    @patch("trading_rl_agent.eval.walk_forward_analyzer.sns")
    def test_generate_visualizations(self, mock_plt, analyzer, sample_data):
        """Test visualization generation."""

        if sample_data is None:
            # Create mock window results
            analyzer.window_results = [
                WindowResult(
                    window_id=0,
                    train_start=pd.Timestamp("2020-01-01"),
                    train_end=pd.Timestamp("2020-05-01"),
                    validation_start=pd.Timestamp("2020-05-02"),
                    validation_end=pd.Timestamp("2020-06-01"),
                    test_start=pd.Timestamp("2020-06-02"),
                    test_end=pd.Timestamp("2020-07-01"),
                    train_metrics={"mae": 0.1},
                    validation_metrics={"mae": 0.15},
                    test_metrics={
                        "sharpe_ratio": 1.2,
                        "total_return": 0.1,
                        "max_drawdown": -0.05,
                    },
                    test_predictions=np.array([0.1, 0.2, 0.3]),
                    test_actuals=np.array([0.1, 0.2, 0.3]),
                    test_returns=np.array([0.01, 0.02, 0.03]),
                )
            ]

        # Mock overall metrics
        analyzer.overall_metrics = {
            "avg_metrics": {
                "sharpe_ratio": 1.2,
                "total_return": 0.1,
                "max_drawdown": -0.05,
            },
            "std_metrics": {
                "sharpe_ratio": 0.1,
                "total_return": 0.02,
                "max_drawdown": 0.01,
            },
            "confidence_intervals": {
                "sharpe_ratio": (1.1, 1.3),
                "total_return": (0.08, 0.12),
                "max_drawdown": (-0.06, -0.04),
            },
            "stability_metrics": {"sharpe_cv": 0.1, "returns_cv": 0.2},
            "performance_degradation": {"sharpe_ratio_degradation_correlation": -0.1},
        }

        # Mock stability analysis
        analyzer.stability_analysis = {
            "normality_test": {"p_value": 0.1, "is_normal": True},
            "stationarity_test": {"p_value": 0.05, "is_stationary": True},
        }

        # Test visualization generation
        analyzer._generate_visualizations()

        # Check that matplotlib was called
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

    def test_calculate_performance_degradation(self, analyzer):
        """Test performance degradation calculation."""
        # Create mock window results with performance metrics
        analyzer.window_results = [
            WindowResult(
                window_id=i,
                train_start=pd.Timestamp("2020-01-01"),
                train_end=pd.Timestamp("2020-05-01"),
                validation_start=pd.Timestamp("2020-05-02"),
                validation_end=pd.Timestamp("2020-06-01"),
                test_start=pd.Timestamp("2020-06-02"),
                test_end=pd.Timestamp("2020-07-01"),
                train_metrics={"mae": 0.1},
                validation_metrics={"mae": 0.15},
                test_metrics={
                    "sharpe_ratio": 1.2 - i * 0.1,  # Decreasing performance
                    "total_return": 0.1 - i * 0.02,
                    "max_drawdown": -0.05 - i * 0.01,
                },
                test_predictions=np.array([0.1, 0.2, 0.3]),
                test_actuals=np.array([0.1, 0.2, 0.3]),
                test_returns=np.array([0.01, 0.02, 0.03]),
            )
            for i in range(5)
        ]

        degradation = analyzer._calculate_performance_degradation()

        assert "sharpe_ratio_degradation_correlation" in degradation
        assert "total_return_degradation_correlation" in degradation
        assert "max_drawdown_degradation_correlation" in degradation

        # Should show negative correlation (degradation)
        assert degradation["sharpe_ratio_degradation_correlation"] < 0

    def test_calculate_stability_metrics(self, analyzer):
        """Test stability metrics calculation."""
        returns_list = [
            np.array([0.01, 0.02, -0.01, 0.03]),
            np.array([0.02, -0.01, 0.01, 0.02]),
            np.array([-0.01, 0.01, 0.02, 0.01]),
        ]

        stability_metrics = analyzer._calculate_stability_metrics(returns_list)

        assert "sharpe_cv" in stability_metrics
        assert "returns_cv" in stability_metrics
        assert "drawdown_cv" in stability_metrics

        # All coefficients of variation should be positive
        for value in stability_metrics.values():
            assert value >= 0

    @patch("builtins.print")
    def test_print_summary(self, mock_print, analyzer):
        """Test summary printing."""
        # Create mock overall metrics
        analyzer.overall_metrics = {
            "avg_metrics": {
                "sharpe_ratio": 1.2,
                "total_return": 0.1,
                "max_drawdown": -0.05,
                "win_rate": 0.6,
            },
            "std_metrics": {
                "sharpe_ratio": 0.1,
                "total_return": 0.02,
                "max_drawdown": 0.01,
                "win_rate": 0.05,
            },
            "stability_metrics": {
                "sharpe_cv": 0.1,
                "returns_cv": 0.2,
            },
            "performance_degradation": {
                "sharpe_ratio_degradation_correlation": -0.1,
            },
        }

        analyzer.print_summary()

        # Check that print was called (summary table creation)
        assert mock_print.called

    def test_save_model(self, analyzer, tmp_path):
        """Test model saving."""
        analyzer.config.model_save_dir = str(tmp_path)
        analyzer.config.save_models = True

        # Create a mock PyTorch model
        model = torch.nn.Linear(10, 1)

        model_path = analyzer._save_model(model, window_id=0)

        assert model_path is not None
        assert Path(model_path).exists()

    @patch("trading_rl_agent.eval.walk_forward_analyzer.OptimizedTrainingManager")
    def test_train_cnn_lstm(self, mock_trainer, analyzer):
        """Test CNN+LSTM training."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randn(20)

        # Mock the trainer
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = {
            "train_metrics": {"mae": 0.1, "rmse": 0.2},
            "val_metrics": {"mae": 0.15, "rmse": 0.25},
        }
        mock_trainer.return_value = mock_trainer_instance

        model, train_metrics, val_metrics = analyzer._train_cnn_lstm(X_train, y_train, X_val, y_val)

        assert train_metrics["mae"] == 0.1
        assert val_metrics["mae"] == 0.15
        mock_trainer_instance.train.assert_called_once()

    def test_evaluate_model(self, analyzer):
        """Test model evaluation."""
        # Create a mock model
        model = Mock()
        model.eval.return_value = None
        model.return_value = torch.tensor([[0.1], [0.2], [0.3]])

        X_test = np.random.randn(3, 10)
        y_test = np.array([0.1, 0.2, 0.3])

        predictions, metrics = analyzer._evaluate_model(model, X_test, y_test)

        assert len(predictions) == 3
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2_score" in metrics
