"""
Tests for Agent Scenario Evaluator

This module tests the comprehensive agent evaluation framework using
synthetic data scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from trade_agent.eval import AgentScenarioEvaluator, MarketScenario, ScenarioResult
from trade_agent.eval.scenario_evaluator import MarketScenarioGenerator


class TestMarketScenario:
    """Test MarketScenario class."""

    def test_valid_scenario_creation(self):
        """Test creating a valid market scenario."""
        scenario = MarketScenario(
            name="Test Scenario",
            description="A test scenario",
            duration_days=100,
            market_regime="trend_following",
            base_volatility=0.02,
            drift=0.001,
        )

        assert scenario.name == "Test Scenario"
        assert scenario.duration_days == 100
        assert scenario.base_volatility == 0.02
        assert scenario.drift == 0.001

    def test_invalid_duration(self):
        """Test scenario creation with invalid duration."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            MarketScenario(
                name="Test",
                description="Test",
                duration_days=0,
                market_regime="trend_following",
                base_volatility=0.02,
                drift=0.001,
            )

    def test_invalid_volatility(self):
        """Test scenario creation with invalid volatility."""
        with pytest.raises(ValueError, match="Volatility must be positive"):
            MarketScenario(
                name="Test",
                description="Test",
                duration_days=100,
                market_regime="trend_following",
                base_volatility=0.0,
                drift=0.001,
            )

    def test_invalid_drawdown(self):
        """Test scenario creation with invalid drawdown."""
        with pytest.raises(ValueError, match="Max drawdown must be between 0 and 1"):
            MarketScenario(
                name="Test",
                description="Test",
                duration_days=100,
                market_regime="trend_following",
                base_volatility=0.02,
                drift=0.001,
                max_drawdown=1.5,  # Invalid
            )


class TestScenarioResult:
    """Test ScenarioResult class."""

    def test_success_evaluation_passed(self):
        """Test success evaluation when criteria are met."""
        scenario = MarketScenario(
            name="Test",
            description="Test",
            duration_days=100,
            market_regime="trend_following",
            base_volatility=0.02,
            drift=0.001,
            min_sharpe_ratio=0.5,
            max_drawdown=0.15,
            min_win_rate=0.4,
            min_profit_factor=1.2,
        )

        result = ScenarioResult(
            scenario=scenario,
            agent_name="test_agent",
            total_return=0.1,
            sharpe_ratio=1.0,  # Above minimum
            sortino_ratio=1.2,
            max_drawdown=0.1,  # Below maximum
            win_rate=0.6,  # Above minimum
            profit_factor=1.5,  # Above minimum
            volatility=0.15,
        )

        success = result.evaluate_success()
        assert success is True
        assert result.passed_criteria is True
        assert len(result.failure_reasons) == 0

    def test_success_evaluation_failed(self):
        """Test success evaluation when criteria are not met."""
        scenario = MarketScenario(
            name="Test",
            description="Test",
            duration_days=100,
            market_regime="trend_following",
            base_volatility=0.02,
            drift=0.001,
            min_sharpe_ratio=0.5,
            max_drawdown=0.15,
            min_win_rate=0.4,
            min_profit_factor=1.2,
        )

        result = ScenarioResult(
            scenario=scenario,
            agent_name="test_agent",
            total_return=0.1,
            sharpe_ratio=0.3,  # Below minimum
            sortino_ratio=0.4,
            max_drawdown=0.2,  # Above maximum
            win_rate=0.3,  # Below minimum
            profit_factor=1.0,  # Below minimum
            volatility=0.15,
        )

        success = result.evaluate_success()
        assert success is False
        assert result.passed_criteria is False
        assert len(result.failure_reasons) == 4  # All criteria failed


class TestMarketScenarioGenerator:
    """Test MarketScenarioGenerator class."""

    def test_trend_following_scenario(self):
        """Test trend following scenario generation."""
        generator = MarketScenarioGenerator(seed=42)

        data = generator.generate_trend_following_scenario(
            duration_days=50,
            trend_strength=0.001,
            volatility=0.015,
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert all(col in data.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])
        assert all(data["close"] > 0)
        assert all(data["volume"] > 0)

    def test_mean_reversion_scenario(self):
        """Test mean reversion scenario generation."""
        generator = MarketScenarioGenerator(seed=42)

        data = generator.generate_mean_reversion_scenario(
            duration_days=50,
            mean_price=100.0,
            reversion_strength=0.1,
            volatility=0.02,
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert all(col in data.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

        # Check that prices stay around mean
        close_prices = data["close"].values
        mean_price = np.mean(close_prices)
        assert 80 < mean_price < 120  # Reasonable range around 100

    def test_volatility_breakout_scenario(self):
        """Test volatility breakout scenario generation."""
        generator = MarketScenarioGenerator(seed=42)

        data = generator.generate_volatility_breakout_scenario(
            duration_days=50,
            base_volatility=0.01,
            breakout_volatility=0.05,
            breakout_probability=0.1,
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert all(col in data.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_crisis_scenario(self):
        """Test crisis scenario generation."""
        generator = MarketScenarioGenerator(seed=42)

        data = generator.generate_crisis_scenario(
            duration_days=50,
            crisis_start=25,
            crisis_duration=10,
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert all(col in data.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_regime_change_scenario(self):
        """Test regime change scenario generation."""
        generator = MarketScenarioGenerator(seed=42)

        data = generator.generate_regime_change_scenario(
            duration_days=50,
            regime_changes=[
                {"start_day": 10, "regime": "trend_up", "duration": 10},
                {"start_day": 20, "regime": "volatile", "duration": 10},
                {"start_day": 30, "regime": "trend_down", "duration": 10},
            ],
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert all(col in data.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])


class TestAgentScenarioEvaluator:
    """Test AgentScenarioEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = AgentScenarioEvaluator(seed=42)

        assert len(evaluator.scenarios) == 5  # Default scenarios
        assert all(isinstance(s, MarketScenario) for s in evaluator.scenarios)
        assert evaluator.results == []

    def test_custom_scenarios_initialization(self):
        """Test evaluator initialization with custom scenarios."""
        custom_scenarios = [
            MarketScenario(
                name="Custom Test",
                description="Custom test scenario",
                duration_days=50,
                market_regime="trend_following",
                base_volatility=0.02,
                drift=0.001,
            )
        ]

        evaluator = AgentScenarioEvaluator(scenarios=custom_scenarios, seed=42)

        assert len(evaluator.scenarios) == 1
        assert evaluator.scenarios[0].name == "Custom Test"

    def test_simple_agent_evaluation(self):
        """Test evaluation of a simple agent."""
        evaluator = AgentScenarioEvaluator(seed=42)

        # Create a simple agent that always returns 0 (hold)
        def simple_agent(features):
            return np.zeros(len(features))

        results = evaluator.evaluate_agent(
            agent=simple_agent,
            agent_name="simple_agent",
        )

        assert "agent_name" in results
        assert results["agent_name"] == "simple_agent"
        assert "scenario_results" in results
        assert len(results["scenario_results"]) == 5  # Number of default scenarios
        assert "aggregate_metrics" in results
        assert "overall_score" in results
        assert "robustness_score" in results
        assert "adaptation_score" in results

    def test_agent_with_predictions(self):
        """Test evaluation of an agent that generates predictions."""
        evaluator = AgentScenarioEvaluator(seed=42)

        # Create an agent that generates random predictions
        def random_agent(features):
            return np.random.normal(0, 1, len(features))

        results = evaluator.evaluate_agent(
            agent=random_agent,
            agent_name="random_agent",
        )

        assert "agent_name" in results
        assert results["agent_name"] == "random_agent"
        assert "scenario_results" in results
        assert len(results["scenario_results"]) == 5

    def test_custom_scenarios_evaluation(self):
        """Test evaluation with custom scenarios."""
        custom_scenarios = [
            MarketScenario(
                name="Short Test",
                description="Short test scenario",
                duration_days=20,
                market_regime="trend_following",
                base_volatility=0.02,
                drift=0.001,
            )
        ]

        evaluator = AgentScenarioEvaluator(scenarios=custom_scenarios, seed=42)

        def simple_agent(features):
            return np.zeros(len(features))

        results = evaluator.evaluate_agent(
            agent=simple_agent,
            agent_name="test_agent",
        )

        assert len(results["scenario_results"]) == 1
        assert results["scenario_results"][0]["scenario"]["name"] == "Short Test"

    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        evaluator = AgentScenarioEvaluator(seed=42)

        # Create an agent that generates positive predictions (buy signals)
        def buy_agent(features):
            return np.ones(len(features))

        results = evaluator.evaluate_agent(
            agent=buy_agent,
            agent_name="buy_agent",
        )

        # Check that metrics are present and reasonable
        aggregate_metrics = results["aggregate_metrics"]
        assert "avg_sharpe_ratio" in aggregate_metrics
        assert "avg_total_return" in aggregate_metrics
        assert "avg_max_drawdown" in aggregate_metrics
        assert "pass_rate" in aggregate_metrics
        assert 0 <= aggregate_metrics["pass_rate"] <= 1

    def test_score_calculations(self):
        """Test that scores are calculated correctly."""
        evaluator = AgentScenarioEvaluator(seed=42)

        def simple_agent(features):
            return np.zeros(len(features))

        results = evaluator.evaluate_agent(
            agent=simple_agent,
            agent_name="test_agent",
        )

        # Check that scores are in reasonable ranges
        assert 0 <= results["overall_score"] <= 1
        assert 0 <= results["robustness_score"] <= 1
        assert 0 <= results["adaptation_score"] <= 1

    def test_report_generation(self):
        """Test report generation."""
        evaluator = AgentScenarioEvaluator(seed=42)

        def simple_agent(features):
            return np.zeros(len(features))

        results = evaluator.evaluate_agent(
            agent=simple_agent,
            agent_name="test_agent",
        )

        report = evaluator.generate_evaluation_report(results)

        assert isinstance(report, str)
        assert "Agent Scenario Evaluation Report" in report
        assert "test_agent" in report
        assert "Executive Summary" in report

    def test_visualization_creation(self):
        """Test visualization creation."""
        evaluator = AgentScenarioEvaluator(seed=42)

        def simple_agent(features):
            return np.zeros(len(features))

        results = evaluator.evaluate_agent(
            agent=simple_agent,
            agent_name="test_agent",
        )

        # Test that visualization doesn't raise errors
        try:
            evaluator.create_visualization(results)
        except Exception as e:
            pytest.fail(f"Visualization creation failed: {e}")

    def test_print_summary(self):
        """Test summary printing."""
        evaluator = AgentScenarioEvaluator(seed=42)

        def simple_agent(features):
            return np.zeros(len(features))

        results = evaluator.evaluate_agent(
            agent=simple_agent,
            agent_name="test_agent",
        )

        # Test that summary printing doesn't raise errors
        try:
            evaluator.print_evaluation_summary(results)
        except Exception as e:
            pytest.fail(f"Summary printing failed: {e}")


class TestIntegration:
    """Integration tests for the scenario evaluator."""

    def test_end_to_end_evaluation(self):
        """Test complete end-to-end evaluation workflow."""
        evaluator = AgentScenarioEvaluator(seed=42)

        # Create multiple agents
        agents = {
            "zero_agent": lambda features: np.zeros(len(features)),
            "random_agent": lambda features: np.random.normal(0, 1, len(features)),
            "trend_agent": lambda features: np.arange(len(features)) / len(features),
        }

        all_results = {}

        for agent_name, agent in agents.items():
            results = evaluator.evaluate_agent(
                agent=agent,
                agent_name=agent_name,
            )
            all_results[agent_name] = results

        # Check that all agents were evaluated
        assert len(all_results) == 3
        assert all("agent_name" in results for results in all_results.values())
        assert all("scenario_results" in results for results in all_results.values())

        # Check that results are different for different agents
        agent_names = list(all_results.keys())
        assert all_results[agent_names[0]]["overall_score"] != all_results[agent_names[1]]["overall_score"]

    def test_scenario_data_generation(self):
        """Test that scenario data generation works for all regimes."""
        evaluator = AgentScenarioEvaluator(seed=42)

        for scenario in evaluator.scenarios:
            data = evaluator._generate_scenario_data(scenario)

            assert isinstance(data, pd.DataFrame)
            assert len(data) == scenario.duration_days
            assert all(col in data.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])
            assert all(data["close"] > 0)
            assert all(data["volume"] > 0)

    def test_feature_preparation(self):
        """Test feature preparation for agent input."""
        evaluator = AgentScenarioEvaluator(seed=42)

        # Create sample data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50),
                "open": np.random.uniform(100, 200, 50),
                "high": np.random.uniform(100, 200, 50),
                "low": np.random.uniform(100, 200, 50),
                "close": np.random.uniform(100, 200, 50),
                "volume": np.random.randint(1000, 10000, 50),
            }
        )

        features = evaluator._prepare_features(data)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(data)
        assert features.shape[1] > 0  # Should have multiple features
        assert not np.any(np.isnan(features))  # No NaN values
        assert not np.any(np.isinf(features))  # No infinite values


if __name__ == "__main__":
    pytest.main([__file__])
