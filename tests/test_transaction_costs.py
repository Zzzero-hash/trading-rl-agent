"""
Tests for the comprehensive transaction cost modeling system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.trading_rl_agent.portfolio.transaction_costs import (
    TransactionCostModel,
    MarketData,
    OrderType,
    MarketCondition,
    BrokerType,
    TransactionCostAnalyzer,
    FlatRateCommission,
    TieredCommission,
    PerShareCommission,
    LinearImpactModel,
    SquareRootImpactModel,
    AdaptiveImpactModel,
    ConstantSlippageModel,
    VolumeBasedSlippageModel,
    SpreadBasedSlippageModel,
    ConstantDelayModel,
    SizeBasedDelayModel,
    MarketConditionDelayModel,
    PartialFillModel,
    CostOptimizationRecommendation,
)


class TestMarketData:
    """Test MarketData class."""
    
    def test_market_data_creation(self):
        """Test MarketData creation and properties."""
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
            sector="Technology",
        )
        
        assert market_data.bid == 99.9
        assert market_data.ask == 100.1
        assert market_data.mid_price == 100.0
        assert market_data.spread == pytest.approx(0.002, rel=1e-3)
        assert market_data.spread_bps == pytest.approx(20.0, rel=1e-3)
    
    def test_spread_calculation(self):
        """Test spread calculation."""
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.0,
            ask=101.0,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        assert market_data.spread == 0.02  # 2% spread
        assert market_data.spread_bps == 200.0  # 200 basis points


class TestCommissionStructures:
    """Test commission structure classes."""
    
    def test_flat_rate_commission(self):
        """Test FlatRateCommission."""
        commission = FlatRateCommission(rate=0.001, min_commission=1.0, max_commission=100.0)
        
        # Test normal case
        assert commission.calculate_commission(10000, 100) == 10.0
        
        # Test minimum commission
        assert commission.calculate_commission(500, 5) == 1.0
        
        # Test maximum commission
        assert commission.calculate_commission(200000, 2000) == 100.0
    
    def test_tiered_commission(self):
        """Test TieredCommission."""
        commission = TieredCommission()
        
        # Test small trade
        assert commission.calculate_commission(5000, 50) == pytest.approx(10.0, rel=1e-3)
        
        # Test medium trade
        assert commission.calculate_commission(50000, 500) == pytest.approx(75.0, rel=1e-3)
        
        # Test large trade
        assert commission.calculate_commission(200000, 2000) == pytest.approx(150.0, rel=1e-3)
    
    def test_per_share_commission(self):
        """Test PerShareCommission."""
        commission = PerShareCommission(rate_per_share=0.005)
        
        # Test normal case
        assert commission.calculate_commission(10000, 100) == 0.5
        
        # Test minimum commission
        assert commission.calculate_commission(1000, 10) == 1.0


class TestMarketImpactModels:
    """Test market impact models."""
    
    def test_linear_impact_model(self):
        """Test LinearImpactModel."""
        model = LinearImpactModel(impact_rate=0.0001)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        impact = model.calculate_impact(100000, market_data)
        expected_impact = 100000 * 0.0001 * (100000 / 1000000)
        assert impact == pytest.approx(expected_impact, rel=1e-3)
    
    def test_square_root_impact_model(self):
        """Test SquareRootImpactModel."""
        model = SquareRootImpactModel(impact_rate=0.00005)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        impact = model.calculate_impact(100000, market_data)
        expected_impact = 100000 * 0.00005 * np.sqrt(100000 / 1000000)
        assert impact == pytest.approx(expected_impact, rel=1e-3)
    
    def test_adaptive_impact_model(self):
        """Test AdaptiveImpactModel."""
        model = AdaptiveImpactModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        impact = model.calculate_impact(100000, market_data)
        assert impact > 0
        assert isinstance(impact, float)


class TestSlippageModels:
    """Test slippage models."""
    
    def test_constant_slippage_model(self):
        """Test ConstantSlippageModel."""
        model = ConstantSlippageModel(slippage_rate=0.0001)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        slippage = model.calculate_slippage(100000, market_data, OrderType.MARKET)
        assert slippage == 10.0
    
    def test_volume_based_slippage_model(self):
        """Test VolumeBasedSlippageModel."""
        model = VolumeBasedSlippageModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        slippage = model.calculate_slippage(100000, market_data, OrderType.MARKET)
        assert slippage > 0
        assert isinstance(slippage, float)
    
    def test_spread_based_slippage_model(self):
        """Test SpreadBasedSlippageModel."""
        model = SpreadBasedSlippageModel(spread_multiplier=0.5)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        slippage = model.calculate_slippage(100000, market_data, OrderType.MARKET)
        expected_slippage = 100000 * market_data.spread * 0.5
        assert slippage == pytest.approx(expected_slippage, rel=1e-3)


class TestExecutionDelayModels:
    """Test execution delay models."""
    
    def test_constant_delay_model(self):
        """Test ConstantDelayModel."""
        model = ConstantDelayModel(delay_seconds=1.0)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        delay = model.calculate_delay(100000, market_data, OrderType.MARKET)
        assert delay == 1.0
    
    def test_size_based_delay_model(self):
        """Test SizeBasedDelayModel."""
        model = SizeBasedDelayModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        delay = model.calculate_delay(100000, market_data, OrderType.MARKET)
        assert delay > 0
        assert isinstance(delay, float)
    
    def test_market_condition_delay_model(self):
        """Test MarketConditionDelayModel."""
        model = MarketConditionDelayModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        delay = model.calculate_delay(100000, market_data, OrderType.MARKET)
        assert delay > 0
        assert isinstance(delay, float)


class TestPartialFillModel:
    """Test PartialFillModel."""
    
    def test_partial_fill_simulation(self):
        """Test partial fill simulation."""
        model = PartialFillModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        filled_quantity, partial_fills = model.simulate_fill(1000, market_data)
        
        assert filled_quantity >= 0
        assert filled_quantity <= 1000
        assert isinstance(partial_fills, list)
        
        if partial_fills:
            assert all(len(fill) == 3 for fill in partial_fills)  # quantity, price, time


class TestTransactionCostModel:
    """Test TransactionCostModel."""
    
    def test_transaction_cost_model_creation(self):
        """Test TransactionCostModel creation."""
        cost_model = TransactionCostModel()
        
        assert cost_model.total_commission == 0.0
        assert cost_model.total_slippage == 0.0
        assert cost_model.total_market_impact == 0.0
        assert cost_model.total_spread_cost == 0.0
        assert cost_model.num_trades == 0
    
    def test_calculate_total_cost(self):
        """Test total cost calculation."""
        cost_model = TransactionCostModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        cost_breakdown = cost_model.calculate_total_cost(
            trade_value=100000,
            quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )
        
        assert "commission" in cost_breakdown
        assert "slippage" in cost_breakdown
        assert "market_impact" in cost_breakdown
        assert "spread_cost" in cost_breakdown
        assert "total_cost" in cost_breakdown
        assert "cost_pct" in cost_breakdown
        assert "delay_seconds" in cost_breakdown
        
        assert cost_breakdown["total_cost"] > 0
        assert cost_breakdown["cost_pct"] > 0
    
    def test_simulate_execution(self):
        """Test execution simulation."""
        cost_model = TransactionCostModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        execution_result = cost_model.simulate_execution(
            requested_quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )
        
        assert execution_result.executed_quantity >= 0
        assert execution_result.executed_price > 0
        assert execution_result.total_cost >= 0
        assert execution_result.delay_seconds >= 0
        assert isinstance(execution_result.partial_fills, list)
        assert isinstance(execution_result.success, bool)
    
    def test_cost_summary(self):
        """Test cost summary generation."""
        cost_model = TransactionCostModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        # Simulate a trade
        cost_model.simulate_execution(
            requested_quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )
        
        summary = cost_model.get_cost_summary()
        
        assert "total_commission" in summary
        assert "total_slippage" in summary
        assert "total_market_impact" in summary
        assert "total_spread_cost" in summary
        assert "total_transaction_costs" in summary
        assert "num_trades" in summary
        assert summary["num_trades"] == 1
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        cost_model = TransactionCostModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        # Simulate several trades to build up history
        for _ in range(10):
            cost_model.simulate_execution(
                requested_quantity=1000,
                market_data=market_data,
                order_type=OrderType.MARKET,
                market_condition=MarketCondition.NORMAL,
            )
        
        recommendations = cost_model.generate_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, CostOptimizationRecommendation)
            assert hasattr(rec, "recommendation_type")
            assert hasattr(rec, "description")
            assert hasattr(rec, "expected_savings")
            assert hasattr(rec, "confidence")
            assert hasattr(rec, "implementation_difficulty")
            assert hasattr(rec, "priority")
    
    def test_broker_model_creation(self):
        """Test broker-specific model creation."""
        for broker_type in BrokerType:
            cost_model = TransactionCostModel.create_broker_model(broker_type)
            assert isinstance(cost_model, TransactionCostModel)
    
    def test_reset(self):
        """Test cost model reset."""
        cost_model = TransactionCostModel()
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        # Simulate a trade
        cost_model.simulate_execution(
            requested_quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )
        
        # Reset
        cost_model.reset()
        
        assert cost_model.total_commission == 0.0
        assert cost_model.total_slippage == 0.0
        assert cost_model.total_market_impact == 0.0
        assert cost_model.total_spread_cost == 0.0
        assert cost_model.num_trades == 0
        assert len(cost_model.trade_history) == 0


class TestTransactionCostAnalyzer:
    """Test TransactionCostAnalyzer."""
    
    def test_analyzer_creation(self):
        """Test TransactionCostAnalyzer creation."""
        cost_model = TransactionCostModel()
        analyzer = TransactionCostAnalyzer(cost_model)
        
        assert analyzer.cost_model == cost_model
    
    def test_cost_trends_analysis(self):
        """Test cost trends analysis."""
        cost_model = TransactionCostModel()
        analyzer = TransactionCostAnalyzer(cost_model)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        # Simulate some trades
        for _ in range(5):
            cost_model.simulate_execution(
                requested_quantity=1000,
                market_data=market_data,
                order_type=OrderType.MARKET,
                market_condition=MarketCondition.NORMAL,
            )
        
        trends = analyzer.analyze_cost_trends()
        
        assert isinstance(trends, dict)
        if trends:  # If there are trends
            assert "cost_trend" in trends
            assert "delay_trend" in trends
            assert "timestamps" in trends
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        cost_model = TransactionCostModel()
        analyzer = TransactionCostAnalyzer(cost_model)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        # Simulate some trades
        for _ in range(3):
            cost_model.simulate_execution(
                requested_quantity=1000,
                market_data=market_data,
                order_type=OrderType.MARKET,
                market_condition=MarketCondition.NORMAL,
            )
        
        metrics = analyzer.calculate_cost_efficiency_metrics()
        
        assert isinstance(metrics, dict)
        if metrics:  # If there are metrics
            assert "cost_efficiency_ratio" in metrics
            assert "avg_cost_per_dollar" in metrics
            assert "cost_per_trade" in metrics
            assert "fill_rate" in metrics
    
    def test_cost_report_generation(self):
        """Test cost report generation."""
        cost_model = TransactionCostModel()
        analyzer = TransactionCostAnalyzer(cost_model)
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=99.9,
            ask=100.1,
            mid_price=100.0,
            volume=100000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )
        
        # Simulate some trades
        for _ in range(2):
            cost_model.simulate_execution(
                requested_quantity=1000,
                market_data=market_data,
                order_type=OrderType.MARKET,
                market_condition=MarketCondition.NORMAL,
            )
        
        report = analyzer.generate_cost_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Transaction Cost Analysis Report" in report


class TestIntegration:
    """Test integration with portfolio management."""
    
    def test_portfolio_manager_integration(self):
        """Test integration with PortfolioManager."""
        from src.trading_rl_agent.portfolio.manager import PortfolioManager, PortfolioConfig
        
        # Create portfolio manager with transaction cost model
        config = PortfolioConfig(broker_type=BrokerType.RETAIL)
        portfolio_manager = PortfolioManager(
            initial_capital=100000,
            config=config,
        )
        
        # Test that the transaction cost model is properly initialized
        assert hasattr(portfolio_manager, 'transaction_cost_model')
        assert isinstance(portfolio_manager.transaction_cost_model, TransactionCostModel)
        
        # Test trade execution with transaction costs
        success = portfolio_manager.execute_trade(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            side="long",
        )
        
        assert success
        assert len(portfolio_manager.transaction_history) > 0
        
        # Test cost analysis
        cost_analysis = portfolio_manager.get_transaction_cost_analysis()
        assert isinstance(cost_analysis, dict)
        assert "cost_summary" in cost_analysis
        
        # Test optimization
        optimization = portfolio_manager.optimize_transaction_costs()
        assert isinstance(optimization, dict)
        assert "recommendations" in optimization


if __name__ == "__main__":
    pytest.main([__file__, "-v"])