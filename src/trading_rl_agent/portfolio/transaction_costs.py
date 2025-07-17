"""
Comprehensive Transaction Cost Modeling System

This module provides realistic transaction cost modeling for backtesting,
including bid-ask spreads, market impact, commission structures, slippage,
execution delays, and cost optimization recommendations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..core.logging import get_logger

logger = get_logger(__name__)


class MarketCondition(Enum):
    """Market condition types affecting transaction costs."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    LIQUID = "liquid"
    ILLIQUID = "illiquid"
    CRISIS = "crisis"


class OrderType(Enum):
    """Order types with different cost implications."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class BrokerType(Enum):
    """Broker types with different commission structures."""
    RETAIL = "retail"
    INSTITUTIONAL = "institutional"
    DISCOUNT = "discount"
    PREMIUM = "premium"
    CRYPTO = "crypto"


@dataclass
class MarketData:
    """Market data snapshot for cost modeling."""
    timestamp: datetime
    bid: float
    ask: float
    mid_price: float
    volume: float
    volatility: float
    avg_daily_volume: float
    market_cap: float
    sector: str = ""
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return (self.ask - self.bid) / self.mid_price
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        return self.spread * 10000


@dataclass
class ExecutionResult:
    """Result of order execution simulation."""
    executed_quantity: float
    executed_price: float
    execution_time: datetime
    partial_fills: List[Tuple[float, float, datetime]]  # quantity, price, time
    total_cost: float
    cost_breakdown: Dict[str, float]
    market_impact: float
    slippage: float
    delay_seconds: float
    success: bool = True
    reason: str = ""


@dataclass
class CostOptimizationRecommendation:
    """Recommendation for cost optimization."""
    recommendation_type: str
    description: str
    expected_savings: float
    confidence: float
    implementation_difficulty: str  # "easy", "medium", "hard"
    priority: str  # "low", "medium", "high", "critical"


class CommissionStructure(ABC):
    """Abstract base class for commission structures."""
    
    @abstractmethod
    def calculate_commission(self, trade_value: float, quantity: float) -> float:
        """Calculate commission for a trade."""
        pass


@dataclass
class FlatRateCommission(CommissionStructure):
    """Flat rate commission structure."""
    rate: float = 0.001  # 0.1%
    min_commission: float = 1.0
    max_commission: float = 1000.0
    
    def calculate_commission(self, trade_value: float, quantity: float) -> float:
        """Calculate flat rate commission."""
        commission = trade_value * self.rate
        return max(self.min_commission, min(commission, self.max_commission))


@dataclass
class TieredCommission(CommissionStructure):
    """Tiered commission structure."""
    tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (10000, 0.002),   # 0.2% for trades up to $10k
        (100000, 0.001),  # 0.1% for trades up to $100k
        (float('inf'), 0.0005)  # 0.05% for trades above $100k
    ])
    min_commission: float = 1.0
    
    def calculate_commission(self, trade_value: float, quantity: float) -> float:
        """Calculate tiered commission."""
        commission = 0.0
        remaining_value = trade_value
        
        for tier_limit, rate in self.tiers:
            if remaining_value <= 0:
                break
            tier_amount = min(remaining_value, tier_limit)
            commission += tier_amount * rate
            remaining_value -= tier_amount
        
        return max(self.min_commission, commission)


@dataclass
class PerShareCommission(CommissionStructure):
    """Per-share commission structure."""
    rate_per_share: float = 0.005  # $0.005 per share
    min_commission: float = 1.0
    max_commission: float = 1000.0
    
    def calculate_commission(self, trade_value: float, quantity: float) -> float:
        """Calculate per-share commission."""
        commission = abs(quantity) * self.rate_per_share
        return max(self.min_commission, min(commission, self.max_commission))


class MarketImpactModel(ABC):
    """Abstract base class for market impact models."""
    
    @abstractmethod
    def calculate_impact(self, order_size: float, market_data: MarketData) -> float:
        """Calculate market impact of an order."""
        pass


@dataclass
class LinearImpactModel(MarketImpactModel):
    """Linear market impact model."""
    impact_rate: float = 0.0001  # 0.01% per $1M traded
    
    def calculate_impact(self, order_size: float, market_data: MarketData) -> float:
        """Calculate linear market impact."""
        volume_ratio = order_size / market_data.avg_daily_volume if market_data.avg_daily_volume > 0 else 0
        return order_size * self.impact_rate * volume_ratio


@dataclass
class SquareRootImpactModel(MarketImpactModel):
    """Square root market impact model (Almgren et al.)."""
    impact_rate: float = 0.00005  # 0.005% per $1M traded
    
    def calculate_impact(self, order_size: float, market_data: MarketData) -> float:
        """Calculate square root market impact."""
        volume_ratio = order_size / market_data.avg_daily_volume if market_data.avg_daily_volume > 0 else 0
        return order_size * self.impact_rate * np.sqrt(volume_ratio)


@dataclass
class AdaptiveImpactModel(MarketImpactModel):
    """Adaptive market impact model based on market conditions."""
    base_impact_rate: float = 0.00005
    volatility_multiplier: float = 2.0
    liquidity_multiplier: float = 1.5
    
    def calculate_impact(self, order_size: float, market_data: MarketData) -> float:
        """Calculate adaptive market impact."""
        volume_ratio = order_size / market_data.avg_daily_volume if market_data.avg_daily_volume > 0 else 0
        
        # Adjust for volatility
        volatility_factor = 1.0 + (market_data.volatility * self.volatility_multiplier)
        
        # Adjust for liquidity (inverse relationship)
        liquidity_factor = 1.0 + (1.0 / (volume_ratio + 0.01)) * self.liquidity_multiplier
        
        impact = order_size * self.base_impact_rate * np.sqrt(volume_ratio) * volatility_factor * liquidity_factor
        return impact


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate slippage for an order."""
        pass


@dataclass
class ConstantSlippageModel(SlippageModel):
    """Constant slippage model."""
    slippage_rate: float = 0.0001  # 0.01%
    
    def calculate_slippage(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate constant slippage."""
        return order_size * self.slippage_rate


@dataclass
class VolumeBasedSlippageModel(SlippageModel):
    """Volume-based slippage model."""
    base_slippage_rate: float = 0.00005
    volume_exponent: float = 0.5
    
    def calculate_slippage(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate volume-based slippage."""
        volume_ratio = order_size / market_data.volume if market_data.volume > 0 else 0
        slippage_rate = self.base_slippage_rate * (volume_ratio ** self.volume_exponent)
        return order_size * slippage_rate


@dataclass
class SpreadBasedSlippageModel(SlippageModel):
    """Spread-based slippage model."""
    spread_multiplier: float = 0.5  # Assume 50% of spread as slippage
    
    def calculate_slippage(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate spread-based slippage."""
        return order_size * market_data.spread * self.spread_multiplier


class ExecutionDelayModel(ABC):
    """Abstract base class for execution delay models."""
    
    @abstractmethod
    def calculate_delay(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate execution delay in seconds."""
        pass


@dataclass
class ConstantDelayModel(ExecutionDelayModel):
    """Constant execution delay model."""
    delay_seconds: float = 1.0
    
    def calculate_delay(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate constant delay."""
        return self.delay_seconds


@dataclass
class SizeBasedDelayModel(ExecutionDelayModel):
    """Size-based execution delay model."""
    base_delay: float = 0.5
    size_multiplier: float = 0.1
    
    def calculate_delay(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate size-based delay."""
        volume_ratio = order_size / market_data.avg_daily_volume if market_data.avg_daily_volume > 0 else 0
        return self.base_delay + (volume_ratio * self.size_multiplier * 3600)  # Convert to seconds


@dataclass
class MarketConditionDelayModel(ExecutionDelayModel):
    """Market condition-based delay model."""
    base_delay: float = 0.5
    volatility_multiplier: float = 2.0
    liquidity_multiplier: float = 1.5
    
    def calculate_delay(self, order_size: float, market_data: MarketData, order_type: OrderType) -> float:
        """Calculate market condition-based delay."""
        volume_ratio = order_size / market_data.avg_daily_volume if market_data.avg_daily_volume > 0 else 0
        
        # Adjust for volatility
        volatility_factor = 1.0 + (market_data.volatility * self.volatility_multiplier)
        
        # Adjust for liquidity
        liquidity_factor = 1.0 + (1.0 / (volume_ratio + 0.01)) * self.liquidity_multiplier
        
        delay = self.base_delay * volatility_factor * liquidity_factor
        
        # Adjust for order type
        if order_type == OrderType.MARKET:
            delay *= 0.5  # Market orders execute faster
        elif order_type == OrderType.LIMIT:
            delay *= 1.5  # Limit orders may take longer
        elif order_type in [OrderType.TWAP, OrderType.VWAP]:
            delay *= 3.0  # Algorithmic orders take longer
        
        return delay


@dataclass
class PartialFillModel:
    """Model for simulating partial fills."""
    min_fill_ratio: float = 0.8
    max_fill_ratio: float = 1.0
    fill_probability: float = 0.9
    
    def simulate_fill(self, requested_quantity: float, market_data: MarketData) -> Tuple[float, List[Tuple[float, float, datetime]]]:
        """Simulate partial fills for an order."""
        if np.random.random() > self.fill_probability:
            return 0.0, []  # No fill
        
        # Determine fill ratio
        fill_ratio = np.random.uniform(self.min_fill_ratio, self.max_fill_ratio)
        filled_quantity = requested_quantity * fill_ratio
        
        if filled_quantity == requested_quantity:
            # Full fill
            return filled_quantity, [(filled_quantity, market_data.mid_price, datetime.now())]
        
        # Partial fill - simulate multiple fills
        fills = []
        remaining = filled_quantity
        current_time = datetime.now()
        
        while remaining > 0:
            # Simulate individual fill
            fill_size = min(remaining, np.random.uniform(0.1, 0.3) * filled_quantity)
            fill_price = market_data.mid_price * (1 + np.random.normal(0, market_data.spread * 0.5))
            fill_time = current_time + timedelta(seconds=np.random.uniform(0, 60))
            
            fills.append((fill_size, fill_price, fill_time))
            remaining -= fill_size
            current_time = fill_time
        
        return filled_quantity, fills


@dataclass
class TransactionCostModel:
    """
    Comprehensive transaction cost model for realistic backtesting.
    
    Features:
    - Configurable commission structures for different brokers
    - Realistic bid-ask spread modeling
    - Market impact modeling based on order size and market conditions
    - Slippage modeling with multiple approaches
    - Execution delay simulation
    - Partial fill simulation
    - Cost optimization recommendations
    """
    
    # Commission structure
    commission_structure: CommissionStructure = field(default_factory=FlatRateCommission)
    
    # Market impact model
    market_impact_model: MarketImpactModel = field(default_factory=LinearImpactModel)
    
    # Slippage model
    slippage_model: SlippageModel = field(default_factory=ConstantSlippageModel)
    
    # Execution delay model
    delay_model: ExecutionDelayModel = field(default_factory=ConstantDelayModel)
    
    # Partial fill model
    partial_fill_model: PartialFillModel = field(default_factory=PartialFillModel)
    
    # Market condition adjustments
    market_condition_multipliers: Dict[MarketCondition, float] = field(default_factory=lambda: {
        MarketCondition.NORMAL: 1.0,
        MarketCondition.VOLATILE: 1.5,
        MarketCondition.LIQUID: 0.8,
        MarketCondition.ILLIQUID: 1.8,
        MarketCondition.CRISIS: 2.5,
    })
    
    # Cost tracking
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_market_impact: float = 0.0
    total_spread_cost: float = 0.0
    total_delays: float = 0.0
    num_trades: int = 0
    
    def __post_init__(self):
        """Initialize the cost model."""
        self.logger = get_logger(self.__class__.__name__)
        self.trade_history: List[Dict[str, Any]] = []
    
    def calculate_total_cost(
        self,
        trade_value: float,
        quantity: float,
        market_data: MarketData,
        order_type: OrderType = OrderType.MARKET,
        market_condition: MarketCondition = MarketCondition.NORMAL,
    ) -> Dict[str, float]:
        """
        Calculate total transaction costs for a trade.
        
        Args:
            trade_value: Dollar value of the trade
            quantity: Number of shares/units traded
            market_data: Current market data
            order_type: Type of order
            market_condition: Current market condition
            
        Returns:
            Dictionary with detailed cost breakdown
        """
        # Apply market condition multiplier
        condition_multiplier = self.market_condition_multipliers.get(market_condition, 1.0)
        
        # Commission
        commission = self.commission_structure.calculate_commission(trade_value, quantity)
        
        # Market impact
        market_impact = self.market_impact_model.calculate_impact(trade_value, market_data)
        market_impact *= condition_multiplier
        
        # Slippage
        slippage = self.slippage_model.calculate_slippage(trade_value, market_data, order_type)
        slippage *= condition_multiplier
        
        # Bid-ask spread cost
        spread_cost = trade_value * market_data.spread
        
        # Execution delay
        delay = self.delay_model.calculate_delay(trade_value, market_data, order_type)
        
        # Total cost
        total_cost = commission + slippage + market_impact + spread_cost
        
        cost_breakdown = {
            "commission": commission,
            "slippage": slippage,
            "market_impact": market_impact,
            "spread_cost": spread_cost,
            "total_cost": total_cost,
            "cost_pct": total_cost / trade_value if trade_value > 0 else 0,
            "delay_seconds": delay,
            "condition_multiplier": condition_multiplier,
        }
        
        return cost_breakdown
    
    def simulate_execution(
        self,
        requested_quantity: float,
        market_data: MarketData,
        order_type: OrderType = OrderType.MARKET,
        market_condition: MarketCondition = MarketCondition.NORMAL,
    ) -> ExecutionResult:
        """
        Simulate order execution with realistic delays and partial fills.
        
        Args:
            requested_quantity: Requested quantity to trade
            market_data: Current market data
            order_type: Type of order
            market_condition: Current market condition
            
        Returns:
            Execution result with detailed information
        """
        # Calculate costs
        trade_value = abs(requested_quantity * market_data.mid_price)
        cost_breakdown = self.calculate_total_cost(
            trade_value, requested_quantity, market_data, order_type, market_condition
        )
        
        # Simulate execution delay
        delay = cost_breakdown["delay_seconds"]
        execution_time = datetime.now() + timedelta(seconds=delay)
        
        # Simulate partial fills
        filled_quantity, partial_fills = self.partial_fill_model.simulate_fill(
            requested_quantity, market_data
        )
        
        # Calculate executed price (weighted average of fills)
        if partial_fills:
            total_value = sum(qty * price for qty, price, _ in partial_fills)
            executed_price = total_value / filled_quantity if filled_quantity > 0 else market_data.mid_price
        else:
            executed_price = market_data.mid_price
            filled_quantity = 0
        
        # Update tracking
        self.total_commission += cost_breakdown["commission"]
        self.total_slippage += cost_breakdown["slippage"]
        self.total_market_impact += cost_breakdown["market_impact"]
        self.total_spread_cost += cost_breakdown["spread_cost"]
        self.total_delays += delay
        self.num_trades += 1
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now(),
            "requested_quantity": requested_quantity,
            "filled_quantity": filled_quantity,
            "executed_price": executed_price,
            "market_data": market_data,
            "order_type": order_type,
            "market_condition": market_condition,
            "cost_breakdown": cost_breakdown,
            "partial_fills": partial_fills,
            "delay_seconds": delay,
        }
        self.trade_history.append(trade_record)
        
        return ExecutionResult(
            executed_quantity=filled_quantity,
            executed_price=executed_price,
            execution_time=execution_time,
            partial_fills=partial_fills,
            total_cost=cost_breakdown["total_cost"],
            cost_breakdown=cost_breakdown,
            market_impact=cost_breakdown["market_impact"],
            slippage=cost_breakdown["slippage"],
            delay_seconds=delay,
            success=filled_quantity > 0,
            reason="Partial fill" if filled_quantity < requested_quantity else "Full fill",
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of all transaction costs."""
        total_costs = (
            self.total_commission + 
            self.total_slippage + 
            self.total_market_impact + 
            self.total_spread_cost
        )
        
        return {
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_market_impact": self.total_market_impact,
            "total_spread_cost": self.total_spread_cost,
            "total_transaction_costs": total_costs,
            "total_delays": self.total_delays,
            "num_trades": self.num_trades,
            "avg_cost_per_trade": total_costs / self.num_trades if self.num_trades > 0 else 0,
            "avg_delay_per_trade": self.total_delays / self.num_trades if self.num_trades > 0 else 0,
        }
    
    def generate_optimization_recommendations(self) -> List[CostOptimizationRecommendation]:
        """
        Generate cost optimization recommendations based on trading history.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if not self.trade_history:
            return recommendations
        
        # Analyze cost components
        total_costs = sum(trade["cost_breakdown"]["total_cost"] for trade in self.trade_history)
        avg_cost = total_costs / len(self.trade_history)
        
        # Commission analysis
        total_commission = sum(trade["cost_breakdown"]["commission"] for trade in self.trade_history)
        commission_pct = total_commission / total_costs if total_costs > 0 else 0
        
        if commission_pct > 0.4:  # Commission is more than 40% of total costs
            recommendations.append(CostOptimizationRecommendation(
                recommendation_type="commission_optimization",
                description="Consider switching to a lower-commission broker or tiered pricing",
                expected_savings=total_commission * 0.3,  # Assume 30% savings
                confidence=0.8,
                implementation_difficulty="medium",
                priority="high" if commission_pct > 0.6 else "medium"
            ))
        
        # Slippage analysis
        total_slippage = sum(trade["cost_breakdown"]["slippage"] for trade in self.trade_history)
        slippage_pct = total_slippage / total_costs if total_costs > 0 else 0
        
        if slippage_pct > 0.3:
            recommendations.append(CostOptimizationRecommendation(
                recommendation_type="slippage_reduction",
                description="Consider using limit orders and breaking large orders into smaller chunks",
                expected_savings=total_slippage * 0.4,
                confidence=0.7,
                implementation_difficulty="hard",
                priority="high" if slippage_pct > 0.5 else "medium"
            ))
        
        # Market impact analysis
        total_impact = sum(trade["cost_breakdown"]["market_impact"] for trade in self.trade_history)
        impact_pct = total_impact / total_costs if total_costs > 0 else 0
        
        if impact_pct > 0.25:
            recommendations.append(CostOptimizationRecommendation(
                recommendation_type="market_impact_reduction",
                description="Implement algorithmic trading strategies (TWAP/VWAP) for large orders",
                expected_savings=total_impact * 0.5,
                confidence=0.8,
                implementation_difficulty="hard",
                priority="high" if impact_pct > 0.4 else "medium"
            ))
        
        # Delay analysis
        avg_delay = sum(trade["delay_seconds"] for trade in self.trade_history) / len(self.trade_history)
        if avg_delay > 5.0:  # Average delay > 5 seconds
            recommendations.append(CostOptimizationRecommendation(
                recommendation_type="execution_speed",
                description="Upgrade to faster execution infrastructure or use market orders",
                expected_savings=total_costs * 0.1,  # Assume 10% savings from faster execution
                confidence=0.6,
                implementation_difficulty="medium",
                priority="medium"
            ))
        
        # Order size optimization
        large_orders = [trade for trade in self.trade_history 
                       if trade["requested_quantity"] > 10000]  # Arbitrary threshold
        if len(large_orders) > len(self.trade_history) * 0.2:  # More than 20% are large orders
            recommendations.append(CostOptimizationRecommendation(
                recommendation_type="order_size_optimization",
                description="Break large orders into smaller chunks to reduce market impact",
                expected_savings=total_costs * 0.15,
                confidence=0.7,
                implementation_difficulty="medium",
                priority="medium"
            ))
        
        return recommendations
    
    def reset(self) -> None:
        """Reset all cost tracking."""
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_market_impact = 0.0
        self.total_spread_cost = 0.0
        self.total_delays = 0.0
        self.num_trades = 0
        self.trade_history.clear()
    
    @classmethod
    def create_broker_model(cls, broker_type: BrokerType) -> "TransactionCostModel":
        """
        Create a transaction cost model configured for a specific broker type.
        
        Args:
            broker_type: Type of broker
            
        Returns:
            Configured transaction cost model
        """
        if broker_type == BrokerType.RETAIL:
            return cls(
                commission_structure=FlatRateCommission(rate=0.002, min_commission=5.0),
                slippage_model=ConstantSlippageModel(slippage_rate=0.0002),
                delay_model=ConstantDelayModel(delay_seconds=2.0),
            )
        elif broker_type == BrokerType.INSTITUTIONAL:
            return cls(
                commission_structure=TieredCommission(),
                slippage_model=VolumeBasedSlippageModel(),
                delay_model=SizeBasedDelayModel(),
            )
        elif broker_type == BrokerType.DISCOUNT:
            return cls(
                commission_structure=PerShareCommission(rate_per_share=0.001),
                slippage_model=SpreadBasedSlippageModel(),
                delay_model=ConstantDelayModel(delay_seconds=1.5),
            )
        elif broker_type == BrokerType.PREMIUM:
            return cls(
                commission_structure=FlatRateCommission(rate=0.0005, min_commission=10.0),
                market_impact_model=AdaptiveImpactModel(),
                slippage_model=VolumeBasedSlippageModel(),
                delay_model=MarketConditionDelayModel(),
            )
        elif broker_type == BrokerType.CRYPTO:
            return cls(
                commission_structure=FlatRateCommission(rate=0.001, min_commission=0.1),
                slippage_model=SpreadBasedSlippageModel(spread_multiplier=0.3),
                delay_model=ConstantDelayModel(delay_seconds=0.1),
            )
        else:
            return cls()  # Default model


class TransactionCostAnalyzer:
    """
    Advanced transaction cost analysis and optimization tools.
    """
    
    def __init__(self, cost_model: TransactionCostModel):
        """Initialize the analyzer with a cost model."""
        self.cost_model = cost_model
        self.logger = get_logger(self.__class__.__name__)
    
    def analyze_cost_trends(self) -> Dict[str, Any]:
        """Analyze cost trends over time."""
        if not self.cost_model.trade_history:
            return {}
        
        df = pd.DataFrame(self.cost_model.trade_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate rolling averages
        window = min(20, len(df))
        if window > 0:
            df['rolling_avg_cost'] = df['cost_breakdown'].apply(lambda x: x['total_cost']).rolling(window).mean()
            df['rolling_avg_delay'] = df['delay_seconds'].rolling(window).mean()
        
        return {
            "cost_trend": df['rolling_avg_cost'].tolist() if 'rolling_avg_cost' in df.columns else [],
            "delay_trend": df['rolling_avg_delay'].tolist() if 'rolling_avg_delay' in df.columns else [],
            "timestamps": df.index.tolist(),
        }
    
    def calculate_cost_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate cost efficiency metrics."""
        if not self.cost_model.trade_history:
            return {}
        
        total_value = sum(
            trade["requested_quantity"] * trade["market_data"].mid_price 
            for trade in self.cost_model.trade_history
        )
        
        total_costs = sum(
            trade["cost_breakdown"]["total_cost"] 
            for trade in self.cost_model.trade_history
        )
        
        return {
            "cost_efficiency_ratio": total_costs / total_value if total_value > 0 else 0,
            "avg_cost_per_dollar": total_costs / total_value if total_value > 0 else 0,
            "cost_per_trade": total_costs / len(self.cost_model.trade_history),
            "fill_rate": sum(1 for trade in self.cost_model.trade_history 
                           if trade["filled_quantity"] > 0) / len(self.cost_model.trade_history),
        }
    
    def generate_cost_report(self) -> str:
        """Generate a comprehensive cost analysis report."""
        summary = self.cost_model.get_cost_summary()
        efficiency_metrics = self.calculate_cost_efficiency_metrics()
        recommendations = self.cost_model.generate_optimization_recommendations()
        
        report = f"""
Transaction Cost Analysis Report
================================

Summary Statistics:
------------------
Total Trades: {summary['num_trades']}
Total Transaction Costs: ${summary['total_transaction_costs']:,.2f}
Average Cost per Trade: ${summary['avg_cost_per_trade']:,.2f}
Average Delay per Trade: {summary['avg_delay_per_trade']:.2f} seconds

Cost Breakdown:
--------------
Commission: ${summary['total_commission']:,.2f} ({(summary['total_commission']/summary['total_transaction_costs']*100):.1f}%)
Slippage: ${summary['total_slippage']:,.2f} ({(summary['total_slippage']/summary['total_transaction_costs']*100):.1f}%)
Market Impact: ${summary['total_market_impact']:,.2f} ({(summary['total_market_impact']/summary['total_transaction_costs']*100):.1f}%)
Spread Cost: ${summary['total_spread_cost']:,.2f} ({(summary['total_spread_cost']/summary['total_transaction_costs']*100):.1f}%)

Efficiency Metrics:
------------------
Cost Efficiency Ratio: {efficiency_metrics.get('cost_efficiency_ratio', 0):.4f}
Average Cost per Dollar: {efficiency_metrics.get('avg_cost_per_dollar', 0):.4f}
Fill Rate: {efficiency_metrics.get('fill_rate', 0):.1%}

Optimization Recommendations:
---------------------------
"""
        
        for i, rec in enumerate(recommendations, 1):
            report += f"""
{i}. {rec.recommendation_type.replace('_', ' ').title()}
   Description: {rec.description}
   Expected Savings: ${rec.expected_savings:,.2f}
   Confidence: {rec.confidence:.1%}
   Priority: {rec.priority.upper()}
   Implementation: {rec.implementation_difficulty}
"""
        
        return report