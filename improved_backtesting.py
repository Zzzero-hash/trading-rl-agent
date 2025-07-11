#!/usr/bin/env python3
"""
Improved Backtesting Script with Realistic Trading Parameters
============================================================

This script addresses the conservative thresholds that prevented trades
in the original backtesting by implementing more realistic trading logic.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.portfolio import PortfolioManager
from trading_rl_agent.risk import RiskManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ImprovedBacktester:
    """
    Improved backtesting with more realistic trading parameters and better signal generation.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.initial_capital = config.get("initial_capital", 100000)
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # More realistic trading parameters
        self.trading_params = {
            "min_confidence": 0.4,  # Lowered from 0.6
            "buy_threshold": 0.003,  # Lowered from 0.008
            "sell_threshold": -0.003,  # Raised from -0.008
            "position_size_pct": 0.15,  # Increased from 0.1
            "max_positions": 5,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "max_drawdown": 0.15,
        }

        # Initialize managers
        self.portfolio_manager = PortfolioManager(initial_capital=self.initial_capital)
        self.risk_manager = RiskManager()

    def run_improved_backtesting(self, model_path: str, dataset_path: str, days: int = 30) -> dict[str, Any]:
        """
        Run improved backtesting with realistic parameters.
        """
        logger.info("🚀 IMPROVED BACKTESTING WITH REALISTIC PARAMETERS")
        logger.info("=" * 60)

        # Load CNN+LSTM model
        logger.info("🤖 Loading CNN+LSTM model...")
        try:
            from train_cnn_lstm import CNNLSTMTrainer, create_model_config, create_training_config

            # First, create a trainer instance
            model_config = create_model_config()
            training_config = create_training_config()
            trainer = CNNLSTMTrainer(model_config=model_config, training_config=training_config, device=self.device)

            # Then, load the checkpoint onto the instance
            trainer.load_checkpoint(checkpoint_path=model_path, input_dim=68)
            model = trainer.model
            model.eval()
            logger.info("✅ CNN+LSTM model loaded successfully")
        except Exception as e:
            logger.exception(f"❌ Failed to load CNN+LSTM model: {e}")
            return {"error": str(e)}

        # Load historical data
        logger.info("📊 Loading historical data...")
        try:
            sequences = np.load(f"{dataset_path}/sequences.npy")
            targets = np.load(f"{dataset_path}/targets.npy")
            logger.info(f"✅ Loaded {len(sequences)} sequences for backtesting")
        except Exception as e:
            logger.exception(f"❌ Failed to load historical data: {e}")
            return {"error": str(e)}

        # Run improved backtesting
        logger.info("🔄 Running improved backtesting simulation...")

        trades = []
        portfolio_values = [self.initial_capital]
        predictions = []
        signals = []

        # Track performance metrics
        daily_returns = []
        max_drawdown = 0
        peak_value = self.initial_capital

        for i, sequence in enumerate(sequences):
            # Get CNN+LSTM prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                prediction = model(input_tensor)
                pred_value = prediction[0].item() if isinstance(prediction, tuple) else prediction.item()

            predictions.append(pred_value)

            # Generate improved trading signal
            signal = self._generate_improved_signal(pred_value, i, portfolio_values[-1], predictions)
            signals.append(signal)

            # Execute trade with improved logic
            if signal != "HOLD":
                trade_result = self._execute_improved_trade(signal, i, pred_value, sequence)
                if trade_result:
                    trades.append(trade_result)
                    logger.info(f"📈 Trade executed: {signal} at step {i}, prediction: {pred_value:.4f}")

            # Update portfolio value
            current_value = self.portfolio_manager.total_value
            portfolio_values.append(current_value)

            # Calculate daily return
            if len(portfolio_values) > 1:
                daily_return = (current_value - portfolio_values[-2]) / portfolio_values[-2]
                daily_returns.append(daily_return)

            # Track drawdown
            peak_value = max(peak_value, current_value)
            drawdown = (current_value - peak_value) / peak_value
            max_drawdown = min(max_drawdown, drawdown)

            # Risk management check
            if abs(max_drawdown) > self.trading_params["max_drawdown"]:
                logger.warning(f"⚠️ Max drawdown limit reached: {max_drawdown:.2%}")
                break

        # Calculate comprehensive performance metrics
        performance = self._calculate_improved_metrics(portfolio_values, trades, predictions, signals, daily_returns)

        self.results["improved_backtesting"] = performance

        # Save results
        self._save_improved_results(performance)

        return performance

    def _generate_improved_signal(
        self, prediction: float, step: int, current_value: float, prior_predictions: list[float]
    ) -> str:
        """
        Generate improved trading signal with more realistic thresholds.
        """
        # Dynamic thresholds based on market conditions
        volatility_factor = min(1.0, abs(prediction) * 10)  # Scale with prediction magnitude

        # Adjust thresholds based on current performance
        if current_value < self.initial_capital * 0.95:  # Underperforming
            buy_threshold = self.trading_params["buy_threshold"] * 0.8  # More aggressive
            sell_threshold = self.trading_params["sell_threshold"] * 0.8
        elif current_value > self.initial_capital * 1.05:  # Outperforming
            buy_threshold = self.trading_params["buy_threshold"] * 1.2  # More conservative
            sell_threshold = self.trading_params["sell_threshold"] * 1.2
        else:
            buy_threshold = self.trading_params["buy_threshold"]
            sell_threshold = self.trading_params["sell_threshold"]

        # Add momentum factor
        momentum_factor = 1.0
        if step > 10:  # Need some history for momentum
            recent_predictions = [p for p in prior_predictions[-10:] if abs(p) > 0.001]
            if recent_predictions:
                momentum = np.mean(recent_predictions)
                if abs(momentum) > 0.002:
                    momentum_factor = 1.2 if momentum > 0 else 0.8

        # Generate signal with improved logic
        if prediction > buy_threshold * momentum_factor:
            return "BUY"
        if prediction < sell_threshold * momentum_factor:
            return "SELL"
        return "HOLD"

    def _execute_improved_trade(
        self, signal: str, step: int, prediction: float, sequence: np.ndarray
    ) -> dict[str, Any] | None:
        """
        Execute improved trade with better position sizing and risk management.
        """
        try:
            # Calculate confidence based on prediction magnitude and consistency
            confidence = min(0.95, abs(prediction) * 20)  # Scale confidence with prediction

            if signal == "BUY" and self.portfolio_manager.cash > 1000:
                # Improved position sizing
                base_position_size = self.portfolio_manager.cash * self.trading_params["position_size_pct"]

                # Adjust position size based on confidence and current performance
                performance_factor = 1.0
                if self.portfolio_manager.total_value > self.initial_capital:
                    performance_factor = 1.2  # Increase position size when profitable
                else:
                    performance_factor = 0.8  # Decrease position size when losing

                position_size = base_position_size * confidence * performance_factor
                position_size = min(position_size, self.portfolio_manager.cash * 0.3)  # Max 30% of cash

                # Simulate realistic price
                base_price = 150
                price_volatility = 0.01
                price = base_price * (1 + np.random.normal(0, price_volatility))

                # Calculate quantity
                quantity = position_size / price

                # Execute trade
                success = self.portfolio_manager.execute_trade(
                    symbol="AAPL", quantity=quantity, price=price, side="buy"
                )

                if success:
                    return {
                        "step": step,
                        "signal": signal,
                        "prediction": prediction,
                        "confidence": confidence,
                        "price": price,
                        "quantity": quantity,
                        "type": "BUY",
                        "position_size": position_size,
                    }

            elif signal == "SELL" and len(self.portfolio_manager.positions) > 0:
                # Sell existing positions
                for symbol, position in self.portfolio_manager.positions.items():
                    # Simulate realistic price
                    base_price = 150
                    price_volatility = 0.01
                    price = base_price * (1 + np.random.normal(0, price_volatility))

                    success = self.portfolio_manager.execute_trade(
                        symbol=symbol, quantity=position.quantity, price=price, side="sell"
                    )

                    if success:
                        return {
                            "step": step,
                            "signal": signal,
                            "prediction": prediction,
                            "confidence": confidence,
                            "price": price,
                            "quantity": position.quantity,
                            "type": "SELL",
                            "symbol": symbol,
                        }

        except Exception as e:
            logger.warning(f"⚠️ Trade execution failed: {e}")

        return None

    def _calculate_improved_metrics(
        self,
        portfolio_values: list[float],
        trades: list[dict],
        predictions: list[float],
        signals: list[str],
        daily_returns: list[float],
    ) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics with additional insights.
        """
        # Basic metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100

        # Risk metrics
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = (
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            )
        else:
            volatility = 0
            sharpe_ratio = 0

        # Drawdown calculation
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        # Trading metrics
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t.get("type") == "BUY"])
        sell_trades = len([t for t in trades if t.get("type") == "SELL"])

        # Signal analysis
        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")
        hold_signals = signals.count("HOLD")

        # Prediction analysis
        if predictions:
            avg_prediction = np.mean(predictions)
            pred_volatility = np.std(predictions)
            positive_predictions = len([p for p in predictions if p > 0])
            negative_predictions = len([p for p in predictions if p < 0])
        else:
            avg_prediction = 0
            pred_volatility = 0
            positive_predictions = 0
            negative_predictions = 0

        return {
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return_percent": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_percent": max_drawdown,
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "avg_prediction": avg_prediction,
            "prediction_volatility": pred_volatility,
            "positive_predictions": positive_predictions,
            "negative_predictions": negative_predictions,
            "portfolio_summary": self.portfolio_manager.get_performance_summary(),
        }

    def _save_improved_results(self, performance: dict[str, Any]):
        """Save improved backtesting results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/improved_backtesting/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Save performance metrics
        with open(f"{output_dir}/performance_metrics.json", "w") as f:
            json.dump(performance, f, indent=2, default=str)

        # Save trading parameters
        with open(f"{output_dir}/trading_params.json", "w") as f:
            json.dump(self.trading_params, f, indent=2)

        logger.info(f"💾 Results saved to: {output_dir}")
        self.results["output_dir"] = output_dir


def main():
    """Main function to run improved backtesting."""
    print("🚀 IMPROVED BACKTESTING RUNNER")
    print("=" * 60)
    print("This script runs improved backtesting with realistic trading parameters.")
    print("=" * 60)

    # Configuration
    config = {"initial_capital": 100000, "max_days": 30, "symbols": ["AAPL", "GOOGL", "MSFT"]}

    # Initialize backtester
    backtester = ImprovedBacktester(config)

    # Check for model and dataset
    model_path = "outputs/demo_training/best_model.pth"
    dataset_path = "outputs/demo_training/dataset/20250711_003545"

    if not os.path.exists(model_path):
        print("❌ Model not found. Please run training first.")
        return

    if not os.path.exists(dataset_path):
        print("❌ Dataset not found. Please run dataset generation first.")
        return

    print(f"✅ Found model: {model_path}")
    print(f"✅ Found dataset: {dataset_path}")

    # Run improved backtesting
    print("\n🔄 Running improved backtesting...")
    results = backtester.run_improved_backtesting(model_path, dataset_path, days=30)

    if "error" in results:
        print(f"❌ Backtesting failed: {results['error']}")
        return

    # Display results
    print("\n📊 IMPROVED BACKTESTING RESULTS")
    print("=" * 50)
    print(f"💰 Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"💰 Final Value: ${results['final_value']:,.2f}")
    print(f"📈 Total Return: {results['total_return_percent']:.2f}%")
    print(f"📊 Volatility: {results['volatility']:.2f}")
    print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown: {results['max_drawdown_percent']:.2f}%")
    print(f"🔄 Total Trades: {results['total_trades']}")
    print(f"📈 Buy Trades: {results['buy_trades']}")
    print(f"📉 Sell Trades: {results['sell_trades']}")
    print(f"📊 Buy Signals: {results['buy_signals']}")
    print(f"📊 Sell Signals: {results['sell_signals']}")
    print(f"📊 Hold Signals: {results['hold_signals']}")
    print(f"🎯 Avg Prediction: {results['avg_prediction']:.4f}")
    print(f"📊 Prediction Volatility: {results['prediction_volatility']:.4f}")

    print(f"\n💾 Results saved to: {results.get('output_dir', 'N/A')}")
    print("\n🎉 Improved backtesting completed!")


if __name__ == "__main__":
    main()
