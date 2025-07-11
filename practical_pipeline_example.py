#!/usr/bin/env python3
"""
Practical Example: Complete Pipeline for Live Trading

This script demonstrates how to use your complete pipeline to:
1. Generate robust, diverse datasets
2. Train CNN+LSTM models
3. Integrate with RL environments
4. Prepare for live trading

Run this script to create a production-ready trading system.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trading_rl_agent.data.robust_dataset_builder import DatasetConfig, RobustDatasetBuilder
from trading_rl_agent.portfolio import PortfolioManager
from trading_rl_agent.risk import RiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_production_dataset():
    """Create a production-ready dataset for live trading."""

    print("🚀 Creating Production Dataset")
    print("=" * 50)

    # Production dataset configuration
    config = DatasetConfig(
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        timeframe="1d",
        real_data_ratio=0.9,  # High real data ratio for production
        min_samples_per_symbol=1500,
        sequence_length=60,
        prediction_horizon=1,
        overlap_ratio=0.8,
        technical_indicators=True,
        sentiment_features=True,
        market_regime_features=True,
        output_dir="data/production_dataset",
        version_tag="production_v1",
    )

    print("📊 Dataset Configuration:")
    print(f"  Symbols: {config.symbols}")
    print(f"  Date Range: {config.start_date} to {config.end_date}")
    print(f"  Real Data Ratio: {config.real_data_ratio:.1%}")
    print(f"  Sequence Length: {config.sequence_length}")

    # Build dataset
    print("\n🔧 Building dataset...")
    builder = RobustDatasetBuilder(config)
    sequences, targets, dataset_info = builder.build_dataset()

    # Dataset analysis
    print("\n📈 Dataset Analysis:")
    print(f"  Total Sequences: {len(sequences):,}")
    print(f"  Features per Timestep: {sequences.shape[-1]}")
    print(f"  Target Correlation: {np.corrcoef(targets, np.arange(len(targets)))[0,1]:.4f}")
    print(f"  Data Completeness: {100 * (1 - np.isnan(sequences).sum()/sequences.size):.1f}%")

    return sequences, targets, dataset_info


def train_cnn_lstm_model(sequences, targets):
    """Train CNN+LSTM model for live trading."""

    print("\n🧠 Training CNN+LSTM Model")
    print("=" * 50)

    # Import training components
    from train_cnn_lstm import CNNLSTMTrainer, create_model_config, create_training_config

    # Create configurations
    model_config = create_model_config()
    training_config = create_training_config()
    training_config["epochs"] = 50  # Reduced for demo

    print("🏗️ Model Architecture:")
    print(f"  CNN Filters: {model_config['cnn_filters']}")
    print(f"  LSTM Units: {model_config['lstm_units']}")
    print(f"  Dropout Rate: {model_config['dropout']}")

    # Initialize trainer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = CNNLSTMTrainer(model_config=model_config, training_config=training_config, device=device)

    print("\n🖥️ Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch Size: {training_config['batch_size']}")
    print(f"  Learning Rate: {training_config['learning_rate']}")

    # Train model
    print("\n🚀 Starting training...")
    model_save_path = "outputs/cnn_lstm_production/best_model.pth"
    training_summary = trainer.train_from_dataset(sequences=sequences, targets=targets, save_path=model_save_path)

    # Training results
    final_metrics = training_summary["final_metrics"]
    print("\n📊 Training Results:")
    print(f"  Best Validation Loss: {training_summary['best_val_loss']:.6f}")
    print(f"  Final MAE: {final_metrics['mae']:.6f}")
    print(f"  Final RMSE: {final_metrics['rmse']:.6f}")
    print(f"  Final Correlation: {final_metrics['correlation']:.4f}")

    return training_summary, model_save_path


def setup_live_trading_components(dataset_info, model_path):
    """Setup components for live trading."""

    print("\n⚡ Setting Up Live Trading Components")
    print("=" * 50)

    # Import components
    from trading_rl_agent.data.robust_dataset_builder import RealTimeDatasetLoader

    # Setup real-time data processor
    dataset_version_dir = dataset_info["dataset_info"]["files"]["sequences"].replace("/sequences.npy", "")
    rt_loader = RealTimeDatasetLoader(dataset_version_dir)

    print(f"📊 Real-Time Loader: {rt_loader}")
    print(f"🤖 Model Path: {model_path}")

    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(
        initial_capital=100000,
        config=None,  # Use default config
    )

    print(f"💰 Portfolio Manager: {portfolio_manager}")
    print(f"  Initial Capital: ${portfolio_manager.initial_capital:,}")
    print(f"  Max Position Size: {portfolio_manager.config.max_position_size:.1%}")

    # Initialize risk manager
    risk_manager = RiskManager()

    print(f"🛡️ Risk Manager: {risk_manager}")
    print(f"  Max Portfolio VaR: {risk_manager.risk_limits.max_portfolio_var:.1%}")
    print(f"  Max Drawdown: {risk_manager.risk_limits.max_drawdown:.1%}")

    return rt_loader, portfolio_manager, risk_manager


def demonstrate_live_trading_workflow(rt_loader, portfolio_manager, risk_manager):
    """Demonstrate the live trading workflow."""

    print("\n🎯 Live Trading Workflow Demonstration")
    print("=" * 50)

    # Simulate incoming market data
    print("📊 Simulating incoming market data...")

    # Create sample market data
    sample_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "open": np.random.normal(150, 5, 100),
            "high": np.random.normal(155, 5, 100),
            "low": np.random.normal(145, 5, 100),
            "close": np.random.normal(150, 5, 100),
            "volume": np.random.normal(1000000, 200000, 100),
        }
    )

    # Process real-time data
    print("🔧 Processing real-time data...")
    try:
        processed_seq = rt_loader.process_realtime_data(sample_data)
        print(f"  ✅ Processed sequence shape: {processed_seq.shape}")
    except Exception as e:
        print(f"  ⚠️ Real-time processing demo: {e}")
        print("  (This is expected in demo mode)")

    # Portfolio operations
    print("\n💰 Portfolio Operations:")

    # Update prices
    current_prices = {"AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 380.0}
    portfolio_manager.update_prices(current_prices)

    print(f"  Current Portfolio Value: ${portfolio_manager.total_value:,.2f}")
    print(f"  Cash: ${portfolio_manager.cash:,.2f}")
    print(f"  Equity: ${portfolio_manager.equity_value:,.2f}")

    # Execute sample trade
    print("\n📈 Executing sample trade...")
    success = portfolio_manager.execute_trade(symbol="AAPL", quantity=100, price=150.0, side="buy")

    if success:
        print("  ✅ Trade executed successfully")
        print(f"  New Portfolio Value: ${portfolio_manager.total_value:,.2f}")
        print(f"  AAPL Position: {portfolio_manager.positions.get('AAPL', 'None')}")
    else:
        print("  ❌ Trade failed")

    # Risk analysis
    print("\n🛡️ Risk Analysis:")

    # Generate risk report
    sample_weights = {"AAPL": 0.3, "GOOGL": 0.3, "MSFT": 0.4}
    risk_report = risk_manager.generate_risk_report(
        portfolio_weights=sample_weights, portfolio_value=portfolio_manager.total_value
    )

    print(f"  Portfolio VaR: {risk_report['portfolio_var']:.2%}")
    print(f"  Max Drawdown: {risk_report['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {risk_report['sharpe_ratio']:.3f}")

    # Check risk limits
    alerts = risk_report.get("alerts", [])
    if alerts:
        print(f"  ⚠️ Risk Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"    - {alert['type']}: {alert['message']}")
    else:
        print("  ✅ No risk alerts - portfolio within limits")


def create_live_trading_config():
    """Create configuration for live trading."""

    print("\n⚙️ Creating Live Trading Configuration")
    print("=" * 50)

    # Live trading configuration
    live_config = {
        "paper_trading": True,  # Start with paper trading
        "real_time_enabled": True,
        "execution_latency_ms": 50,
        "risk_checks_enabled": True,
        "monitoring_enabled": True,
        "alert_thresholds": {"max_drawdown": 0.05, "max_daily_loss": 0.02, "position_concentration": 0.15},
        "data_sources": {"primary": "alpaca", "backup": "yfinance"},
        "broker_config": {
            "broker": "alpaca",
            "paper_trading": True,
            "api_key": "YOUR_API_KEY",
            "secret_key": "YOUR_SECRET_KEY",
        },
    }

    print("🎯 Live Trading Configuration:")
    print(f"  Paper Trading: {live_config['paper_trading']}")
    print(f"  Real-Time Enabled: {live_config['real_time_enabled']}")
    print(f"  Execution Latency: {live_config['execution_latency_ms']}ms")
    print(f"  Risk Checks: {live_config['risk_checks_enabled']}")
    print(f"  Primary Data Source: {live_config['data_sources']['primary']}")
    print(f"  Broker: {live_config['broker_config']['broker']}")

    # Save configuration
    output_dir = Path("outputs/live_trading_config")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "live_trading_config.json").open("w") as f:
        json.dump(live_config, f, indent=2)

    print(f"\n💾 Configuration saved to: {output_dir / 'live_trading_config.json'}")

    return live_config


def generate_trading_signals_example():
    """Generate example trading signals."""

    print("\n📊 Trading Signals Example")
    print("=" * 50)

    # Simulate model predictions
    predictions = np.random.normal(0.001, 0.02, 100)  # Daily returns
    confidence_scores = np.random.uniform(0.3, 0.9, 100)

    print("🎯 Signal Generation:")
    print(f"  Mean Prediction: {np.mean(predictions):.4f}")
    print(f"  Prediction Std: {np.std(predictions):.4f}")
    print(f"  Mean Confidence: {np.mean(confidence_scores):.3f}")

    # Generate trading signals
    signals = []
    for pred, conf in zip(predictions, confidence_scores):
        if conf > 0.7:  # High confidence threshold
            if pred > 0.01:  # Strong positive signal
                signal = "STRONG_BUY"
            elif pred > 0.005:  # Moderate positive signal
                signal = "BUY"
            elif pred < -0.01:  # Strong negative signal
                signal = "STRONG_SELL"
            elif pred < -0.005:  # Moderate negative signal
                signal = "SELL"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"  # Low confidence

        signals.append(signal)

    # Signal analysis
    signal_counts = pd.Series(signals).value_counts()
    print("\n📈 Signal Distribution:")
    for signal, count in signal_counts.items():
        print(f"  {signal}: {count} ({count/len(signals)*100:.1f}%)")

    return signals


def main():
    """Main execution function."""

    print("🎯 PRACTICAL PIPELINE EXAMPLE")
    print("=" * 60)
    print("This script demonstrates the complete pipeline for live trading.")
    print("=" * 60)

    try:
        # Step 1: Create production dataset
        sequences, targets, dataset_info = create_production_dataset()

        # Step 2: Train CNN+LSTM model
        training_summary, model_path = train_cnn_lstm_model(sequences, targets)

        # Step 3: Setup live trading components
        rt_loader, portfolio_manager, risk_manager = setup_live_trading_components(dataset_info, model_path)

        # Step 4: Demonstrate live trading workflow
        demonstrate_live_trading_workflow(rt_loader, portfolio_manager, risk_manager)

        # Step 5: Create live trading configuration
        live_config = create_live_trading_config()

        # Step 6: Generate trading signals example
        signals = generate_trading_signals_example()

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📊 Summary:")
        print(f"  Dataset Size: {len(sequences):,} sequences")
        print(f"  Model Performance: {training_summary['final_metrics']['correlation']:.4f} correlation")
        print(f"  Portfolio Value: ${portfolio_manager.total_value:,.2f}")
        print(f"  Risk Status: {'✅ Safe' if not risk_manager.risk_alerts else '⚠️ Alerts'}")
        print(f"  Live Trading: {'✅ Ready' if live_config['paper_trading'] else '🚀 Production'}")

        print("\n🚀 Next Steps:")
        print("  1. Configure your broker API keys")
        print("  2. Start with paper trading")
        print("  3. Monitor performance and risk metrics")
        print("  4. Gradually increase position sizes")
        print("  5. Deploy to production when ready")

        print("\n📁 Output Files:")
        print("  - Dataset: data/production_dataset/")
        print("  - Model: outputs/cnn_lstm_production/best_model.pth")
        print("  - Config: outputs/live_trading_config/live_trading_config.json")

        print("\n🎯 Your trading system is ready for live trading!")

    except Exception as e:
        print(f"❌ Pipeline example failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
