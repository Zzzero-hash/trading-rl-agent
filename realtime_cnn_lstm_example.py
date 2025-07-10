"""
Real-time Data Streaming Example for CNN+LSTM

This script demonstrates how to use the robust dataset pipeline for real-time
market data processing and prediction.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trading_rl_agent.data.robust_dataset_builder import RealTimeDatasetLoader
from trading_rl_agent.data.synthetic import fetch_synthetic_data
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

logger = logging.getLogger(__name__)


class RealTimeCNNLSTMPredictor:
    """Real-time predictor using trained CNN+LSTM model."""

    def __init__(self, model_checkpoint: str, dataset_version_dir: str):
        self.model_checkpoint = model_checkpoint
        self.dataset_version_dir = dataset_version_dir

        # Load real-time data processor
        self.rt_loader = RealTimeDatasetLoader(dataset_version_dir)

        # Load trained model
        self._load_model()

        # Track prediction history for monitoring
        self.prediction_history = []

    def _load_model(self):
        """Load the trained CNN+LSTM model."""

        # Handle PyTorch 2.6+ weights_only behavior for sklearn scalers
        import torch.serialization
        from sklearn.preprocessing._data import RobustScaler

        torch.serialization.add_safe_globals([RobustScaler])

        try:
            checkpoint = torch.load(self.model_checkpoint, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(self.model_checkpoint, map_location="cpu", weights_only=False)

        # Get input dimension from metadata
        input_dim = len(self.rt_loader.feature_columns)

        # Initialize and load model
        self.model = CNNLSTMModel(
            input_dim=input_dim,
            config=checkpoint["model_config"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(
            f"Model loaded: {input_dim} features, {checkpoint['epoch']} epochs trained",
        )

    def predict_next_return(self, market_data: pd.DataFrame) -> dict:
        """Predict next period return from market data."""

        try:
            # Process data through same pipeline as training
            processed_sequence = self.rt_loader.process_realtime_data(market_data)

            # Make prediction
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(processed_sequence)
                prediction = self.model(sequence_tensor).item()

            # Calculate confidence (simplified using prediction magnitude)
            confidence = min(1.0, abs(prediction) / 0.05)  # Normalize to 0-1

            # Generate trading signal
            signal = self._generate_signal(prediction, confidence)

            # Log prediction
            result = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "prediction": prediction,
                "confidence": confidence,
                "signal": signal,
                "market_data_shape": market_data.shape,
            }

            self.prediction_history.append(result)
        except Exception:
            logger.exception("Prediction failed")
            return {
                "timestamp": pd.Timestamp.now().isoformat(),
                "prediction": 0.0,
                "confidence": 0.0,
                "signal": "HOLD",
                "error": "Prediction failed",
            }
        else:
            return result

    def _generate_signal(self, prediction: float, confidence: float) -> str:
        """Generate trading signal from prediction."""

        # Thresholds
        min_confidence = 0.3
        strong_signal_threshold = 0.02  # 2% return prediction
        weak_signal_threshold = 0.005  # 0.5% return prediction

        if confidence < min_confidence:
            return "HOLD"

        if prediction > strong_signal_threshold:
            return "STRONG_BUY"
        if prediction > weak_signal_threshold:
            return "BUY"
        if prediction < -strong_signal_threshold:
            return "STRONG_SELL"
        if prediction < -weak_signal_threshold:
            return "SELL"
        return "HOLD"

    def get_prediction_summary(self) -> dict:
        """Get summary of recent predictions."""

        if not self.prediction_history:
            return {"error": "No predictions made yet"}

        recent_predictions = self.prediction_history[-20:]  # Last 20 predictions

        predictions = [p["prediction"] for p in recent_predictions]
        confidences = [p["confidence"] for p in recent_predictions]
        signals = [p["signal"] for p in recent_predictions]

        return {
            "total_predictions": len(self.prediction_history),
            "recent_predictions": len(recent_predictions),
            "avg_prediction": np.mean(predictions),
            "std_prediction": np.std(predictions),
            "avg_confidence": np.mean(confidences),
            "signal_distribution": {signal: signals.count(signal) for signal in set(signals)},
            "latest_prediction": recent_predictions[-1] if recent_predictions else None,
        }


def simulate_realtime_trading(
    predictor: RealTimeCNNLSTMPredictor,
    symbols: list[str],
    n_iterations: int = 50,
) -> dict:
    """Simulate real-time trading with synthetic data."""

    logger.info(
        f"🔄 Starting real-time trading simulation for {n_iterations} iterations...",
    )

    results = []

    for i in range(n_iterations):
        # Simulate receiving new market data (in reality this would come from an API)
        # We'll use different symbols in rotation
        symbol = symbols[i % len(symbols)]

        # Generate fresh market data (simulating real-time feed)
        market_data = fetch_synthetic_data(
            n_samples=100,  # Window of recent data
            timeframe="hour",
            volatility=0.02,
            symbol=symbol,
        )

        # Add symbol column for processing
        market_data["symbol"] = symbol

        # Make prediction
        prediction_result = predictor.predict_next_return(market_data)
        prediction_result["symbol"] = symbol
        prediction_result["iteration"] = i

        results.append(prediction_result)

        # Log every 10 iterations
        if i % 10 == 0:
            summary = predictor.get_prediction_summary()
            logger.info(
                f"Iteration {i}: {symbol} -> {prediction_result['signal']} "
                f"(pred: {prediction_result['prediction']:.4f}, "
                f"conf: {prediction_result['confidence']:.2f})",
            )
            logger.info(f"  Summary: {summary['signal_distribution']}")

        # Simulate processing delay
        time.sleep(0.1)

    # Final summary
    final_summary = predictor.get_prediction_summary()

    logger.info("✅ Real-time simulation completed!")
    logger.info(f"📊 Final summary: {final_summary}")

    return {
        "predictions": results,
        "summary": final_summary,
        "symbols_tested": symbols,
        "total_iterations": n_iterations,
    }


def main():
    """Main real-time example."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Check if we have a trained model
    model_path = "outputs/cnn_lstm_training/best_model.pth"
    dataset_path = "outputs/cnn_lstm_training/dataset"

    if not Path(model_path).exists():
        logger.error(f"❌ Model not found at {model_path}")
        logger.info("Please run the training pipeline first:")
        logger.info("python train_cnn_lstm.py")
        return

    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset not found at {dataset_path}")
        logger.info("Please run the training pipeline first to generate dataset")
        return

    try:
        # Find the actual dataset version directory
        dataset_base = Path(dataset_path)
        dataset_versions = [d for d in dataset_base.iterdir() if d.is_dir()]

        if not dataset_versions:
            logger.error("No dataset versions found")
            return

        # Use the most recent dataset version
        latest_dataset = max(dataset_versions, key=lambda x: x.name)

        logger.info("🚀 Initializing real-time predictor...")
        logger.info(f"Model: {model_path}")
        logger.info(f"Dataset: {latest_dataset}")

        # Initialize predictor
        predictor = RealTimeCNNLSTMPredictor(
            model_checkpoint=model_path,
            dataset_version_dir=str(latest_dataset),
        )

        # Run simulation
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        simulation_results = simulate_realtime_trading(
            predictor=predictor,
            symbols=symbols,
            n_iterations=30,
        )

        # Save results
        output_path = "outputs/realtime_simulation_results.json"
        with Path(output_path).open(output_path, "w") as f:
            json.dump(simulation_results, f, indent=2, default=str)

        logger.info(f"💾 Simulation results saved to {output_path}")

        # Example of single prediction
        logger.info("\n🎯 Single prediction example:")
        test_data = fetch_synthetic_data(n_samples=100, symbol="TEST")
        test_data["symbol"] = "TEST"

        single_result = predictor.predict_next_return(test_data)
        logger.info(f"Test prediction: {single_result}")

    except Exception:
        logger.exception("❌ Real-time example failed")
        raise


if __name__ == "__main__":
    main()
