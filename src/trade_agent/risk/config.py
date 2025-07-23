# Configuration options for regime definitions and risk parameter adjustments

REGIME_DEFINITIONS = {
    "bull_market": {
        "volatility_threshold": 0.1,
        "volume_threshold": 1000000,
        "order_book_imbalance_threshold": 0.2
    },
    "bear_market": {
        "volatility_threshold": 0.2,
        "volume_threshold": 500000,
        "order_book_imbalance_threshold": 0.5
    },
    "sideways_market": {
        "volatility_threshold": 0.05,
        "volume_threshold": 750000,
        "order_book_imbalance_threshold": 0.1
    }
}

RISK_PARAMETERS = {
    "position_sizing": {
        "max_risk_per_trade": 0.02,  # 2% of account balance
        "min_position_size": 1,
        "max_position_size": 100
    },
    "stop_loss": {
        "default_distance": 0.03,  # 3% from entry price
        "max_distance": 0.1
    },
    "strategy_activation": {
        "activation_threshold": 0.7  # Probability threshold for activating strategy
    }
}
