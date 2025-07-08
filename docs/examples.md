# Examples

This section provides practical examples of using the Trading RL Agent.

## Basic Training Example

```python
from trading_rl_agent.agents.sac_agent import SACAgent
from trading_rl_agent.envs.trading_env import TradingEnv
from trading_rl_agent.agents.configs import SACConfig

# Create environment
env_config = {
    'dataset_paths': ['data/advanced_trading_dataset_*.csv'],
    'window_size': 50,
    'initial_balance': 10000,
    'transaction_cost': 0.001,
    'use_cnn_lstm_features': True
}
env = TradingEnv(env_config)

# Create agent
agent_config = SACConfig(
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=1000000,
    tau=0.005
)
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    config=agent_config
)

# Train the agent
for episode in range(1000):
    state, _ = env.reset()
    episode_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.replay_buffer.add(state, action, reward, next_state, terminated)

        if len(agent.replay_buffer) > agent.config.batch_size:
            agent.train()

        state = next_state
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

## Data Processing Example

```python
from trading_rl_agent.data.features import generate_features
from trading_rl_agent.data.live import fetch_live_data
import pandas as pd

# Load historical data
data = pd.read_csv('data/historical_data.csv')

# Generate technical indicators
features_df = generate_features(data, config={
    'sma_windows': [5, 10, 20],
    'ema_windows': [12, 26],
    'rsi_window': 14,
    'bollinger_window': 20,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
})

print(f"Generated {len(features_df.columns)} features")
print(features_df.head())
```

## Hyperparameter Optimization Example

```python
from trading_rl_agent.optimization.ray_tune_optimizer import RayTuneOptimizer
from ray import tune

# Define hyperparameter search space
search_space = {
    'learning_rate': tune.loguniform(1e-5, 1e-3),
    'batch_size': tune.choice([64, 128, 256]),
    'buffer_size': tune.choice([100000, 500000, 1000000]),
    'tau': tune.uniform(0.001, 0.01),
    'gamma': tune.uniform(0.95, 0.99)
}

# Initialize optimizer
optimizer = RayTuneOptimizer(
    search_space=search_space,
    metric='episode_reward_mean',
    mode='max'
)

# Run optimization
results = optimizer.optimize(
    train_function=train_agent,
    num_samples=50,
    max_concurrent_trials=4,
    time_budget_s=3600  # 1 hour
)

print(f"Best config: {results.best_config}")
```

## Live Trading Example

```python
from trading_rl_agent.agents.sac_agent import SACAgent
from trading_rl_agent.data.live import LiveDataProvider
from trading_rl_agent.trading.portfolio import Portfolio
import time

# Load trained agent
agent = SACAgent.load('models/best_sac_agent.pkl')

# Initialize live data provider
data_provider = LiveDataProvider(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    api_key='your_api_key'
)

# Initialize portfolio
portfolio = Portfolio(
    initial_balance=10000,
    risk_limit=0.02  # 2% risk per trade
)

# Trading loop
while True:
    # Get current market state
    current_data = data_provider.get_current_data()
    state = preprocess_state(current_data)

    # Get action from agent
    action = agent.select_action(state, add_noise=False)

    # Execute trade
    portfolio.execute_action(action, current_data)

    # Log performance
    portfolio.log_performance()

    # Wait for next interval
    time.sleep(60)  # Trade every minute
```

## Custom Environment Example

```python
from trading_rl_agent.envs.trading_env import TradingEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomTradingEnv(TradingEnv):
    """Custom trading environment with additional features."""

    def __init__(self, config):
        super().__init__(config)

        # Add custom observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features + 5),  # +5 for custom features
            dtype=np.float32
        )

    def _get_observation(self):
        """Get observation with custom features."""
        base_obs = super()._get_observation()

        # Add custom features
        portfolio_value = self.portfolio_value / self.initial_balance
        cash_ratio = self.cash / self.portfolio_value
        position_ratio = self.position / self.portfolio_value

        # Risk metrics
        volatility = np.std(self.price_history[-20:]) if len(self.price_history) >= 20 else 0
        momentum = (self.current_price - self.price_history[-10]) / self.price_history[-10] if len(self.price_history) >= 10 else 0

        custom_features = np.array([
            portfolio_value,
            cash_ratio,
            position_ratio,
            volatility,
            momentum
        ])

        # Combine base observation with custom features
        return np.concatenate([base_obs, custom_features])

    def _calculate_reward(self, action):
        """Custom reward function."""
        base_reward = super()._calculate_reward(action)

        # Add risk-adjusted reward
        sharpe_ratio = self._calculate_sharpe_ratio()
        risk_penalty = abs(action) * 0.01  # Penalize large positions

        return base_reward + sharpe_ratio * 0.1 - risk_penalty

    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio for risk adjustment."""
        if len(self.portfolio_history) < 2:
            return 0

        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        if len(returns) < 2:
            return 0

        return np.mean(returns) / (np.std(returns) + 1e-8)

# Usage
env = CustomTradingEnv(config)
```

## Model Evaluation Example

```python
from trading_rl_agent.evaluation.evaluator import ModelEvaluator
from trading_rl_agent.utils.metrics import calculate_trading_metrics
import matplotlib.pyplot as plt

# Load trained model
agent = SACAgent.load('models/best_model.pkl')

# Initialize evaluator
evaluator = ModelEvaluator(
    agent=agent,
    test_env=test_env,
    num_episodes=100
)

# Run evaluation
results = evaluator.evaluate()

# Calculate metrics
metrics = calculate_trading_metrics(
    portfolio_values=results['portfolio_values'],
    actions=results['actions'],
    rewards=results['rewards']
)

print("Evaluation Results:")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")

# Plot performance
plt.figure(figsize=(12, 6))
plt.plot(results['portfolio_values'])
plt.title('Portfolio Performance')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value')
plt.show()
```

## Backtesting Example

```python
from trading_rl_agent.backtesting.backtester import Backtester
from trading_rl_agent.data.historical import HistoricalDataProvider
from datetime import datetime

# Load historical data
data_provider = HistoricalDataProvider()
data = data_provider.get_data(
    symbols=['AAPL'],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Initialize backtester
backtester = Backtester(
    data=data,
    initial_balance=100000,
    transaction_cost=0.001,
    slippage=0.0005
)

# Load strategy
agent = SACAgent.load('models/trained_agent.pkl')

# Run backtest
results = backtester.run(
    agent=agent,
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31)
)

# Analyze results
print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Annual Return: {results['annual_return']:.2%}")
print(f"Volatility: {results['volatility']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")

# Generate report
backtester.generate_report(results, 'backtest_report.html')
```

## Advanced Configuration Example

```python
from trading_rl_agent.agents.configs import SACConfig
from trading_rl_agent.optimization.config import OptimizationConfig
from trading_rl_agent.data.config import DataConfig

# Advanced SAC configuration
sac_config = SACConfig(
    # Network architecture
    actor_hidden_dims=[400, 300],
    critic_hidden_dims=[400, 300],

    # Training parameters
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=1000000,

    # SAC specific
    entropy_coeff=0.2,
    target_entropy='auto',

    # Regularization
    weight_decay=1e-5,
    dropout_rate=0.1,

    # Training schedule
    warmup_steps=10000,
    learning_rate_decay=0.99,
    target_update_frequency=1000
)

# Data processing configuration
data_config = DataConfig(
    # Technical indicators
    sma_windows=[5, 10, 20, 50],
    ema_windows=[12, 26, 50],
    rsi_window=14,
    bollinger_window=20,
    macd_config={'fast': 12, 'slow': 26, 'signal': 9},

    # Volume indicators
    volume_sma_window=20,
    volume_rsi_window=14,

    # Volatility indicators
    atr_window=14,
    volatility_window=30,

    # Feature engineering
    price_differences=[1, 3, 5],
    returns_windows=[1, 5, 10],
    rolling_stats_windows=[10, 20, 50],

    # Preprocessing
    normalize_features=True,
    standardize_returns=True,
    handle_missing='interpolate'
)

# Optimization configuration
opt_config = OptimizationConfig(
    # Search algorithm
    algorithm='hyperopt',

    # Resource allocation
    num_samples=100,
    max_concurrent_trials=4,
    cpu_per_trial=2,
    gpu_per_trial=0.25,

    # Early stopping
    early_stopping=True,
    patience=10,
    min_improvement=0.01,

    # Checkpointing
    save_checkpoints=True,
    checkpoint_frequency=10
)
```
