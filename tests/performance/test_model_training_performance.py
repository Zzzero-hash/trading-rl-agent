"""
Performance tests for model training components.

Tests include:
- Ensemble training performance
- Policy optimization speed
- Training pipeline efficiency
- Memory usage during training
- GPU utilization
- Training convergence speed
"""

import time

import numpy as np
import pytest
import torch

from trading_rl_agent.agents.advanced_policy_optimization import AdvancedPPO, AdvancedPPOConfig
from trading_rl_agent.agents.configs import EnsembleConfig, PPOConfig, SACConfig
from trading_rl_agent.agents.ensemble_trainer import EnsembleTrainer


class TestModelTrainingPerformance:
    """Performance tests for model training components."""

    @pytest.fixture
    def training_data(self, benchmark_data):
        """Prepare training data for model training tests."""
        # Convert benchmark data to training format
        test_data = benchmark_data.copy()

        # Group by symbol and create sequences
        sequences = []
        for symbol in test_data["symbol"].unique()[:10]:  # Use 10 symbols for training
            symbol_data = test_data[test_data["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("timestamp")

            # Create features
            symbol_data["returns"] = symbol_data["close"].pct_change()
            symbol_data["volume_ma"] = symbol_data["volume"].rolling(20).mean()
            symbol_data["price_ma"] = symbol_data["close"].rolling(20).mean()

            # Drop NaN values
            symbol_data = symbol_data.dropna()

            if len(symbol_data) > 100:  # Only use symbols with sufficient data
                sequences.append(symbol_data)

        return sequences

    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.ml
    def test_ensemble_training_performance(self, training_data, performance_monitor):
        """Test ensemble training performance."""
        # Prepare training configuration
        config = EnsembleConfig(
            agents={
                "ppo": PPOConfig(
                    learning_rate=3e-4,
                    batch_size=64,
                    n_steps=2048,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    normalize_advantage=True,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    use_sde=False,
                    sde_sample_freq=-1,
                    target_kl=None,
                    tensorboard_log=None,
                    policy_kwargs=None,
                    verbose=0,
                    seed=None,
                    device="auto",
                    _init_setup_model=True,
                ),
                "sac": SACConfig(
                    learning_rate=3e-4,
                    batch_size=256,
                    buffer_size=1000000,
                    learning_starts=100,
                    train_freq=1,
                    gradient_steps=1,
                    tau=0.005,
                    gamma=0.99,
                    ent_coef="auto",
                    target_entropy="auto",
                    use_sde=False,
                    sde_sample_freq=-1,
                    use_sde_at_warmup=False,
                    policy_kwargs=None,
                    verbose=0,
                    seed=None,
                    device="auto",
                    _init_setup_model=True,
                ),
            },
            ensemble_method="weighted_average",
            weight_update_frequency=100,
            diversity_penalty=0.1,
            consensus_threshold=0.7,
        )

        # Create simple environment creator for testing
        def create_test_env():
            class TestEnv:
                def __init__(self):
                    self.observation_space = type("obj", (object,), {"shape": (20,), "dtype": np.float32})()
                    self.action_space = type("obj", (object,), {"shape": (3,), "dtype": np.float32})()

                def reset(self):
                    return np.random.random(20)

                def step(self, action):
                    return np.random.random(20), np.random.random(), False, {}

            return TestEnv()

        # Initialize ensemble trainer
        trainer = EnsembleTrainer(
            config=config,
            env_creator=create_test_env,
            save_dir="test_outputs/ensemble_performance",
            device="cpu",  # Use CPU for consistent testing
        )

        performance_monitor.start_monitoring()

        # Benchmark training
        def train_ensemble():
            trainer.create_agents()
            return trainer.train_ensemble(
                total_iterations=50,  # Reduced for performance testing
                eval_frequency=10,
                save_frequency=25,
                early_stopping_patience=10,
            )

        # Measure performance
        start_time = time.time()
        result = train_ensemble()
        end_time = time.time()

        performance_monitor.record_measurement("ensemble_training_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert end_time - start_time < 300  # Should complete within 5 minutes
        assert metrics["peak_memory_mb"] < 4096  # Should use less than 4GB

        # Log performance metrics
        print("Ensemble training performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  CPU usage: {metrics['average_cpu_percent']:.1f}%")

    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.ml
    def test_policy_optimization_performance(self, training_data, performance_monitor):
        """Test policy optimization performance."""

        # Create simple policy and value networks for testing
        class SimplePolicy(torch.nn.Module):
            def __init__(self, input_dim=20, output_dim=3):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, 64)
                self.fc2 = torch.nn.Linear(64, 64)
                self.fc3 = torch.nn.Linear(64, output_dim)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return torch.softmax(self.fc3(x), dim=-1)

        class SimpleValue(torch.nn.Module):
            def __init__(self, input_dim=20):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, 64)
                self.fc2 = torch.nn.Linear(64, 64)
                self.fc3 = torch.nn.Linear(64, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        # Initialize policy optimizer
        policy_net = SimplePolicy()
        value_net = SimpleValue()
        config = AdvancedPPOConfig(
            learning_rate=3e-4,
            batch_size=64,
            buffer_size=100000,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        optimizer = AdvancedPPO(policy_net, value_net, config, device="cpu")

        # Prepare training data
        states = []
        actions = []
        rewards = []

        for data in training_data[:5]:  # Use first 5 sequences
            # Create synthetic training data
            n_steps = len(data)
            for i in range(n_steps - 1):
                state = np.random.random(20)  # 20-dimensional state
                action = np.random.random(3)  # 3-dimensional action
                reward = np.random.random()

                states.append(state)
                actions.append(action)
                rewards.append(reward)

        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        performance_monitor.start_monitoring()

        # Benchmark policy optimization
        def optimize_policy():
            # Convert data to tensors
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)

            # Create synthetic episode data
            dones = torch.zeros(len(rewards))
            dones[-1] = 1  # Mark end of episode

            # Create old log probs (synthetic)
            old_log_probs = torch.randn(len(actions))

            # Run a few optimization steps
            total_loss = 0
            for _ in range(10):  # 10 epochs
                loss_info = optimizer.update(
                    states=states_tensor,
                    actions=actions_tensor,
                    rewards=rewards_tensor,
                    dones=dones,
                    old_log_probs=old_log_probs,
                )
                total_loss += loss_info.get("total_loss", 0)

            return {"total_loss": total_loss}

        # Measure performance
        start_time = time.time()
        result = optimize_policy()
        end_time = time.time()

        performance_monitor.record_measurement("policy_optimization_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log performance metrics
        print("Policy optimization performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Training samples: {len(states)}")

    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.ml
    def test_training_pipeline_efficiency(self, training_data, performance_monitor):
        """Test training pipeline efficiency."""
        from trading_rl_agent.training.trainer import TrainingManager

        # Prepare training data
        train_data = []
        val_data = []

        for i, data in enumerate(training_data[:8]):  # Use 8 sequences
            if i < 6:  # 75% for training
                train_data.append(data)
            else:  # 25% for validation
                val_data.append(data)

        # Initialize training manager
        trainer = TrainingManager(
            model_type="cnn_lstm",
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            learning_rate=1e-3,
            device="cpu",
        )

        performance_monitor.start_monitoring()

        # Benchmark training pipeline
        def run_training_pipeline():
            return trainer.train(
                train_data=train_data, val_data=val_data, epochs=5, batch_size=32, early_stopping_patience=3
            )

        # Measure performance
        start_time = time.time()
        result = run_training_pipeline()
        end_time = time.time()

        performance_monitor.record_measurement("training_pipeline_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert end_time - start_time < 120  # Should complete within 2 minutes
        assert metrics["peak_memory_mb"] < 2048  # Should use less than 2GB

        # Log performance metrics
        print("Training pipeline efficiency:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Training sequences: {len(train_data)}")
        print(f"  Validation sequences: {len(val_data)}")

    @pytest.mark.performance
    @pytest.mark.memory
    @pytest.mark.ml
    def test_memory_usage_during_training(self, training_data, memory_profiler):
        """Test memory usage during training."""
        from trading_rl_agent.training.trainer import TrainingManager

        # Prepare training data
        train_data = training_data[:5]  # Use 5 sequences for memory testing

        # Initialize training manager
        trainer = TrainingManager(
            model_type="cnn_lstm",
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            learning_rate=1e-3,
            device="cpu",
        )

        # Profile memory usage during training
        def train_with_memory_monitoring():
            return trainer.train(
                train_data=train_data,
                val_data=train_data[:1],  # Use one sequence for validation
                epochs=3,
                batch_size=16,
                early_stopping_patience=2,
            )

        memory_metrics = memory_profiler(train_with_memory_monitoring)

        # Assertions
        assert memory_metrics["max_memory_mb"] < 3072  # Should use less than 3GB
        assert memory_metrics["avg_memory_mb"] < 1536  # Average should be less than 1.5GB

        # Log memory metrics
        print("Memory usage during training:")
        print(f"  Max memory: {memory_metrics['max_memory_mb']:.2f} MB")
        print(f"  Avg memory: {memory_metrics['avg_memory_mb']:.2f} MB")
        print(f"  Min memory: {memory_metrics['min_memory_mb']:.2f} MB")

    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.ml
    def test_gpu_utilization(self, training_data, performance_monitor):
        """Test GPU utilization during training (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for testing")

        from trading_rl_agent.training.trainer import TrainingManager

        # Prepare training data
        train_data = training_data[:5]

        # Initialize training manager with GPU
        trainer = TrainingManager(
            model_type="cnn_lstm",
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            learning_rate=1e-3,
            device="cuda",
        )

        performance_monitor.start_monitoring()

        # Benchmark GPU training
        def train_on_gpu():
            return trainer.train(
                train_data=train_data, val_data=train_data[:1], epochs=3, batch_size=32, early_stopping_patience=2
            )

        # Measure performance
        start_time = time.time()
        result = train_on_gpu()
        end_time = time.time()

        performance_monitor.record_measurement("gpu_training_complete")
        metrics = performance_monitor.stop_monitoring()

        # Get GPU memory usage
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB

        # Assertions
        assert result is not None
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert gpu_memory_allocated < 2048  # Should use less than 2GB GPU memory

        # Log performance metrics
        print("GPU utilization:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  GPU memory allocated: {gpu_memory_allocated:.2f} MB")
        print(f"  GPU memory reserved: {gpu_memory_reserved:.2f} MB")
        print(f"  CPU memory peak: {metrics['peak_memory_mb']:.2f} MB")

    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.ml
    def test_training_convergence_speed(self, training_data, performance_monitor):
        """Test training convergence speed."""
        from trading_rl_agent.training.trainer import TrainingManager

        # Prepare training data
        train_data = training_data[:6]
        val_data = training_data[6:8] if len(training_data) > 6 else training_data[:1]

        # Initialize training manager
        trainer = TrainingManager(
            model_type="cnn_lstm",
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            learning_rate=1e-3,
            device="cpu",
        )

        performance_monitor.start_monitoring()

        # Track convergence
        convergence_epochs = []
        final_losses = []

        # Run multiple training sessions to measure convergence
        def train_and_track_convergence():
            history = trainer.train(
                train_data=train_data,
                val_data=val_data,
                epochs=10,
                batch_size=32,
                early_stopping_patience=5,
                return_history=True,
            )

            # Find convergence point (when loss stops improving significantly)
            val_losses = history.get("val_loss", [])
            if len(val_losses) > 1:
                for i in range(1, len(val_losses)):
                    if abs(val_losses[i] - val_losses[i - 1]) < 0.001:
                        convergence_epochs.append(i)
                        break
                else:
                    convergence_epochs.append(len(val_losses))
            else:
                convergence_epochs.append(1)

            final_losses.append(val_losses[-1] if val_losses else float("inf"))
            return history

        # Measure performance
        start_time = time.time()
        result = train_and_track_convergence()
        end_time = time.time()

        performance_monitor.record_measurement("convergence_tracking_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert end_time - start_time < 180  # Should complete within 3 minutes
        assert len(convergence_epochs) > 0
        assert convergence_epochs[0] <= 10  # Should converge within 10 epochs

        # Log performance metrics
        print("Training convergence speed:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Convergence epoch: {convergence_epochs[0]}")
        print(f"  Final validation loss: {final_losses[0]:.6f}")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")

    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.ml
    def test_batch_processing_efficiency(self, training_data, performance_monitor):
        """Test batch processing efficiency during training."""
        from trading_rl_agent.training.trainer import TrainingManager

        # Prepare training data
        train_data = training_data[:4]

        # Test different batch sizes
        batch_sizes = [16, 32, 64, 128]
        batch_times = []

        for batch_size in batch_sizes:
            trainer = TrainingManager(
                model_type="cnn_lstm",
                input_size=20,
                hidden_size=64,
                num_layers=2,
                output_size=3,
                learning_rate=1e-3,
                device="cpu",
            )

            performance_monitor.start_monitoring()

            # Benchmark batch processing
            def train_with_batch_size(batch_size=batch_size, trainer=trainer, train_data=train_data):
                return trainer.train(
                    train_data=train_data,
                    val_data=train_data[:1],
                    epochs=2,
                    batch_size=batch_size,
                    early_stopping_patience=1,
                )

            # Measure performance
            start_time = time.time()
            result = train_with_batch_size()
            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)

            performance_monitor.record_measurement(f"batch_size_{batch_size}_complete")
            metrics = performance_monitor.stop_monitoring()

            # Log batch performance
            print(f"Batch size {batch_size} performance:")
            print(f"  Time: {batch_time:.2f} seconds")
            print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")

        # Find optimal batch size
        optimal_batch_size = batch_sizes[np.argmin(batch_times)]

        # Assertions
        assert len(batch_times) == len(batch_sizes)
        assert min(batch_times) < 30  # Fastest should complete within 30 seconds

        # Log optimal batch size
        print(f"Optimal batch size: {optimal_batch_size}")
        print(f"Fastest training time: {min(batch_times):.2f} seconds")

    @pytest.mark.performance
    @pytest.mark.benchmark
    @pytest.mark.ml
    def test_model_checkpointing_performance(self, training_data, performance_monitor, tmp_path):
        """Test model checkpointing performance."""
        from trading_rl_agent.training.trainer import TrainingManager

        # Prepare training data
        train_data = training_data[:3]

        # Initialize training manager
        trainer = TrainingManager(
            model_type="cnn_lstm",
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            learning_rate=1e-3,
            device="cpu",
        )

        checkpoint_path = tmp_path / "model_checkpoint.pth"

        performance_monitor.start_monitoring()

        # Benchmark checkpointing
        def train_with_checkpointing():
            # Train for a few epochs
            trainer.train(
                train_data=train_data, val_data=train_data[:1], epochs=2, batch_size=32, early_stopping_patience=1
            )

            # Save checkpoint
            trainer.save_checkpoint(str(checkpoint_path))

            # Load checkpoint
            loaded_trainer = TrainingManager(
                model_type="cnn_lstm",
                input_size=20,
                hidden_size=64,
                num_layers=2,
                output_size=3,
                learning_rate=1e-3,
                device="cpu",
            )
            loaded_trainer.load_checkpoint(str(checkpoint_path))

            return loaded_trainer

        # Measure performance
        start_time = time.time()
        result = train_with_checkpointing()
        end_time = time.time()

        performance_monitor.record_measurement("checkpointing_complete")
        metrics = performance_monitor.stop_monitoring()

        # Assertions
        assert result is not None
        assert end_time - start_time < 60  # Should complete within 60 seconds
        assert checkpoint_path.exists()  # Checkpoint file should exist
        assert metrics["peak_memory_mb"] < 1024  # Should use less than 1GB

        # Log performance metrics
        print("Model checkpointing performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory peak: {metrics['peak_memory_mb']:.2f} MB")
        print(f"  Checkpoint size: {checkpoint_path.stat().st_size / 1024:.2f} KB")
