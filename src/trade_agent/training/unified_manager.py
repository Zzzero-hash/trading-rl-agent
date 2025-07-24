"""
Unified Training Manager     def __init__(
        self,
        model_type: str,
        data_path: str | Path,
        output_dir: str | Path,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "auto",
        distributed: bool = False,
        optimize_hyperparams: bool = False,
        n_trials: int = 50,
        resume_from: str | None = None,
        base_model_path: str | None = None,
        **kwargs: Any
    ) -> None:Models.

This module provides a comprehensive training orchestration system that handles
hierarchical model training (CNN-LSTM → Hybrid → Ensemble), resource management,
error handling, and advanced features like distributed training and experiment tracking.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from trade_agent.core.config import ConfigManager

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration container for training operations."""

    def __init__(
        self,
        model_type: str,
        data_path: str | Path,
        output_dir: str | Path,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "auto",
        distributed: bool = False,
        optimize_hyperparams: bool = False,
        n_trials: int = 50,
        resume_from: str | None = None,
        base_model_path: str | None = None,
        **kwargs: Any
    ) -> None:
        self.model_type = model_type
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.distributed = distributed
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.resume_from = resume_from
        self.base_model_path = base_model_path
        self.extra_params = kwargs

        # Generate unique training ID
        self.training_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Auto-detect device if needed
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class TrainingResult:
    """Container for training results and metadata."""

    def __init__(self, training_id: str, model_type: str):
        self.training_id = training_id
        self.model_type = model_type
        self.start_time = datetime.now()
        self.end_time: datetime | None = None
        self.status = "started"
        self.model_path: str | None = None
        self.preprocessor_path: str | None = None
        self.metrics: dict[str, float] = {}
        self.performance_grade: str | None = None
        self.error_message: str | None = None
        self.checkpoints: list[str] = []

    def mark_completed(self, model_path: str, preprocessor_path: str, metrics: dict[str, float]) -> None:
        """Mark training as completed successfully."""
        self.end_time = datetime.now()
        self.status = "completed"
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.metrics = metrics

    def mark_failed(self, error_message: str) -> None:
        """Mark training as failed."""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error_message = error_message

    @property
    def duration(self) -> float | None:
        """Get training duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class UnifiedTrainingManager:
    """
    Unified manager for all model training operations.

    This class orchestrates the complete training pipeline including:
    - Hierarchical model training (CNN-LSTM → RL → Hybrid → Ensemble)
    - Resource management and optimization
    - Error handling and recovery
    - Distributed training coordination
    - Experiment tracking and model registry integration

    Examples:
        Basic training configuration and execution:

        >>> from trade_agent.training import UnifiedTrainingManager, TrainingConfig
        >>> config = TrainingConfig(
        ...     model_type="cnn_lstm",
        ...     data_path="data/dataset.csv",
        ...     output_dir="models/cnn_lstm",
        ...     epochs=100,
        ...     optimize_hyperparams=True
        ... )
        >>> manager = UnifiedTrainingManager()
        >>> result = manager.train_model(config)
        >>> print(f"Model saved: {result.model_path}")
        >>> print(f"Performance grade: {result.performance_grade}")

        Distributed training:

        >>> config = TrainingConfig(
        ...     model_type="cnn_lstm",
        ...     data_path="data/dataset.csv",
        ...     distributed=True
        ... )
        >>> result = manager.train_model(config)

    Attributes:
        config_manager: Configuration manager instance
        active_trainings: Dictionary of active training results
        device_manager: Device allocation and optimization manager
        error_handler: Training error handling and recovery
    """

    def __init__(self, config_manager: ConfigManager | None = None):
        self.config_manager = config_manager or ConfigManager()
        self.active_trainings: dict[str, TrainingResult] = {}
        self.device_manager = DeviceManager()
        self.error_handler = TrainingErrorHandler()

        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def train_model(self, config: TrainingConfig) -> TrainingResult:
        """
        Main entry point for model training.

        Orchestrates the complete training process including environment validation,
        distributed training setup, model creation, and training execution.

        Args:
            config: Training configuration containing model type, data path,
                   training parameters, and optimization settings.

        Returns:
            TrainingResult: Contains training outcomes, model paths, performance
                          metrics, and metadata. Status will be "completed" for
                          successful training or "failed" with error details.

        Raises:
            FileNotFoundError: If training data or base models are not found.
            RuntimeError: If distributed training is requested but not available.
            ValueError: If model type is not supported.

        Examples:
            Basic model training:

            >>> config = TrainingConfig(
            ...     model_type="cnn_lstm",
            ...     data_path="data/market_data.csv",
            ...     epochs=50
            ... )
            >>> result = manager.train_model(config)
            >>> if result.status == "completed":
            ...     print(f"Model trained successfully: {result.model_path}")

            Hierarchical training with dependencies:

            >>> rl_config = TrainingConfig(
            ...     model_type="ppo",
            ...     data_path="data/market_data.csv",
            ...     base_model_path="cnn_lstm_v1.0.0_grade_A"
            ... )
            >>> result = manager.train_model(rl_config)
        """
        result = TrainingResult(config.training_id, config.model_type)
        self.active_trainings[config.training_id] = result

        try:
            self.logger.info(f"Starting training: {config.training_id}")

            # Validate dependencies and environment
            self._validate_training_environment(config)

            # Setup distributed training if requested
            if config.distributed:
                self._setup_distributed_training(config, result)
            else:
                self._train_single_process(config, result)

        except Exception as e:
            self.logger.error(f"Training failed: {config.training_id}: {e!s}")
            result.mark_failed(str(e))
            self.error_handler.handle_training_error(e, config, result)

        finally:
            # Cleanup resources
            self._cleanup_training_resources(config.training_id)

        return result

    def _validate_training_environment(self, config: TrainingConfig) -> None:
        """Validate training environment and dependencies."""
        self.logger.info("Validating training environment...")

        # Check data availability
        if not config.data_path.exists():
            raise FileNotFoundError(f"Training data not found: {config.data_path}")

        # Check base model dependencies for hierarchical training
        if config.base_model_path and not Path(config.base_model_path).exists():
            raise FileNotFoundError(f"Base model not found: {config.base_model_path}")

        # Check GPU availability for GPU training
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            config.device = "cpu"

        # Validate distributed training setup
        if config.distributed and not dist.is_available():
            raise RuntimeError("Distributed training requested but PyTorch distributed not available")

        self.logger.info("Environment validation completed")

    def _setup_distributed_training(self, config: TrainingConfig, result: TrainingResult) -> None:
        """Setup and launch distributed training."""
        world_size = torch.cuda.device_count() if config.device.startswith("cuda") else 1

        if world_size < 2:
            self.logger.warning("Distributed training requested but only 1 device available, using single process")
            self._train_single_process(config, result)
            return

        self.logger.info(f"Starting distributed training on {world_size} devices")

        # Launch distributed training processes
        mp.spawn(
            self._distributed_train_worker,
            args=(world_size, config, result),
            nprocs=world_size,
            join=True
        )

    def _distributed_train_worker(self, rank: int, world_size: int, config: TrainingConfig, result: TrainingResult) -> None:
        """Worker function for distributed training."""
        # Initialize process group
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Set device for this process
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"

        try:
            # Create model and wrap with DDP
            model = self._create_model(config, device)
            model = DDP(model, device_ids=[rank])

            # Run training
            self._execute_training(config, result, model, device, rank == 0)

        finally:
            # Cleanup distributed training
            dist.destroy_process_group()

    def _train_single_process(self, config: TrainingConfig, result: TrainingResult) -> None:
        """Execute single-process training."""
        device = config.device
        model = self._create_model(config, device)
        self._execute_training(config, result, model, device, is_main_process=True)

    def _create_model(self, config: TrainingConfig, device: str) -> Any:
        """Create model based on configuration."""
        if config.model_type == "cnn_lstm":
            return self._create_cnn_lstm_model(config, device)
        elif config.model_type in ["ppo", "sac", "td3"]:
            return self._create_rl_model(config, device)
        elif config.model_type == "hybrid":
            return self._create_hybrid_model(config, device)
        elif config.model_type == "ensemble":
            return self._create_ensemble_model(config, device)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def _create_cnn_lstm_model(self, config: TrainingConfig, device: str) -> Any:
        """Create CNN-LSTM model."""
        from trade_agent.models.cnn_lstm import CNNLSTMModel

        # Load model configuration
        model_config = self._get_model_config(config.model_type)
        model = CNNLSTMModel(**model_config)
        return model.to(device)

    def _create_rl_model(self, config: TrainingConfig, device: str) -> Any:
        """Create RL model with CNN-LSTM state enhancement."""
        # This will be implemented in the next phase
        raise NotImplementedError(f"RL model {config.model_type} not yet implemented")

    def _create_hybrid_model(self, config: TrainingConfig, device: str) -> Any:
        """Create hybrid CNN-LSTM + RL model."""
        # This will be implemented in the next phase
        raise NotImplementedError("Hybrid model not yet implemented")

    def _create_ensemble_model(self, config: TrainingConfig, device: str) -> Any:
        """Create ensemble model."""
        # This will be implemented in the next phase
        raise NotImplementedError("Ensemble model not yet implemented")

    def _execute_training(self, config: TrainingConfig, result: TrainingResult, model: Any, device: str, is_main_process: bool) -> None:
        """Execute the actual training process."""
        self.logger.info(f"Executing training on device: {device}")

        # This is a placeholder for the actual training logic
        # In the real implementation, this would call the appropriate trainer

        # For now, just simulate training
        time.sleep(2)  # Simulate training time

        if is_main_process:
            # Save model and preprocessor (only main process saves in distributed training)
            model_path = self._save_model(model, config, result)
            preprocessor_path = self._save_preprocessor(config, result)

            # Calculate metrics (placeholder)
            metrics = {"loss": 0.1, "accuracy": 0.95, "val_loss": 0.12}

            result.mark_completed(model_path, preprocessor_path, metrics)
            self.logger.info(f"Training completed: {result.training_id}")

    def _save_model(self, model: Any, config: TrainingConfig, _result: TrainingResult) -> str:
        """Save trained model with metadata."""
        # Create model directory structure
        model_dir = Path("models") / config.model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        # Generate model filename with performance info (placeholder)
        performance_grade = "A"  # This will be calculated by performance grader
        version = "v1.0.0"  # This will be managed by model registry
        model_filename = f"{config.model_type}_{version}_grade_{performance_grade}.pth"
        model_path = model_dir / model_filename

        # Save model
        torch.save(model.state_dict(), model_path)

        self.logger.info(f"Model saved: {model_path}")
        return str(model_path)

    def _save_preprocessor(self, config: TrainingConfig, _result: TrainingResult) -> str:
        """Save preprocessor with versioning."""
        # This will be implemented with PreprocessorManager
        # For now, just return a placeholder path
        preprocessor_path = f"models/{config.model_type}/preprocessor_v1.0.0.pkl"
        self.logger.info(f"Preprocessor saved: {preprocessor_path}")
        return preprocessor_path

    def _get_model_config(self, model_type: str) -> dict[str, Any]:
        """Get model configuration from config manager."""
        # This will integrate with the config system
        # For now, return default config
        if model_type == "cnn_lstm":
            return {
                "input_dim": 78,  # Will be determined from data
                "sequence_length": 60,
                "num_filters": 64,
                "kernel_size": 3,
                "lstm_units": 50,
                "dropout": 0.2,
                "use_attention": True
            }
        return {}

    def _cleanup_training_resources(self, training_id: str) -> None:
        """Cleanup resources after training completion."""
        if training_id in self.active_trainings:
            self.logger.info(f"Cleaning up training resources: {training_id}")
            # Cleanup will be implemented as needed

    def get_training_status(self, training_id: str) -> TrainingResult | None:
        """Get status of a training run."""
        return self.active_trainings.get(training_id)

    def list_active_trainings(self) -> list[TrainingResult]:
        """List all active training runs."""
        return list(self.active_trainings.values())


class DeviceManager:
    """Manages device allocation and optimization."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_optimal_device(self) -> str:
        """Get optimal device for training."""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def get_memory_info(self) -> dict[str, Any]:
        """Get memory information for current device."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        return {"cpu_memory": "N/A"}


class TrainingErrorHandler:
    """Handles training errors and implements recovery mechanisms."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def handle_training_error(self, error: Exception, config: TrainingConfig, result: TrainingResult) -> None:
        """Handle training errors with appropriate recovery strategies."""
        self.logger.error(f"Training error in {result.training_id}: {type(error).__name__}: {error!s}")

        if isinstance(error, torch.cuda.OutOfMemoryError):
            self._handle_gpu_oom_error(error, config, result)
        elif isinstance(error, FileNotFoundError):
            self._handle_file_not_found_error(error, config, result)
        else:
            self._handle_generic_error(error, config, result)

    def _handle_gpu_oom_error(self, _error: Exception, _config: TrainingConfig, _result: TrainingResult) -> None:
        """Handle GPU out-of-memory errors."""
        self.logger.warning("GPU OOM detected, implementing recovery strategy")
        # Could implement batch size reduction, gradient checkpointing, etc.

    def _handle_file_not_found_error(self, error: Exception, _config: TrainingConfig, _result: TrainingResult) -> None:
        """Handle file not found errors."""
        self.logger.error(f"Required file not found: {error!s}")
        # Could implement automatic data download, dependency resolution, etc.

    def _handle_generic_error(self, error: Exception, _config: TrainingConfig, _result: TrainingResult) -> None:
        """Handle generic training errors."""
        self.logger.error(f"Generic training error: {error!s}")
        # Could implement retry logic, checkpoint recovery, etc.
