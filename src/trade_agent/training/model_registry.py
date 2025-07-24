"""
Model Registry for Trade Agent Models.

This module provides comprehensive model management including versioning,
dependency tracking, performance grading, and metadata management.
"""

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ModelType(Enum):
    """Supported model types."""
    CNN_LSTM = "cnn_lstm"
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class PerformanceGrade(Enum):
    """Performance grade scale."""
    S = "S"    # Exceptional (>95th percentile)
    A_PLUS = "A+"  # Excellent (90-95th percentile)
    A = "A"    # Very Good (80-90th percentile)
    A_MINUS = "A-"  # Good (70-80th percentile)
    B_PLUS = "B+"  # Above Average (60-70th percentile)
    B = "B"    # Average (50-60th percentile)
    B_MINUS = "B-"  # Below Average (40-50th percentile)
    C = "C"    # Poor (30-40th percentile)
    D = "D"    # Very Poor (20-30th percentile)
    F = "F"    # Failed (<20th percentile)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    model_type: str
    version: str
    performance_grade: str | None
    created_at: datetime
    training_config: dict[str, Any]
    metrics: dict[str, float]
    dataset_info: dict[str, Any]
    dependencies: list[str]  # References to base models
    file_paths: dict[str, str]  # model, preprocessor, config files
    training_duration: float | None
    hardware_info: dict[str, Any]
    git_commit: str | None
    model_size_mb: float
    checksum: str
    tags: list[str]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ModelRegistry:
    """
    Centralized registry for all trained models.

    Provides version control, dependency tracking, and comprehensive
    metadata management for all model types in the training pipeline.
    """

    def __init__(self, registry_root: Path | None = None):
        self.registry_root = Path(registry_root or "models")
        self.registry_file = self.registry_root / "registry.json"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize registry structure
        self._initialize_registry()

        # Load existing registry
        self._models: dict[str, ModelMetadata] = self._load_registry()

    def _initialize_registry(self) -> None:
        """Initialize registry directory structure."""
        self.registry_root.mkdir(exist_ok=True)

        # Create subdirectories for each model type
        for model_type in ModelType:
            (self.registry_root / model_type.value).mkdir(exist_ok=True)

        # Create additional directories
        (self.registry_root / "archived").mkdir(exist_ok=True)
        (self.registry_root / "temp").mkdir(exist_ok=True)

        self.logger.info(f"Registry initialized at: {self.registry_root}")

    def _load_registry(self) -> dict[str, ModelMetadata]:
        """Load existing registry from disk."""
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file) as f:
                data = json.load(f)

            models = {}
            for model_id, model_data in data.items():
                models[model_id] = ModelMetadata.from_dict(model_data)

            self.logger.info(f"Loaded {len(models)} models from registry")
            return models

        except Exception as e:
            self.logger.error(f"Failed to load registry: {e!s}")
            return {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            data = {}
            for model_id, metadata in self._models.items():
                data[model_id] = metadata.to_dict()

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.debug("Registry saved to disk")

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e!s}")

    def register_model(
        self,
        model_type: str,
        model_path: str,
        preprocessor_path: str,
        training_config: dict[str, Any],
        metrics: dict[str, float],
        dataset_info: dict[str, Any],
        dependencies: list[str] | None = None,
        training_duration: float | None = None,
        hardware_info: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str = ""
    ) -> str:
        """
        Register a new model in the registry.

        Args:
            model_type: Type of model (cnn_lstm, ppo, etc.)
            model_path: Path to model file
            preprocessor_path: Path to preprocessor file
            training_config: Training configuration used
            metrics: Performance metrics
            dataset_info: Information about training dataset
            dependencies: List of base model IDs this model depends on
            training_duration: Training time in seconds
            hardware_info: Hardware configuration used
            tags: Additional tags for categorization
            notes: Additional notes

        Returns:
            Unique model ID
        """
        # Generate version and model ID
        version = self._generate_version(model_type)
        performance_grade = self._calculate_performance_grade(metrics, model_type)
        model_id = self._generate_model_id(model_type, version, performance_grade)

        # Create organized file structure
        file_paths = self._organize_model_files(
            model_id, model_type, model_path, preprocessor_path, training_config
        )

        # Calculate model checksum
        checksum = self._calculate_checksum(file_paths["model"])

        # Get model size
        model_size_mb = Path(file_paths["model"]).stat().st_size / (1024 * 1024)

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            version=version,
            performance_grade=performance_grade.value if performance_grade else None,
            created_at=datetime.now(),
            training_config=training_config,
            metrics=metrics,
            dataset_info=dataset_info,
            dependencies=dependencies or [],
            file_paths=file_paths,
            training_duration=training_duration,
            hardware_info=hardware_info or {},
            git_commit=self._get_git_commit(),
            model_size_mb=model_size_mb,
            checksum=checksum,
            tags=tags or [],
            notes=notes
        )

        # Register model
        self._models[model_id] = metadata
        self._save_registry()

        self.logger.info(f"Registered model: {model_id}")
        return model_id

    def _generate_version(self, model_type: str) -> str:
        """Generate semantic version for model."""
        existing_versions = []
        for metadata in self._models.values():
            if metadata.model_type == model_type:
                existing_versions.append(metadata.version)

        if not existing_versions:
            return "v1.0.0"

        # Extract major versions and increment
        major_versions = []
        for version in existing_versions:
            try:
                major = int(version.split(".")[0][1:])  # Remove 'v' prefix
                major_versions.append(major)
            except (ValueError, IndexError):
                continue

        if major_versions:
            next_major = max(major_versions) + 1
            return f"v{next_major}.0.0"

        return "v1.0.0"

    def _calculate_performance_grade(
        self,
        metrics: dict[str, float],
        model_type: str
    ) -> PerformanceGrade | None:
        """Calculate performance grade based on metrics."""
        # Get benchmark metrics for this model type
        benchmarks = self._get_performance_benchmarks(model_type)

        if not benchmarks:
            return None

        # Calculate composite score (simplified version)
        score = 0.0
        weight_sum = 0.0

        for metric_name, value in metrics.items():
            if metric_name in benchmarks:
                benchmark = benchmarks[metric_name]
                weight = benchmark.get("weight", 1.0)

                # Normalize score (higher is better for most metrics)
                if benchmark.get("higher_is_better", True):
                    normalized = min(value / benchmark.get("excellent", 1.0), 1.0)
                else:
                    normalized = max(1.0 - (value / benchmark.get("poor", 1.0)), 0.0)

                score += normalized * weight
                weight_sum += weight

        if weight_sum == 0:
            return None

        final_score = score / weight_sum

        # Map score to grade
        if final_score >= 0.95:
            return PerformanceGrade.S
        elif final_score >= 0.90:
            return PerformanceGrade.A_PLUS
        elif final_score >= 0.80:
            return PerformanceGrade.A
        elif final_score >= 0.70:
            return PerformanceGrade.A_MINUS
        elif final_score >= 0.60:
            return PerformanceGrade.B_PLUS
        elif final_score >= 0.50:
            return PerformanceGrade.B
        elif final_score >= 0.40:
            return PerformanceGrade.B_MINUS
        elif final_score >= 0.30:
            return PerformanceGrade.C
        elif final_score >= 0.20:
            return PerformanceGrade.D
        else:
            return PerformanceGrade.F

    def _get_performance_benchmarks(self, model_type: str) -> dict[str, dict[str, Any]]:
        """Get performance benchmarks for model type."""
        benchmarks = {
            "cnn_lstm": {
                "accuracy": {"weight": 2.0, "excellent": 0.95, "poor": 0.50, "higher_is_better": True},
                "loss": {"weight": 1.5, "excellent": 0.01, "poor": 0.5, "higher_is_better": False},
                "val_loss": {"weight": 1.5, "excellent": 0.02, "poor": 0.6, "higher_is_better": False},
                "r2_score": {"weight": 1.0, "excellent": 0.90, "poor": 0.30, "higher_is_better": True}
            },
            "ppo": {
                "reward": {"weight": 2.0, "excellent": 1000, "poor": 100, "higher_is_better": True},
                "sharpe_ratio": {"weight": 2.0, "excellent": 2.0, "poor": 0.5, "higher_is_better": True},
                "max_drawdown": {"weight": 1.5, "excellent": 0.05, "poor": 0.30, "higher_is_better": False}
            },
            "ensemble": {
                "ensemble_accuracy": {"weight": 2.5, "excellent": 0.98, "poor": 0.70, "higher_is_better": True},
                "diversity_score": {"weight": 1.0, "excellent": 0.8, "poor": 0.2, "higher_is_better": True}
            }
        }

        return benchmarks.get(model_type, {})

    def _generate_model_id(
        self,
        model_type: str,
        version: str,
        performance_grade: PerformanceGrade | None
    ) -> str:
        """Generate unique model ID."""
        grade_str = performance_grade.value if performance_grade else "ungraded"
        return f"{model_type}_{version}_grade_{grade_str}"

    def _organize_model_files(
        self,
        model_id: str,
        model_type: str,
        model_path: str,
        preprocessor_path: str,
        training_config: dict[str, Any]
    ) -> dict[str, str]:
        """Organize model files in registry structure."""
        model_dir = self.registry_root / model_type

        # Define target paths
        target_model_path = model_dir / f"{model_id}.pth"
        target_preprocessor_path = model_dir / f"preprocessor_{model_id.split('_', 2)[2]}.pkl"  # Extract version part
        target_config_path = model_dir / f"config_{model_id.split('_', 2)[2]}.json"
        target_metadata_path = model_dir / f"metadata_{model_id.split('_', 2)[2]}.json"

        # Copy files to organized structure
        shutil.copy2(model_path, target_model_path)
        shutil.copy2(preprocessor_path, target_preprocessor_path)

        # Save training config
        with open(target_config_path, "w") as f:
            json.dump(training_config, f, indent=2, default=str)

        return {
            "model": str(target_model_path),
            "preprocessor": str(target_preprocessor_path),
            "config": str(target_config_path),
            "metadata": str(target_metadata_path)
        }

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of model file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.registry_root.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def get_model(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        model_type: str | None = None,
        grade_filter: str | None = None,
        limit: int | None = None
    ) -> list[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self._models.values())

        # Filter by type
        if model_type:
            models = [m for m in models if m.model_type == model_type]

        # Filter by grade
        if grade_filter:
            models = [m for m in models if m.performance_grade == grade_filter]

        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        # Apply limit
        if limit:
            models = models[:limit]

        return models

    def get_model_dependencies(self, model_id: str) -> list[ModelMetadata]:
        """Get all dependencies for a model."""
        model = self.get_model(model_id)
        if not model:
            return []

        dependencies = []
        for dep_id in model.dependencies:
            dep_model = self.get_model(dep_id)
            if dep_model:
                dependencies.append(dep_model)

        return dependencies

    def validate_model_integrity(self, model_id: str) -> bool:
        """Validate model file integrity using checksum."""
        model = self.get_model(model_id)
        if not model:
            return False

        try:
            current_checksum = self._calculate_checksum(model.file_paths["model"])
            return current_checksum == model.checksum
        except Exception as e:
            self.logger.error(f"Failed to validate model {model_id}: {e!s}")
            return False

    def archive_model(self, model_id: str) -> bool:
        """Archive a model (move to archived directory)."""
        model = self.get_model(model_id)
        if not model:
            return False

        try:
            archive_dir = self.registry_root / "archived"

            # Move all model files to archive
            for file_type, file_path in model.file_paths.items():
                if Path(file_path).exists():
                    archive_path = archive_dir / Path(file_path).name
                    shutil.move(file_path, archive_path)
                    model.file_paths[file_type] = str(archive_path)

            # Update registry
            self._save_registry()

            self.logger.info(f"Archived model: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to archive model {model_id}: {e!s}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """Permanently delete a model."""
        model = self.get_model(model_id)
        if not model:
            return False

        try:
            # Delete all model files
            for file_path in model.file_paths.values():
                if Path(file_path).exists():
                    Path(file_path).unlink()

            # Remove from registry
            del self._models[model_id]
            self._save_registry()

            self.logger.info(f"Deleted model: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e!s}")
            return False

    def get_best_model(self, model_type: str, metric: str = "performance_grade") -> ModelMetadata | None:
        """Get the best performing model of a given type."""
        models = self.list_models(model_type=model_type)

        if not models:
            return None

        if metric == "performance_grade":
            # Sort by performance grade (S > A+ > A > ... > F)
            grade_order = {grade.value: i for i, grade in enumerate(PerformanceGrade)}
            models.sort(key=lambda m: grade_order.get(m.performance_grade or "F", 99))
            return models[0] if models else None

        # Sort by specific metric
        models_with_metric = [m for m in models if metric in m.metrics]
        if not models_with_metric:
            return None

        # Assume higher is better for most metrics
        models_with_metric.sort(key=lambda m: m.metrics[metric], reverse=True)
        return models_with_metric[0]

    def export_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Export comprehensive model information."""
        model = self.get_model(model_id)
        if not model:
            return None

        info = model.to_dict()

        # Add dependency information
        dependencies = self.get_model_dependencies(model_id)
        info["dependency_details"] = [dep.to_dict() for dep in dependencies]

        # Add integrity check
        info["integrity_valid"] = self.validate_model_integrity(model_id)

        return info
