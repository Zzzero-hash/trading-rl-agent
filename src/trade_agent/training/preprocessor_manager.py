"""
Preprocessor Manager for Trade Agent Models.

This module manages preprocessing pipelines, ensuring reproducible model inference
through versioned preprocessing components that are saved alongside models.
"""

import hashlib
import json
import logging
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PreprocessorMetadata:
    """Metadata for preprocessing pipeline."""
    preprocessor_id: str
    version: str
    created_at: datetime
    pipeline_steps: list[str]
    input_features: list[str]
    output_features: list[str]
    scaling_method: str
    feature_engineering_steps: list[str]
    data_validation_rules: dict[str, Any]
    preprocessing_time: float | None
    checksum: str
    compatibility_hash: str  # Hash of input data schema

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessorMetadata":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class PreprocessingPipeline:
    """
    Versioned preprocessing pipeline that can be saved and loaded.

    This class encapsulates all preprocessing steps needed to transform
    raw data into model-ready format, ensuring reproducible inference.
    """

    def __init__(self, preprocessor_id: str):
        self.preprocessor_id = preprocessor_id
        self.steps: list[Callable] = []
        self.step_names: list[str] = []
        self.fitted_transformers: dict[str, Any] = {}
        self.input_schema: dict[str, Any] | None = None
        self.output_schema: dict[str, Any] | None = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_step(self, step_name: str, transform_func: Callable[..., Any], **kwargs: Any) -> None:
        """Add a preprocessing step to the pipeline."""
        self.steps.append(lambda data: transform_func(data, **kwargs))
        self.step_names.append(step_name)
        self.logger.debug(f"Added preprocessing step: {step_name}")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline on data and transform it."""
        self.input_schema = self._extract_schema(data)
        result = data.copy()

        start_time = datetime.now()

        for i, (step, step_name) in enumerate(zip(self.steps, self.step_names, strict=False)):
            try:
                self.logger.debug(f"Executing step {i+1}/{len(self.steps)}: {step_name}")
                result = step(result)

                if result is None or result.empty:
                    raise ValueError(f"Step {step_name} returned empty result")

            except Exception as e:
                self.logger.error(f"Error in preprocessing step {step_name}: {e!s}")
                raise

        self.output_schema = self._extract_schema(result)
        processing_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(f"Pipeline fit_transform completed in {processing_time:.2f}s")
        return result

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        if self.input_schema is None:
            raise ValueError("Pipeline must be fitted before transform")

        # Validate input schema compatibility
        self._validate_input_schema(data)

        result = data.copy()

        for step, step_name in zip(self.steps, self.step_names, strict=False):
            try:
                result = step(result)
                if result is None or result.empty:
                    raise ValueError(f"Step {step_name} returned empty result")
            except Exception as e:
                self.logger.error(f"Error in preprocessing step {step_name}: {e!s}")
                raise

        return result

    def _extract_schema(self, data: pd.DataFrame) -> dict[str, Any]:
        """Extract schema information from data."""
        return {
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "shape": data.shape,
            "index_type": str(type(data.index).__name__),
            "has_nulls": data.isnull().any().to_dict()
        }

    def _validate_input_schema(self, data: pd.DataFrame) -> None:
        """Validate that input data matches expected schema."""
        if self.input_schema is None:
            return  # No schema to validate against

        current_schema = self._extract_schema(data)

        # Check columns
        expected_cols = set(self.input_schema["columns"])
        actual_cols = set(current_schema["columns"])

        if not expected_cols.issubset(actual_cols):
            missing_cols = expected_cols - actual_cols
            raise ValueError(f"Missing expected columns: {missing_cols}")

        # Check data types for common columns
        for col in expected_cols.intersection(actual_cols):
            expected_dtype = self.input_schema["dtypes"][col]
            actual_dtype = current_schema["dtypes"][col]

            # Allow some flexibility in dtype matching
            if not self._dtypes_compatible(expected_dtype, actual_dtype):
                self.logger.warning(
                    f"Column {col} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
                )

    def _dtypes_compatible(self, expected: str, actual: str) -> bool:
        """Check if data types are compatible."""
        # Define compatible dtype groups
        numeric_types = {"int64", "int32", "float64", "float32"}
        object_types = {"object", "string"}

        if expected in numeric_types and actual in numeric_types:
            return True
        if expected in object_types and actual in object_types:
            return True

        return expected == actual


class PreprocessorManager:
    """
    Manager for preprocessing pipelines with versioning and persistence.

    Handles saving, loading, and version management of preprocessing pipelines
    to ensure reproducible model inference across different environments.
    """

    def __init__(self, preprocessor_root: Path | None = None):
        self.preprocessor_root = Path(preprocessor_root or "models/preprocessors")
        self.registry_file = self.preprocessor_root / "preprocessor_registry.json"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize directory structure
        self._initialize_structure()

        # Load existing registry
        self._preprocessors: dict[str, PreprocessorMetadata] = self._load_registry()

    def _initialize_structure(self) -> None:
        """Initialize preprocessor directory structure."""
        self.preprocessor_root.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different model types
        model_types = ["cnn_lstm", "ppo", "sac", "td3", "hybrid", "ensemble"]
        for model_type in model_types:
            (self.preprocessor_root / model_type).mkdir(exist_ok=True)

        self.logger.info(f"Preprocessor manager initialized at: {self.preprocessor_root}")

    def _load_registry(self) -> dict[str, PreprocessorMetadata]:
        """Load preprocessor registry from disk."""
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file) as f:
                data = json.load(f)

            preprocessors = {}
            for proc_id, proc_data in data.items():
                preprocessors[proc_id] = PreprocessorMetadata.from_dict(proc_data)

            self.logger.info(f"Loaded {len(preprocessors)} preprocessors from registry")
            return preprocessors

        except Exception as e:
            self.logger.error(f"Failed to load preprocessor registry: {e!s}")
            return {}

    def _save_registry(self) -> None:
        """Save preprocessor registry to disk."""
        try:
            data = {}
            for proc_id, metadata in self._preprocessors.items():
                data[proc_id] = metadata.to_dict()

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.debug("Preprocessor registry saved to disk")

        except Exception as e:
            self.logger.error(f"Failed to save preprocessor registry: {e!s}")

    def create_pipeline(
        self,
        model_type: str,
        version: str | None = None
    ) -> PreprocessingPipeline:
        """Create a new preprocessing pipeline."""
        if version is None:
            version = self._generate_version(model_type)

        preprocessor_id = f"{model_type}_preprocessor_{version}"
        pipeline = PreprocessingPipeline(preprocessor_id)

        self.logger.info(f"Created preprocessing pipeline: {preprocessor_id}")
        return pipeline

    def create_standard_pipeline(
        self,
        model_type: str,
        scaling_method: str = "robust",
        include_technical_indicators: bool = True,
        include_sentiment: bool = True,
        sequence_length: int | None = None
    ) -> PreprocessingPipeline:
        """Create a standard preprocessing pipeline for a model type."""
        pipeline = self.create_pipeline(model_type)

        # Add standard preprocessing steps based on model type
        if model_type == "cnn_lstm":
            self._add_cnn_lstm_steps(
                pipeline, scaling_method, include_technical_indicators,
                include_sentiment, sequence_length
            )
        elif model_type in ["ppo", "sac", "td3"]:
            self._add_rl_steps(
                pipeline, scaling_method, include_technical_indicators, include_sentiment
            )
        elif model_type == "hybrid":
            self._add_hybrid_steps(
                pipeline, scaling_method, include_technical_indicators, include_sentiment
            )
        elif model_type == "ensemble":
            self._add_ensemble_steps(
                pipeline, scaling_method, include_technical_indicators, include_sentiment
            )

        return pipeline

    def _add_cnn_lstm_steps(
        self,
        pipeline: PreprocessingPipeline,
        scaling_method: str,
        include_technical_indicators: bool,
        include_sentiment: bool,
        sequence_length: int | None
    ) -> None:
        """Add CNN-LSTM specific preprocessing steps."""
        # Data validation
        pipeline.add_step("validate_data", self._validate_market_data)

        # Handle missing values
        pipeline.add_step("handle_missing", self._handle_missing_values, method="forward_fill")

        # Feature engineering
        if include_technical_indicators:
            pipeline.add_step("technical_indicators", self._add_technical_indicators)

        if include_sentiment:
            pipeline.add_step("sentiment_features", self._add_sentiment_features)

        # Scaling
        pipeline.add_step("scale_features", self._scale_features, method=scaling_method)

        # Sequence creation for LSTM
        if sequence_length:
            pipeline.add_step(
                "create_sequences",
                self._create_sequences,
                sequence_length=sequence_length
            )

        # Final validation
        pipeline.add_step("final_validation", self._validate_model_input, model_type="cnn_lstm")

    def _add_rl_steps(
        self,
        pipeline: PreprocessingPipeline,
        scaling_method: str,
        include_technical_indicators: bool,
        include_sentiment: bool
    ) -> None:
        """Add RL agent specific preprocessing steps."""
        # Similar to CNN-LSTM but without sequence creation
        pipeline.add_step("validate_data", self._validate_market_data)
        pipeline.add_step("handle_missing", self._handle_missing_values, method="forward_fill")

        if include_technical_indicators:
            pipeline.add_step("technical_indicators", self._add_technical_indicators)

        if include_sentiment:
            pipeline.add_step("sentiment_features", self._add_sentiment_features)

        # State normalization for RL
        pipeline.add_step("normalize_states", self._normalize_rl_states, method=scaling_method)

        # Add reward shaping features
        pipeline.add_step("reward_features", self._add_reward_features)

        pipeline.add_step("final_validation", self._validate_model_input, model_type="rl")

    def _add_hybrid_steps(self, pipeline: PreprocessingPipeline, scaling_method: str, include_technical_indicators: bool, include_sentiment: bool) -> None:
        """Add hybrid model specific preprocessing steps."""
        # Combination of CNN-LSTM and RL preprocessing
        pipeline.add_step("validate_data", self._validate_market_data)
        pipeline.add_step("handle_missing", self._handle_missing_values, method="forward_fill")

        if include_technical_indicators:
            pipeline.add_step("technical_indicators", self._add_technical_indicators)

        if include_sentiment:
            pipeline.add_step("sentiment_features", self._add_sentiment_features)

        # Dual preprocessing for both components
        pipeline.add_step("scale_features", self._scale_features, method=scaling_method)
        pipeline.add_step("prepare_hybrid_inputs", self._prepare_hybrid_inputs)

        pipeline.add_step("final_validation", self._validate_model_input, model_type="hybrid")

    def _add_ensemble_steps(self, pipeline: PreprocessingPipeline, scaling_method: str, include_technical_indicators: bool, include_sentiment: bool) -> None:
        """Add ensemble model specific preprocessing steps."""
        # Preprocessing for ensemble inputs
        pipeline.add_step("validate_data", self._validate_market_data)
        pipeline.add_step("handle_missing", self._handle_missing_values, method="forward_fill")

        if include_technical_indicators:
            pipeline.add_step("technical_indicators", self._add_technical_indicators)

        if include_sentiment:
            pipeline.add_step("sentiment_features", self._add_sentiment_features)

        # Ensemble-specific feature engineering
        pipeline.add_step("ensemble_features", self._add_ensemble_features)
        pipeline.add_step("scale_features", self._scale_features, method=scaling_method)

        pipeline.add_step("final_validation", self._validate_model_input, model_type="ensemble")

    # Preprocessing step implementations (placeholders for now)
    def _validate_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate market data format and quality."""
        # Implement market data validation logic
        self.logger.debug("Validating market data...")
        return data

    def _handle_missing_values(self, data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """Handle missing values in the data."""
        self.logger.debug(f"Handling missing values with method: {method}")
        if method == "forward_fill":
            return data.fillna(method="ffill")
        elif method == "backward_fill":
            return data.fillna(method="bfill")
        elif method == "mean":
            return data.fillna(data.mean())
        else:
            return data.dropna()

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data."""
        self.logger.debug("Adding technical indicators...")
        # This would integrate with existing technical indicator calculation
        return data

    def _add_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to the data."""
        self.logger.debug("Adding sentiment features...")
        # This would integrate with existing sentiment analysis
        return data

    def _scale_features(self, data: pd.DataFrame, method: str = "robust") -> pd.DataFrame:
        """Scale features using specified method."""
        self.logger.debug(f"Scaling features with method: {method}")
        # This would integrate with DataStandardizer
        return data

    def _normalize_rl_states(self, data: pd.DataFrame, _method: str = "robust") -> pd.DataFrame:
        """Normalize states for RL agents."""
        self.logger.debug("Normalizing RL states...")
        return data

    def _add_reward_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add reward shaping features for RL."""
        self.logger.debug("Adding reward features...")
        return data

    def _create_sequences(self, data: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
        """Create sequences for LSTM input."""
        self.logger.debug(f"Creating sequences with length: {sequence_length}")
        # Implement sequence creation logic
        return data

    def _prepare_hybrid_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare inputs for hybrid models."""
        self.logger.debug("Preparing hybrid inputs...")
        return data

    def _add_ensemble_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add ensemble-specific features."""
        self.logger.debug("Adding ensemble features...")
        return data

    def _validate_model_input(self, data: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """Final validation for model input."""
        self.logger.debug(f"Final validation for {model_type} model input...")
        return data

    def save_pipeline(
        self,
        pipeline: PreprocessingPipeline,
        model_type: str,
        _training_data_info: dict[str, Any],
        processing_time: float | None = None
    ) -> str:
        """Save preprocessing pipeline to disk."""
        # Generate file paths
        model_dir = self.preprocessor_root / model_type
        pipeline_file = model_dir / f"{pipeline.preprocessor_id}.pkl"
        metadata_file = model_dir / f"{pipeline.preprocessor_id}_metadata.json"

        # Save pipeline object
        with open(pipeline_file, "wb") as f:
            pickle.dump(pipeline, f)

        # Calculate checksum
        checksum = self._calculate_checksum(str(pipeline_file))

        # Create metadata
        metadata = PreprocessorMetadata(
            preprocessor_id=pipeline.preprocessor_id,
            version=pipeline.preprocessor_id.split("_")[-1],
            created_at=datetime.now(),
            pipeline_steps=pipeline.step_names,
            input_features=list(pipeline.input_schema["columns"]) if pipeline.input_schema else [],
            output_features=list(pipeline.output_schema["columns"]) if pipeline.output_schema else [],
            scaling_method="robust",  # This should be extracted from pipeline
            feature_engineering_steps=pipeline.step_names,
            data_validation_rules={},  # This could be extracted from validation steps
            preprocessing_time=processing_time,
            checksum=checksum,
            compatibility_hash=self._calculate_compatibility_hash(pipeline.input_schema)
        )

        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        # Register in registry
        self._preprocessors[pipeline.preprocessor_id] = metadata
        self._save_registry()

        self.logger.info(f"Saved preprocessing pipeline: {pipeline.preprocessor_id}")
        return str(pipeline_file)

    def load_pipeline(self, preprocessor_id: str) -> PreprocessingPipeline | None:
        """Load preprocessing pipeline from disk."""
        if preprocessor_id not in self._preprocessors:
            self.logger.error(f"Preprocessor not found: {preprocessor_id}")
            return None

        metadata = self._preprocessors[preprocessor_id]

        # Determine file path from metadata or construct it
        pipeline_file = None
        for model_type in ["cnn_lstm", "ppo", "sac", "td3", "hybrid", "ensemble"]:
            potential_path = self.preprocessor_root / model_type / f"{preprocessor_id}.pkl"
            if potential_path.exists():
                pipeline_file = potential_path
                break

        if not pipeline_file:
            self.logger.error(f"Pipeline file not found for: {preprocessor_id}")
            return None

        try:
            with open(pipeline_file, "rb") as f:
                # Note: This pickle.load is used only for internal preprocessing pipelines
                # created by this application, not external untrusted data
                pipeline = pickle.load(f)  # nosec B301

            # Validate integrity
            current_checksum = self._calculate_checksum(str(pipeline_file))
            if current_checksum != metadata.checksum:
                self.logger.warning(f"Checksum mismatch for preprocessor: {preprocessor_id}")

            self.logger.info(f"Loaded preprocessing pipeline: {preprocessor_id}")
            return pipeline  # type: ignore[no-any-return]

        except Exception as e:
            self.logger.error(f"Failed to load preprocessor {preprocessor_id}: {e!s}")
            return None

    def _generate_version(self, model_type: str) -> str:
        """Generate version for new preprocessor."""
        existing_versions = []
        for metadata in self._preprocessors.values():
            if model_type in metadata.preprocessor_id:
                existing_versions.append(metadata.version)

        if not existing_versions:
            return "v1.0.0"

        # Simple version increment
        version_numbers = []
        for version in existing_versions:
            try:
                major = int(version.split(".")[0][1:])  # Remove 'v' prefix
                version_numbers.append(major)
            except (ValueError, IndexError):
                continue

        if version_numbers:
            next_version = max(version_numbers) + 1
            return f"v{next_version}.0.0"

        return "v1.0.0"

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _calculate_compatibility_hash(self, schema: dict[str, Any] | None) -> str:
        """Calculate hash for input schema compatibility."""
        if not schema:
            return ""

        # Create a deterministic hash of the schema
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def list_preprocessors(self, model_type: str | None = None) -> list[PreprocessorMetadata]:
        """List available preprocessors."""
        preprocessors = list(self._preprocessors.values())

        if model_type:
            preprocessors = [p for p in preprocessors if model_type in p.preprocessor_id]

        # Sort by creation time (newest first)
        preprocessors.sort(key=lambda p: p.created_at, reverse=True)

        return preprocessors

    def get_preprocessor_info(self, preprocessor_id: str) -> dict[str, Any] | None:
        """Get detailed preprocessor information."""
        if preprocessor_id not in self._preprocessors:
            return None

        metadata = self._preprocessors[preprocessor_id]
        return metadata.to_dict()

    def validate_preprocessor_integrity(self, preprocessor_id: str) -> bool:
        """Validate preprocessor file integrity."""
        if preprocessor_id not in self._preprocessors:
            return False

        metadata = self._preprocessors[preprocessor_id]

        # Find the preprocessor file
        for model_type in ["cnn_lstm", "ppo", "sac", "td3", "hybrid", "ensemble"]:
            pipeline_file = self.preprocessor_root / model_type / f"{preprocessor_id}.pkl"
            if pipeline_file.exists():
                try:
                    current_checksum = self._calculate_checksum(str(pipeline_file))
                    return current_checksum == metadata.checksum
                except Exception:
                    return False

        return False
