#!/usr/bin/env python3
"""Complete fix for remaining MyPy errors."""

import re
from pathlib import Path


def fix_unified_manager() -> None:
    """Fix unified_manager.py type annotations."""
    file_path = Path("src/trade_agent/training/unified_manager.py")
    content = file_path.read_text()

    # Define all the method fixes
    method_fixes = [
        # Fix the ones that need -> None
        (r"(def mark_failed\(self, error_message: str\)):", r"\1 -> None:"),
        (r"(def _validate_training_environment\(self, config: TrainingConfig\)):", r"\1 -> None:"),
        (r"(def _setup_distributed_training\(self, config: TrainingConfig, result: TrainingResult\)):", r"\1 -> None:"),
        (r"(def _distributed_train_worker\(self, rank: int, world_size: int, config: TrainingConfig, result: TrainingResult\)):", r"\1 -> None:"),
        (r"(def _train_single_process\(self, config: TrainingConfig, result: TrainingResult\)):", r"\1 -> None:"),
        (r"(def _cleanup_training_resources\(self, training_id: str\)):", r"\1 -> None:"),
        (r"(def __init__\(self\)):", r"\1 -> None:"),
        (r"(def handle_training_error\(self, error: Exception, config: TrainingConfig, result: TrainingResult\)):", r"\1 -> None:"),
        (r"(def _handle_gpu_oom_error\(self, _error: Exception, _config: TrainingConfig, _result: TrainingResult\)):", r"\1 -> None:"),
        (r"(def _handle_file_not_found_error\(self, error: Exception, _config: TrainingConfig, _result: TrainingResult\)):", r"\1 -> None:"),
        (r"(def _handle_generic_error\(self, error: Exception, _config: TrainingConfig, _result: TrainingResult\)):", r"\1 -> None:"),

        # Fix the ones that return models (Any)
        (r"(def _create_model\(self, config: TrainingConfig, device: str\)):", r"\1 -> Any:"),
        (r"(def _create_cnn_lstm_model\(self, config: TrainingConfig, device: str\)):", r"\1 -> Any:"),
        (r"(def _create_rl_model\(self, config: TrainingConfig, device: str\)):", r"\1 -> Any:"),
        (r"(def _create_hybrid_model\(self, config: TrainingConfig, device: str\)):", r"\1 -> Any:"),
        (r"(def _create_ensemble_model\(self, config: TrainingConfig, device: str\)):", r"\1 -> Any:"),

        # Fix _execute_training which has both missing return type and untyped model param
        (r"def _execute_training\(self, config: TrainingConfig, result: TrainingResult, model, device: str, is_main_process: bool\):",
         r"def _execute_training(self, config: TrainingConfig, result: TrainingResult, model: Any, device: str, is_main_process: bool) -> None:"),
    ]

    for pattern, replacement in method_fixes:
        content = re.sub(pattern, replacement, content)

    file_path.write_text(content)
    print("Fixed unified_manager.py")


def fix_preprocessor_manager() -> None:
    """Fix preprocessor_manager.py type annotations."""
    file_path = Path("src/trade_agent/training/preprocessor_manager.py")
    content = file_path.read_text()

    method_fixes = [
        (r"(def _add_cnn_lstm_steps\(\s*self,\s*pipeline: PreprocessingPipeline,\s*scaling: str = \"robust\"\s*\)):", r"\1 -> None:"),
        (r"(def _add_rl_steps\(\s*self,\s*pipeline: PreprocessingPipeline,\s*scaling: str = \"robust\"\s*\)):", r"\1 -> None:"),
        (r"(def _add_hybrid_steps\(self, pipeline: PreprocessingPipeline, scaling: str = \"robust\"\)):", r"\1 -> None:"),
        (r"(def _add_ensemble_steps\(self, pipeline: PreprocessingPipeline, scaling: str = \"robust\"\)):", r"\1 -> None:"),
    ]

    for pattern, replacement in method_fixes:
        content = re.sub(pattern, replacement, content)

    file_path.write_text(content)
    print("Fixed preprocessor_manager.py")


def fix_training_init() -> None:
    """Fix training/__init__.py type ignore."""
    file_path = Path("src/trade_agent/training/__init__.py")
    content = file_path.read_text()

    # Fix the type ignore comment
    content = re.sub(
        r"EnhancedCNNLSTMTrainer = None  # type: ignore\[assignment\]",
        "EnhancedCNNLSTMTrainer = None  # type: ignore[misc,assignment]",
        content
    )

    file_path.write_text(content)
    print("Fixed training/__init__.py")


if __name__ == "__main__":
    fix_unified_manager()
    fix_preprocessor_manager()
    fix_training_init()
    print("All MyPy fixes applied!")
