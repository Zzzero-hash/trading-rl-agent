import sys
from unittest.mock import patch
import runpy

import pytest

from trading_rl_agent.main import main
pytestmark = pytest.mark.integration


def test_main_when_main(sample_config_files):
    env_path, model_path, trainer_path = sample_config_files
    test_args = [
        "main.py",
        "--env-config",
        env_path,
        "--model-config",
        model_path,
        "--trainer-config",
        trainer_path,
    ]
    with (
        patch.object(sys, "argv", test_args),
-        patch("src.main.Trainer") as mock_trainer_class,
-        patch("src.main.main") as mock_main,
+        patch("src.main.Trainer"),
+        patch("src.main.main"),
    ):
        runpy.run_path("src/main.py")
        assert callable(main)
