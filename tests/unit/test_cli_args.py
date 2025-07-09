import pytest

from trading_rl_agent.main import build_parser

pytestmark = pytest.mark.unit


def test_parser_required_args(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--env-config",
            str(tmp_path / "env.yaml"),
            "--model-config",
            str(tmp_path / "model.yaml"),
            "--trainer-config",
            str(tmp_path / "trainer.yaml"),
        ]
    )
    assert args.env_config.endswith("env.yaml")
    assert args.model_config.endswith("model.yaml")
    assert args.trainer_config.endswith("trainer.yaml")
    assert args.seed == 42
    assert args.save_dir == "outputs"


def test_parser_all_args(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--env-config",
            "env.yaml",
            "--model-config",
            "model.yaml",
            "--trainer-config",
            "trainer.yaml",
            "--seed",
            "123",
            "--save-dir",
            str(tmp_path),
            "--train",
            "--eval",
        ]
    )
    assert args.seed == 123
    assert args.save_dir == str(tmp_path)
    assert args.train is True
    assert args.eval is True


def test_parser_missing_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
