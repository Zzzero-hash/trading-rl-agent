import yaml
from src.training.train_cnn_lstm import TrainingConfig, load_template_config


def test_template_file_parses():
    cfg = load_template_config()
    assert 'data' in cfg
    assert 'model' in cfg
    assert 'training' in cfg


def test_training_config_from_template():
    cfg = TrainingConfig.from_template()
    # values taken from template_config.yaml
    assert cfg.train_path == 'data/train.csv'
    assert cfg.val_path == 'data/val.csv'
    assert cfg.test_path == 'data/test.csv'
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 0.001
    assert cfg.epochs == 10
    assert cfg.model_config is not None
    assert cfg.model_config.get('cnn_filters') == [32, 64]
