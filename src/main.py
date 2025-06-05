import argparse
import yaml
from agents.trainer import Trainer
import psutil
import platform
from datetime import datetime

def main():

    

    parser = argparse.ArgumentParser(description='Train or evaluate an RL trading agent')
    parser.add_argument('--env-config', type=str, required=True, help='Path to environment config YAML')
    parser.add_argument('--model-config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--trainer-config', type=str, required=True, help='Path to trainer config YAML')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Directory to save models and logs')
    parser.add_argument('--train', action='store_true', help='Run training loop')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--tune', action='store_true', help='Run Ray Tune hyperparameter search')
    args = parser.parse_args()

    if args.tune:
        from agents.tune import run_tune
        run_tune([args.env_config, args.model_config, args.trainer_config])
        return

    # Load configurations
    with open(args.env_config) as f:
        env_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.trainer_config) as f:
        trainer_cfg = yaml.safe_load(f)

    trainer = Trainer(env_cfg, model_cfg, trainer_cfg, seed=args.seed, save_dir=args.save_dir)
    if args.train:
        trainer.train()
    if args.eval:
        trainer.evaluate()
    if args.test:
        print("Running tests...")
        trainer.test()


if __name__ == '__main__':
    main()