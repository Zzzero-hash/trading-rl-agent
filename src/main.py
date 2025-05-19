import argparse
import yaml
from agents.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate an RL trading agent')
    parser.add_argument('--env-config', type=str, required=True, help='Path to environment config YAML')
    parser.add_argument('--model-config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--trainer-config', type=str, required=True, help='Path to trainer config YAML')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Directory to save models and logs')
    parser.add_argument('--train', action='store_true', help='Run training loop')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    args = parser.parse_args()

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


if __name__ == '__main__':
    main()