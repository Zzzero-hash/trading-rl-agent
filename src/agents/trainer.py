# Basic Trainer implementation
import os
import ray

class Trainer:
    def __init__(self, env_cfg, model_cfg, trainer_cfg, seed=42, save_dir='outputs'):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.seed = seed
        self.save_dir = save_dir

        self.ray_address = self.trainer_cfg.get('ray_address') if isinstance(self.trainer_cfg, dict) else None
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address)
            else:
                ray.init()

        os.makedirs(self.save_dir, exist_ok=True)
        print(
            f"Initialized Trainer with seed={self.seed}, save_dir='{self.save_dir}', ray_address={self.ray_address}"
        )

    def train(self):
        print(f"[Trainer] Starting training with configs:\n  env={self.env_cfg}\n  model={self.model_cfg}\n  trainer={self.trainer_cfg}")
        # TODO: implement training loop

    def evaluate(self):
        print(f"[Trainer] Starting evaluation using models in '{self.save_dir}'")
        # TODO: implement evaluation logic