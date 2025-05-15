import pytest
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # print the full configuration
    print(OmegaConf.to_yaml(cfg))
    # pass cfg to your trainer or entry logic
    from agents.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()