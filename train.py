from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from diffltm.artifacts import completed_training
from diffltm.trainer import train


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig):
    cfg = OmegaConf.to_container(config, resolve=True)
    run = Path(cfg["output_dir"])
    if (run / "config.json").exists():
        completed_training(run, config=cfg)
        print(f"Reusing completed training: {run}", flush=True)
        return
    train(cfg)


if __name__ == "__main__":
    main()
