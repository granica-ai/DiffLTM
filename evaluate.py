import hydra
from omegaconf import DictConfig, OmegaConf

from diffltm.evaluation import evaluate


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(config: DictConfig):
    evaluate(OmegaConf.to_container(config, resolve=True))


if __name__ == "__main__":
    main()
