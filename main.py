import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="conf/hydra", config_name="config")
def main(cfg: DictConfig):
    cfg = DictConfig(dict(cfg))
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
