from omegaconf import DictConfig, OmegaConf


class ModelConfig:
    def __init__(self, cfg: DictConfig):
        self.T = OmegaConf.select(cfg, "model.T", default=1)

