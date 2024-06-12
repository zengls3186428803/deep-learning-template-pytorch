import torch
from omegaconf import DictConfig, OmegaConf


class ModelConfig:import torch
from omegaconf import DictConfig, OmegaConf


class ModelConfig:
    def __init__(self, cfg: DictConfig):
        self.T = OmegaConf.select(cfg, "model.T", default=1)

    import torch
    from omegaconf import DictConfig, OmegaConf

    class ModelConfig:
        def __init__(self, cfg: DictConfig):
            self.T = OmegaConf.select(cfg, "model.T", default=1)

    def __init__(self, cfg: DictConfig):
        self.T = OmegaConf.select(cfg, "model.T", default=1)
