import torch
from omegaconf import DictConfig, OmegaConf


class DataConfig:
    def __init__(self, cfg: DictConfig):

        self.batch_size = OmegaConf.select(cfg, "data.batch_size", default=128)
        self.test_batch_size = OmegaConf.select(cfg, "data.test_batch_size", default=1000)
