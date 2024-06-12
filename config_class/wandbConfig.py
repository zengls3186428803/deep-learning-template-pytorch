from omegaconf import DictConfig, OmegaConf


class WandbConfig:
    def __init__(self, cfg: DictConfig):
        self.project = OmegaConf.select(cfg, "wandb.project", default="neural-ode")
        self.entity = OmegaConf.select(cfg, "wandb.entity", default="superposed-tree")
        self.dir = OmegaConf.select(cfg, "wandb.dir", default="wandb_outputs")
        self.mode = OmegaConf.select(cfg, "wandb.mode", default="online")
        self.group = OmegaConf.select(cfg, "wandb.group", default="test_group")
