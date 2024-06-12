import torch
from omegaconf import DictConfig, OmegaConf


class AlgorithmConfig:
    def __init__(self, cfg: DictConfig):
        # 对于复杂的类型，只持有类的定义，不持有实例
        self.learning_rate = OmegaConf.select(cfg, "algorithm.learning_rate", default=0.1)
        self.optimizer_name = OmegaConf.select(cfg, "algorithm.optimizer", default="SGD")
        self.optimizer = eval("torch.optim." + self.optimizer_name)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR
        self.optimizers = (self.optimizer, self.scheduler)
        self.loss_fn = torch.nn.CrossEntropyLoss
