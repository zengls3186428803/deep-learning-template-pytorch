from class_train.trainer import Trainer
import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from class_config.algorithmConfig import AlgorithmConfig
from my_utils.decorator import wandb_loger
from class_config.trainConfig import TrainConfig
from class_config.dataConfig import DataConfig


class GPTTrainer(Trainer):
    def __init__(
            self,
            model: torch.nn.Module,
            dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
            config: TrainConfig,
            algorithm_config: AlgorithmConfig,
            data_config: DataConfig = None,
    ):
        super().__init__(
            model,
            dataloaders,
            config,
            algorithm_config,
            data_config
        )

