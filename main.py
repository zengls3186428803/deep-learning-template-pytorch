from neuralNetwork import ImageClassificationModel
import hydra
from omegaconf import DictConfig, OmegaConf
from get_dataloaders import get_fashion_mnist_loaders, get_mnist_loaders
from trainer import Trainer, TrainConfig
from config_class.algorithmConfig import AlgorithmConfig
from config_class.dataConfig import DataConfig
from config_class.wandbConfig import WandbConfig
from config_class.modelConfig import ModelConfig
import wandb
import time
from utils.seed_all import seed_everything


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # =========================translate DictConfig to class-Config=====================
    print(OmegaConf.to_yaml(cfg))
    train_config = TrainConfig(cfg)
    algorithm_config = AlgorithmConfig(cfg)
    data_config = DataConfig(cfg)
    model_config = ModelConfig(cfg)

    # =======================seed =======================================================
    seed_everything(train_config.seed)

    # ===============prepare model and data===============================================
    model = ImageClassificationModel(in_features=28 * 28, out_features=10, T=model_config.T)
    train_loader, train_eval_loader, test_loader = get_fashion_mnist_loaders(
        data_aug=False,
        batch_size=data_config.batch_size,
        test_batch_size=data_config.test_batch_size
    )

    # =======================wandb config=======================================
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    wandb_config = WandbConfig(cfg)
    wandb.init(
        project=wandb_config.project,
        group=wandb_config.group,
        entity=wandb_config.entity,
        name="T=" + str(model_config.T) + "," + timestamp,
        config=dict(
            num_epochs=train_config.num_epochs,
            learning_rate=algorithm_config,
            optimizer=algorithm_config.optimizer_name,
            T=model_config.T,
        ),
        mode=wandb_config.mode,
        reinit=True,
    )

    # ================================trainer==========================================
    trainer = Trainer(
        model=model,
        dataloaders=(train_loader, train_eval_loader, test_loader),
        config=train_config,
        algorithm_config=algorithm_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
    pass
