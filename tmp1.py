import hydra
from omegaconf import DictConfig, OmegaConf
from get_dataloaders import (get_fashion_mnist_loaders_5)
from class_train.gcmTrainer import CovTrainer
from class_config.algorithmConfig import AlgorithmConfig
from class_config.dataConfig import DataConfig
from class_config.wandbConfig import WandbConfig
from class_config.modelConfig import ModelConfig
from class_config.trainConfig import TrainConfig
import wandb
import time
from my_utils.seed_all import seed_everything
from class_model.imageClassificationODEModel import ImageClassificationModel


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
    train_loader, train_eval_loader, test_loader, dataloader_one, dataloader_full = get_fashion_mnist_loaders_5(
        data_aug=False,
        batch_size=data_config.batch_size,
        test_batch_size=data_config.test_batch_size
    )
    # model = AvilaClassificationModel(T=model_config.T)
    # train_loader, train_eval_loader, test_loader, dataloader_one, dataloader_full = get_avila_loaders_5(
    #     batch_size=data_config.batch_size,
    #     test_batch_size=data_config.test_batch_size
    # )

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

    # train_config.device = "cpu"
    # ================================trainer==========================================
    trainer = CovTrainer(
        model=model,
        dataloaders=(train_loader, train_eval_loader, test_loader),
        config=train_config,
        algorithm_config=algorithm_config,
        data_config=data_config,
        dataloader_one=dataloader_one,
        dataloader_full=dataloader_full
    )
    trainer.train()


if __name__ == "__main__":
    main()
    pass
