import os
import time
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from get_dataloader.douban import get_douban_dataloader
from class_model.gpt2Tokenizer import GPT2Tokenizer
from class_model.gpt2 import GPT2
from class_train.gptTrainer import GPTTrainer
from class_config.trainConfig import TrainConfig
from class_config.algorithmConfig import AlgorithmConfig
from class_config.dataConfig import DataConfig
from class_config.wandbConfig import WandbConfig
from class_config.modelConfig import GPT2Config
from my_utils.seed_all import seed_everything
from my_utils.data_processer import get_collate_fn

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def data_generator(dataloader):
    for batch in dataloader:
        for data in batch:
            yield data


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train_config = TrainConfig(cfg)
    algorithm_config = AlgorithmConfig(cfg)
    data_config = DataConfig(cfg)
    seed_everything(train_config.seed)
    wandb_config = WandbConfig(cfg)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    wandb.init(
        project=wandb_config.project,
        group=wandb_config.group,
        entity=wandb_config.entity,
        name=timestamp,
        config=dict(
            num_epochs=train_config.num_epochs,
            learning_rate=algorithm_config,
            optimizer=algorithm_config.optimizer_name,
        ),
        mode=wandb_config.mode,
        reinit=True,
    )

    train_loader, eval_loader, test_loader, dataloader_one = get_douban_dataloader(
        data_config.batch_size,
        data_config.test_batch_size
    )
    gpt2tokenizer = GPT2Tokenizer()
    gpt2tokenizer.build_vocab([text for text in data_generator(train_loader)])
    gpt2tokenizer.save_state_dict()
    gpt2tokenizer.load_state_dict()
    print(type([k for k, _ in gpt2tokenizer.inv_vocab.items()][0]))
    print("vocab_size=", gpt2tokenizer.get_vocab_size())
    gpt2config = GPT2Config(cfg)
    gpt = GPT2(
        vocab_size=gpt2tokenizer.get_vocab_size(),
        embed_dim=gpt2config.embed_dim,
        num_head=gpt2config.n_head,
        num_block_gpt=gpt2config.n_block_gpt,
        max_pos=gpt2config.max_pos,
        batch_first=gpt2config.batch_first,
        dropout=gpt2config.dropout
    )

    trainer = GPTTrainer(
        model=gpt,
        dataloaders=(train_loader, eval_loader, test_loader),
        config=train_config,
        algorithm_config=algorithm_config,
        data_config=data_config,
        collate_fn=get_collate_fn(gpt2tokenizer, 500),
        tokenizer=gpt2tokenizer
    )
    trainer.train()


if __name__ == '__main__':
    main()
