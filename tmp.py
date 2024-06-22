import os
import time

import hydra
import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from get_dataloader.douban import get_douban_dataloader
from class_model.gpt2Tokenizer import GPT2Tokenizer
from class_model.gpt2 import GPT2
from class_train.gptTrainer import GPTTrainer
from class_config.trainConfig import TrainConfig
from class_config.algorithmConfig import AlgorithmConfig
from class_config.dataConfig import DataConfig
from class_config.wandbConfig import WandbConfig
from my_utils.seed_all import seed_everything
from omegaconf import DictConfig, OmegaConf

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class GPT2Config:
    EMBED_DIM = 10
    N_HEAD = 1
    DROPOUT = 0.1
    N_BLOCK_GPT = 1
    BATCH_FIRST = True
    BATCH_SIZE = 64
    MAX_GEN_LEN = 128
    MAX_POS = 5000


def data_generator(dataloader):
    for batch in dataloader:
        for data in batch:
            yield data


def transform_text_to_tensor(text: str, tokenizer: GPT2Tokenizer):
    return torch.Tensor(
        tokenizer.convert_token_to_id(
            tokenizer.tokenize(text) + [tokenizer.eos_token]
        )
    )


def get_collate_fn(tokenizer: GPT2Tokenizer):
    def collate_fn(batch):
        collated_batch = []
        for sample in batch:
            collated_batch.append(transform_text_to_tensor(sample.rstrip("\n"), tokenizer))
        collated_batch = pad_sequence(
            collated_batch,
            padding_value=tokenizer.convert_token_to_id([tokenizer.pad_token])[0],
            batch_first=True
        )
        return collated_batch.long()

    return collate_fn


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
    # gpt2tokenizer.build_vocab([text for text in data_generator(train_loader)])
    # gpt2tokenizer.save_state_dict()
    gpt2tokenizer.load_state_dict()
    print(gpt2tokenizer.get_vocab_size())
    gpt2config = GPT2Config()
    gpt = GPT2(
        vocab_size=gpt2tokenizer.get_vocab_size(),
        embed_dim=gpt2config.EMBED_DIM,
        num_head=gpt2config.N_HEAD,
        num_block_gpt=gpt2config.N_BLOCK_GPT,
        max_pos=gpt2config.MAX_POS,
        batch_first=gpt2config.BATCH_FIRST,
        dropout=gpt2config.DROPOUT
    )

    trainer = GPTTrainer(
        model=gpt,
        dataloaders=(train_loader, eval_loader, test_loader),
        config=train_config,
        algorithm_config=algorithm_config,
        data_config=data_config,
        collate_fn=get_collate_fn(gpt2tokenizer),
        tokenizer=gpt2tokenizer
    )
    trainer.train()


if __name__ == '__main__':
    main()
