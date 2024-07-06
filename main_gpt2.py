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
    train_config.device = "cuda"
    from tools_for_quant_offload.forward_hook import OffloadHookContext
    from tools_for_quant_offload.graph_hook import OffloadSavedTensorHook
    with OffloadHookContext(
            model=gpt,
            device="cuda",
            no_split_module_classes=["GPT2TransformerBlock"],
            enable=True,
            with_backward_hook=False,
            num_block=2,
    ):
        with torch.autograd.graph.saved_tensors_hooks(
                pack_hook=OffloadSavedTensorHook.pack,
                unpack_hook=OffloadSavedTensorHook.unpack,
        ):
            flag = False
            for batch in train_loader:
                batch = get_collate_fn(tokenizer=gpt2tokenizer, max_len=20)(batch)
                # (batch_size, batch_seq_len)
                x = batch[:, :-1]
                y = batch[:, 1:].to("cuda")
                seq_len = x.shape[1]
                attention_mask = (torch.ones(seq_len, seq_len) - torch.triu(torch.ones(seq_len, seq_len))).type(
                    torch.bool).transpose(0, 1)
                padding_mask = (x == gpt2tokenizer.pad_id)
                att_mask, pad_mask = attention_mask, padding_mask
                att_mask = att_mask.to(train_config.device)
                pad_mask = pad_mask.to(train_config.device)
                o = gpt(x, att_mask, pad_mask)
                o: torch.Tensor
                y: torch.Tensor
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(o.reshape(-1, o.shape[-1]), y.reshape(-1))
                if not flag:
                    from my_utils.cgraph import get_compute_graph
                    get_compute_graph(gpt, input={"x": x, "tgt_mask": att_mask, "tgt_key_padding_mask": pad_mask}, )
                    flag = True
                # =============================================
                loss.backward()
                from tools_for_quant_offload.resource_monitor import show_gpu_and_cpu_memory
                show_gpu_and_cpu_memory()

            # trainer = GPTTrainer(
            #     model=gpt,
            #     dataloaders=(train_loader, eval_loader, test_loader),
            #     config=train_config,
            #     algorithm_config=algorithm_config,
            #     data_config=data_config,
            #     collate_fn=get_collate_fn(gpt2tokenizer, 500),
            #     tokenizer=gpt2tokenizer
            # )
            # trainer.train()


if __name__ == '__main__':
    main()
