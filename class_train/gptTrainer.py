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
            collate_fn=None,
            tokenizer=None
    ):
        super().__init__(
            model,
            dataloaders,
            config,
            algorithm_config,
            data_config
        )
        self.collate_fn = collate_fn
        self.tokenizer = tokenizer

    @wandb_loger(desc="")
    def train_a_batch(self, x, y):
        self.model.train()
        x = x.to(self.device).long()
        y = y.to(self.device).long()
        # ==============================================
        att_mask, pad_mask = self.get_masks(x)
        o = self.model(x, att_mask, pad_mask)
        o: torch.Tensor
        y: torch.Tensor
        loss = self.loss_fn(o.reshape(-1, o.shape[-1]), y.reshape(-1))
        # =============================================
        self.loss_list.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()
        self.step += 1

        result = {
            "train_step_loss": loss_val,
            "step": self.step,
        }
        print(result)

        if self.step % self.config.evaluate_interval_steps == 0:
            flag = f"evaluate_set per {self.config.evaluate_interval_steps} step : "
            result_eval = self.evaluate(
                self.evaluate_dataloader,
                flag=flag
            )
            result.update(result_eval)
            flag = f"test_set per {self.config.evaluate_interval_steps} step : "
            result_eval = self.evaluate(
                self.test_dataloader,
                flag=flag
            )
            result.update(result_eval)
            flag = f"evaluate per {self.config.evaluate_interval_steps} step : "
            result[flag + "step"] = self.step
        return result

    @wandb_loger(desc="")
    def train_a_epoch(self):
        total = len(self.train_dataloader)
        p_bar = tqdm(total=total, desc="step(iteration)([mini]-batch)")
        for batch in self.train_dataloader:
            batch = self.collate_fn(batch)
            # (batch_size, batch_seq_len)
            x = batch[:, :-1]
            y = batch[:, 1:]
            self.train_a_batch(x, y)
            p_bar.update(1)
        self.epoch += 1
        result = {
            "epoch": self.epoch,
        }
        if self.epoch % self.config.evaluate_interval_epochs == 0:
            flag = f"evaluate_set per {self.config.evaluate_interval_epochs} epoch : "
            result_eval = self.evaluate(
                self.evaluate_dataloader,
                flag=flag,
            )
            result.update(result_eval)
            flag = f"test_set per {self.config.evaluate_interval_epochs} epoch : "
            result_eval = self.evaluate(
                self.test_dataloader,
                flag=flag,
            )
            result.update(result_eval)
            flag = f"evaluate per {self.config.evaluate_interval_epochs} epoch : "
            result[flag + "epoch"] = self.epoch
        return result

    def get_masks(self, data: torch.Tensor):
        from class_model.gpt2Tokenizer import GPT2Tokenizer
        self.tokenizer: GPT2Tokenizer
        seq_len = data.shape[1]
        attention_mask = (torch.ones(seq_len, seq_len) - torch.triu(torch.ones(seq_len, seq_len))).type(
            torch.bool).transpose(0, 1)
        padding_mask = (data == self.tokenizer.pad_id)
        return attention_mask, padding_mask

    def evaluate(self, dataloader: DataLoader, flag=""):
        self.model.eval()
        with torch.no_grad():
            correct_total = 0
            all_total = 0
            loss_total = 0
            n_batch = len(dataloader)
            p_bar = tqdm(total=n_batch, desc="evaluate")
            for batch in dataloader:
                batch = self.collate_fn(batch)
                # (batch_size, batch_seq_len)
                x = batch[:, :-1]
                y = batch[:, 1:]
                x = x.to(self.config.device)
                y = y.to(self.config.device)
                # ==============================================
                att_mask, pad_mask = self.get_masks(x)
                o = self.model(x, att_mask, pad_mask)
                o: torch.Tensor
                y: torch.Tensor
                logits = o.reshape(-1, o.shape[-1])
                y = y.reshape(-1)
                loss = self.loss_fn(logits, y)
                # =============================================
                accurate, correct, batch_size = self.compute_accurate(logits, y)
                all_total += batch_size
                correct_total += correct
                loss_total += loss.item()
                p_bar.update(1)
            print(flag +
                  f"average_loss={loss_total / n_batch},average_acc={correct_total / all_total},correct_total/all_total={correct_total}/{all_total}")
            average_loss = loss_total / n_batch
            return {
                flag + "average_loss": average_loss,
                flag + "acc": correct_total / all_total,
                flag + "correct_total": correct_total,
                flag + "all_total": all_total,
            }
