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
from class_model.gpt2Tokenizer import GPT2Tokenizer


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
        self.loss_fn = algorithm_config.loss_fn(ignore_index=self.tokenizer.pad_id)

    # @wandb_loger(desc="")
    def train_a_batch(self, x, y):
        print("=" * 20)
        self.model.train()
        # print("x.shape=", x.shape)
        x = x.to(self.device).long()
        y = y.to(self.device).long()
        # print("train_a_batch x.device", x.device)
        # ==============================================
        att_mask, pad_mask = self.get_masks(x)
        att_mask = att_mask.to(self.device)
        pad_mask = pad_mask.to(self.device)
        o = self.model(x, att_mask, pad_mask)
        o: torch.Tensor
        y: torch.Tensor
        loss = self.loss_fn(o.reshape(-1, o.shape[-1]), y.reshape(-1))
        # =============================================
        self.loss_list.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        from my_utils.resouces_monitor import show_gpu_and_cpu_memory
        show_gpu_and_cpu_memory()
        self.optimizer.step()
        loss_val = loss.item()
        self.step += 1

        result = {
            "train_step_loss": loss_val,
            "step": self.step,
        }

        print(result)
        return
        print(self.text_generate(prompt="天空突然下雨"))

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
                att_mask = att_mask.to(self.device)
                pad_mask = pad_mask.to(self.device)
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

    def greedy_decode(self, max_gen_len: int, prompt: str):
        self.tokenizer: GPT2Tokenizer
        x = self.collate_fn([prompt])
        x = x.to(self.config.device)
        self.model.eval()
        with torch.no_grad():
            for i in range(max_gen_len):
                att_mask, pad_mask = self.get_masks(x)
                # ==============================================
                att_mask, pad_mask = self.get_masks(x)
                att_mask = att_mask.to(self.device)
                pad_mask = pad_mask.to(self.device)
                out = self.model(x, att_mask, pad_mask)
                _, next_word = torch.max(out[:, -1, :], dim=1)
                next_word = next_word.item()
                x = torch.cat([x, torch.ones(1, 1).type_as(x).fill_(next_word)], dim=1)
                if next_word == self.tokenizer.eos_id:
                    break
        return x[0]

    def text_generate(self, prompt: str) -> str:
        x = self.greedy_decode(max_gen_len=300, prompt=prompt)
        x = x.tolist()
        self.tokenizer: GPT2Tokenizer
        token_list = self.tokenizer.convert_id_to_token(x)
        text = "".join(token_list).replace(self.tokenizer.eos_token, "")
        return text
