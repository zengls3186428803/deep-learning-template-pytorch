from class_train.trainer import Trainer
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from class_config.algorithmConfig import AlgorithmConfig
from my_utils.decorator import wandb_loger
from class_config.trainConfig import TrainConfig
from class_config.dataConfig import DataConfig
from torch.utils.hooks import RemovableHandle
from wandb import Table


class CovTrainer(Trainer):
    def __init__(
            self,
            model: torch.nn.Module,
            dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
            config: TrainConfig,
            algorithm_config: AlgorithmConfig,
            data_config: DataConfig,
            dataloader_one: DataLoader,
            dataloader_full: DataLoader,
    ):
        super().__init__(
            model=model,
            dataloaders=dataloaders,
            config=config,
            algorithm_config=algorithm_config,
            data_config=data_config)
        self.dataloader_one = dataloader_one
        self.dataloader_full = dataloader_full
        self.cache = dict()

    @wandb_loger(desc="")
    def train_a_batch(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        o = self.model(x)
        loss = self.loss_fn(o, y)
        self.loss_list.append(loss.item())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss_val = loss.item()
        self.step += 1

        result = {
            "train_step_loss": loss_val,
            "step": self.step,
        }

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
            result[flag] = self.step
            flag = f"calculate per {self.config.evaluate_interval_steps} step : "
            result[flag + "trace"] = self.get_trace_of_gcm(self.model, self.dataloader_one)
        if self.step == self.config.calculate_eigenvalue_at_step:
            flag = f"calculate per {self.config.evaluate_interval_steps} step : "
            grad_covariance = self.get_covariance_matrix(self.model, self.dataloader_one, self.dataloader_full)
            eigenvalue_list = self.get_eigenvalue(grad_covariance)
            eigenvalue_table = Table(
                columns=["eigenvalue " + str(i) for i in range(1, len(eigenvalue_list) + 1)],
                data=[eigenvalue_list]
            )
            result[flag + "eigenvalue_table"] = eigenvalue_table
        return result

    def get_grad_vector(self) -> (list, int):
        # return tuple(parameter_vector, the number of Module that have grad)
        result = torch.Tensor([]).to(self.device)
        cnt = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    vec_p = torch.flatten(p.grad)
                    result = torch.concat([result, vec_p], dim=0)
                    cnt += 1
        return result.requires_grad_(False).to(self.device)

    def get_covariance_matrix(self, model: torch.nn.Module, dataloader_one: DataLoader,
                              dataloader_full: DataLoader):
        grad_covariance_one = None
        n = len(dataloader_one)
        p_bar = tqdm(total=n, desc="calculating first covariance matrix:")
        for x, y in dataloader_one:
            x = x.to(self.device)
            y = y.to(self.device)
            o = model(x)
            loss = self.loss_fn(o, y)
            loss.backward()
            grad_vector = self.get_grad_vector()
            grad_covariance = torch.einsum("i,j->ij", grad_vector, grad_vector)
            if grad_covariance_one is None:
                grad_covariance_one = torch.zeros_like(grad_covariance)
            grad_covariance_one += (1 / n) * grad_covariance
            self.optimizer.zero_grad()
            p_bar.update(1)

        p_bar = tqdm(total=n, desc="calculating second covariance matrix:")
        for x, y in dataloader_full:
            x = x.to(self.device)
            y = y.to(self.device)
            o = model(x)
            loss = self.loss_fn(o, y)
            loss = (1 / n) * loss
            loss.backward()
            p_bar.update(1)
        grad_vector = self.get_grad_vector()
        self.optimizer.zero_grad()
        grad_covariance_full = torch.einsum("i,j->ij", grad_vector, grad_vector)

        gcm = (grad_covariance_one - grad_covariance_full).to(self.device).requires_grad_(False)
        b = self.data_config.batch_size
        gcm = ((n - b) / (b * (n - 1))) * gcm
        return gcm

    def get_eigenvalue(self, matrix: torch.Tensor, flag=""):
        p_bar = tqdm(total=1, desc="calculating eigenvalue")
        eigenvalue = torch.linalg.eigvalsh(matrix)
        eigenvalue: torch.Tensor
        p_bar.update(1)
        return eigenvalue.tolist()

    def get_trace_of_gcm(self, model: torch.nn.Module, dataloader_one, dataloader_full=None):
        self.register_hook_for_gcm_trace()
        n = len(dataloader_one)
        p_bar = tqdm(total=n, desc="calculate trace of gcm")
        trace_1 = None
        for x, y in dataloader_one:
            x = x.to(self.device)
            y = y.to(self.device)
            o = model(x)
            loss = self.loss_fn(o, y)
            loss = loss / n
            loss.backward()
            grad_vector = self.get_grad_vector()
            grad_vector = grad_vector * n
            if trace_1 is None:
                trace_1 = torch.zeros(1).to(self.device)
            trace_1 += (1 / n) * torch.sum(grad_vector * grad_vector)
            self.optimizer.zero_grad()
            p_bar.update(1)
        b = self.data_config.batch_size
        trace_2 = torch.zeros(1).to(self.device)
        for name, p in self.model.named_parameters():
            trace_1 += torch.sum(self.cache[name]["grad"] * self.cache[name]["grad"])
        trace = ((n - b) / (b * (n - 1))) * (trace_1 - trace_2)
        self.remove_hook_for_gcm_trace()
        return trace

    def register_hook_for_gcm_trace(self):
        with torch.no_grad():
            for name, p in tqdm(self.model.named_parameters(), desc="register hook for gcm trace"):
                if p.requires_grad:
                    self.cache[name] = dict()
                    self.cache[name]["grad"] = torch.zeros_like(p)
                    self.cache[name]["handle"] = p.register_hook(self.get_hook_for_trace(name=name))

    def remove_hook_for_gcm_trace(self):
        with torch.no_grad():
            for name, p in tqdm(self.model.named_parameters(), desc="remove hook for gcm trace"):
                if p.requires_grad and isinstance(self.cache[name]["handle"], RemovableHandle):
                    self.cache[name]["handle"].remove()

    def get_hook_for_trace(self, name):
        def grad_hook(grad):
            with torch.no_grad():
                if self.cache.get(name, None) is not None:
                    self.cache[name]["grad"] += grad
                else:
                    self.cache[name]["grad"] = grad
            return grad

        return grad_hook
