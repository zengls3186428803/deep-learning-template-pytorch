import os

import accelerate
from accelerate import DeepSpeedPlugin
from torch.utils.data import DataLoader, Dataset
import deepspeed
import torch
from torch.nn import MSELoss
from torch.optim import SGD
import hydra

from deep_speed.util import get_offload_model_using_deep_speed


class DistributionEnvironmentContext:
    import os
    def __init__(self):
        pass

    def __enter__(self):
        self.init_dist_environment()
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy_dist_environment()
        pass

    def init_dist_environment(self):
        # os.environ["MASTER_ADDR"] = "127.0.0.1"
        # os.environ["MASTER_PORT"] = "9988"
        # os.environ["LOCAL_RANK"] = "0"
        # os.environ["RANK"] = "0"
        # os.environ["WORLD_SIZE"] = "1"

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "9988"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        torch.distributed.init_process_group("nccl")

    def destroy_dist_environment(self):
        from torch.distributed import destroy_process_group
        torch.distributed.destroy_process_group()


dim = 1024


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = dict()
        self.data["x"] = torch.randn(30, dim)
        self.data["y"] = torch.randn(30, dim)

    def __getitem__(self, item):
        return self.data["x"][item], self.data["y"][item]

    def __len__(self):
        return len(self.data["y"])


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nb = 200
        self.fc = torch.nn.ModuleList(
            [
                torch.nn.Linear(dim, dim, bias=False)
                for i in range(self.nb)
            ]
        )

    def forward(self, x: torch.Tensor):
        for m in self.fc:
            # print(m)
            x = m(x)
        return x


cache = dict()


def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    cached_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated()
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Cached Memory: {cached_memory:.2f} GB")
    print(f"max_memory_allocated {max_allocated / (1024 ** 3)} GB")


def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    print(cfg)
    model = M()
    for n, p in model.named_parameters():
        p.register_hook(get_record_gradient_hook(model, cache))
    dataset = MyDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=3)
    optimizer = SGD(model.parameters(), lr=1e-2)
    loss_fn = MSELoss()
    offload_proportion = 0.5
    # model = get_offload_model_using_deep_speed(
    #     model=model,
    #     dataset=dataset,
    #     offload_proportion=offload_proportion,
    #     offload_strategy=None,
    # )
    plugin = DeepSpeedPlugin(
        model,
        zero_stage=3,
        offload_param_device="cpu",
        # offload_optimizer_device="cpu",
    )
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True
            # },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "reduce_bucket_size": 1e5,
            "stage3_prefetch_bucket_size": 1e5,
            "stage3_param_persistence_threshold": int(
                (1 - offload_proportion) * sum(p.numel() for p in model.parameters())),
            "stage3_max_live_parameters": int((1 - offload_proportion) * sum(p.numel() for p in model.parameters())),
            "stage3_max_reuse_distance": 0,
        },
        "train_batch_size": 5,
        "train_micro_batch_size_per_gpu": 5,
        "wall_clock_breakdown": False,
        "zero_force_ds_cpu_optimizer": False,
        "zero_allow_untested_optimizer": True,
        "logging": {
            "level": "info"
        },
        "profiling": {
            "enabled": True,
            "profile_step": 1,
            "module_name": "deepspeed",
            "report_name": "ds_report.html"
        }
    }
    plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 5
    plugin.deepspeed_config.update(
        ds_config
    )

    accelerator_tmp = accelerate.Accelerator(deepspeed_plugin=plugin)
    # model.to(accelerator.device)
    model = accelerator_tmp.prepare(model)
    # with torch.autograd.graph.save_on_cpu():
    for i in range(5000):
        for x, y in dataloader:
            # x = x.to("cuda")
            # y = y.to("cuda")
            o = model(x)
            loss = loss_fn(o, y)
            print_gpu_memory()
            loss.backward()
            print_gpu_memory()
            # accelerator.backward(loss)
            loss: torch.Tensor
            print(loss)


if __name__ == '__main__':
    # with DistributionEnvironmentContext():
    main()
