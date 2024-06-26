import os

import accelerate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from my_utils.dispatch import get_dispatch_model
from accelerate import Accelerator

os.environ["HYDRA_FULL_ERROR"] = "1"


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024 * 8, 1024 * 8, bias=False)
        self.fc2 = torch.nn.Linear(1024 * 8, 1024 * 8, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def get_hook(name):
    def hook(grad):
        print(f"{name} hook is called, p is at {p.device}")
        return grad

    return hook


def offload_model_to_cpu(model: torch.nn.Module, device_map):
    offloaded_modules = []

    for name, module in model.named_modules():
        print(name)
        module.to(device_map[name])
        offloaded_modules.append(name)

    print(f"Offloaded modules to CPU: {offloaded_modules}")


from accelerate.hooks import (attach_execution_device_hook, attach_align_device_hook, remove_hook_from_submodules)


def main(cfg: DictConfig = None):
    model = M()
    assert 1 == 2
    # =========================translate DictConfig to class-Config=====================
    device = "cuda"
    model = M().cpu()
    # attach_align_device_hook(model, offload=True, execution_device=device)
    print([{name: parameter} for name, parameter in model.named_parameters()])
    model = get_dispatch_model(model)
    print("after")
    print([{name: parameter} for name, parameter in model.named_parameters()])
    accelerator = Accelerator()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # (model, optimizer) = accelerator.prepare([model, optimizer])
    x = torch.randn(1, 1024, requires_grad=True, device=device)
    y = torch.randn(1, 1024, requires_grad=True, device=device)
    loss_fn = torch.nn.MSELoss()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    # accelerator.backward(loss)
    optimizer.step()
    print("loss", loss)
    print("x.grad", x.grad)
    print("y.grad", y.grad)
    print("get_grad")
    for n, p in model.named_parameters():
        print(f"{n}, p.device={p.device}, p.grad={p.grad}")


if __name__ == "__main__":
    main()
    pass
