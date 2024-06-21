import os

import accelerate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from my_utils.cgraph import get_dispatch_model
from accelerate import Accelerator

os.environ["HYDRA_FULL_ERROR"] = "1"


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            *[torch.nn.Linear(1024, 1024, bias=False) for i in range(256)]
        )

    def forward(self, x):
        return self.fc(x)


def get_hook(name):
    def hook(grad):
        print(f"{name} hook is called, p is at {p.device}")
        return grad

    return hook


def offload_model_to_cpu(model, device_map):
    offloaded_modules = []

    for name, module in model.named_children():
        print(name, module)
        module.to(device_map[name])
        offloaded_modules.append(name)

    print(f"Offloaded modules to CPU: {offloaded_modules}")


from accelerate.hooks import (attach_execution_device_hook, attach_align_device_hook, remove_hook_from_submodules)


def main(cfg: DictConfig = None):
    # =========================translate DictConfig to class-Config=====================
    device = "cuda"
    model = M().cpu()
    attach_align_device_hook(model, offload=True, execution_device=device)
    accelerator = Accelerator()
    (model,) = accelerator.prepare([model])
    x = torch.randn(1, 1024, requires_grad=True, device=device)
    y = torch.randn(1, 1024, requires_grad=True, device=device)
    loss_fn = torch.nn.MSELoss()
    pred = model(x)
    loss = loss_fn(pred, y)
    accelerator.backward(loss)
    print(loss)
    print(x.grad)
    print(y.grad)
    remove_hook_from_submodules(model)

    # for k, v in model.state_dict().items():
    #     print(k, v.cpu())


if __name__ == "__main__":
    main()
    pass
