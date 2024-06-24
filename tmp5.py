import os

import accelerate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from accelerate import Accelerator
from accelerate.hooks import (attach_execution_device_hook, attach_align_device_hook, remove_hook_from_submodules)
from peft import prepare_model_for_kbit_training
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


def get_dispatch_model(model: torch.nn.Module, strategy="auto", device_map=None, max_memory=None, offload_dir=None,
                       state_dict=None):
    import accelerate
    from accelerate.big_modeling import dispatch_model, load_checkpoint_and_dispatch
    from accelerate.big_modeling import cpu_offload
    match strategy:
        case "only_cpu":
            model = cpu_offload(model)
        case "auto":
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
            )
            print("auto device map is ", device_map)
            # model = load_checkpoint_and_dispatch(model)
            model = dispatch_model(
                model,
                device_map=device_map,
                offload_dir=offload_dir,
                state_dict=state_dict,
            )
        case "device_map":
            assert device_map is not None, "device map is None"
            model = dispatch_model(model, device_map=device_map)
        case _:
            print("No offload policy is specified, so do nothing")
    return model


def main(cfg: DictConfig = None):
    model = M()
    assert 1 == 2
    # =========================translate DictConfig to class-Config=====================
    device = "cuda"
    model = M().cpu()

    print("after")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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
