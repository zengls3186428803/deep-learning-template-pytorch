import accelerate.hooks
import torch
import torch.nn as nn
from accelerate import Accelerator


def get_hook(name):
    print(name)

    def hook(grad):
        print(f"{name} hook is called, p is at {p.device}")
        return grad

    return hook


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc255 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc255(x)
        return x


def offload_model_to_cpu(model, device_map):
    offloaded_modules = []

    for name, module in model.named_children():
        if name in device_map and device_map[name] == 'cpu':
            module.to('cpu')
            offloaded_modules.append(name)

    print(f"Offloaded modules to CPU: {offloaded_modules}")


from accelerate.hooks import attach_execution_device_hook
from accelerate.hooks import attach_align_device_hook, remove_hook_from_submodules

# 示例用法
if __name__ == "__main__":
    # 初始化模型并将其放置在 CUDA 设备上
    model = M().cpu()
    model.fc1.to("cuda")
    model.fc255.to("cpu")
    device = "cuda"
    print("====pre-offload======================")
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
    # attach_execution_device_hook(model, execution_device=device)
    print("===register-hook=====================")
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         p.register_hook(get_hook(n))
    # attach_align_device_hook(model, execution_device=device, offload=True)
    print("-------------------------------")
    model.train()
    # 打印每个模块的设备信息以确认是否成功 offload
    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")

    # 使用 Accelerate 加速器准备模型
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare([model, optimizer])
    x = torch.randn(1, 1024, requires_grad=True, device=device)
    y = torch.randn(1, 1024, requires_grad=True, device=device)
    for i in range(10):
        optimizer: torch.optim.Optimizer
        optimizer.zero_grad()
        loss_fn = torch.nn.MSELoss()
        pred = model(x)
        loss = loss_fn(pred, y)
        accelerator.backward(loss)
        print("loss", loss)
        optimizer.step()

    for n, p in model.named_parameters():
        print(n, p.grad)

        remove_hook_from_submodules(model)

        # for k, v in model.state_dict().items():
        #     print(k, v.cpu())
