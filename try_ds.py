from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
import deepspeed
import torch
from torch.nn import MSELoss
from torch.optim import SGD
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam

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
        self.nb = 120
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
        "reduce_bucket_size": 1000,
        "stage3_prefetch_bucket_size": 1000,
        "stage3_param_persistence_threshold": 0.5,
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
    },
    "train_batch_size": 5,
    "train_micro_batch_size_per_gpu": 5,
    "wall_clock_breakdown": False,
    "zero_force_ds_cpu_optimizer": False,
    "zero_allow_untested_optimizer": True
}


def get_hook(name):
    def hook(grad):
        print(name, grad.device)

    return hook


model = M()
for n, p in model.named_parameters():
    p.register_hook(get_hook(n))
dataset = MyDataset()
# dataloader = DataLoader(dataset=dataset, batch_size=3)
optimizer = SGD(model.parameters(), lr=1e-2)
loss_fn = MSELoss()
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
ratio = 0.5
model, optimizer, dataloader, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    training_data=dataset,
    config=ds_config,
    model_parameters=list(model.parameters())[:int(ratio * len(list(model.parameters())))],
)
for i in range(5000):
    for x, y in dataloader:
        x = x.to("cuda")
        y = y.to("cuda")
        o = model(x)
        loss = loss_fn(o, y)
        model.backward(loss)
        model.step()
        print(loss)
