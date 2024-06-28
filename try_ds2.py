from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
import torch

class MyModel(torch.nn.Module):
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


# Enable CPU Offloading
model = MyModel()
trainer = Trainer(gpus=4, plugins="deepspeed_stage_3_offload", precision=16)
trainer.fit(model)

# Enable CPU Offloading, and offload parameters to CPU
model = MyModel()
trainer = Trainer(
    gpus=4,
    plugins=DeepSpeedPlugin(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
        remote_device="nvme",
        offload_params_device="nvme",
        offload_optimizer_device="nvme",
        nvme_path="/local_nvme",
    ),
    precision=16,
)
trainer.fit(model)
