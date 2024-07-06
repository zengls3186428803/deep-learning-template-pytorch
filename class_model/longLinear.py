import torch


class LongLinearModel(torch.nn.Module):
    def __init__(self, n_layers=200, dim=1024):
        super().__init__()
        self.nb = n_layers
        self.dim = dim
        self.fc = torch.nn.ModuleList(
            [
                torch.nn.Linear(dim, dim, bias=False)
                for i in range(self.nb)
            ]
        )

    def forward(self, x: torch.Tensor):
        i = 0
        for m in self.fc:
            # print(f"i={i},x.device={x.device}")
            x = m(x)
            i += 1
        return x
