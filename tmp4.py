import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset

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
            x = m(x)
        return x


def main():
    model = M().cuda()
    x = torch.randn(20, dim, requires_grad=True).cuda()
    y = torch.randn(20, dim, requires_grad=True).cuda()
    loss_fn = MSELoss()
    with torch.autograd.graph.save_on_cpu():
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        print(loss)


if __name__ == '__main__':
    main()
