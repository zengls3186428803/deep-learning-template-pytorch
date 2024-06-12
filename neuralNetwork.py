from typing import Iterator
import torch
from torch import nn
from torch.nn import Parameter
from torchdiffeq import odeint

device = "cuda" if torch.cuda.is_available() else "cpu"


# device = "cpu"

class DiagonalNeuralNetwork(nn.Module):
    def __init__(
            self,
            num_layers=1,
            num_features=10,
            share_parameters=False,
            device="cuda"
    ):
        super().__init__()
        self.blocks = list()
        self.num_layers = num_layers
        self.share_parameters = share_parameters
        if share_parameters:
            block = torch.ones(num_features, 1, requires_grad=True, device=device)
            # block = torch.randn(num_features, 1, requires_grad=True, device=device)
            for i in range(num_layers):
                self.blocks.append(block)
        else:
            for i in range(num_layers):
                block = torch.randn(num_features, 1, requires_grad=True, device=device)
                self.blocks.append(block)

    def forward(self, x: torch.Tensor):
        for i in range(self.num_layers):
            x = self.blocks[i] * x
        return x

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self.share_parameters:
            yield self.blocks[0]
        else:
            for block in self.blocks:
                yield block


class ConvODEfunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=1, bias=False)

    def forward(self, t, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class ConvODEBlock(nn.Module):
    def __init__(self, ode_func, T=1):
        super().__init__()
        self.ode_func = ode_func
        self.integration_time = torch.arange(start=0, end=T + 1, step=1, dtype=torch.float)
        self.T = T

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.ode_func, x, self.integration_time)
        return out[self.T]


class ImageClassificationModel(nn.Module):
    def __init__(self, in_features, out_features, T=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flattenLayer = nn.Flatten()

        self.odeBlock = ConvODEBlock(ode_func=ConvODEfunc(dim=1), T=T)
        self.outLinear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x):
        x = self.odeBlock(x)
        x = self.flattenLayer(x)
        x = self.outLinear(x)
        return x


def test_DiagonalNeuralNetwork():
    model = DiagonalNeuralNetwork(num_layers=10, share_parameters=True, device=device)
    model = model.to(device)
    model.parameters()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    x = torch.ones(10, 3) * 2
    x = x.to(device)
    label = torch.ones(10, 3)
    label = label.to(device)
    for i in range(100):
        y = model(x)
        loss = torch.nn.functional.mse_loss(y, label)
        loss.backward()
        # print(model.block.grad)
        print(loss)
        optimizer.step()
        optimizer.zero_grad()
    print(model.blocks)


class ODEfunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, t, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, ode_func, T=1):
        super().__init__()
        self.ode_func = ode_func
        self.integration_time = torch.arange(start=0, end=T + 1, step=1, dtype=torch.float)
        self.T = T

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.ode_func, x, self.integration_time)
        return out[self.T]


class AvilaClassificationModel(nn.Module):
    def __init__(self, in_features=10, out_features=12, T=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.odeBlock = ODEBlock(ode_func=ODEfunc(dim=in_features), T=T)
        self.out_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.odeBlock(x)
        x = self.out_linear(x)
        return x


if __name__ == "__main__":
    pass
    # test_DiagonalNeuralNetwork()
