import torch
from torch import nn
from torchdiffeq import odeint

device = "cuda" if torch.cuda.is_available() else "cpu"


# device = "cpu"

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
