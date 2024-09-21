import torch
from torch import nn
from torchdiffeq import odeint


class ODEfunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, t, x):
        x = self.fc(x)
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


class DNNClassificationODEModel(nn.Module):
    def __init__(self, in_features=10, out_features=12, ode_features=10, T=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ode_features = ode_features
        self.in_layer = nn.Linear(in_features=in_features, out_features=ode_features)
        self.odeBlock = ODEBlock(ode_func=ODEfunc(dim=ode_features), T=T)
        self.out_layer = nn.Linear(ode_features, out_features)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.odeBlock(x)
        x = self.out_layer(x)
        return x
