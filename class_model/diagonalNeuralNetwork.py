from typing import Iterator
import torch
from torch import nn
from torch.nn import Parameter

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
