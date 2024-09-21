import torch
import torch.nn as nn


class FFNNeuralNetwork(nn.Module):
    def __init__(self, in_features=10, out_features=12, n_layers=2):
        super().__init__()
        layers = []
        current_in_features = in_features

        for _ in range(n_layers):
            layers.append(nn.Linear(current_in_features, 20))
            layers.append(nn.ReLU())
            current_in_features = 20
        layers.append(nn.Linear(current_in_features, out_features))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def IrisFFNNeuralNetwork(in_features=4, out_features=3, n_layers=2):
    return FFNNeuralNetwork(
        in_features=in_features, out_features=out_features, n_layers=n_layers
    )
