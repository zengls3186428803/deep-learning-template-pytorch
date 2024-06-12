import torch
import torch.nn as nn


linear_layer = nn.Linear(in_features=10, out_features=5)


print(f"Weight dtype: {linear_layer.weight.dtype}")
print(f"Bias dtype: {linear_layer.bias.dtype}")
