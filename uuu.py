import torch
import torch.nn as nn

# 定义一个线性层
linear_layer = nn.Linear(in_features=10, out_features=5)

# 打印线性层的权重和偏置的数据类型
print(f"Weight dtype: {linear_layer.weight.dtype}")
print(f"Bias dtype: {linear_layer.bias.dtype}")