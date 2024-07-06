import numpy as np
import torch
from torchviz import make_dot


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


x = torch.randn(3, 10, requires_grad=True)
y = torch.randn(3, 10, requires_grad=True)
model = Model()
pred = model(x)
loss_fn = torch.nn.MSELoss()
loss = loss_fn(pred, y)
print(loss)
model.to("cuda")
graph = make_dot(loss, params={f"weight({model.linear.weight.device})": model.linear.weight}.update({'x': x, 'y': y}),
                 show_saved=True, show_attrs=True)
loss.backward()


print(model.linear.weight.grad)
print(model.linear.weight.device)
print(model.linear.weight.grad.device)
print(x.grad)
print(x.grad.device)
