import copy

import torch

from bitsandbytes.nn.modules import Linear4bit, Linear8bitLt
from torch import nn
from torchviz import make_dot

from my_utils.cgraph import get_compute_graph

m4 = Linear4bit(1, 1, quant_type="nf4", bias=False)
m = nn.Linear(1, 1, bias=False)
m4.load_state_dict(m.state_dict())
m4 = m4.cuda()
# m4 = m.cuda()

cache = list()


def pack_hook(x):
    print(f"Packing {x}")
    cache.append(x)
    return x


def unpack_hook(x):
    print("Unpacking", x)
    return x


z = torch.ones((1, 1), requires_grad=True, device=0)
x = z * 2
y = torch.ones((1, 1), device=0) * 3
print("weight=", m4.weight)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    loss_fn = nn.MSELoss()
    pred = m4(x)
    loss = loss_fn(pred, y)
    make_dot(
        loss,
        dict(
            m4.named_parameters()
        ),
        show_attrs=True
    ).render()
    loss: torch.Tensor
    loss.retain_grad()
    loss.backward()

print(cache)
