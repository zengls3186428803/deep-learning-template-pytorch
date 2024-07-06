import torch
from class_model.longLinear import LongLinearModel
from tools_for_quant_offload.resource_monitor import show_gpu_and_cpu_memory

model = LongLinearModel(n_layers=200)
with torch.autograd.graph.save_on_cpu():
    model.to("cuda")
    print("pre-forward" + "=" * 20)
    show_gpu_and_cpu_memory()
    x = torch.randn(10, 1024).cuda()
    pred = model(x)
    print("after-forward" + "=" * 20)
    show_gpu_and_cpu_memory()
    model.to("cpu")
    print("after move" + "=" * 20)
    show_gpu_and_cpu_memory()
