import torch

x = torch.Tensor([1, 2]).cuda()
packed = x.device, x.dtype, x.shape, x.untyped_storage().cpu()

origin_device, origin_dtype, origin_shape, storage = packed

y = torch.empty(size=origin_shape, dtype=origin_dtype, device=origin_device)
y.untyped_storage()
print(y)
