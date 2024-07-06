import torch


class OffloadSavedTensorHook:
    offload_device = "cpu"

    @staticmethod
    def unpack(data):
        origin_device, x = data
        x = x.to(origin_device)
        return x

    @staticmethod
    def pack(x):
        origin_device = x.device
        return origin_device, x.to(OffloadSavedTensorHook.offload_device)
