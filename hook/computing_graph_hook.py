import random


class SavedTensorOffloadHook:
    offload_probability = 1
    device = "cpu"

    @staticmethod
    def unpack(x):
        origin_device, x = x
        x = x.to(origin_device)
        return x

    @staticmethod
    def pack(x):
        p = random.random()
        if p <= SavedTensorOffloadHook.offload_probability:
            return x.device, x.to(SavedTensorOffloadHook.device)
        else:
            return x.device, x
