import torch.nn
from .model_offload import ModelOffloadHookContext
from .gradient_offload import GradientOffloadHookContext


class OffloadContext:
    def __init__(
            self,
            model: torch.nn.Module,
            named_grads: dict,
            num_block=2,
            no_split_module_classes=None,
            enable_gradient_offload=True,
            enable_model_offload=True,

    ):
        self.modelOffloadHookContext = ModelOffloadHookContext(
            model=model,
            no_split_module_classes=no_split_module_classes,
            num_block=num_block,
            enable=enable_model_offload,
            # =========================
            device="cuda",
            strategy="block",
            with_backward_hook=False
        )
        self.gradientOffloadHookContext = GradientOffloadHookContext(
            model=model,
            enable=enable_gradient_offload,
            record_dict=named_grads,
        )

    def __enter__(self):
        self.modelOffloadHookContext.__enter__()
        self.gradientOffloadHookContext.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.modelOffloadHookContext.__exit__(exc_type, exc_val, exc_tb)
        self.gradientOffloadHookContext.__exit__(exc_type, exc_val, exc_tb)
