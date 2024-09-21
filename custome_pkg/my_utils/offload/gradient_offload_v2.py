import torch.nn


def get_offload_grad_hook(offload_attr_name="offloaded_grad", offload_device="cpu", init_iters="init_iters", *arg,
                          **kwargs):
    def offload_grad_hook(x):
        # print("offload_grad_hook is called")
        if x.grad is not None:
            grad = x.grad
            x.grad = None
            if not hasattr(x, offload_attr_name):
                setattr(x, offload_attr_name, grad.to(offload_device))
                setattr(x, init_iters, 0)
            else:
                accumulated_grad = getattr(x, offload_attr_name)
                setattr(x, offload_attr_name, accumulated_grad + grad.to(offload_device))
                init_niter = getattr(x, init_iters)
                setattr(x, init_iters, init_niter + 1)

    return offload_grad_hook


class GradientOffloadHookContext:
    def __init__(self, model: torch.nn.Module, enable: bool, *args, **kwargs):
        self.enable = enable
        if not enable:
            return
        self.model = model
        self.offload_attr_name = "offloaded_grad"
        self.offload_device = "cpu"
        self.handle_list = list()

    def __enter__(self):
        if not self.enable:
            return
        for n, p in self.model.named_parameters():
            handle = p.register_post_accumulate_grad_hook(
                hook=GradientOffloadHookContext.get_offload_grad_hook(
                    offload_attr_name=self.offload_attr_name,
                    offload_device=self.offload_device
                )
            )
            self.handle_list.append(handle)
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        for handle in self.handle_list:
            handle.remove()
        pass

    @staticmethod
    def get_offload_grad_hook(offload_attr_name="offloaded_grad", offload_device="cpu", init_iters="init_iters", *arg,
                              **kwargs):
        def offload_grad_hook(x):
            # print("offload_grad_hook is called")
            if x.grad is not None:
                grad = x.grad
                x.grad = None
                if not hasattr(x, offload_attr_name):
                    setattr(x, offload_attr_name, grad.to(offload_device))
                    setattr(x, init_iters, 0)
                else:
                    accumulated_grad = getattr(x, offload_attr_name)
                    setattr(x, offload_attr_name, accumulated_grad + grad.to(offload_device))
                    init_niter = getattr(x, init_iters)
                    setattr(x, init_iters, init_niter + 1)

        return offload_grad_hook