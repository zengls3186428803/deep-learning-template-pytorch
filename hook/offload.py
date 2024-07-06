import random
import torch


def get_backward_hook(pre=True):
    def pre_hook(module, grad_output):
        print(f"pre_backward_hook: module.name={type(module)},grad_output.device={[arg.device for arg in grad_output]}")
        if len(list(module.parameters())) > 0:
            print(f"next(module.parameters()).device = {[p.device for p in module.parameters()]}")
        return grad_output

    def after_hook(module, grad_input, grad_output):
        print(
            f"after_backward_hook: module.name={type(module)},\
            grad_output.device={[arg.device if isinstance(arg, torch.Tensor) else arg for arg in grad_output]}, \
            grad_input.device={[arg.device if isinstance(arg, torch.Tensor) else arg for arg in grad_input]}")
        if len(list(module.parameters())) > 0:
            print(f"next(module.parameters()).device = {[p.device for p in module.parameters()]}")
        return grad_input

    if pre:
        return pre_hook
    else:
        return after_hook


class ForwardHookForDevice:
    def __init__(self):
        pass

    @staticmethod
    def get_align_device_pre_forward_hook(device="cuda", with_kwargs=False):
        """
        ensure same device for input and module
        """

        def hook(module: torch.nn.Module, args):
            if device is not None:
                align_device = device
            elif len(list(module.parameters())) > 0:
                align_device = next(module.parameters()).device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def hook_with_kwargs(module: torch.nn.Module, args, kwargs):
            if device is not None:
                align_device = device
            elif len(list(module.parameters())) > 0:
                align_device = next(module.parameters()).device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            _kwargs = dict()
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    _kwargs[k] = v.to(align_device)
                else:
                    _kwargs[k] = v
            kwargs = _kwargs
            return args, kwargs

        if with_kwargs:
            return hook_with_kwargs
        else:
            return hook

    @staticmethod
    def get_forward_hook(pre: bool, device=None, with_kwargs=False):
        """
        device is executing device
        origin_device is the device where tensor is saved after forward
        """
        origin_device = "cpu"
        if device is not None:
            device = device
        else:
            device = "cuda"

        def pre_hook(module: torch.nn.Module, args):
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def after_hook(module: torch.nn.Module, args, output):
            module.to(origin_device)

            output = output.to(origin_device) if isinstance(output, torch.Tensor) else output
            if isinstance(output, tuple):
                output = tuple(o.to(origin_device) if isinstance(o, torch.Tensor) else o for o in output)
            return output

        def pre_hook_with_kwargs(module, args, kwargs):
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in kwargs.items()}
            return args, kwargs

        def after_hook_with_kwargs(module, args, kwargs, output):
            module.to(origin_device)
            output = output.to(origin_device) if isinstance(output, torch.Tensor) else output
            if isinstance(output, tuple):
                output = tuple(o.to(origin_device) if isinstance(o, torch.Tensor) else o for o in output)
            return output

        if pre and with_kwargs:
            return pre_hook_with_kwargs
        elif pre and not with_kwargs:
            return pre_hook
        elif not pre and with_kwargs:
            return after_hook_with_kwargs
        elif not pre and not with_kwargs:
            return after_hook

    @staticmethod
    def get_full_name_list(model):
        full_name_list = list()

        def _get_full_name_list(module, parent_name=''):

            """
            get full name list of all submodule. result is self.
            """
            if len(list(module.named_children())) == 0:
                full_name_list.append(parent_name)
            for name, sub_module in module.named_children():
                full_name = f'{parent_name}.{name}' if parent_name else name
                _get_full_name_list(sub_module, full_name)

        _get_full_name_list(model)

        return full_name_list

    @staticmethod
    def get_module_list(model, no_split_module_classes=None):
        module_list = list()

        def _get_module_list(module: torch.nn.Module, parent_name=""):
            flag = False
            if module.__class__.__name__ in no_split_module_classes:
                flag = True
            if flag:
                module_list.append(parent_name)
                return
            if len(list(module.named_children())) == 0:
                module_list.append(parent_name)
                return

            for name, sub_module in module.named_children():
                extend_name = f"{parent_name}.{name}" if parent_name else name
                _get_module_list(sub_module, extend_name)

        _get_module_list(model)
        return module_list


class OffloadForwardHookContext(ForwardHookForDevice):
    def __init__(self, model, offload_proportion=0.5, device="cuda", no_split_module_classes=None, max_memory=None):
        super().__init__()
        self.device = device  # device for executing
        self.model = model
        if no_split_module_classes is None:
            no_split_module_classes = ["LlamaDecoderLayer", "GPT2TransformerBlock"]
        self.module_list = ForwardHookForDevice.get_module_list(model, no_split_module_classes=no_split_module_classes)
        self.offload_list = self.module_list[:int(offload_proportion * len(self.module_list))]
        self.handle_list = list()
        print(f"model_list:{self.module_list}")
        print(f"offload_module:{self.offload_list}")

    def __enter__(self):
        print("__enter__(self):")
        self._register_forward_hook(self.model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__(self, exc_type, exc_val, exc_tb)")
        for handle in self.handle_list:
            handle.remove()

    def _register_forward_hook(self, module: torch.nn.Module, parent_name=''):
        print(f"_register_forward_hook(self, module, parent_name={parent_name}")
        # if parent_name in self.module_list:
        #     handle = module.register_full_backward_pre_hook(hook=get_backward_hook())
        #     self.handle_list.append(handle)
        #     handle = module.register_full_backward_hook(hook=get_backward_hook(pre=False))
        #     self.handle_list.append(handle)

        if parent_name in self.offload_list:
            handle = module.register_forward_pre_hook(
                self.get_forward_hook(pre=True, device=self.device, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            handle = module.register_forward_hook(
                self.get_forward_hook(pre=False, device=self.device, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            return
        elif parent_name in self.module_list:
            handle = module.register_forward_pre_hook(
                self.get_align_device_pre_forward_hook(device="cuda", with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            return
        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            self._register_forward_hook(sub_module, full_name)


class SavedTensorOffloadHook:
    offload_probability = 0.5
    device = "cpu"

    @staticmethod
    def unpack(x):
        origin_device, xx = x
        xx = xx.to(origin_device)
        return xx

    @staticmethod
    def pack(x):
        p = random.random()
        if p <= SavedTensorOffloadHook.offload_probability:
            return x.device, x.to(SavedTensorOffloadHook.device)
        else:
            return x.device, x
