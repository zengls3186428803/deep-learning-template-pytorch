from typing import List

import torch
import numpy as np
from tools_for_quant_offload.graph_hook import OffloadSavedTensorHook


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


class OffloadHookContext(ForwardHookForDevice):
    def __init__(
            self,
            model,
            offload_proportion=0.5,
            device="cuda",
            no_split_module_classes=None,
            with_backward_hook=False,  # for print
            enable=False,
            num_block: int = 2,
            strategy="block"  # enum["module","block"],
    ):
        self.enable = enable
        if not enable:
            return
        super().__init__()
        self.strategy = strategy
        self.num_block = num_block
        self.device = device  # computing device for offloaded modules
        self.with_backward_hook = with_backward_hook
        self.model = model
        if no_split_module_classes is None:
            no_split_module_classes = ["LlamaDecoderLayer", "GPT2TransformerBlock"]
        self.module_list = ForwardHookForDevice.get_module_list(model, no_split_module_classes=no_split_module_classes)
        print(f"module_list:{self.module_list}")
        self.offload_list = self.module_list[:int(offload_proportion * len(self.module_list))]
        # print(f"offload_module:{self.offload_list}")
        self.handle_list = list()
        self.module_info = self.get_partition_block(self.module_list, self.num_block)
        print(self.module_info)

    def __enter__(self):
        if not self.enable:
            return
        print("__enter__(self):")
        if self.strategy == "module":
            self.register_forward_hook_by_module(self.model)
        else:
            self.register_hook_by_block(self.model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        print("__exit__(self, exc_type, exc_val, exc_tb)")
        for handle in self.handle_list:
            handle.remove()

    def register_hook_by_block(self, module: torch.nn.Module, parent_name=''):
        if self.with_backward_hook and parent_name in self.module_list:
            handle = module.register_full_backward_pre_hook(hook=self.get_backward_hook(pre=True))
            self.handle_list.append(handle)
            handle = module.register_full_backward_hook(hook=self.get_backward_hook(pre=False))
            self.handle_list.append(handle)
        if parent_name in self.module_list:
            print(f"register_hook_by_block(self, module, parent_name={parent_name}")
            # forward hook==============================================================
            handle = module.register_forward_pre_hook(
                hook=self.get_forward_hook_by_block(info=self.module_info[parent_name], pre=True, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            handle = module.register_forward_hook(
                hook=self.get_forward_hook_by_block(info=self.module_info[parent_name], pre=False, with_kwargs=True),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            # backward hook==============================================================
            handle = module.register_full_backward_pre_hook(
                hook=self.get_backward_hook_by_block(info=self.module_info[parent_name], pre=True)
            )
            self.handle_list.append(handle)
            handle = module.register_full_backward_hook(
                hook=self.get_backward_hook_by_block(info=self.module_info[parent_name], pre=False)
            )
            self.handle_list.append(handle)
            return

        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            self.register_hook_by_block(sub_module, full_name)

    @staticmethod
    def get_forward_hook_by_block(info: dict, pre=True, device="cuda", with_kwargs=True):
        if device is None:
            device = "cuda"
        offload_device = "cpu"
        first_block_flag = info["first_block_flag"]
        last_block_flag = info["last_block_flag"]
        first_module_flag = info["first_module_flag"]
        last_module_flag = info["last_module_flag"]

        def pre_hook_with_kwargs(module, args, kwargs):
            # model
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in kwargs.items()}
            # saved_tensor,such as activations.
            if not last_block_flag and first_module_flag:
                print(f"set OffloadSavedTensorHook.offload_device = offload_device:{offload_device}")
                OffloadSavedTensorHook.offload_device = offload_device
            elif last_block_flag and first_module_flag:
                print(f"set OffloadSavedTensorHook.offload_device = device:{device}")
                OffloadSavedTensorHook.offload_device = device
            return args, kwargs

        def after_hook_with_kwargs(module, args, kwargs, output):
            if not last_block_flag:
                module.to(offload_device)
                pass
                # output = output.to(offload_device) if isinstance(output, torch.Tensor) else output
                # if isinstance(output, tuple):
                #     output = tuple(o.to(offload_device) if isinstance(o, torch.Tensor) else o for o in output)
            elif last_block_flag:
                module.to(device)
                pass
                # output = output.to(device) if isinstance(output, torch.Tensor) else output
                # if isinstance(output, tuple):
                #     output = tuple(o.to(device) if isinstance(o, torch.Tensor) else o for o in output)
            return output

        if pre:
            return pre_hook_with_kwargs
        else:
            return after_hook_with_kwargs

    @staticmethod
    def get_backward_hook_by_block(info: dict, pre=True, device="cuda"):
        if device is None:
            device = "cuda"
        offload_device = "cpu"
        first_block_flag = info["first_block_flag"]
        last_block_flag = info["last_block_flag"]
        first_module_flag = info["first_module_flag"]
        last_module_flag = info["last_module_flag"]

        def pre_hook(module, grad_output):
            from tools_for_quant_offload.resource_monitor import show_gpu_and_cpu_memory
            show_gpu_and_cpu_memory()
            print(
                f"pre_backward_hook: module.name={type(module)},\
                grad_output.device={[arg.device if isinstance(arg, torch.Tensor) else arg for arg in grad_output]}")
            if len(list(module.parameters())) > 0:
                print(f"module.device = {[p.device for p in module.parameters()]}")
            module.to(device)
            print("$$$$$")
            print(
                f"pre_backward_hook: module.name={type(module)},\
                grad_output.device={[arg.device if isinstance(arg, torch.Tensor) else arg for arg in grad_output]}")
            if len(list(module.parameters())) > 0:
                print(f"module.device = {[p.device for p in module.parameters()]}")
            # grad_output = tuple(grad.to(device) if isinstance(grad, torch.Tensor) else grad for grad in grad_output)
            return grad_output

        def after_hook(module, grad_input, grad_output):
            # grad_input = tuple(grad.to(device) if isinstance(grad, torch.Tensor) else grad for grad in grad_input)
            if not first_block_flag:
                module.to(offload_device)
            else:
                pass
            print(
                f"after_backward_hook: module.name={type(module)},\
                grad_output.device={[arg.device if isinstance(arg, torch.Tensor) else arg for arg in grad_output]}, \
                grad_input.device={[arg.device if isinstance(arg, torch.Tensor) else arg for arg in grad_input]}")
            if len(list(module.parameters())) > 0:
                print(f"module.device = {[p.device for p in module.parameters()]}")
            return grad_input

        if pre:
            return pre_hook
        else:
            return after_hook

    @staticmethod
    def get_backward_hook(pre=True):
        def pre_hook(module, grad_output):
            print(
                f"pre_backward_hook: module.name={type(module)},grad_output.device={[arg.device if isinstance(arg, torch.Tensor) else arg for arg in grad_output]}")
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

    def register_forward_hook_by_module(self, module: torch.nn.Module, parent_name=''):
        print(f"_register_forward_hook(self, module, parent_name={parent_name}")
        if self.with_backward_hook and parent_name in self.module_list:
            handle = module.register_full_backward_pre_hook(hook=self.get_backward_hook())
            self.handle_list.append(handle)
            handle = module.register_full_backward_hook(hook=self.get_backward_hook(pre=False))
            self.handle_list.append(handle)

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
            self.register_forward_hook_by_module(sub_module, full_name)

    @staticmethod
    def get_partition_block(module_list: list, num_block: int) -> dict:
        block_list = list()
        module_groups = [list(e) for e in np.array_split(module_list, num_block)]
        for i in range(num_block):
            block = dict()
            block["module_list"] = module_groups[i]
            block["first_block_flag"] = True if i == 0 else False
            block["last_block_flag"] = True if i == (num_block - 1) else False
            block_list.append(block)
        module_info = dict()
        for block in block_list:
            n_module = len(block["module_list"])
            for i in range(n_module):
                module_name = block["module_list"][i]
                module_info[module_name] = dict()
                module_info[module_name].update({
                    "first_block_flag": block["first_block_flag"],
                    "last_block_flag": block["last_block_flag"],
                    "first_module_flag": True if i == 0 else False,
                    "last_module_flag": True if i == (n_module - 1) else False,
                })
        return module_info
