import torch
import accelerate


class OffloadForwardHookContext:

    def __init__(self, model, offload_proportion=0.5, device="cuda", max_memory=None, no_split_module_classes=None):
        if max_memory is None:
            max_memory = {0: "4GB", "cpu": "10GB"}
        self.model = model
        self.offload_list = list()
        self.handle_list = list()
        self.device = device
        self.device_map = accelerate.infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes
        )
        self.module_list = self.device_map.keys()
        for k, v in self.device_map.items():
            if not isinstance(v, int):
                self.offload_list.append(k)
        print(f"device_map:{self.device_map}")
        print(f"offload:{self.offload_list}")
        print(f"all:{self.module_list}")

    def __enter__(self):
        print("__enter__(self):")
        self._register_forward_hook(self.model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__(self, exc_type, exc_val, exc_tb)")
        for handle in self.handle_list:
            handle.remove()

    def _register_forward_hook(self, module, parent_name=''):
        print(f"_register_forward_hook(self, module, parent_name={parent_name}")
        if parent_name in self.offload_list:
            handle = module.register_forward_pre_hook(self.get_forward_hook(pre=True))
            self.handle_list.append(handle)
            handle = module.register_forward_hook(self.get_forward_hook(pre=False))
            self.handle_list.append(handle)
            return

        elif parent_name in self.module_list:
            handle = module.register_forward_pre_hook(self.get_align_device_pre_forward_hook(device="cuda"))
            self.handle_list.append(handle)
            return
        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            self._register_forward_hook(sub_module, full_name)

    @staticmethod
    def get_align_device_pre_forward_hook(device="cuda", with_kwargs=False):
        """
        ensure same device for input and module
        """

        def hook(module: torch.nn.Module, args):
            if len(list(module.parameters())) > 0:
                if len(list(args)) > 0:
                    print(
                        f"align_device_hook is called, module device={next(module.parameters()).device}, input device={next(iter(args)).device}")
                else:
                    print(
                        f"align_device_hook is called, module device={next(module.parameters()).device}, no input")
            else:
                print(
                    f"align_device_hook is called, module hasn't weight, input device={next(iter(args)).device}")
            if device is not None:
                align_device = device
            elif len(list(module.parameters())) > 0:
                align_device = next(module.parameters()).device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            print(tuple(arg.device if isinstance(arg, torch.Tensor) else arg for arg in args))
            return args

        def hook_with_kwargs(module, args, kwargs):
            if len(list(module.parameters())) > 0:
                if len(list(args)) > 0:
                    print(
                        f"align_device_hook is called, module device={next(module.parameters()).device}, input device={next(iter(args)).device}")
                else:
                    print(
                        f"align_device_hook is called, module device={next(module.parameters()).device}, no input")
            else:
                print(
                    f"align_device_hook is called, module hasn't weight, input device={next(iter(args)).device}")
            if device is not None:
                align_device = device
            elif len(list(module.parameters())) > 0:
                align_device = next(module.parameters()).device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = dict(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in kwargs.items())
            print(tuple(arg.device if isinstance(arg, torch.Tensor) else arg for arg in args))
            return args, kwargs

        if with_kwargs:
            return hook_with_kwargs
        else:
            return hook

    @staticmethod
    def get_forward_hook(pre: bool, device=None):
        """
        device is execute device
        """
        origin_device = "cpu"
        if device is not None:
            device = device
        else:
            device = "cuda"

        def pre_hook(module: torch.nn.Module, args):
            if len(list(module.parameters())) > 0:
                print(
                    f"pre_hook is called, module device={next(module.parameters()).device}, input device={next(iter(args)).device}")
            else:
                print(
                    f"pre_hook is called, module hasn't weight, input device={next(iter(args)).device}")
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def after_hook(module: torch.nn.Module, args, output):
            module.to(origin_device)
            output = output.to(origin_device) if isinstance(output, torch.Tensor) else output
            if isinstance(output, tuple):
                output = tuple(o.to(origin_device) if isinstance(o, torch.Tensor) else o for o in output)

            if len(list(module.parameters())) > 0:
                print(
                    f"after_hook is called, device={next(module.parameters()).device}, input device={next(iter(output)).device}")
            else:
                print(f"after_hook is called, model hasn't weight, input device={next(iter(output)).device}")
            print(output)
            print(f"typed of output is {type(output)}")
            return output

        if pre:
            return pre_hook
        else:
            return after_hook


class ForwardHookForOffloadContext:
    def __init__(self, model, offload_proportion=0.5, device="cuda"):
        self.model = model
        self.module_list = list()
        self.get_full_name(self.model)
        self.offload_list = self.module_list[:int(offload_proportion * len(self.module_list))]
        #        print(f"self.offload_list={self.offload_list}")
        self.handle_list = list()
        #        print("module_list", self.module_list)
        self.device = device

    def __enter__(self):
        #        print("__enter__(self):")
        self._register_forward_hook(self.model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        #        print("__exit__(self, exc_type, exc_val, exc_tb)")
        for handle in self.handle_list:
            handle.remove()

    def _register_forward_hook(self, module, parent_name=''):

        if len(list(module.named_children())) == 0:
            #            print("_register_forward_hook")
            if parent_name in self.offload_list:
                handle = module.register_forward_pre_hook(self.get_forward_hook(pre=True, device="cpu"))
                self.handle_list.append(handle)
                handle = module.register_forward_hook(self.get_forward_hook(pre=False, device="cpu"))
                self.handle_list.append(handle)
            else:
                handle = module.register_forward_pre_hook(self.get_align_device_pre_forward_hook(device="cuda"))
                self.handle_list.append(handle)
            return
        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            self._register_forward_hook(sub_module, full_name)

    @staticmethod
    def get_align_device_pre_forward_hook(device="cuda"):
        """
        resident in gpu. used for part that don't be offloaded.
        """

        def hook(module: torch.nn.Module, args):
            # print(f",{module}")
            # print(f"align_device_hook is called, module device={next(module.parameters()).device}, input device={next(iter(args)).device}")
            if device is None:
                align_device = next(module.parameters()).device
            else:
                align_device = device
            module.to(device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        return hook

    @staticmethod
    def get_forward_hook(pre: bool, device=None):
        origin_device = "cpu"
        if device is not None:
            device = device
        else:
            device = "cuda"

        def pre_hook(module: torch.nn.Module, args):
            """
            # cpu->gpu
            Args:
                module: torch.nn.Module
                args: inputs of module

            Returns: modified inputs

            """

            # print(f"pre_hook is called, module device={next(module.parameters()).device}, input device={next(iter(args)).device}")
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def after_hook(module: torch.nn.Module, args, output):
            """
            gpu->cpu
            Args:
                module:  torch.nn.Module
                args: inputs of module
                output: outputs of module

            Returns: modified outputs.

            """
            module.to(origin_device)

            output = output.to(origin_device) if isinstance(output, torch.Tensor) else output
            # print(f"after_hook is called, device={next(module.parameters()).device}, input device={next(iter(output)).device}")
            return output

        if pre:
            return pre_hook
        else:
            return after_hook

    @staticmethod
    def get_inputs_align_device_hook(device="cuda"):
        """
        ensure same device for module with input and without weight. such as loss_fn and activate_fn
        """

        def hook(module: torch.nn.Module, args):
            # print(f"inputs_align_device_hook is called,input device={next(iter(args)).device}")
            if device is None:
                align_device = next(iter(args)).device
            else:
                align_device = device
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        return hook

    def get_full_name(self, module, parent_name=''):
        """
        get full name list of all submodule.
        """
        if len(list(module.named_children())) == 0:
            self.module_list.append(parent_name)
        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            self.get_full_name(sub_module, full_name)


class LLMForwardHookForOffloadContext:
    def __init__(self, model, offload_proportion=0.5, device="cuda", max_memory=None, no_split_module_classes=None):
        if max_memory is None:
            max_memory = {0: "4GB", "cpu": "10GB"}
        self.model = model
        self.offload_list = list()
        self.handle_list = list()
        self.device = device
        self.device_map = accelerate.infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes
        )
        self.module_list = self.device_map.keys()
        for k, v in self.device_map.items():
            if not isinstance(v, int):
                self.offload_list.append(k)
        print(f"device_map:{self.device_map}")
        print(f"offload:{self.offload_list}")
        print(f"all:{self.module_list}")

    def __enter__(self):
        print("__enter__(self):")
        self._register_forward_hook(self.model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__(self, exc_type, exc_val, exc_tb)")
        for handle in self.handle_list:
            handle.remove()

    def _register_forward_hook(self, module, parent_name=''):
        print(f"_register_forward_hook(self, module, parent_name={parent_name}")
        if parent_name in self.offload_list:
            handle = module.register_forward_pre_hook(self.get_forward_hook(pre=True))
            self.handle_list.append(handle)
            handle = module.register_forward_hook(self.get_forward_hook(pre=False))
            self.handle_list.append(handle)
            return

        elif parent_name in self.module_list:
            handle = module.register_forward_pre_hook(self.get_align_device_pre_forward_hook(device="cuda"))
            self.handle_list.append(handle)
            return
        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            self._register_forward_hook(sub_module, full_name)

    @staticmethod
    def get_align_device_pre_forward_hook(device="cuda"):
        """
        resident in gpu. used for the part that don't be offloaded.
        """

        def hook(module: torch.nn.Module, args):
            # print(f",{module}")
            if len(list(module.parameters())) > 0:
                if len(list(args)) > 0:
                    print(
                        f"align_device_hook is called, module device={next(module.parameters()).device}, input device={next(iter(args)).device}")
                else:
                    print(
                        f"align_device_hook is called, module device={next(module.parameters()).device}, no input")
            else:
                print(
                    f"align_device_hook is called, module hasn't weight, input device={next(iter(args)).device}")
            if device is not None:
                align_device = device
            elif len(list(module.parameters())) > 0:
                align_device = next(module.parameters()).device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            print(tuple(arg.device if isinstance(arg, torch.Tensor) else arg for arg in args))
            return args

        return hook

    @staticmethod
    def get_forward_hook(pre: bool, device=None):
        origin_device = "cpu"
        if device is not None:
            device = device
        else:
            device = "cuda"

        def pre_hook(module: torch.nn.Module, args):
            """
            # cpu->gpu
            Args:
                module: torch.nn.Module
                args: inputs of module

            Returns: modified inputs

            """

            if len(list(module.parameters())) > 0:
                print(
                    f"pre_hook is called, module device={next(module.parameters()).device}, input device={next(iter(args)).device}")
            else:
                print(
                    f"pre_hook is called, module hasn't weight, input device={next(iter(args)).device}")
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def after_hook(module: torch.nn.Module, args, output):
            """
            gpu->cpu
            Args:
                module:  torch.nn.Module
                args: inputs of module
                output: outputs of module

            Returns: modified outputs.

            """
            module.to(origin_device)
            output = output.to(origin_device) if isinstance(output, torch.Tensor) else output
            if isinstance(output, tuple):
                output = tuple(o.to(origin_device) if isinstance(o, torch.Tensor) else o for o in output)

            if len(list(module.parameters())) > 0:
                print(
                    f"after_hook is called, device={next(module.parameters()).device}, input device={next(iter(output)).device}")
            else:
                print(f"after_hook is called, model hasn't weight, input device={next(iter(output)).device}")
            print(output)
            print(f"typed of output is {type(output)}")
            return output

        if pre:
            return pre_hook
        else:
            return after_hook

    @staticmethod
    def get_inputs_align_device_hook(device="cuda"):
        """
        ensure same device for module with input and without weight. such as loss_fn and activate_fn
        """

        def hook(module: torch.nn.Module, args):
            print(f"inputs_align_device_hook is called,input device={next(iter(args)).device}")
            if device is None:
                align_device = next(iter(args)).device
            else:
                align_device = device
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        return hook

    def get_full_name(self, module, parent_name=''):
        """
        get full name list of all submodule.
        """
        if len(list(module.named_children())) == 0:
            self.module_list.append(parent_name)
        for name, sub_module in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            self.get_full_name(sub_module, full_name)
