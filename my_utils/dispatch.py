import torch


def get_dispatch_model(model: torch.nn.Module, strategy="auto", device_map=None, max_memory=None, offload_dir=None,
                       state_dict=None):
    import accelerate
    from accelerate.big_modeling import dispatch_model
    from accelerate.big_modeling import cpu_offload
    match strategy:
        case "only_cpu":
            model = cpu_offload(model)
        case "auto":
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
            )
            print("auto device map is ", device_map)
            model = dispatch_model(
                model,
                device_map=device_map,
                offload_dir=offload_dir,
                state_dict=state_dict,
            )
        case "device_map":
            assert device_map is not None, "device map is None"
            model = dispatch_model(model, device_map=device_map)
        case _:
            print("No offload policy is specified, so do nothing")
    return model
