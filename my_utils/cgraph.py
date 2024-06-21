import torch
from torchviz import make_dot
import os


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


def get_compute_graph(model: torch.nn.Module,
                      input_shape=None,
                      input: dict = None,
                      dir: str = "compute_graph",
                      filename: str = "simple_net_graph",
                      format: str = "pdf"
                      ):
    """
    generate the computing graph of model (format default is pdf)
    """
    print("os.getcwd()", os.getcwd())
    # torch.autograd.set_detect_anomaly(True)
    assert input is not None or input_shape is not None, "error: input is None and input_shape is None"
    if input is None:
        example_input = torch.randn(input_shape)
    else:
        example_input = input
    out = model(**example_input)

    def extract_tensors(output):
        if isinstance(output, torch.Tensor):
            return output
        elif hasattr(output, "logits"):
            return output.logits
        elif hasattr(output, "loss"):
            return output.loss
        else:
            raise ValueError("Unsupported output type")

    output = extract_tensors(out)
    make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    ).render(dir + "/" + filename, format=format)
