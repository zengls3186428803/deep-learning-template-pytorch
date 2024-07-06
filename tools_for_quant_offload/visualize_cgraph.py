import torch


def extract_tensors(
        output: dict | torch.Tensor,
) -> torch.Tensor:
    """
    extract tensors from a dict of result whose form is like {"logits: Tensor([...])"}
    Args:
        output: dict or tensor
    Returns: Tensor
    """
    if isinstance(output, torch.Tensor):
        return output
    elif hasattr(output, "logits"):
        return output.logits
    elif hasattr(output, "loss"):
        return output.loss
    else:
        raise ValueError("Unsupported output type")


def get_compute_graph(
        model: torch.nn.Module,
        input_shape=None,
        mode_input: dict = None,
        computing_graph_dir: str = "compute_graph",
        filename: str = "simple_net_graph",
        file_format: str = "pdf"
) -> None:
    """

    Args:
        model: your torch model
        input_shape: if you not give input, please specify an input shape
        mode_input: input should be a dict that has key "x", such as input can be {"x",Tensor([1.2, 2.2, 3.3, 4.4])}
        computing_graph_dir: the directory where computing graph of model is saved
        filename: file name of computing graph to save
        file_format: the format of file of computing graph

    Returns: None
    """
    from torchviz import make_dot
    # torch.autograd.set_detect_anomaly(True)
    assert mode_input is not None or input_shape is not None, "error: input is None and input_shape is None"
    if mode_input is None:
        example_input = torch.randn(input_shape)
    else:
        example_input = mode_input
    out = model(**example_input)

    output = extract_tensors(out)
    make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=True
    ).render(
        computing_graph_dir + "/" + filename,
        format=file_format,
    )
