import torch
from torchviz import make_dot
import os


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
