import torch
from peft.tuners.lora.bnb import (
    Linear8bitLt as LoraLinear8bitLt,
    Linear4bit as LoraLinear4bit,
    Linear8bitLt,
    Linear4bit
)
from torch.nn import Linear


def get_float_weight(model: torch.nn.Module):
    model: torch.nn.Linear

    device = model.weight.device
    in_features = model.in_features
    with torch.no_grad():
        I = torch.eye(in_features).to(device)
        w = model(I)
        if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
            w -= model.bias
        w = torch.transpose(w, 0, 1)
    w.requires_grad = model.weight.requires_grad
    return w


def replace_module_with_linear(model: torch.nn.Module, target):
    for name, module in model.named_children():
        if isinstance(module, target):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            new_module = torch.nn.Linear(in_features, out_features, bias)
            with torch.no_grad():
                new_module.weight.data = get_float_weight(module).data
                if bias:
                    new_module.bias.data = (
                        module.bias if module.bias is not None else None
                    )
            setattr(model, name, new_module)

        else:
            replace_module_with_linear(module, target)


def dequantize(model, dtype):
    target = None
    if dtype == "int8":
        target = LoraLinear8bitLt
    elif dtype == "nf4":
        target = LoraLinear4bit
    replace_module_with_linear(model=model, target=target)


class Config:
    in_features = 3
    out_features = 4
    device = "cuda"


def main():
    config = Config()
    print("get float weight")
    linear = Linear(config.in_features, config.out_features)
    print(linear.weight)
    w = get_float_weight(model=linear)
    print(w)
    print("int8 quant==============================================")
    m = Linear8bitLt(config.in_features, config.out_features, has_fp16_weights=False)
    m.load_state_dict(linear.state_dict())
    m.to(config.device)
    print(m.weight)
    w = get_float_weight(model=m)
    print(linear.weight)
    print(w)
    print("nf4 quant==============================================")
    m = Linear4bit(config.in_features, config.out_features)
    m.load_state_dict(linear.state_dict())
    m.to(config.device)
    print(m.weight)
    print(linear.weight)
    w = get_float_weight(model=m)
    print(w)


if __name__ == '__main__':
    main()
