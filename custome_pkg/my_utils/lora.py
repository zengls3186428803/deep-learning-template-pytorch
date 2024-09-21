from typing import List
from peft import get_peft_model, LoraConfig
import torch


def find_linear_modules(model) -> List[str]:
    linear_cls = torch.nn.Linear
    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def get_lora_model(model, lora_config):
    taget_modules = find_linear_modules(model)
    # print(f"Target modules for LoRA: {taget_modules}")
    # print(model)
    lora_config = LoraConfig(
        r=lora_config["lora_r"],
        target_modules=taget_modules,
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
    )
    model = get_peft_model(model, lora_config)
    return model
