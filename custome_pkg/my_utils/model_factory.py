import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def get_text_to_text_model(
    model_name: str,
    model_type: str = "CausalLM",
    dtype: str = "bf16",
    tokenizer: str = None,
    flash_attention: bool = False,
):
    assert model_type in ["CausalLM", "ConditionalGeneration"]
    match model_type:
        case "CausalLM":
            auto_model_class = AutoModelForCausalLM
        case "ConditionalGeneration":
            auto_model_class = AutoModelForSeq2SeqLM
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
    model_config = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    if flash_attention:
        model_config["attn_implementation"] = "flash_attention_2"
    match dtype:
        case "fp32":
            model_config["torch_dtype"] = torch.float32
        case "bf16":
            model_config["torch_dtype"] = torch.bfloat16
        case "int8":
            quant_8bit_config = dict(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            model_config["quantization_config"] = quant_8bit_config
        case "nf4":
            quant_4bit_config = dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_config["quantization_config"] = quant_4bit_config
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")
    model = auto_model_class.from_pretrained(**model_config)
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
