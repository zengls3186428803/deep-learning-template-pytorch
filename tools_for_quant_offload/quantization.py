import torch
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def quantize(
        model: torch.nn.Module,
        quant_type: str = "nf4"
) -> torch.nn.Module:
    """
    Convert a high-precision floating-point model into a low-precision model
    Args:
        model: pytorch model
        quant_type: support nf4 and int8
    Returns: quantized model
    """
    match quant_type:
        case "int8":
            quant_k_bit_config = dict(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        case "nf4":
            quant_k_bit_config = dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        case _:
            raise ValueError("Wrong dtype")
    quant_k_bit_config = BnbQuantizationConfig(**quant_k_bit_config)
    return load_and_quantize_model(model, bnb_quantization_config=quant_k_bit_config)


def get_quant_model(
        model_name: str,
        model_type: str,
        dtype: str,
        tokenizer: str = None,
        flash_attention: bool = False,
):
    assert model_type in ["CausalLM", "ConditionalGeneration"]
    auto_model_class = AutoModelForCausalLM if model_type == "CausalLM" else AutoModelForSeq2SeqLM
    model_config = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    if flash_attention:
        model_config["attn_implementation"] = "flash_attention_2"
    match dtype:
        case "int8":
            quant_k_bit_config = dict(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True
            )
        case "nf4":
            quant_k_bit_config = dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        case _:
            raise ValueError("Wrong dtype")
    model_config["quantization_config"] = quant_k_bit_config
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
