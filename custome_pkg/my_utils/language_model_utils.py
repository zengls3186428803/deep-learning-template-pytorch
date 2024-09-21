import numpy as np
import torch
from transformers import EvalPrediction


def get_data_collator(
    tokenizer, max_length=-1, ignore_masked_token=True, model_type="causal"
):
    """
    Shifting the inputs and labels to align them happens inside the model,
    so the data collator just copies the inputs to create the labels.
    """

    def huggingface_causal_collator(batch):
        ignore_index = -100
        collated_batch = list()
        for sample in batch:
            collated_batch.append(sample["x"] + " " + sample["y"] + tokenizer.eos_token)
        encodings = tokenizer(
            collated_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length if max_length != -1 else None,
        )
        # tokens_list = [tokenizer.tokenize(text) for text in collated_batch]
        # print(f"Tokens list: {tokens_list}")
        # Calculate input text length in tokens
        input_text_length_list = list()
        for sample in batch:
            input_text_length_list.append(
                len(tokenizer(sample["x"], return_tensors="pt")["input_ids"][0])
            )
        labels = encodings["input_ids"].clone()
        for i, l in enumerate(input_text_length_list):
            labels[i, :l] = ignore_index
        if ignore_masked_token:
            # The attention mask here should be padding mask.
            # 1 indicates valid tokens and 0 indicates padding in huggingface.
            # This convention is the opposite of how the Pytorch Transformers library uses attention_mask.
            labels[encodings["attention_mask"] == 0] = ignore_index
        results = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
        return results

    def huggingface_autoregressive_collator(batch):
        ignore_index = -100
        collated_batch = list()
        for sample in batch:
            collated_batch.append(sample["x"] + " " + sample["y"] + tokenizer.eos_token)
        encodings = tokenizer(
            collated_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length if max_length != -1 else None,
        )
        labels = encodings["input_ids"].clone()
        if ignore_masked_token:
            # The attention mask here should be padding mask.
            # 1 indicates valid tokens and 0 indicates padding in huggingface.
            # This convention is the opposite of how the Pytorch Transformers library uses attention_mask.
            labels[encodings["attention_mask"] == 0] = ignore_index
        results = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
        return results

    def huggingface_seq2seq_collator(batch):
        ignore_index = -100
        inputs = tokenizer(
            batch["x"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length if max_length != -1 else None,
        )
        outputs = tokenizer(
            batch["y"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length if max_length != -1 else None,
        )

        results = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": outputs["input_ids"],
            "decoder_attention_mask": outputs["attention_mask"],
        }

        if ignore_masked_token:
            # The attention mask here should be padding mask.
            # 1 indicates valid tokens and 0 indicates padding in huggingface.
            # This convention is the opposite of how the Pytorch Transformers library uses attention_mask.
            results["labels"][outputs["attention_mask"] == 0] = ignore_index

        return results

    match model_type:
        case "causal":
            return huggingface_causal_collator
        case "seq2seq":
            return huggingface_seq2seq_collator
        case "autoregressive":
            return huggingface_autoregressive_collator
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")


def compute_metrics(eval_prediction: EvalPrediction):

    predictions, _labels = eval_prediction.predictions
    predictions = predictions[:, :-1]
    labels = eval_prediction.label_ids[:, 1:]

    predictions = predictions.flatten().astype(np.int32)
    labels = labels.flatten().astype(np.int32)

    correct_predictions = np.sum((predictions == labels))
    total_samples = len(labels)

    accuracy = correct_predictions / total_samples
    return {
        "accuracy": accuracy,
    }


def preprocess_logits_for_metrics(logits, labels):
    prediction_ids = torch.argmax(logits, dim=-1)
    return prediction_ids, labels
