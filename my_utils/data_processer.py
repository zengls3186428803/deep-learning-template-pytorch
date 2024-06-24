import torch
from torch.nn.utils.rnn import pad_sequence

from class_model.gpt2Tokenizer import GPT2Tokenizer


def get_collate_fn(tokenizer: GPT2Tokenizer, max_len: int = 500):
    def transform_text_to_tensor(text: str, tokenizer: GPT2Tokenizer):
        return torch.Tensor(
            tokenizer.convert_token_to_id(
                tokenizer.tokenize(text) + [tokenizer.eos_token]
            )
        )

    def collate_fn(batch):
        collated_batch = []
        for sample in batch:
            collated_batch.append(transform_text_to_tensor(sample.rstrip("\n"), tokenizer))
        collated_batch = pad_sequence(
            collated_batch,
            padding_value=tokenizer.convert_token_to_id([tokenizer.pad_token])[0],
            batch_first=True
        )
        return collated_batch.long()[:, :max_len]

    return collate_fn
