import torch
from .tokenizer import get_collate_fn, get_masks


def greedy_decode(model, tokenizer, prompt, max_len, device):
    model.eval()
    with torch.no_grad():
        input_seq = get_collate_fn(tokenizer, max_len=500, train=False)([prompt])["x"]
        output_seq = []
        for _ in range(max_len):
            input_seq = input_seq.to(device)
            att_mask, pad_mask = get_masks(input_seq, tokenizer)
            att_mask = att_mask.to(device)
            pad_mask = pad_mask.to(device)
            model.to(device)
            logits = model(input_seq, att_mask, pad_mask)
            logits = logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            output_seq.append(next_token.item())
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_id:
                break

    return output_seq
