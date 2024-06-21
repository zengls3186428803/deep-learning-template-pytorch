from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
# from gpt import GPT2
from torch import Tensor
from torchtext.vocab import build_vocab_from_iterator
from myDataset import SongCi
from torch.utils.data import DataLoader
import torch
import math
import tqdm
import sys

# torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = "cpu"
EMBED_DIM = 512
N_HEAD = 8
DROPOUT = 0.1
N_BLOCK_GPT = 3
BATCH_FIRST = True
BATCH_SIZE = 64
MAX_GEN_LEN = 128
learning_rate = 0.1


# =======================token_transform=================================
def token_transform(data):
    return list(data)


def yield_tokens(data_iter):
    for data in data_iter:
        yield token_transform(data)


# =======================vocab_transform=================================
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
train_dataset = SongCi(is_train=True)
data_iter = iter(train_dataset)
vocab_transform = build_vocab_from_iterator(yield_tokens(data_iter),
                                            min_freq=1,
                                            specials=special_symbols,
                                            special_first=True)

vocab_transform.set_default_index(UNK_IDX)
VOCAB_SIZE = len(vocab_transform)
print("vocab_size=", VOCAB_SIZE, flush=True)


# =======================tensor_transform=================================
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# ======================text_transform====================================
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


text_transform = sequential_transforms(token_transform,
                                       vocab_transform,
                                       tensor_transform)


# ============================GPT========================================
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        pe = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_dim)))
        pe = pe.unsqueeze(0).to(DEVICE)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.embed_dim)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)  # x.size()=(batch_size, seq_len, embed_dim)
        # self.pe[:, :seq_len]等价于self.pe[:, :seq_len，:],  pytorch应该重载了[]，[,,]相当于[][][]
        x = x + self.pe[:, :seq_len].clone().detach()  # 这一步会对PE广播
        return x


class NormLayer(torch.nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.size = embed_dim
        self.alpha = torch.nn.Parameter(torch.ones(self.size)).to(DEVICE)
        self.bias = torch.nn.Parameter(torch.zeros(self.size)).to(DEVICE)
        self.eps = eps

    def forward(self, x):
        # 这里的减法和除法是逐元素操作。
        # -1表示沿最后一个维度
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_dim=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = torch.nn.Linear(embed_dim, intermediate_dim).to(DEVICE)
        self.act = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(intermediate_dim, embed_dim).to(DEVICE)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class GPT2AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_head, dropout, batch_first=True):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_head,
            dropout=dropout,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            batch_first=batch_first
        ).to(DEVICE)
        self.linear = torch.nn.Linear(embed_dim, embed_dim).to(DEVICE)
        self.dropout = torch.nn.Dropout(p=dropout)

    # decoder-only的GPT只有target-attention
    def forward(self, x, tgt_mask, key_padding_mask):
        x, _ = self.att(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=tgt_mask,
            average_attn_weights=True,
            is_causal=False
        )
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GPT2TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_head, dropout, batch_first=True):
        super().__init__()
        self.ln1 = NormLayer(embed_dim=embed_dim).to(DEVICE)
        self.att = GPT2AttentionBlock(
            embed_dim=embed_dim,
            num_head=num_head,
            dropout=dropout,
            batch_first=batch_first
        ).to(DEVICE)
        #
        self.ln2 = NormLayer(embed_dim=embed_dim).to(DEVICE)
        self.ff = FeedForwardBlock(
            embed_dim=embed_dim,
            intermediate_dim=2048,
            dropout=dropout
        ).to(DEVICE)

    def forward(self, x, tgt_mask, key_padding_mask):
        x_residual = x
        x = self.ln1(x)
        x = self.att(x, tgt_mask=tgt_mask, key_padding_mask=key_padding_mask)
        x = x_residual + x
        x_residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = x_residual + x
        return x


class GPT2(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_head, dropout, num_block_gpt=3, max_seq_len=5000, batch_first=True):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, embed_dim).to(DEVICE)
        self.pte = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len).to(DEVICE)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.gpt_blocks = [
            GPT2TransformerBlock(
                embed_dim=embed_dim, num_head=num_head, dropout=dropout, batch_first=batch_first
            ).to(DEVICE)
            for i in range(num_block_gpt)
        ]
        self.ln = NormLayer(embed_dim).to(DEVICE)
        self.linear = torch.nn.Linear(embed_dim, vocab_size).to(DEVICE)

    def forward(self, x, tgt_mask, tgt_key_padding_mask):
        x = self.wte(x)
        x = self.pte(x)
        x = self.dropout(x)
        for gpt_block in self.gpt_blocks:
            x = gpt_block(x, tgt_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.ln(x)
        x = self.linear(x)
        return x


# ================================generate=================================
def greedy_decode(model, max_len, start_symbol, prompt):
    tgt_input = text_transform(prompt)
    tgt_input: Tensor
    tgt_input = tgt_input[:-1]
    tgt_input = tgt_input.view(1, -1).to(DEVICE)
    for i in range(max_len - 1):
        tgt_mask, tgt_key_padding_mask = create_mask(tgt_input, PAD_IDX)
        out = model(tgt_input, tgt_mask, tgt_key_padding_mask)
        _, next_word = torch.max(out[:, -1], dim=1)
        next_word = next_word.item()
        tgt_input = torch.cat([tgt_input, torch.ones(1, 1).type_as(tgt_input).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return tgt_input


def topk_sampling_decode(model, max_len, start_symbol, prompt, k=5):
    # 转换输入序列为Tensor，并准备初始的输入序列
    tgt_input = text_transform(prompt)
    tgt_input = tgt_input[:-1]  # 移除结束标记
    tgt_input = tgt_input.view(1, -1).to(DEVICE)  # 转换形状为 (1, sequence_length)
    generated_sequence = []
    for i in range(max_len - 1):
        tgt_mask, tgt_key_padding_mask = create_mask(tgt_input, PAD_IDX)
        out = model(tgt_input, tgt_mask, tgt_key_padding_mask)
        # 使用 Top-K 抽样从输出中采样下一个单词
        topk_probs, topk_indices = torch.topk(out[:, -1], k)  # topk_probs: (1, k), topk_indices: (1, k)
        sampled_index = torch.multinomial(torch.exp(topk_probs), 1).item()  # 从 topk_probs 中抽样一个单词的索引
        next_word = topk_indices[0][sampled_index].item()  # 获取抽样的单词的索引
        generated_sequence.append(next_word)
        tgt_input = torch.cat([tgt_input, torch.ones(1, 1).type_as(tgt_input).fill_(next_word)],
                              dim=1)  # 新序列形状: (1, sequence_length+1)
        if next_word == EOS_IDX:
            break

    return generated_sequence


def beam_search_decode(model, max_len, start_symbol, prompt, beam_width=5):
    tgt_input = text_transform(prompt)
    tgt_input: Tensor
    tgt_input = tgt_input[:-1]
    tgt_input = tgt_input.view(1, -1).to(DEVICE)
    # Create beam search candidates
    candidates = [(tgt_input, 0)]  # (sequence, score)
    for i in range(max_len - 1):
        new_candidates = []
        for seq, score in candidates:
            tgt_mask, tgt_key_padding_mask = create_mask(seq, PAD_IDX)
            out = model(seq, tgt_mask, tgt_key_padding_mask)
            _, top_indices = torch.topk(out[:, -1], beam_width)
            for j in range(beam_width):
                next_word = top_indices[0][j].item()
                new_seq = torch.cat([seq, torch.ones(1, 1).type_as(seq).fill_(next_word)], dim=1)
                new_score = score + out[:, -1][0][next_word].item()
                new_candidates.append((new_seq, new_score))
        # Select top beam_width candidates
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:beam_width]
        # Check if any candidate has reached EOS_IDX
        for seq, score in candidates:
            if seq[0][-1].item() == EOS_IDX:
                return seq
    # If max_len reached without EOS_IDX, return the best candidate
    return candidates[0][0]


def generate(model, prompt):
    model.eval()
    tgt_tokens = beam_search_decode(model=model, max_len=MAX_GEN_LEN, start_symbol=BOS_IDX, prompt=prompt).flatten()
    return ("".join(vocab_transform.lookup_tokens(list(tgt_tokens.cpu().numpy()))).
            replace("<bos>", "").replace("<eos>", ""))


def create_mask(tgt: Tensor, pad_idx: int):
    seq_len = tgt.size(1)
    padding_mask = (tgt == pad_idx)
    attention_mask = (torch.ones(seq_len, seq_len) - torch.triu(torch.ones(seq_len, seq_len))).type(
        torch.bool).transpose(0, 1)
    return attention_mask.to(DEVICE), padding_mask.to(DEVICE)


# ===============================model=========================================
model = GPT2(vocab_size=VOCAB_SIZE,
             embed_dim=EMBED_DIM,
             num_head=N_HEAD,
             dropout=DROPOUT,
             num_block_gpt=N_BLOCK_GPT,
             batch_first=BATCH_FIRST)

model = model.to(DEVICE)
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)


# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ===============================train============================================

def collate_fn(batch):
    tgt_batch = []
    for tgt_sample in batch:
        tgt_batch.append(text_transform(tgt_sample.rstrip("\n")))
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=BATCH_FIRST)
    return tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    p_bar = tqdm.tqdm(total=len(train_dataloader), desc="batch")
    for tgt in train_dataloader:
        p_bar.update(1)
        sys.stdout.flush()
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_att_mask, tgt_key_padding_mask = create_mask(tgt_input, PAD_IDX)
        logits = model(tgt_input, tgt_att_mask, tgt_key_padding_mask)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        losses += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(generate(model, "天空突然下雨"), "\ttrain_loss=", loss.item())
        sys.stdout.flush()
    return losses / len(list(train_dataloader))


test_dataset = train_dataset


def evaluate(model):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for tgt in val_dataloader:
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_att_mask, tgt_key_padding_mask = create_mask(tgt_input, PAD_IDX)
        logits = model(tgt_input, tgt_att_mask, tgt_key_padding_mask)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        losses += loss.item()
    return losses / len(list(val_dataloader))


NUM_EPOCHS = 100
for epoch in tqdm.tqdm(range(1, NUM_EPOCHS + 1), desc="epoch"):
    start_time = timer()
    train_loss = 0
    train_loss = train_epoch(model, optimizer)
    val_loss = 0
    end_time = timer()
    # val_loss = evaluate(model)
    print((
        f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    sys.stdout.flush()