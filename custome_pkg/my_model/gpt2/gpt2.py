import torch
import math


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        pe = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_dim)))
        pe = pe.unsqueeze(0)
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
        self.alpha = torch.nn.Parameter(torch.ones(self.size))
        self.bias = torch.nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        # 这里的减法和除法是逐元素操作。
        # -1表示沿最后一个维度
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_dim=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = torch.nn.Linear(embed_dim, intermediate_dim)
        self.act = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(intermediate_dim, embed_dim)
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
        )
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
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
        self.ln1 = NormLayer(embed_dim=embed_dim)
        self.att = GPT2AttentionBlock(
            embed_dim=embed_dim,
            num_head=num_head,
            dropout=dropout,
            batch_first=batch_first
        )
        #
        self.ln2 = NormLayer(embed_dim=embed_dim)
        self.ff = FeedForwardBlock(
            embed_dim=embed_dim,
            intermediate_dim=2048,
            dropout=dropout
        )

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
    def __init__(self, vocab_size, embed_dim, num_head, dropout, num_block_gpt=3, max_pos=5000, batch_first=True):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, embed_dim)
        self.pte = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_pos)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.gpt_blocks = torch.nn.ModuleList([
            GPT2TransformerBlock(
                embed_dim=embed_dim, num_head=num_head, dropout=dropout, batch_first=batch_first
            )
            for i in range(num_block_gpt)
        ])
        self.ln = NormLayer(embed_dim)
        self.linear = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, tgt_mask, tgt_key_padding_mask):
        x = self.wte(x)
        x = self.pte(x)
        x = self.dropout(x)
        for gpt_block in self.gpt_blocks:
            x = gpt_block(x, tgt_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.ln(x)
        x = self.linear(x)
        return x
