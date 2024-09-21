import torch


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, embed_dim, intermediate_dim=2048, dropout=0.0):
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


class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_head, dropout=0, batch_first=True):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_head,
            dropout=dropout,
            bias=False,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            batch_first=batch_first
        )
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    # decoder-only的GPT只有target-attention
    def forward(self, x, tgt_mask, key_padding_mask):
        x, att_weight = self.att(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=tgt_mask,
            average_attn_weights=True,
            is_causal=False
        )
        x = self.linear(x)
        x = self.dropout(x)
        # print(att_weight)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_head, dropout, batch_first=True):
        super().__init__()
        self.att = AttentionBlock(
            embed_dim=embed_dim,
            num_head=num_head,
            dropout=dropout,
            batch_first=batch_first
        )
        #
        self.ff = FeedForwardBlock(
            embed_dim=embed_dim,
            intermediate_dim=2048,
            dropout=dropout
        )

    def forward(self, x, tgt_mask, key_padding_mask):
        x_residual = x
        x = self.att(x, tgt_mask=tgt_mask, key_padding_mask=key_padding_mask)
        x = x_residual + x
        x_residual = x
        x = self.ff(x)
        x = x_residual + x
        return x


class CausalLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_head, dropout=0, num_block=3, max_pos=5000, batch_first=True):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_head=num_head, dropout=dropout, batch_first=batch_first
            )
            for i in range(num_block)
        ])
        self.linear = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, tgt_mask, tgt_key_padding_mask):
        x = self.wte(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, tgt_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.linear(x)
        return x
