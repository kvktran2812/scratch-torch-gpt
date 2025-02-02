import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, in_channels, n_heads, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.embedding_size = embedding_size
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.head_size = embedding_size // n_heads
        
        self.c_attn = nn.Linear(embedding_size, embedding_size * 3)
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.ln_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.embedding_size, 2)
        
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        if self.flash:
            value = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            attn = q @ k.transpose(-2, -1) * (1 / math.sqrt(self.head_size))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            value = attn @ v
        value = value.transpose(1, 2).contiguous().view(B, T, C)

        value = self.proj(value)
        value = self.ln_dropout(value)
        return value