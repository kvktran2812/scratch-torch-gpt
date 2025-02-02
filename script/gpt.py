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
    
class FeedForward(nn.Module):
    def __init__(self, in_channels, factor, dropout=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.ln1 = nn.Linear(in_channels, in_channels * factor)
        self.ln2 = nn.Linear(in_channels * factor, in_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.relu(self.ln1(x))
        x = self.ln2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, embedding_size, n_heads, dropout=0.0):
        super().__init__()
        head_size = embedding_size // n_heads
        self.multi_head_attn = MultiHeadAttention(embedding_size, in_channels, n_heads, dropout)
        self.ffwd = FeedForward(embedding_size, 4, dropout)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        x = x + self.multi_head_attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, in_channels, vocab_size, embedding_size, n_heads, n_layers, dropout=0.0):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(in_channels, embedding_size)
        self.blocks = nn.Sequential(*[Block(in_channels, embedding_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        device = x.device
        B,T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        token_embedding = self.token_embedding_table(x)
        pos_embedding = self.position_embedding_table(pos)
        
        x = self.dropout(token_embedding + pos_embedding)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x