"""
This is more-or-less the custom base transformer class.

This file contains the following model layer 'classes':
- DecoderOnlyTransformer: The base transformer
- Head: Single self-attention head
- MHA: Multi-head self-attention
- FFN: Feed-forward network
- Block: Single SA-transformer block (decoder)

For more info on each class, refer to their docstrings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Single self-attention head of head-size"""
    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float = 0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expected shape: (B,T,C)
        B, T, C = x.shape

        k = self.key(x)  # (B,T,C) | C: head_size
        q = self.query(x)  # (B,T,C) | C: head_size
        v = self.value(x)  # (B,T,C) | C: head_size

        out = q @ k.transpose(-2, -1) * T ** -0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)

        out = out.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        out = F.softmax(out, dim=-1)
        out = self.dropout(out)
        out = out @ v  # (B,T,T) @ (B,T,C) -> (B,T,C) | C: head_size

        return out


class MultiHeadAttention(nn.Module):
    """Multi Head Attention..."""
    def __init__(self, num_heads, head_size, n_embed):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.linear(out)
        return out

class FFN(nn.Module):
    """Feed-forward netork, as desc by the paper"""
    def __init__(self, num_heads, n_embed):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, num_heads * n_embed),
            nn.ReLU(),
            nn.Linear(num_heads * n_embed, n_embed),
        )  # (B,T,n_embed) -> (B,T,n_embed*n_heads) -> (B,T,n_embed)

    def forward(self, x):
        out = self.ffn(x)
        return out


class Block(nn.Module):
    """Single decoder-only transformer block"""

    def __init__(self,
                 num_heads: int,
                 n_embed: int,
                 dropout: float = 0.2,
                 _head_size: int or None = None,
                 layer_norm_first: bool = True,
                 _activate_dropout: bool = True):
        """
        Args:
            num_heads (int): Number of attention heads.
            n_embed (int): Embedding dim
            dropout (float): Dropout prob (default = 0.2)
            layer_norm_first (bool): Performs layer-norm before SA if True (default = True)

        Experimental args (start with '_param_name' | these are some custom features/add-ons to the transformer):
            _activate_dropout (bool): Use dropout layer if True (default: True)
            _head_size (int): Embedding dim per SA-Head.

        Individual head_sizes' are inferred using: (n_embed // num_heads), if a _head_size is not provided.
        Contains a single MHA layer as this will be used as a Decoder-only transformer.

        Default value for (layer_norm_first = True) by convention, most decoder-only transformers tend to perform better.

        NOTE: The dropout layer is not implemented yet, as its well-supported on MPS+compile. Will add it when using CUDA.
        """
        super().__init__()

        if _head_size is None:
            _head_size = n_embed // num_heads

        self.mha = MultiHeadAttention(num_heads, _head_size,
                                      n_embed)  # Only one MHA, not adding 'mha-block' for enc-dec transformer
        self.ffn = FFN(num_heads, n_embed)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        if self.layer_norm_first:
            x = self.ln1(x)
            x = x + self.mha(x)
            x = self.ln2(x)
            x = x + self.ffn(x)
            return x

        x = self.mha(x)
        x = x + self.ln1(x)
        x = self.ffn(x)
        x = x + self.ln2(x)
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer with n-transformer blocks"""

    def __init__(self,
                 num_blocks: int,
                 num_heads: int,
                 vocab_size: int,
                 block_size: int,
                 n_embed: int,
                 device: torch.device,
                 head_size: int or None = None):
        """
        Args:
            num_blocks (int): Number of transformer blocks.
            num_heads (int): Number of SA-Heads per block.
            vocab_size (int): Size of the vocab.
            block_size (int): Context length.
            n_embed (int): Embedding size.
            device (torch.device): Device to move pos-embeddings to.
            head_size (int): Individual head dim (default = None) | Read through Block's docstring.

        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)

        self.blocks = nn.ModuleList(
            [Block(num_heads=num_heads,
                   n_embed=n_embed,
                   _head_size=head_size) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(n_embed)

        self.ff = nn.Linear(n_embed, vocab_size)  # Final FF for pred

        self.device = device

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)  # (B,T,C) | C = n_embed
        pos_emb = self.pos_emb(torch.arange(T, device=self.device))  # (T,C) | C = n_embed
        x = tok_emb + pos_emb  # (B,T,C) + (T,C) -> (B,T,C) | C = n_embed

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        self.ff(x)

        return x

