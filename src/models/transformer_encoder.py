import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.models.transformer_attention import MultiHeadAttention
from src.models.positional_encoding import PositionalEncoding
from src.models.transformer_feedforward import PositionwiseFeedForward

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer:
    1. Multi-Head Self-Attention
    2. Add & LayerNorm
    3. Position-wise Feed Forward
    4. Add & LayerNorm
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Multi-Head Self-Attention + Residual Connection + LayerNorm
        attn_out, attn_weights = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_out))

        # 2. Feed-Forward Network + Residual Connection + LayerNorm
        ffn_out = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ffn_out))

        return src, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of N Transformer Encoder Layers with Embedding and Positional Encoding.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        # Embedding scaled by sqrt(d_model) as in Vaswani et al. (2017)
        x = self.embedding(src) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)

        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask=src_mask)
            attn_weights_list.append(attn_weights)

        x = self.norm(x)
        return x, attn_weights_list
