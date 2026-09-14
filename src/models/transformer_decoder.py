import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from src.models.transformer_attention import MultiHeadAttention
from src.models.positional_encoding import PositionalEncoding
from src.models.transformer_feedforward import PositionwiseFeedForward

class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer:
    1. Masked Multi-Head Self-Attention
    2. Add & LayerNorm
    3. Multi-Head Cross-Attention (Encoder-Decoder Attention)
    4. Add & LayerNorm
    5. Position-wise Feed Forward
    6. Add & LayerNorm
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
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Masked Self-Attention
        self_attn_out, self_attn_weights = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))

        # 2. Cross-Attention (Q from decoder, K/V from encoder memory)
        cross_attn_out, cross_attn_weights = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))

        # 3. Feed Forward Network
        ffn_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))

        return tgt, self_attn_weights, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Stack of N Transformer Decoder Layers with Embedding and Positional Encoding.
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
            TransformerDecoderLayer(
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
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        x = self.embedding(tgt) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)

        self_attn_list = []
        cross_attn_list = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
            self_attn_list.append(self_attn_weights)
            cross_attn_list.append(cross_attn_weights)

        x = self.norm(x)
        return x, self_attn_list, cross_attn_list
