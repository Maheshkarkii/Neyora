import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + Mask ) * V
    
    Tensor Dimensions:
    - Q: [B, H, L_q, d_k]
    - K: [B, H, L_k, d_k]
    - V: [B, H, L_v, d_v] (where L_v == L_k)
    - Mask: [B, 1, L_q, L_k] or broadcastable boolean/float mask
    - Output: [B, H, L_q, d_v]
    - Attention Weights: [B, H, L_q, L_k]
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        # 1. Compute dot product scores: [B, H, L_q, L_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply attention mask if provided
        if mask is not None:
            # If mask is boolean where True = masked position, fill with -1e9
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -1e9)
            else:
                scores = scores + mask

        # 3. Softmax over key sequence length dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted sum of values: [B, H, L_q, d_v]
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    Tensor Dimensions:
    - Input Q: [B, L_q, d_model]
    - Input K: [B, L_k, d_model]
    - Input V: [B, L_v, d_model]
    - Output:  [B, L_q, d_model]
    - Attention Weights: [B, num_heads, L_q, L_k]
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Query, Key, Value, and Output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key.size(1)
        len_v = value.size(1)

        # 1. Project inputs: [B, L, D] -> [B, L, H, D_h] -> [B, H, L, D_h]
        q = self.w_q(query).view(batch_size, len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(batch_size, len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(batch_size, len_v, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention: [B, H, L_q, D_h], [B, H, L_q, L_k]
        attn_output, attn_weights = self.attention(q, k, v, mask=mask)

        # 3. Concatenate heads: [B, H, L_q, D_h] -> [B, L_q, H, D_h] -> [B, L_q, D]
        concat = attn_output.transpose(1, 2).contiguous().view(batch_size, len_q, self.d_model)

        # 4. Final linear projection: [B, L_q, D]
        output = self.w_o(concat)
        output = self.dropout(output)

        return output, attn_weights
