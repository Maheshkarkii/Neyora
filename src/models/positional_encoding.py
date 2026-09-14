import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Tensor Dimensions:
    - Input:  [B, L, D]
    - Output: [B, L, D]
    """
    def __init__(self, d_model: int = 256, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as persistent non-trainable buffer: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)
