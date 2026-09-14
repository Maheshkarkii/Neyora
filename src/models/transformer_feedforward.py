import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    
    Tensor Dimensions:
    - Input:  [B, L, d_model]
    - Hidden: [B, L, ffn_dim]
    - Output: [B, L, d_model]
    """
    def __init__(self, d_model: int = 256, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN applied independently at each sequence position
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
