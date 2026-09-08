import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    """
    LSTM Encoder for Seq2Seq Neural Machine Translation.

    Why We Use nn.Embedding:
    Token IDs (e.g. 12, 45, 87) are discrete categorical integers with no continuous metric relationships
    (e.g., ID 87 is not "larger" or 7x the magnitude of ID 12). nn.Embedding maps each discrete index
    into a learnable continuous dense vector in R^E where geometric distance and orientation capture
    semantic affinities learned during training.

    What Hidden State (h) and Cell State (c) Represent:
    - Hidden State (h_t in R^H): The short-term working memory output at time step t, carrying active contextual representations.
    - Cell State (c_t in R^H): The long-term memory conveyor belt regulated by LSTM input/forget/output gates, preserving
      information across long sequence distances without vanishing gradient degradation.

    Tensor Shapes (batch_first=True):
    - Input `src`        : [Batch_Size (B), Source_Length (S)], dtype torch.long
    - Embedded `embedded`: [B, S, Embedding_Dim (E)], dtype torch.float32
    - Outputs `outputs`  : [B, S, Hidden_Dim (H)], dtype torch.float32
    - Hidden State `h_n` : [Num_Layers, B, Hidden_Dim (H)], dtype torch.float32
    - Cell State `c_n`   : [Num_Layers, B, Hidden_Dim (H)], dtype torch.float32
    """
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hid_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # src: [B, S]
        embedded = self.dropout(self.embedding(src)) # [B, S, E]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [B, S, H]
        # hidden : [num_layers, B, H]
        # cell   : [num_layers, B, H]
        return outputs, (hidden, cell)
