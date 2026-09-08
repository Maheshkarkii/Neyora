import torch
import torch.nn as nn
from typing import Tuple

class Decoder(nn.Module):
    """
    LSTM Decoder with Teacher Forcing Support.

    Why No Softmax in Output Layer:
    PyTorch nn.CrossEntropyLoss internally combines LogSoftmax and NLLLoss in a single numerically stable function
    using the log-sum-exp trick. Applying Softmax explicitly before CrossEntropyLoss would cause redundant operations
    and numeric underflow/instability.

    Tensor Shapes (batch_first=True):
    - Single Step Input `input_token`: [B, 1] or [B], dtype torch.long
    - Embedded `embedded`            : [B, 1, Embedding_Dim (E)], dtype torch.float32
    - LSTM Output `output`           : [B, 1, Hidden_Dim (H)], dtype torch.float32
    - Logits `prediction`            : [B, Output_Vocab_Size (V)], dtype torch.float32
    - Hidden State `hidden`          : [Num_Layers, B, Hidden_Dim (H)], dtype torch.float32
    - Cell State `cell`              : [Num_Layers, B, Hidden_Dim (H)], dtype torch.float32
    """
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hid_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0
    ):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # input_token: [B] or [B, 1]
        if input_token.ndim == 1:
            input_token = input_token.unsqueeze(1) # [B, 1]

        embedded = self.dropout(self.embedding(input_token)) # [B, 1, E]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell)) # output: [B, 1, H]
        prediction = self.fc_out(output.squeeze(1)) # [B, output_dim]

        return prediction, (hidden, cell)
