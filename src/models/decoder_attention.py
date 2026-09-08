import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.models.attention import BahdanauAttention

class AttentiveDecoder(nn.Module):
    """
    LSTM Decoder with Integrated Bahdanau Attention.

    Tensor Shapes (batch_first=True):
    - `input_token`    : [B] or [B, 1], previous token index
    - `hidden`         : [num_layers, B, H], decoder hidden state
    - `cell`           : [num_layers, B, H], decoder cell state
    - `encoder_outputs`: [B, S, H], full sequence of encoder hidden states
    - `mask`           : [B, S], padding mask
    - Outputs:
        * `prediction`: [B, V], unnormalized vocabulary logits
        * `(new_h, new_c)`: [num_layers, B, H]
        * `attn_weights`: [B, S], attention distribution over source tokens
    """
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0
    ):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.num_layers = num_layers

        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # LSTM receives concatenation of embedded target token [E] and context vector [H]
        self.lstm = nn.LSTM(
            emb_dim + enc_hid_dim,
            dec_hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Linear projection combines LSTM output [H], context [H], and embedding [E]
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if input_token.ndim == 1:
            input_token = input_token.unsqueeze(1) # [B, 1]

        # 1. Embed previous target token
        embedded = self.dropout(self.embedding(input_token)) # [B, 1, E]

        # 2. Compute attention using top-layer decoder hidden state
        # hidden[-1] is [B, H]
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask=mask)
        # context: [B, 1, H], attn_weights: [B, S]

        # 3. Concatenate embedded token and context vector for LSTM input
        lstm_input = torch.cat((embedded, context), dim=2) # [B, 1, E + H]

        # 4. Decoder LSTM forward step
        output, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden, cell)) # output: [B, 1, H]

        # 5. Concatenate output, context, and embedding for vocabulary projection
        combined = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1) # [B, H + H + E]
        prediction = self.fc_out(combined) # [B, output_dim]

        return prediction, (new_hidden, new_cell), attn_weights
