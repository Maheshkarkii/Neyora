import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.

    Mathematical Formulation:
    1. Score Calculation (Energy):
       e_{t, i} = v_a^T tanh(W_s * s_{t-1} + W_h * h_i)
       where:
         s_{t-1} in R^H is the decoder hidden state from the previous step.
         h_i in R^H is the encoder hidden state at source position i.
         W_s, W_h in R^{H x H} are learned linear projections.
         v_a in R^H is the learned attention weight vector.

    2. Padding Masking:
       e_{t, i} = -1e9 for all i where source token is <PAD>.

    3. Normalized Attention Weights:
       alpha_{t, i} = exp(e_{t, i}) / sum_{j=1}^S exp(e_{t, j})
       (Softmax ensures weights form a valid probability distribution summing to 1.0).

    4. Context Vector:
       c_t = sum_{i=1}^S alpha_{t, i} * h_i in R^H

    Tensor Shapes:
    - `decoder_hidden` : [B, H] (from top layer of decoder hidden state)
    - `encoder_outputs`: [B, S, H]
    - `mask`           : [B, S] (Bool or Byte tensor where True/1 indicates PAD)
    - `attention_weights`: [B, S]
    - `context`        : [B, 1, H]
    """
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: Optional[int] = None):
        super().__init__()
        attn_dim = attn_dim or dec_hid_dim
        self.W_h = nn.Linear(enc_hid_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_hid_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden: [B, H]
        # encoder_outputs: [B, S, H]
        src_len = encoder_outputs.shape[1]

        # 1. Project decoder state and encoder outputs into attention space
        # W_s(decoder_hidden): [B, attn_dim] -> unsqueeze to [B, 1, attn_dim]
        # W_h(encoder_outputs): [B, S, attn_dim]
        dec_proj = self.W_s(decoder_hidden).unsqueeze(1) # [B, 1, attn_dim]
        enc_proj = self.W_h(encoder_outputs)            # [B, S, attn_dim]

        # 2. Additive combination + tanh non-linearity
        energy = torch.tanh(dec_proj + enc_proj)        # [B, S, attn_dim]
        scores = self.v_a(energy).squeeze(2)            # [B, S]

        # 3. Apply Padding Mask (assign large negative value so softmax yields ~0.0)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # 4. Softmax normalization across source positions S
        attention_weights = F.softmax(scores, dim=1)    # [B, S]

        # 5. Compute context vector via weighted sum: [B, 1, S] x [B, S, H] -> [B, 1, H]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # [B, 1, H]

        return context, attention_weights
