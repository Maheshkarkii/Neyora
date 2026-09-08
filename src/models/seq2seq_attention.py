import random
import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.models.encoder import Encoder
from src.models.decoder_attention import AttentiveDecoder

class Seq2SeqAttention(nn.Module):
    """
    End-to-End Sequence-to-Sequence Architecture with Bahdanau Attention.

    Tensor Shapes:
    - `src`         : [B, S], Source Nepali tokens
    - `tgt`         : [B, T], Target English tokens
    - `outputs`     : [B, T, V], Predicted logits over English vocabulary
    - `attentions`  : [B, T, S], Attention matrices across all generation steps
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: AttentiveDecoder,
        pad_idx: int = 0,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

        assert encoder.hid_dim == decoder.dec_hid_dim, "Encoder and Decoder hidden dimensions must match!"
        assert encoder.num_layers == decoder.num_layers, "Encoder and Decoder layers must match!"

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Creates boolean mask where True indicates <PAD> positions."""
        return (src == self.pad_idx) # [B, S]

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = src.shape[0]
        max_tgt_len = tgt.shape[1]
        src_len = src.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # Tensors to store outputs and attention weights
        outputs = torch.zeros(batch_size, max_tgt_len, tgt_vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, max_tgt_len, src_len, device=self.device)

        # 1. Mask and encode entire source sequence
        mask = self.create_mask(src) # [B, S]
        encoder_outputs, (hidden, cell) = self.encoder(src)
        # encoder_outputs: [B, S, H]
        # hidden, cell   : [num_layers, B, H]

        # 2. First input token is <SOS>
        decoder_input = tgt[:, 0] # [B]

        # 3. Sequential decoding
        for t in range(1, max_tgt_len):
            prediction, (hidden, cell), attn_weights = self.decoder(
                input_token=decoder_input,
                hidden=hidden,
                cell=cell,
                encoder_outputs=encoder_outputs,
                mask=mask
            )
            outputs[:, t, :] = prediction
            attentions[:, t, :] = attn_weights

            use_teacher_forcing = (random.random() < teacher_forcing_ratio) and self.training
            top1_prediction = prediction.argmax(dim=1)
            decoder_input = tgt[:, t] if use_teacher_forcing else top1_prediction

        return outputs, attentions
