import random
import torch
import torch.nn as nn
from typing import Tuple
from src.models.encoder import Encoder
from src.models.decoder import Decoder

class Seq2Seq(nn.Module):
    """
    End-to-End Sequence-to-Sequence Architecture.

    Why Teacher Forcing is Used:
    Autoregressive decoders condition each step on the previous output. Early in training, the model's predictions
    are almost entirely incorrect; feeding wrong predictions causes cascading errors and destabilizes gradient flow.
    Teacher forcing feeds the ground-truth previous token with probability `teacher_forcing_ratio`, accelerating
    convergence and stabilizing representation learning.

    Tensor Shapes:
    - `src`    : [B, S], Source Nepali token sequences
    - `tgt`    : [B, T], Target English token sequences (with <SOS> and <EOS>)
    - `outputs`: [B, T, Output_Vocab_Size], Logits over the target English vocabulary across all timesteps
    """
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: torch.device
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Encoder and Decoder hidden dimensions must match!"
        assert encoder.num_layers == decoder.num_layers, "Encoder and Decoder number of layers must match!"

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        max_tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # Tensor to store decoder logits: [B, T, V]
        outputs = torch.zeros(batch_size, max_tgt_len, tgt_vocab_size, device=self.device)

        # 1. Encode source sequence to obtain final hidden and cell states
        _, (hidden, cell) = self.encoder(src)

        # 2. First input to decoder is the <SOS> token (index 0 of target)
        decoder_input = tgt[:, 0] # [B]

        # 3. Autoregressive decoding loop for timesteps 1 to max_tgt_len-1
        for t in range(1, max_tgt_len):
            prediction, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = prediction # Store logits at step t

            # Decide whether to use teacher forcing
            use_teacher_forcing = (random.random() < teacher_forcing_ratio) and self.training
            top1_prediction = prediction.argmax(dim=1) # [B]

            # Next input is ground-truth target token if teacher forcing, else model's top-1 prediction
            decoder_input = tgt[:, t] if use_teacher_forcing else top1_prediction

        return outputs
