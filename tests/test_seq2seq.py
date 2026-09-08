import pytest
import torch
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

def test_seq2seq_forward_shapes():
    B, S, T, E, H, layers = 4, 8, 12, 64, 128, 2
    src_vocab, tgt_vocab = 40, 50
    device = torch.device("cpu")

    enc = Encoder(src_vocab, E, H, num_layers=layers)
    dec = Decoder(tgt_vocab, E, H, num_layers=layers)
    model = Seq2Seq(enc, dec, device=device)

    src = torch.randint(0, src_vocab, (B, S))
    tgt = torch.randint(0, tgt_vocab, (B, T))

    # With teacher forcing
    out_tf = model(src, tgt, teacher_forcing_ratio=1.0)
    assert out_tf.shape == (B, T, tgt_vocab)

    # Without teacher forcing
    out_no_tf = model(src, tgt, teacher_forcing_ratio=0.0)
    assert out_no_tf.shape == (B, T, tgt_vocab)
