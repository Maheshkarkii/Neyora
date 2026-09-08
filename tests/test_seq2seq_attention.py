import pytest
import torch
from src.models.encoder import Encoder
from src.models.decoder_attention import AttentiveDecoder
from src.models.seq2seq_attention import Seq2SeqAttention

def test_seq2seq_attention_forward():
    B, S, T, E, H, layers = 2, 6, 8, 64, 128, 2
    src_vocab, tgt_vocab = 50, 60
    device = torch.device("cpu")

    enc = Encoder(src_vocab, E, H, num_layers=layers)
    dec = AttentiveDecoder(tgt_vocab, E, H, H, num_layers=layers)
    model = Seq2SeqAttention(enc, dec, pad_idx=0, device=device)

    src = torch.randint(1, src_vocab, (B, S))
    src[:, -2:] = 0 # add padding
    tgt = torch.randint(1, tgt_vocab, (B, T))

    outputs, attentions = model(src, tgt, teacher_forcing_ratio=0.5)

    assert outputs.shape == (B, T, tgt_vocab)
    assert attentions.shape == (B, T, S)
    assert not torch.isnan(outputs).any()
    assert not torch.isnan(attentions).any()
