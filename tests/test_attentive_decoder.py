import pytest
import torch
from src.models.decoder_attention import AttentiveDecoder

def test_attentive_decoder_step():
    B, S, E, H, layers, out_vocab = 3, 7, 64, 128, 2, 80
    decoder = AttentiveDecoder(
        output_dim=out_vocab,
        emb_dim=E,
        enc_hid_dim=H,
        dec_hid_dim=H,
        num_layers=layers,
        dropout=0.1
    )

    input_tokens = torch.randint(0, out_vocab, (B,))
    hidden = torch.randn(layers, B, H)
    cell = torch.randn(layers, B, H)
    enc_outputs = torch.randn(B, S, H)
    mask = torch.zeros(B, S, dtype=torch.bool)

    prediction, (new_h, new_c), attn_weights = decoder(
        input_tokens, hidden, cell, enc_outputs, mask=mask
    )

    assert prediction.shape == (B, out_vocab)
    assert new_h.shape == (layers, B, H)
    assert new_c.shape == (layers, B, H)
    assert attn_weights.shape == (B, S)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(B), atol=1e-5)
