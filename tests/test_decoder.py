import pytest
import torch
from src.models.decoder import Decoder

def test_decoder_step_shapes():
    B, E, H, layers = 4, 64, 128, 2
    out_vocab_size = 60
    decoder = Decoder(output_dim=out_vocab_size, emb_dim=E, hid_dim=H, num_layers=layers, dropout=0.1)

    input_tokens = torch.randint(0, out_vocab_size, (B,))
    hidden = torch.randn(layers, B, H)
    cell = torch.randn(layers, B, H)

    prediction, (new_hidden, new_cell) = decoder(input_tokens, hidden, cell)

    assert prediction.shape == (B, out_vocab_size)
    assert new_hidden.shape == (layers, B, H)
    assert new_cell.shape == (layers, B, H)
