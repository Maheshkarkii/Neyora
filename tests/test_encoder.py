import pytest
import torch
from src.models.encoder import Encoder

def test_encoder_forward_shapes():
    B, S, E, H, layers = 4, 10, 64, 128, 2
    vocab_size = 50
    encoder = Encoder(input_dim=vocab_size, emb_dim=E, hid_dim=H, num_layers=layers, dropout=0.1)

    src = torch.randint(0, vocab_size, (B, S))
    outputs, (hidden, cell) = encoder(src)

    assert outputs.shape == (B, S, H)
    assert hidden.shape == (layers, B, H)
    assert cell.shape == (layers, B, H)
    assert outputs.dtype == torch.float32
