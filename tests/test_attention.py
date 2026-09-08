import pytest
import torch
from src.models.attention import BahdanauAttention

def test_bahdanau_attention_shapes_and_sums():
    B, S, H = 4, 10, 128
    attn = BahdanauAttention(enc_hid_dim=H, dec_hid_dim=H)

    dec_hidden = torch.randn(B, H)
    enc_outputs = torch.randn(B, S, H)

    context, weights = attn(dec_hidden, enc_outputs)

    # 1. Output shapes
    assert context.shape == (B, 1, H)
    assert weights.shape == (B, S)

    # 2. Weights sum to 1.0
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(B), atol=1e-5)

    # 3. No NaNs
    assert not torch.isnan(context).any()
    assert not torch.isnan(weights).any()

def test_attention_padding_mask():
    B, S, H = 2, 6, 64
    attn = BahdanauAttention(enc_hid_dim=H, dec_hid_dim=H)

    dec_hidden = torch.randn(B, H)
    enc_outputs = torch.randn(B, S, H)

    # Mask last 2 tokens for batch item 0, and last 3 tokens for item 1
    mask = torch.tensor([
        [False, False, False, False, True, True],
        [False, False, False, True, True, True]
    ])

    context, weights = attn(dec_hidden, enc_outputs, mask=mask)

    # Padded positions must receive negligible probability (< 1e-6)
    assert weights[0, 4].item() < 1e-6
    assert weights[0, 5].item() < 1e-6
    assert weights[1, 3].item() < 1e-6
    assert weights[1, 4].item() < 1e-6
    assert weights[1, 5].item() < 1e-6

    # Valid positions must sum to 1.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)
