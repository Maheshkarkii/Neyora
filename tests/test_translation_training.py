import pytest
import torch
import torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

def test_loss_backward_and_clipping():
    B, S, T, E, H = 2, 5, 6, 32, 64
    src_vocab, tgt_vocab = 30, 30
    pad_idx = 0
    device = torch.device("cpu")

    enc = Encoder(src_vocab, E, H, pad_idx=pad_idx)
    dec = Decoder(tgt_vocab, E, H, pad_idx=pad_idx)
    model = Seq2Seq(enc, dec, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    src = torch.tensor([[2, 4, 5, 3, 0], [2, 6, 7, 8, 3]])
    tgt = torch.tensor([[2, 10, 11, 3, 0, 0], [2, 12, 13, 14, 15, 3]])

    optimizer.zero_grad()
    output = model(src, tgt, teacher_forcing_ratio=0.5)

    out_dim = output.shape[-1]
    loss = criterion(output[:, 1:, :].reshape(-1, out_dim), tgt[:, 1:].reshape(-1))

    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert grad_norm > 0.0
    optimizer.step()
