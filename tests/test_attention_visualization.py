import pytest
import os
import torch
from src.evaluation.attention_visualization import plot_attention_matrix, print_text_attention_alignment

def test_plot_attention_matrix(tmp_path):
    attn = torch.softmax(torch.randn(5, 4), dim=-1)
    src_tokens = ["म", "आज", "कलेज", "<EOS>"]
    tgt_tokens = ["i", "go", "to", "college", "<EOS>"]
    save_path = str(tmp_path / "heatmap.png")

    plot_attention_matrix(attn, src_tokens, tgt_tokens, save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
