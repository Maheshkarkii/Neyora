import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # Headless backend
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_attention_matrix(
    attention_matrix: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str],
    save_path: Optional[str] = "reports/attention_heatmap.png",
    title: str = "Attention Alignment Heatmap"
) -> None:
    """
    Renders and saves a heatmap of attention weights between target and source tokens.
    attention_matrix shape: [tgt_len, src_len]
    """
    os.makedirs(os.path.dirname(save_path) if save_path else "reports", exist_ok=True)
    attn = attention_matrix.cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens)*0.8), max(5, len(tgt_tokens)*0.6)))
    cax = ax.matshow(attn, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))

    ax.set_xticklabels(src_tokens, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(tgt_tokens, fontsize=10)

    ax.set_xlabel("Source Nepali Tokens", fontsize=11, labelpad=10)
    ax.set_ylabel("Generated English Tokens", fontsize=11)
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def print_text_attention_alignment(
    attention_matrix: torch.Tensor,
    src_tokens: List[str],
    tgt_tokens: List[str]
) -> None:
    """Prints a text-based alignment table showing which source token received highest attention."""
    attn = attention_matrix.cpu().detach().numpy()
    print("\n--- ATTENTION ALIGNMENT SUMMARY ---")
    for t_idx, tgt_tok in enumerate(tgt_tokens):
        weights = attn[t_idx]
        max_s_idx = int(np.argmax(weights))
        top_weight = weights[max_s_idx]
        aligned_src = src_tokens[max_s_idx] if max_s_idx < len(src_tokens) else "?"
        print(f"Generated: '{tgt_tok:<12}' -> Attended: '{aligned_src:<15}' (weight: {top_weight:.3f})")
    print("----------------------------------\n")
