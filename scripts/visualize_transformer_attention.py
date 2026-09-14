import os
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer import Transformer
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.inference.translate_transformer import translate_transformer_greedy

def plot_transformer_cross_attention(
    attention_matrix: torch.Tensor,
    src_tokens: list,
    tgt_tokens: list,
    save_path: str = "reports/transformer_attention_heatmap.png"
):
    """Plots and saves cross-attention heatmap for the Transformer model."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    attn = attention_matrix.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(attn, cmap="Blues")
    fig.colorbar(cax)

    # Set axes labels
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(tgt_tokens, fontsize=10)

    ax.set_xlabel("Source Nepali Tokens", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Generated English Tokens", fontsize=11, fontweight="bold")
    ax.set_title("Transformer Cross-Attention Alignment Heatmap", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to '{save_path}'.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/best_transformer.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    src_vocab = TranslationVocabulary(ckpt["src_vocab"])
    tgt_vocab = TranslationVocabulary(ckpt["tgt_vocab"])

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    cfg = ckpt.get("config", {}).get("model", {})
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.get("d_model", 128),
        num_heads=cfg.get("num_heads", 4),
        num_encoder_layers=cfg.get("num_encoder_layers", 2),
        num_decoder_layers=cfg.get("num_decoder_layers", 2),
        ffn_dim=cfg.get("ffn_dim", 256),
        dropout=0.0,
        pad_idx=src_vocab.pad_idx,
        device=device
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sentence = "नमस्ते म नेपालबाट आएको हुँ।" if len(sys.argv) <= 1 else " ".join(sys.argv[1:])
    pred_str, attn_mat, s_toks, t_toks = translate_transformer_greedy(
        model=model,
        sentence=sentence,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        device=device
    )

    print(f"Source Nepali:      {sentence}")
    print(f"Transformer Pred:   {pred_str}")
    print(f"Source tokens:      {s_toks}")
    print(f"Target tokens:      {t_toks}")

    if attn_mat.numel() > 0 and len(t_toks) > 0 and len(s_toks) > 0:
        # Match slice dimensions: [len(t_toks), len(s_toks)]
        slice_attn = attn_mat[:len(t_toks), :len(s_toks)]
        plot_transformer_cross_attention(slice_attn, s_toks, t_toks)

if __name__ == "__main__":
    main()
