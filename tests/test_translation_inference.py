import pytest
import os
import torch
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.training.train_translation import build_seq2seq_model
from src.inference.translate import translate_sentence, compute_sentence_bleu
from src.utils.checkpoint import save_checkpoint, load_checkpoint

def test_inference_greedy_decoding(tmp_path):
    src_vocab = TranslationVocabulary()
    tgt_vocab = TranslationVocabulary()
    src_vocab.build_from_texts(["म कलेज जान्छु ।"])
    tgt_vocab.build_from_texts(["i go to college ."])

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    model = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=32,
        hid_dim=64,
        num_layers=1,
        device=torch.device("cpu")
    )

    translation = translate_sentence(model, "म कलेज जान्छु ।", src_tok, tgt_tok, max_len=10)
    assert isinstance(translation, str)

def test_checkpoint_roundtrip(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    save_checkpoint(
        state={"epoch": 5, "val_loss": 0.42, "model_state_dict": {}},
        checkpoint_dir=ckpt_dir,
        filename="test.pt"
    )
    loaded = load_checkpoint(os.path.join(ckpt_dir, "test.pt"))
    assert loaded["epoch"] == 5
    assert loaded["val_loss"] == 0.42

def test_bleu_computation():
    ref = ["i", "go", "to", "college"]
    hyp = ["i", "go", "to", "college"]
    score = compute_sentence_bleu(ref, hyp)
    assert score == 1.0

    hyp_bad = ["they", "play", "football"]
    score_bad = compute_sentence_bleu(ref, hyp_bad)
    assert score_bad == 0.0
