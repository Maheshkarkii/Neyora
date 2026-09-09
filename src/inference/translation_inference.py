import os
import time
import torch
from typing import Dict, Any, Optional, Tuple, List
from src.models.seq2seq_attention import Seq2SeqAttention
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.training.train_attention import build_seq2seq_attention_model
from src.utils.checkpoint import load_checkpoint
from src.inference.translate_attention import translate_with_attention
from src.utils.logger import get_logger

logger = get_logger("translation_inference")

class TranslationInferenceEngine:
    """
    Inference engine for Nepali-to-English Neural Machine Translation (NMT)
    with Bahdanau Additive Attention.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_lstm_attention.pt",
        src_vocab_path: str = "data/processed/nmt/src_vocab.json",
        tgt_vocab_path: str = "data/processed/nmt/tgt_vocab.json",
        device: Optional[torch.device] = None,
        max_output_length: int = 30
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_output_length = max_output_length

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Translation checkpoint not found at {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        
        # Load vocabularies from checkpoint if available, otherwise from json files
        if "src_vocab" in checkpoint and "tgt_vocab" in checkpoint:
            self.src_vocab = TranslationVocabulary(checkpoint["src_vocab"])
            self.tgt_vocab = TranslationVocabulary(checkpoint["tgt_vocab"])
        else:
            self.src_vocab = TranslationVocabulary.load(src_vocab_path)
            self.tgt_vocab = TranslationVocabulary.load(tgt_vocab_path)

        self.src_tokenizer = TranslationTokenizer(self.src_vocab, is_nepali=True)
        self.tgt_tokenizer = TranslationTokenizer(self.tgt_vocab, is_nepali=False)

        # Model configuration
        model_config = checkpoint.get("config", {}).get("model", {})
        emb_dim = model_config.get("embedding_dim", 128)
        hid_dim = model_config.get("hidden_dim", 256)
        num_layers = model_config.get("num_layers", 2)
        dropout = 0.0 # Inference dropout

        self.model = build_seq2seq_attention_model(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            emb_dim=emb_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=self.src_vocab.pad_idx,
            device=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"Translation model successfully loaded from {checkpoint_path} on device: {self.device}")

    @torch.no_grad()
    def translate(
        self,
        nepali_text: str,
        max_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Translates a Nepali text string into English.
        Returns:
            Dict containing translated text, attention matrix, token lists, and latency.
        """
        start_time = time.perf_counter()
        gen_max_len = max_len or self.max_output_length

        if not nepali_text or not nepali_text.strip():
            return {
                "translation": "",
                "attention_matrix": None,
                "src_tokens": [],
                "tgt_tokens": [],
                "latency_sec": time.perf_counter() - start_time
            }

        translation, attn_matrix, src_token_strs, tgt_token_strs = translate_with_attention(
            model=self.model,
            sentence=nepali_text,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer,
            max_len=gen_max_len,
            device=self.device
        )

        latency_sec = time.perf_counter() - start_time

        return {
            "translation": translation,
            "attention_matrix": attn_matrix,
            "src_tokens": src_token_strs,
            "tgt_tokens": tgt_token_strs,
            "latency_sec": latency_sec
        }
