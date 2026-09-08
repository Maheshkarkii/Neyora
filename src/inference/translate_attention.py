import torch
from typing import Tuple, List
from src.models.seq2seq_attention import Seq2SeqAttention
from src.data.tokenizer import TranslationTokenizer
from src.evaluation.metrics import calculate_sentence_bleu

def translate_with_attention(
    model: Seq2SeqAttention,
    sentence: str,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    max_len: int = 30,
    device: torch.device = torch.device("cpu")
) -> Tuple[str, torch.Tensor, List[str], List[str]]:
    """
    Translates a Nepali sentence using greedy decoding with attention tracking.
    Returns:
      (translated_sentence, attention_matrix, src_token_strs, tgt_token_strs)
    """
    model.eval()
    with torch.no_grad():
        src_tokens = src_tokenizer.encode(sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0) # [1, S]
        mask = model.create_mask(src_tensor)

        encoder_outputs, (hidden, cell) = model.encoder(src_tensor)

        current_token = torch.tensor([tgt_tokenizer.vocab.sos_idx], dtype=torch.long, device=device)
        translated_token_ids: List[int] = []
        attention_steps: List[torch.Tensor] = []

        for _ in range(max_len):
            prediction, (hidden, cell), attn_weights = model.decoder(
                input_token=current_token,
                hidden=hidden,
                cell=cell,
                encoder_outputs=encoder_outputs,
                mask=mask
            )
            top1 = prediction.argmax(dim=1).item()
            attention_steps.append(attn_weights.squeeze(0)) # [S]

            if top1 == tgt_tokenizer.vocab.eos_idx:
                break

            translated_token_ids.append(top1)
            current_token = torch.tensor([top1], dtype=torch.long, device=device)

        # Stack attention weights along time: [T_gen, S]
        if attention_steps:
            attention_matrix = torch.stack(attention_steps, dim=0) # [T, S]
        else:
            attention_matrix = torch.empty((0, len(src_tokens)))

        src_token_strs = [src_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in src_tokens]
        tgt_token_strs = [tgt_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in translated_token_ids]
        translated_str = tgt_tokenizer.decode(translated_token_ids, skip_special_tokens=True)

    return translated_str, attention_matrix, src_token_strs, tgt_token_strs


def evaluate_attention_corpus_bleu(
    model: Seq2SeqAttention,
    dataloader: torch.utils.data.DataLoader,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    device: torch.device
) -> float:
    """Evaluates corpus BLEU score for Attention Seq2Seq."""
    model.eval()
    scores: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            src_batch, _, tgt_batch, _, raw_srcs, raw_tgts = batch
            for i in range(len(raw_srcs)):
                pred_str, _, _, _ = translate_with_attention(
                    model=model,
                    sentence=raw_srcs[i],
                    src_tokenizer=src_tokenizer,
                    tgt_tokenizer=tgt_tokenizer,
                    device=device
                )
                ref_tokens = raw_tgts[i].strip().lower().split()
                hyp_tokens = pred_str.strip().lower().split()
                scores.append(calculate_sentence_bleu(ref_tokens, hyp_tokens))

    return float(sum(scores) / max(1, len(scores))) * 100.0
