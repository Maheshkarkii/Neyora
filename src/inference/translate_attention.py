import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
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

        if attention_steps:
            attention_matrix = torch.stack(attention_steps, dim=0) # [T, S]
        else:
            attention_matrix = torch.empty((0, len(src_tokens)))

        src_token_strs = [src_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in src_tokens]
        tgt_token_strs = [tgt_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in translated_token_ids]
        translated_str = tgt_tokenizer.decode(translated_token_ids, skip_special_tokens=True)

    return translated_str, attention_matrix, src_token_strs, tgt_token_strs


def translate_with_beam_search(
    model: Seq2SeqAttention,
    sentence: str,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    beam_width: int = 5,
    max_len: int = 30,
    length_penalty_alpha: float = 0.6,
    device: torch.device = torch.device("cpu")
) -> Tuple[str, torch.Tensor, List[str], List[str]]:
    """
    Translates a Nepali sentence using Beam Search Decoding with length penalty.
    """
    model.eval()
    with torch.no_grad():
        src_tokens = src_tokenizer.encode(sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0) # [1, S]
        mask = model.create_mask(src_tensor)

        encoder_outputs, (enc_hidden, enc_cell) = model.encoder(src_tensor)

        # Beam state: (cumulative_log_prob, [token_ids], hidden, cell, [attn_tensors])
        sos_idx = tgt_tokenizer.vocab.sos_idx
        eos_idx = tgt_tokenizer.vocab.eos_idx

        # Initialize with <SOS>
        active_beams = [(0.0, [sos_idx], enc_hidden, enc_cell, [])]
        completed_beams = []

        for step in range(max_len):
            candidates = []
            for score, tokens, hidden, cell, attn_list in active_beams:
                last_tok = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                prediction, (next_hidden, next_cell), attn_weights = model.decoder(
                    input_token=last_tok,
                    hidden=hidden,
                    cell=cell,
                    encoder_outputs=encoder_outputs,
                    mask=mask
                )
                log_probs = F.log_softmax(prediction, dim=-1).squeeze(0) # [V]
                topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_width)

                for k in range(beam_width):
                    cand_tok = topk_indices[k].item()
                    cand_score = score + topk_log_probs[k].item()
                    cand_tokens = tokens + [cand_tok]
                    cand_attn = attn_list + [attn_weights.squeeze(0)]

                    if cand_tok == eos_idx:
                        # Length penalty normalization
                        length = len(cand_tokens) - 1 # excluding SOS
                        lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
                        normalized_score = cand_score / max(1e-6, lp)
                        completed_beams.append((normalized_score, cand_tokens, cand_attn))
                    else:
                        candidates.append((cand_score, cand_tokens, next_hidden, next_cell, cand_attn))

            # Prune to top beam_width active candidates
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            active_beams = candidates[:beam_width]

            if len(completed_beams) >= beam_width:
                break

        # Select best hypothesis
        if completed_beams:
            completed_beams.sort(key=lambda x: x[0], reverse=True)
            best_score, best_tokens, best_attns = completed_beams[0]
            # Strip SOS and EOS
            output_tokens = [t for t in best_tokens if t not in (sos_idx, eos_idx)]
        else:
            active_beams.sort(key=lambda x: x[0], reverse=True)
            _, best_tokens, _, _, best_attns = active_beams[0]
            output_tokens = [t for t in best_tokens if t not in (sos_idx, eos_idx)]

        if best_attns:
            attention_matrix = torch.stack(best_attns, dim=0)
        else:
            attention_matrix = torch.empty((0, len(src_tokens)))

        src_token_strs = [src_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in src_tokens]
        tgt_token_strs = [tgt_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in output_tokens]
        translated_str = tgt_tokenizer.decode(output_tokens, skip_special_tokens=True)

    return translated_str, attention_matrix, src_token_strs, tgt_token_strs


def evaluate_attention_corpus_bleu(
    model: Seq2SeqAttention,
    dataloader: torch.utils.data.DataLoader,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    device: torch.device,
    use_beam_search: bool = False,
    beam_width: int = 5
) -> float:
    """Evaluates corpus BLEU score for Attention Seq2Seq with greedy or beam search."""
    model.eval()
    scores: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            src_batch, _, tgt_batch, _, raw_srcs, raw_tgts = batch
            for i in range(len(raw_srcs)):
                if use_beam_search:
                    pred_str, _, _, _ = translate_with_beam_search(
                        model=model,
                        sentence=raw_srcs[i],
                        src_tokenizer=src_tokenizer,
                        tgt_tokenizer=tgt_tokenizer,
                        beam_width=beam_width,
                        device=device
                    )
                else:
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
