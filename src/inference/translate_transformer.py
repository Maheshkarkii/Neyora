import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
from src.models.transformer import Transformer
from src.data.tokenizer import TranslationTokenizer
from src.evaluation.metrics import calculate_sentence_bleu

def translate_transformer_greedy(
    model: Transformer,
    sentence: str,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    max_len: int = 35,
    device: torch.device = torch.device("cpu")
) -> Tuple[str, torch.Tensor, List[str], List[str]]:
    """
    Autoregressive greedy decoding for the Transformer model.
    Returns:
        (translated_text, final_layer_cross_attention_matrix, src_tokens, tgt_tokens)
    """
    model.eval()
    with torch.inference_mode():
        src_tokens = src_tokenizer.encode(sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0) # [1, S]
        src_mask = model.make_src_mask(src_tensor)

        # 1. Encode source sentence once
        memory, _ = model.encoder(src_tensor, src_mask=src_mask)

        # 2. Autoregressive decoding starting with <SOS>
        sos_idx = tgt_tokenizer.vocab.sos_idx
        eos_idx = tgt_tokenizer.vocab.eos_idx
        tgt_indices = [sos_idx]
        cross_attention_weights = None

        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0) # [1, T_curr]
            tgt_mask = model.make_tgt_mask(tgt_tensor)

            dec_out, _, cross_attns = model.decoder(
                tgt=tgt_tensor,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
            logits = model.output_projection(dec_out) # [1, T_curr, V_tgt]
            next_token_id = logits[:, -1, :].argmax(dim=-1).item()

            if cross_attns:
                # Last layer cross-attention averaged across heads: [1, H, T_curr, S] -> [T_curr, S]
                cross_attention_weights = cross_attns[-1].mean(dim=1).squeeze(0)

            if next_token_id == eos_idx:
                break

            tgt_indices.append(next_token_id)

        output_tokens = [t for t in tgt_indices if t not in (sos_idx, eos_idx)]
        src_token_strs = [src_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in src_tokens]
        tgt_token_strs = [tgt_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in output_tokens]
        translated_str = tgt_tokenizer.decode(output_tokens, skip_special_tokens=True)

        if cross_attention_weights is None:
            cross_attention_weights = torch.empty((0, len(src_tokens)))

    return translated_str, cross_attention_weights, src_token_strs, tgt_token_strs


def translate_transformer_beam_search(
    model: Transformer,
    sentence: str,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    beam_width: int = 5,
    max_len: int = 35,
    length_penalty_alpha: float = 0.6,
    device: torch.device = torch.device("cpu")
) -> Tuple[str, torch.Tensor, List[str], List[str]]:
    """
    Autoregressive beam search decoding for the Transformer model.
    """
    model.eval()
    with torch.inference_mode():
        src_tokens = src_tokenizer.encode(sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = model.make_src_mask(src_tensor)

        memory, _ = model.encoder(src_tensor, src_mask=src_mask)

        sos_idx = tgt_tokenizer.vocab.sos_idx
        eos_idx = tgt_tokenizer.vocab.eos_idx

        # Beams store: (cumulative_log_prob, [token_ids])
        active_beams = [(0.0, [sos_idx])]
        completed_beams = []

        for step in range(max_len):
            candidates = []
            for score, tokens in active_beams:
                tgt_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                tgt_mask = model.make_tgt_mask(tgt_tensor)

                dec_out, _, _ = model.decoder(
                    tgt=tgt_tensor,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_mask=src_mask
                )
                logits = model.output_projection(dec_out[:, -1:, :]) # [1, 1, V_tgt]
                log_probs = F.log_softmax(logits, dim=-1).squeeze()

                topk_probs, topk_indices = torch.topk(log_probs, k=beam_width)
                for k in range(beam_width):
                    cand_tok = topk_indices[k].item()
                    cand_score = score + topk_probs[k].item()
                    cand_seq = tokens + [cand_tok]

                    if cand_tok == eos_idx:
                        lp = ((5.0 + len(cand_seq) - 1) / 6.0) ** length_penalty_alpha
                        completed_beams.append((cand_score / lp, cand_seq))
                    else:
                        candidates.append((cand_score, cand_seq))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            active_beams = candidates[:beam_width]

            if len(completed_beams) >= beam_width:
                break

        if completed_beams:
            completed_beams.sort(key=lambda x: x[0], reverse=True)
            best_seq = completed_beams[0][1]
        else:
            active_beams.sort(key=lambda x: x[0], reverse=True)
            best_seq = active_beams[0][1]

        output_tokens = [t for t in best_seq if t not in (sos_idx, eos_idx)]
        src_token_strs = [src_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in src_tokens]
        tgt_token_strs = [tgt_tokenizer.vocab.idx2token.get(t, "<UNK>") for t in output_tokens]
        translated_str = tgt_tokenizer.decode(output_tokens, skip_special_tokens=True)

    return translated_str, torch.empty((0, len(src_tokens))), src_token_strs, tgt_token_strs
