import torch
from typing import List, Union
from src.models.seq2seq import Seq2Seq
from src.data.tokenizer import TranslationTokenizer
from collections import Counter
import math

def translate_sentence(
    model: Seq2Seq,
    sentence: str,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    max_len: int = 30,
    device: torch.device = torch.device("cpu")
) -> str:
    """
    Translates a single Nepali sentence to English using Greedy Decoding.

    Inference Policy:
    During inference, the ground-truth target is completely unavailable.
    The decoder feeds its own argmax prediction from time step t into time step t+1.
    """
    model.eval()
    with torch.no_grad():
        src_tokens = src_tokenizer.encode(sentence, add_special_tokens=True)
        src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0) # [1, S]

        _, (hidden, cell) = model.encoder(src_tensor)

        # Start token <SOS>
        current_token = torch.tensor([tgt_tokenizer.vocab.sos_idx], dtype=torch.long, device=device) # [1]
        translated_token_ids: List[int] = []

        for _ in range(max_len):
            prediction, (hidden, cell) = model.decoder(current_token, hidden, cell)
            top1 = prediction.argmax(dim=1).item()

            if top1 == tgt_tokenizer.vocab.eos_idx:
                break

            translated_token_ids.append(top1)
            current_token = torch.tensor([top1], dtype=torch.long, device=device)

    return tgt_tokenizer.decode(translated_token_ids, skip_special_tokens=True)


def compute_sentence_bleu(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """Computes standard sentence BLEU score with brevity penalty and modified n-gram precision."""
    if not hypothesis:
        return 0.0
    if not reference:
        return 0.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)

    # Brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / max(1, hyp_len)))

    precisions = []
    for n in range(1, max_n + 1):
        if hyp_len < n:
            continue
        hyp_ngrams = [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]

        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped_count = sum(min(count, ref_counts.get(ng, 0)) for ng, count in hyp_counts.items())
        total_count = max(1, len(hyp_ngrams))
        precisions.append(clipped_count / total_count)

    if not precisions or min(precisions) == 0:
        return 0.0

    log_sum = sum(math.log(p) for p in precisions) / len(precisions)
    return float(bp * math.exp(log_sum))


def evaluate_corpus_bleu(
    model: Seq2Seq,
    dataloader: torch.utils.data.DataLoader,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    device: torch.device
) -> float:
    """Evaluates average BLEU score across an evaluation dataloader."""
    model.eval()
    scores: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            src_batch, _, tgt_batch, _, raw_srcs, raw_tgts = batch
            src_batch = src_batch.to(device)

            for i in range(len(raw_srcs)):
                pred_str = translate_sentence(
                    model,
                    raw_srcs[i],
                    src_tokenizer,
                    tgt_tokenizer,
                    device=device
                )
                ref_tokens = raw_tgts[i].strip().lower().split()
                hyp_tokens = pred_str.strip().lower().split()
                score = compute_sentence_bleu(ref_tokens, hyp_tokens)
                scores.append(score)

    return float(sum(scores) / max(1, len(scores))) * 100.0
