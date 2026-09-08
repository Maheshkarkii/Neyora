import math
from collections import Counter
from typing import List

def calculate_sentence_bleu(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """Calculates smoothed sentence-level BLEU score."""
    if not hypothesis or not reference:
        return 0.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / max(1, hyp_len)))

    precisions = []
    for n in range(1, max_n + 1):
        if hyp_len < n:
            continue
        hyp_ngrams = [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)
        clipped_count = sum(min(count, ref_counts.get(ng, 0)) for ng, count in hyp_counts.items())
        precisions.append(clipped_count / max(1, len(hyp_ngrams)))

    if not precisions or min(precisions) == 0:
        return 0.0

    log_sum = sum(math.log(p) for p in precisions) / len(precisions)
    return float(bp * math.exp(log_sum))
