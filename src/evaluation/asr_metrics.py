from typing import List, Union

def levenshtein_distance(seq1: Union[str, List[str]], seq2: Union[str, List[str]]) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)
    for i in range(1, n + 1):
        curr_row[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + cost
            )
        prev_row = list(curr_row)
    return prev_row[m]

def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist) / len(ref_chars)

def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist) / len(ref_words)

def compute_corpus_cer(references: List[str], hypotheses: List[str]) -> float:
    assert len(references) == len(hypotheses)
    total_dist = 0
    total_ref_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        total_dist += levenshtein_distance(ref_chars, hyp_chars)
        total_ref_len += len(ref_chars)
    if total_ref_len == 0:
        return 0.0
    return float(total_dist) / total_ref_len

def compute_corpus_wer(references: List[str], hypotheses: List[str]) -> float:
    assert len(references) == len(hypotheses)
    total_dist = 0
    total_ref_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_dist += levenshtein_distance(ref_words, hyp_words)
        total_ref_len += len(ref_words)
    if total_ref_len == 0:
        return 0.0
    return float(total_dist) / total_ref_len
