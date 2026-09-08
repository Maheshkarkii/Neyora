import torch
import torch.nn as nn
from src.models.asr_model import NepaliASR
from src.models.cnn_extractor import CNNExtractor
from src.data.vocabulary import ASRVocabulary
from src.data.tokenizer import ASRTokenizer
from src.decoding.ctc_decoder import CTCDecoder
from src.evaluation.asr_metrics import compute_cer, compute_wer, compute_corpus_cer, compute_corpus_wer

def test_cnn_extractor_shapes():
    cnn = CNNExtractor(in_channels=1, n_mels=80, cnn_out_channels=64)
    x = torch.randn(4, 80, 200)
    in_lens = torch.tensor([200, 180, 100, 50], dtype=torch.long)
    feats, out_lens = cnn(x, in_lens)
    assert feats.shape[0] == 4
    assert feats.shape[1] == out_lens[0].item()
    assert feats.shape[2] == cnn.out_dim

def test_asr_model_forward_and_ctc():
    model = NepaliASR(vocab_size=50, n_mels=80)
    specs = torch.randn(2, 80, 150)
    in_lens = torch.tensor([150, 120], dtype=torch.long)
    log_probs, sub_lens = model(specs, in_lens)

    assert log_probs.shape[0] == 2
    assert log_probs.shape[2] == 50

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    targets = torch.tensor([[5, 10, 15], [7, 8, 0]], dtype=torch.long)
    tgt_lens = torch.tensor([3, 2], dtype=torch.long)

    log_probs_t = log_probs.transpose(0, 1)
    loss = criterion(log_probs_t, targets, sub_lens, tgt_lens)
    assert not torch.isnan(loss)
    loss.backward()

def test_ctc_decoder_deduplication():
    vocab = ASRVocabulary()
    tokenizer = ASRTokenizer(vocab)
    decoder = CTCDecoder(tokenizer, blank_idx=0, pad_idx=1)

    # Rawlogits indices with duplicates and blanks
    raw_indices = [0, 0, 3, 3, 0, 24, 24, 29, 0]
    filtered = decoder.decode_indices(raw_indices)
    assert filtered == [3, 24, 29]

def test_asr_metrics_cer_wer():
    ref = "data"
    hyp = "date"
    # Cer: 1 sub / 4 chars = 0.25
    assert compute_cer(ref, hyp) == 0.25
    assert compute_wer("all day", "all night") == 0.5

    # Corpus metrics
    cer = compute_corpus_cer(["abc", "def"], ["abc", "deg"])
    assert round(cer, 4) == round(1 / 6, 4)
