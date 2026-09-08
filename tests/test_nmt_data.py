import pytest
import torch
from src.nmt.text_cleaner import ParallelTextCleaner
from src.nmt.tokenizer import NMTTokenizer
from src.nmt.dataset import TranslationCollateFn

def test_parallel_text_cleaner():
    cleaner = ParallelTextCleaner()
    src = "नमस्ते ,   तपाईँलाई कस्तो छ ?"
    tgt = "Hello ,  how ARE you ?"
    
    c_src = cleaner.clean_nepali(src)
    c_tgt = cleaner.clean_english(tgt)
    
    assert "नमस्ते" in c_src
    assert c_tgt == "hello , how are you ?"
    
    is_valid, _ = cleaner.validate_pair(c_src, c_tgt)
    assert is_valid

def test_nmt_tokenizer():
    tok = NMTTokenizer()
    texts = ["hello how are you", "what is your name"]
    tok.build_vocab(texts)
    
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.sos_id == 2
    assert tok.eos_id == 3
    
    encoded = tok.encode("hello how", add_special_tokens=True)
    assert encoded[0] == tok.sos_id
    assert encoded[-1] == tok.eos_id
    
    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == "hello how"

def test_nmt_collate_fn():
    collate = TranslationCollateFn(src_pad_id=0, tgt_pad_id=0)
    sample_batch = [
        {"src_ids": torch.tensor([2, 5, 6, 3]), "tgt_ids": torch.tensor([2, 10, 11, 12, 13, 3]), "raw_src": "क", "raw_tgt": "a"},
        {"src_ids": torch.tensor([2, 7, 8, 9, 10, 3]), "tgt_ids": torch.tensor([2, 14, 3]), "raw_src": "ख", "raw_tgt": "b"}
    ]
    src_padded, src_lens, tgt_padded, tgt_lens, raw_s, raw_t = collate(sample_batch)
    
    assert src_padded.shape == (2, 6)
    assert tgt_padded.shape == (2, 6)
    assert torch.equal(src_lens, torch.tensor([4, 6]))
    assert torch.equal(tgt_lens, torch.tensor([6, 3]))
    assert src_padded[0, 4].item() == 0
    assert tgt_padded[1, 3].item() == 0
