import pytest
import torch
from src.data.collate import ASRCollateFn, TranslationCollateFn

def test_asr_collate_shapes():
    collate = ASRCollateFn(pad_id=1)
    batch = [
        {"mel_spec": torch.randn(80, 50), "tokens": torch.tensor([4, 5, 6]), "text": "म", "duration": 1.0, "speaker_id": "s1", "audio_path": "a.wav"},
        {"mel_spec": torch.randn(80, 80), "tokens": torch.tensor([7, 8, 9, 10]), "text": "कलेज", "duration": 1.6, "speaker_id": "s2", "audio_path": "b.wav"}
    ]
    specs, targets, in_lens, tgt_lens, texts = collate(batch)

    assert specs.shape == (2, 80, 80) # [B, n_mels, max_time]
    assert targets.shape == (2, 4)     # [B, max_target_tokens]
    assert torch.equal(in_lens, torch.tensor([50, 80]))
    assert torch.equal(tgt_lens, torch.tensor([3, 4]))
    assert targets[0, 3].item() == 1 # Pad token

def test_translation_collate_shapes():
    collate = TranslationCollateFn(src_pad_id=0, tgt_pad_id=0)
    batch = [
        {"src_ids": torch.tensor([2, 5, 6, 3]), "tgt_ids": torch.tensor([2, 10, 11, 12, 13, 3]), "raw_src": "a", "raw_tgt": "b"},
        {"src_ids": torch.tensor([2, 7, 8, 9, 10, 3]), "tgt_ids": torch.tensor([2, 14, 3]), "raw_src": "c", "raw_tgt": "d"}
    ]
    src_batch, src_lens, tgt_batch, tgt_lens, raw_s, raw_t = collate(batch)

    assert src_batch.shape == (2, 6) # [B, max_src_len]
    assert tgt_batch.shape == (2, 6) # [B, max_tgt_len]
    assert torch.equal(src_lens, torch.tensor([4, 6]))
    assert torch.equal(tgt_lens, torch.tensor([6, 3]))
    assert src_batch[0, 4].item() == 0 # Source Pad
    assert tgt_batch[1, 3].item() == 0 # Target Pad
