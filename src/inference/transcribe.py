import os
import torch
from typing import Dict, Any, Optional
from src.data.preprocessing import AudioPreprocessor
from src.data.vocabulary import ASRVocabulary
from src.data.tokenizer import ASRTokenizer
from src.models.asr_model import NepaliASR
from src.decoding.ctc_decoder import CTCDecoder

class NepaliTranscriber:
    """
    Inference Engine for Nepali ACR (Automatic Speech Recognition).
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_nepali_asr.pt",
        vocab_path: str = "data/processed/asr/vocab.json",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)
        self.vocab = ASRVocabulary.load(vocab_path)
        self.tokenizer = ASRTokenizer(self.vocab)
        self.decoder = CTCDecoder(self.tokenizer, blank_idx=self.vocab.blank_idx, pad_idx=self.vocab.pad_idx)

        self.model = NepaliASR(
            vocab_size=len(self.vocab),
            n_mels=80,
            cnn_out_channels=64,
            rnn_hidden_size=256,
            rnn_num_layers=2,
            rnn_dropout=0.0,
            bidirectional=True
        ).to(self.device)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        waveform, _ = self.preprocessor.load_audio(audio_path)
        mel_spec = self.preprocessor.extract_mel_spectrogram(waveform, augment=False) # [80, Time]
        mel_spec_batch = mel_spec.unsqueeze(0).to(self.device)
        input_length = torch.tensor([mel_spec.shape[1]], dtype=torch.long, device=self.device)

        log_probs, sub_lens = self.model(mel_spec_batch, input_length)
        transcriptions = self.decoder.decode_greedy(log_probs, sub_lens)
        return transcriptions[0] if transcriptions else ""
