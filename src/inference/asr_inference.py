import os
import time
import torch
from typing import Dict, Any, Optional, Tuple, List, Union
from src.data.preprocessing import AudioPreprocessor
from src.data.vocabulary import ASRVocabulary
from src.data.tokenizer import ASRTokenizer
from src.models.asr_model import NepaliASR
from src.decoding.ctc_decoder import CTCDecoder
from src.utils.logger import get_logger

logger = get_logger("asr_inference")

class ASRInferenceEngine:
    """
    Inference engine for Nepali Automatic Speech Recognition (ASR).
    Encapsulates preprocessing, neural model forward pass, and CTC decoding.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_nepali_asr.pt",
        vocab_path: str = "data/processed/asr/vocab.json",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)
        
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"ASR vocabulary not found at {vocab_path}")
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

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ASR checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"ASR model successfully loaded from {checkpoint_path} on device: {self.device}")

    @torch.no_grad()
    def transcribe(
        self,
        audio_input: Union[str, torch.Tensor],
        sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Transcribes audio from file path or waveform tensor.
        Returns:
            Dict containing transcription, latency, audio duration, and diagnostic stats.
        """
        start_time = time.perf_counter()
        
        if isinstance(audio_input, str):
            waveform, sr = self.preprocessor.load_audio(audio_input)
        else:
            waveform = audio_input
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if sample_rate and sample_rate != self.preprocessor.target_sample_rate:
                import torchaudio.transforms as T
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.preprocessor.target_sample_rate)
                waveform = resampler(waveform)

        duration_sec = waveform.shape[-1] / self.preprocessor.target_sample_rate
        mel_spec = self.preprocessor.extract_mel_spectrogram(waveform, augment=False) # [n_mels, Time]
        mel_spec_batch = mel_spec.unsqueeze(0).to(self.device)
        input_length = torch.tensor([mel_spec.shape[1]], dtype=torch.long, device=self.device)

        log_probs, sub_lens = self.model(mel_spec_batch, input_length) # log_probs: [B, T_sub, V]
        transcriptions = self.decoder.decode_greedy(log_probs, sub_lens)
        transcription = transcriptions[0] if transcriptions else ""

        # Raw heuristic diagnostic confidence estimation (uncalibrated softmax)
        # Note: Greedy CTC raw max softmax probabilities are uncalibrated and intended for diagnostics only.
        probs = torch.exp(log_probs[0, :sub_lens[0]]) # [T_sub, V]
        max_probs, argmax_idx = torch.max(probs, dim=-1)
        non_blank_mask = (argmax_idx != self.vocab.blank_idx) & (argmax_idx != self.vocab.pad_idx)
        
        if non_blank_mask.any():
            diagnostic_conf = float(max_probs[non_blank_mask].mean().item())
        else:
            diagnostic_conf = float(max_probs.mean().item()) if len(max_probs) > 0 else 0.0

        latency_sec = time.perf_counter() - start_time

        return {
            "transcription": transcription,
            "duration_sec": duration_sec,
            "latency_sec": latency_sec,
            "diagnostic_confidence": diagnostic_conf,
            "confidence_note": "Uncalibrated diagnostic score based on greedy CTC frame softmax mean (not true posterior probability)."
        }
