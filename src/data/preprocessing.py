import os
import re
import unicodedata
from typing import Tuple, Optional, Dict, Any, List
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np

class AudioPreprocessor:
    """
    Audio feature extraction and augmentation pipeline:
    Audio -> Load waveform -> Resample -> Normalize -> Mel-Spectrogram -> Model-ready tensor.
    
    Tensor shape convention:
    - Mel-spectrogram output: [n_mels, time] for single item, [B, n_mels, time] in batch collate.
    """
    def __init__(
        self,
        target_sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        apply_spec_augment: bool = False,
        silence_threshold_rms: float = 0.005
    ):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.apply_spec_augment = apply_spec_augment
        self.silence_threshold_rms = silence_threshold_rms

        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0
        )

        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Loads an audio file dynamically from disk.
        Returns:
            waveform: FloatTensor [1, num_samples], normalized in [-1.0, 1.0]
            sample_rate: int (resampled to target_sample_rate)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            data, sr = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)

        # Convert multi-channel to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Normalize amplitude to [-1.0, 1.0]
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        return waveform, self.target_sample_rate

    def is_silence(self, waveform: torch.Tensor, threshold: Optional[float] = None) -> bool:
        """
        Checks whether the given waveform is silence using Root-Mean-Square (RMS) energy.
        """
        thresh = threshold or self.silence_threshold_rms
        if waveform.numel() == 0:
            return True
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()
        return bool(rms < thresh)

    def extract_mel_spectrogram(self, waveform: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        Extracts Log-Mel Spectrogram features.
        Input: waveform of shape [1, T_samples] or [T_samples]
        Output: Log-Mel spectrogram of shape [n_mels, time_frames]
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_transform(waveform).squeeze(0) # [n_mels, time]
        log_mel = torch.log(mel_spec + 1e-6)

        if augment and self.apply_spec_augment:
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)

        return log_mel

    def augment_waveform(
        self,
        waveform: torch.Tensor,
        noise_level: float = 0.005,
        gain_db: float = 0.0
    ) -> torch.Tensor:
        """Applies waveform-level augmentations: additive Gaussian noise and gain scaling."""
        aug_wave = waveform.clone()
        if gain_db != 0.0:
            scale = 10.0 ** (gain_db / 20.0)
            aug_wave = aug_wave * scale
        if noise_level > 0:
            noise = torch.randn_like(aug_wave) * noise_level
            aug_wave = aug_wave + noise
        max_val = torch.max(torch.abs(aug_wave))
        if max_val > 1.0:
            aug_wave = aug_wave / max_val
        return aug_wave

    def chunk_audio(
        self,
        waveform: torch.Tensor,
        chunk_duration_sec: float = 10.0,
        overlap_sec: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Splits long waveform into overlapping segments for robust long-audio processing.
        """
        chunk_samples = int(chunk_duration_sec * self.target_sample_rate)
        step_samples = int((chunk_duration_sec - overlap_sec) * self.target_sample_rate)
        total_samples = waveform.shape[-1]

        if total_samples <= chunk_samples:
            return [waveform]

        chunks = []
        for start in range(0, total_samples, step_samples):
            end = min(start + chunk_samples, total_samples)
            chunks.append(waveform[:, start:end])
            if end == total_samples:
                break
        return chunks


class NepaliTextCleaner:
    """
    Cleans and normalizes Nepali Devanagari text.
    Handles Unicode NFC normalization, Danda preservation, and non-Devanagari removal.
    """
    def __init__(self):
        self.devanagari_regex = re.compile(r"[\u0900-\u097F]")

    def normalize(self, text: Optional[str]) -> str:
        if not text:
            return ""
        # 1. Unicode NFC Normalization (canonical decomposition & composition)
        text = unicodedata.normalize("NFC", text.strip())
        # 2. Standardize Danda (।) and punctuation
        text = text.replace("||", "।").replace("!", "।").replace("?", "।").replace(".", "।")
        # 3. Whitelist Devanagari script, danda, and spaces
        text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
        # 4. Collapse extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_valid_nepali(self, text: str) -> bool:
        if not text or len(text.strip()) == 0:
            return False
        dev_chars = len(self.devanagari_regex.findall(text))
        return dev_chars > 0 and (dev_chars / max(1, len(text.replace(" ", "")))) > 0.75


class EnglishTextCleaner:
    """
    Cleans and normalizes English translation target text.
    """
    def normalize(self, text: Optional[str]) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text.strip().lower())
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        # Space-separate standard punctuation
        text = re.sub(r"([.,!?;:\"()])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def validate_audio_file(audio_path: str, min_duration: float = 0.5, max_duration: float = 20.0) -> Tuple[bool, str]:
    """Validates existence, format, readable frames, and duration of an audio file."""
    if not os.path.exists(audio_path):
        return False, "File does not exist"
    try:
        info = sf.info(audio_path)
        if info.duration < min_duration:
            return False, f"Duration too short ({info.duration:.2f}s < {min_duration}s)"
        if info.duration > max_duration:
            return False, f"Duration too long ({info.duration:.2f}s > {max_duration}s)"
        if info.frames == 0:
            return False, "Zero audio frames"
        return True, "Valid"
    except Exception as e:
        return False, f"Corrupted audio: {str(e)}"


def validate_parallel_pair(
    src: str,
    tgt: str,
    min_src: int = 1,
    max_src: int = 50,
    min_tgt: int = 1,
    max_tgt: int = 50,
    max_ratio: float = 2.5
) -> Tuple[bool, str]:
    """Validates length bounds and token ratio of a (Nepali, English) pair."""
    if not src or not tgt:
        return False, "Empty sentence"
    s_tokens = src.split()
    t_tokens = tgt.split()
    if len(s_tokens) < min_src or len(s_tokens) > max_src:
        return False, f"Source token length ({len(s_tokens)}) out of bounds [{min_src}, {max_src}]"
    if len(t_tokens) < min_tgt or len(t_tokens) > max_tgt:
        return False, f"Target token length ({len(t_tokens)}) out of bounds [{min_tgt}, {max_tgt}]"
    ratio = max(len(s_tokens), len(t_tokens)) / max(1, min(len(s_tokens), len(t_tokens)))
    if ratio > max_ratio:
        return False, f"Length ratio ({ratio:.2f}) exceeds {max_ratio}"
    return True, "Valid"
