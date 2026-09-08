import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import os
from typing import Tuple, Optional

class AudioPreprocessor:
    """
    Audio feature extraction pipeline:
    1. Audio Loading (via soundfile or torchaudio)
    2. Mono conversion & Target Sample Rate Resampling (default: 16000 Hz)
    3. Log-Mel Spectrogram computation
    4. Optional SpecAugment (Time & Frequency masking)
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
    ):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.apply_spec_augment = apply_spec_augment

        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )

        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Loads an audio file and returns (waveform, sample_rate). Waveform shape: [1, num_samples]."""
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

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        return waveform, self.target_sample_rate

    def extract_features(self, waveform: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        Extracts Log-Mel Spectrogram features.
        Input: waveform of shape [1, T_samples] or [T_samples]
        Output: Log-Mel spectrogram of shape [n_mels, T_frames]
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        mel_spec = self.mel_transform(waveform).squeeze(0) # [n_mels, T_frames]
        log_mel = torch.log(mel_spec + 1e-6)
        
        if augment and self.apply_spec_augment:
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)
            
        return log_mel
