import os
import tempfile
import soundfile as sf
import torch
from typing import Tuple, Dict, Any, Optional
from src.data.preprocessing import AudioPreprocessor

class AudioService:
    """
    Service responsible for audio upload validation, normalization,
    silence detection, and temporary file management.
    """
    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_size_mb: float = 25.0,
        max_duration_sec: float = 120.0,
        silence_threshold: float = 0.005
    ):
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=target_sample_rate,
            silence_threshold_rms=silence_threshold
        )
        self.max_size_mb = max_size_mb
        self.max_duration_sec = max_duration_sec

    def validate_and_load(self, file_bytes: bytes, filename: str) -> Tuple[torch.Tensor, float, bool]:
        """
        Validates raw audio bytes, saves temporarily, extracts normalized mono waveform,
        checks duration/silence, and cleans up temp files.
        Returns:
            (waveform, duration_sec, is_silent)
        """
        if len(file_bytes) == 0:
            raise ValueError("Audio file is empty (0 bytes).")

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > self.max_size_mb:
            raise ValueError(f"File size ({size_mb:.2f}MB) exceeds limit of {self.max_size_mb}MB.")

        suffix = os.path.splitext(filename)[1] or ".wav"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            info = sf.info(temp_path)
            if info.frames == 0:
                raise ValueError("Audio file contains zero playable frames.")
            if info.duration > self.max_duration_sec:
                raise ValueError(f"Audio duration ({info.duration:.1f}s) exceeds limit of {self.max_duration_sec}s.")

            waveform, _ = self.preprocessor.load_audio(temp_path)
            is_silent = self.preprocessor.is_silence(waveform)
            duration_sec = waveform.shape[-1] / self.preprocessor.target_sample_rate

            return waveform, duration_sec, is_silent
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
