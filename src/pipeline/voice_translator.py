import os
import time
import torch
from typing import Dict, Any, Optional, Tuple, List, Union
from src.inference.asr_inference import ASRInferenceEngine
from src.inference.translation_inference import TranslationInferenceEngine
from src.utils.logger import get_logger

logger = get_logger("voice_translator")

class NepaliVoiceTranslator:
    """
    Production-Ready End-to-End Pipeline for Nepali Voice Translation:
    Input Audio (.wav) -> Audio Preprocessing -> Mel-Spectrogram -> CNN-BiLSTM-CTC ASR
    -> Nepali Transcription -> Devanagari Tokenizer -> Seq2Seq Attention NMT -> English Translation.
    
    Supports:
    - Greedy & Beam Search decoding for both ASR and NMT
    - Audio chunking for long recording processing
    - Root-Mean-Square Energy Silence Detection
    - torch.inference_mode() execution
    """
    def __init__(
        self,
        asr_checkpoint: str = "checkpoints/best_nepali_asr.pt",
        translation_checkpoint: str = "checkpoints/best_lstm_attention.pt",
        asr_vocab_path: str = "data/processed/asr/vocab.json",
        nmt_src_vocab_path: str = "data/processed/nmt/src_vocab.json",
        nmt_tgt_vocab_path: str = "data/processed/nmt/tgt_vocab.json",
        device: Optional[Union[str, torch.device]] = None,
        asr_decoder_type: str = "greedy",
        translation_decoder_type: str = "greedy",
        beam_width: int = 5
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.asr_checkpoint = asr_checkpoint
        self.translation_checkpoint = translation_checkpoint
        self.asr_vocab_path = asr_vocab_path
        self.nmt_src_vocab_path = nmt_src_vocab_path
        self.nmt_tgt_vocab_path = nmt_tgt_vocab_path
        self.asr_decoder_type = asr_decoder_type
        self.translation_decoder_type = translation_decoder_type
        self.beam_width = beam_width

        logger.info(f"Initializing NepaliVoiceTranslator on device: {self.device}")
        self.asr_engine = self.load_asr_model()
        self.translation_engine = self.load_translation_model()
        
        print(f"ASR model loaded")
        print(f"Translation model loaded")
        print(f"Device: {self.device}")

    def load_asr_model(self) -> ASRInferenceEngine:
        """Loads and prepares the Nepali ASR model."""
        return ASRInferenceEngine(
            checkpoint_path=self.asr_checkpoint,
            vocab_path=self.asr_vocab_path,
            device=self.device,
            decoder_type=self.asr_decoder_type,
            beam_width=self.beam_width
        )

    def load_translation_model(self) -> TranslationInferenceEngine:
        """Loads and prepares the Nepali-English Seq2Seq Attention model."""
        return TranslationInferenceEngine(
            checkpoint_path=self.translation_checkpoint,
            src_vocab_path=self.nmt_src_vocab_path,
            tgt_vocab_path=self.nmt_tgt_vocab_path,
            device=self.device,
            decoder_type=self.translation_decoder_type,
            beam_width=self.beam_width
        )

    def translate_text(
        self,
        nepali_text: str,
        max_len: int = 30,
        decoder_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Direct text-to-text translation (for Mode 1 ground-truth evaluation)."""
        return self.translation_engine.translate(
            nepali_text=nepali_text,
            max_len=max_len,
            decoder_type=decoder_type
        )

    @torch.inference_mode()
    def translate_audio(
        self,
        audio_input: Union[str, torch.Tensor],
        sample_rate: Optional[int] = None,
        max_len: int = 30,
        diagnostic: bool = False,
        asr_decoder_type: Optional[str] = None,
        translation_decoder_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end pipeline: Audio -> Nepali Transcription -> English Translation.
        Preserves intermediate stages, handles silence, and computes latency metrics.
        """
        total_start = time.perf_counter()
        logger.info(f"Request started for audio input: {audio_input if isinstance(audio_input, str) else 'Tensor'}")

        # 1. ASR Stage
        asr_res = self.asr_engine.transcribe(
            audio_input,
            sample_rate=sample_rate,
            decoder_type=asr_decoder_type
        )
        nepali_text = asr_res["transcription"]
        audio_duration = asr_res["duration_sec"]
        asr_latency = asr_res["latency_sec"]
        logger.info(f"ASR completed. Transcription: '{nepali_text}' (Latency: {asr_latency:.4f}s)")

        # Silence / Empty Audio Handler
        if not nepali_text or not nepali_text.strip():
            total_latency = time.perf_counter() - total_start
            rtf = (total_latency / audio_duration) if audio_duration > 0 else 0.0
            return {
                "transcription": "",
                "translation": "",
                "audio_duration_sec": round(audio_duration, 4),
                "asr_latency_sec": round(asr_latency, 4),
                "translation_latency_sec": 0.0,
                "total_latency_sec": round(total_latency, 4),
                "real_time_factor": round(rtf, 4),
                "asr_confidence": round(asr_res["diagnostic_confidence"], 4),
                "confidence_note": asr_res["confidence_note"],
                "status": "silence_or_empty_audio"
            }

        # 2. Translation Stage
        nmt_res = self.translation_engine.translate(
            nepali_text,
            max_len=max_len,
            decoder_type=translation_decoder_type
        )
        english_text = nmt_res["translation"]
        nmt_latency = nmt_res["latency_sec"]
        logger.info(f"Translation completed. Translation: '{english_text}' (Latency: {nmt_latency:.4f}s)")

        total_latency = time.perf_counter() - total_start
        rtf = (total_latency / audio_duration) if audio_duration > 0 else 0.0
        logger.info(f"Request completed. Total latency: {total_latency:.4f}s | RTF: {rtf:.4f}")

        result: Dict[str, Any] = {
            "transcription": nepali_text,
            "translation": english_text,
            "audio_duration_sec": round(audio_duration, 4),
            "asr_latency_sec": round(asr_latency, 4),
            "translation_latency_sec": round(nmt_latency, 4),
            "total_latency_sec": round(total_latency, 4),
            "real_time_factor": round(rtf, 4),
            "asr_confidence": round(asr_res["diagnostic_confidence"], 4),
            "confidence_note": asr_res["confidence_note"],
            "asr_decoder_used": asr_res.get("decoder_type", self.asr_decoder_type),
            "translation_decoder_used": nmt_res.get("decoder_type", self.translation_decoder_type),
            "status": "success"
        }

        if diagnostic:
            result["diagnostic"] = {
                "src_tokens": nmt_res["src_tokens"],
                "tgt_tokens": nmt_res["tgt_tokens"],
                "attention_matrix": nmt_res["attention_matrix"],
                "raw_audio_input": audio_input if isinstance(audio_input, str) else "Waveform Tensor"
            }

        return result

    @torch.inference_mode()
    def translate_long_audio(
        self,
        audio_path: str,
        chunk_duration_sec: float = 10.0,
        overlap_sec: float = 1.0,
        max_len: int = 40
    ) -> Dict[str, Any]:
        """
        Processes long audio files safely by chunking into segments, transcribing each,
        concatenating intermediate transcripts, and generating the full translation.
        """
        start_time = time.perf_counter()
        waveform, sr = self.asr_engine.preprocessor.load_audio(audio_path)
        chunks = self.asr_engine.preprocessor.chunk_audio(
            waveform,
            chunk_duration_sec=chunk_duration_sec,
            overlap_sec=overlap_sec
        )

        segment_transcripts: List[str] = []
        asr_total_time = 0.0

        for idx, chunk in enumerate(chunks):
            asr_res = self.asr_engine.transcribe(chunk)
            asr_total_time += asr_res["latency_sec"]
            text = asr_res["transcription"].strip()
            if text:
                segment_transcripts.append(text)

        combined_nepali = " ".join(segment_transcripts)
        
        if combined_nepali:
            nmt_res = self.translation_engine.translate(combined_nepali, max_len=max_len)
            english_text = nmt_res["translation"]
            nmt_time = nmt_res["latency_sec"]
        else:
            english_text = ""
            nmt_time = 0.0

        total_latency = time.perf_counter() - start_time
        audio_dur = waveform.shape[-1] / sr

        return {
            "transcription": combined_nepali,
            "translation": english_text,
            "num_chunks": len(chunks),
            "chunk_transcriptions": segment_transcripts,
            "audio_duration_sec": round(audio_dur, 4),
            "asr_latency_sec": round(asr_total_time, 4),
            "translation_latency_sec": round(nmt_time, 4),
            "total_latency_sec": round(total_latency, 4),
            "real_time_factor": round(total_latency / max(1e-6, audio_dur), 4)
        }

    @staticmethod
    def analyze_error_propagation(
        ground_truth_nepali: str,
        predicted_nepali: str,
        ground_truth_english: str,
        predicted_english: str,
        standalone_english_from_gt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Diagnoses error sources in the pipeline.
        """
        from src.evaluation.asr_metrics import compute_cer, compute_wer
        from src.evaluation.metrics import calculate_sentence_bleu

        cer = compute_cer(ground_truth_nepali, predicted_nepali)
        wer = compute_wer(ground_truth_nepali, predicted_nepali)
        
        gt_eng_tokens = ground_truth_english.strip().lower().split()
        pred_eng_tokens = predicted_english.strip().lower().split()
        e2e_bleu = calculate_sentence_bleu(gt_eng_tokens, pred_eng_tokens)

        asr_passed = (wer <= 0.20 or cer <= 0.15 or predicted_nepali.strip() == ground_truth_nepali.strip())
        trans_passed = (e2e_bleu >= 0.40 or predicted_english.strip() == ground_truth_english.strip())

        if standalone_english_from_gt:
            gt_nmt_tokens = standalone_english_from_gt.strip().lower().split()
            mode1_bleu = calculate_sentence_bleu(gt_eng_tokens, gt_nmt_tokens)
            mode1_passed = (mode1_bleu >= 0.40 or standalone_english_from_gt.strip() == ground_truth_english.strip())
        else:
            mode1_passed = trans_passed

        if asr_passed and trans_passed:
            classification = "Correct"
            reason = "Both ASR transcription and English translation succeeded accurately."
        elif not asr_passed and mode1_passed:
            classification = "ASR error"
            reason = "ASR transcribed text incorrectly, propagating error to translation; translation from GT text was accurate."
        elif asr_passed and not trans_passed:
            classification = "Translation error"
            reason = "ASR transcribed text correctly, but translation model failed to produce accurate English."
        else:
            classification = "Both"
            reason = "Both ASR transcription failed and translation model struggled on this sample."

        return {
            "classification": classification,
            "reason": reason,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "bleu": round(e2e_bleu * 100.0, 2)
        }
