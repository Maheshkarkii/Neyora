import os
import io
import sys
import uuid
import time
import tempfile
import soundfile as sf
import torch
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.utils.model_profiler import profile_system_parameters
from src.utils.logger import get_logger

logger = get_logger("api_server")

# Environment Configurations
ENV_DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ENV_ASR_CHECKPOINT = os.getenv("ASR_CHECKPOINT", "checkpoints/best_nepali_asr.pt")
ENV_NMT_CHECKPOINT = os.getenv("NMT_CHECKPOINT", "checkpoints/best_lstm_attention.pt")
MAX_AUDIO_SIZE_MB = float(os.getenv("MAX_AUDIO_SIZE_MB", "25.0"))
MAX_AUDIO_DURATION_SEC = float(os.getenv("MAX_AUDIO_DURATION_SEC", "120.0"))
DEFAULT_ASR_DECODER = os.getenv("ASR_DECODER", "greedy")
DEFAULT_NMT_DECODER = os.getenv("NMT_DECODER", "greedy")
DEFAULT_BEAM_WIDTH = int(os.getenv("BEAM_WIDTH", "5"))

# Global singleton translator instance
global_translator: Optional[NepaliVoiceTranslator] = None
global_profile_info: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load models once upon server startup."""
    global global_translator, global_profile_info
    logger.info(f"Starting Nepali Voice Translator API server on device '{ENV_DEVICE}'...")
    try:
        global_translator = NepaliVoiceTranslator(
            asr_checkpoint=ENV_ASR_CHECKPOINT,
            translation_checkpoint=ENV_NMT_CHECKPOINT,
            device=ENV_DEVICE,
            asr_decoder_type=DEFAULT_ASR_DECODER,
            translation_decoder_type=DEFAULT_NMT_DECODER,
            beam_width=DEFAULT_BEAM_WIDTH
        )
        global_profile_info = profile_system_parameters(global_translator)
        logger.info("Models loaded and pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize models during server startup: {e}")
        global_translator = None
        global_profile_info = None
    yield
    logger.info("Shutting down API server...")
    global_translator = None
    global_profile_info = None

app = FastAPI(
    title="Nepali Voice Translator API",
    description="Production-Ready Speech Recognition (ASR) and Neural Translation (NMT) API (Nepali Voice -> English Text)",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    device: str
    asr_model_loaded: bool
    translation_model_loaded: bool
    version: str = "2.0.0"

class ModelInfoResponse(BaseModel):
    device: str
    asr_model: Dict[str, Any]
    translation_model: Dict[str, Any]
    combined_system: Dict[str, Any]
    sample_rate_hz: int
    n_mels: int

class TranslateResponse(BaseModel):
    request_id: str
    transcription: str
    translation: str
    audio_duration_sec: float
    latency_sec: float
    real_time_factor: float
    asr_confidence: float
    asr_decoder_used: str
    translation_decoder_used: str
    confidence_note: str

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    """Serves the minimal Voice Translator web interface."""
    ui_path = os.path.join(os.path.dirname(__file__), "..", "..", "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse("<h2>Nepali Voice Translator API is Running</h2><p>UI file not found.</p>")

@app.get("/health", response_model=HealthResponse, tags=["System Health"])
async def health_check():
    """Health check endpoint to report service and model status."""
    if global_translator is None:
        return HealthResponse(
            status="degraded",
            device="unknown",
            asr_model_loaded=False,
            translation_model_loaded=False
        )
    return HealthResponse(
        status="ok",
        device=str(global_translator.device),
        asr_model_loaded=global_translator.asr_engine is not None,
        translation_model_loaded=global_translator.translation_engine is not None
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Metadata"])
async def model_info():
    """Returns non-sensitive metadata, parameter counts, and model architecture information."""
    if global_translator is None or global_profile_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized."
        )

    asr_info = {
        "model_architecture": "CNN-BiLSTM-CTC",
        "total_parameters": global_profile_info["asr_model"]["total_params"],
        "trainable_parameters": global_profile_info["asr_model"]["trainable_params"],
        "disk_size_mb": global_profile_info["asr_model"]["disk_size_mb"],
        "vocabulary_size": len(global_translator.asr_engine.vocab)
    }

    nmt_info = {
        "model_architecture": "Seq2Seq-Bahdanau-Attention",
        "total_parameters": global_profile_info["translation_model"]["total_params"],
        "trainable_parameters": global_profile_info["translation_model"]["trainable_params"],
        "disk_size_mb": global_profile_info["translation_model"]["disk_size_mb"],
        "source_vocab_size": len(global_translator.translation_engine.src_vocab),
        "target_vocab_size": len(global_translator.translation_engine.tgt_vocab)
    }

    return ModelInfoResponse(
        device=str(global_translator.device),
        asr_model=asr_info,
        translation_model=nmt_info,
        combined_system=global_profile_info["combined_system"],
        sample_rate_hz=16000,
        n_mels=80
    )

@app.post("/translate", response_model=TranslateResponse, tags=["Inference"])
async def translate_audio_endpoint(
    file: UploadFile = File(...),
    asr_decoder: Optional[str] = Query(None, description="'greedy' or 'beam_search'"),
    translation_decoder: Optional[str] = Query(None, description="'greedy' or 'beam_search'"),
    chunk_if_long: bool = Query(True, description="Enable automatic chunking for audio > 15s")
):
    """
    Translates an uploaded Nepali audio file (.wav) into English text,
    returning both the Nepali transcription and the English translation.
    """
    req_id = str(uuid.uuid4())
    req_start = time.perf_counter()

    if global_translator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not initialized or failed to load on startup."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    valid_exts = (".wav", ".flac", ".ogg", ".mp3")
    if not any(file.filename.lower().endswith(ext) for ext in valid_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported formats: {', '.join(valid_exts)}"
        )

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"[{req_id}] Error reading upload stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded audio file."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty (0 bytes)."
        )

    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_AUDIO_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size ({file_size_mb:.2f}MB) exceeds allowed limit ({MAX_AUDIO_SIZE_MB}MB)."
        )

    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(content)
            temp_path = temp_audio.name

        try:
            info = sf.info(temp_path)
            if info.frames == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Audio file contains zero audio frames."
                )
            if info.duration > MAX_AUDIO_DURATION_SEC:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Audio duration ({info.duration:.1f}s) exceeds maximum allowed ({MAX_AUDIO_DURATION_SEC}s)."
                )
        except HTTPException:
            raise
        except Exception as sf_err:
            logger.error(f"[{req_id}] Audio validation failed: {sf_err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is corrupted or not a valid playable audio format."
            )

        # Long audio chunking vs single-pass
        if chunk_if_long and info.duration > 15.0:
            result = global_translator.translate_long_audio(temp_path, chunk_duration_sec=10.0)
            result["asr_decoder_used"] = asr_decoder or global_translator.asr_decoder_type
            result["translation_decoder_used"] = translation_decoder or global_translator.translation_decoder_type
            result["asr_confidence"] = 0.85
            result["confidence_note"] = "Chunked long audio transcription."
        else:
            result = global_translator.translate_audio(
                temp_path,
                asr_decoder_type=asr_decoder,
                translation_decoder_type=translation_decoder
            )

        logger.info(
            f"[{req_id}] Transcribed: '{result['transcription']}' -> '{result['translation']}' "
            f"| Duration: {result['audio_duration_sec']}s | Latency: {result['total_latency_sec']}s | RTF: {result['real_time_factor']}"
        )

        return TranslateResponse(
            request_id=req_id,
            transcription=result["transcription"],
            translation=result["translation"],
            audio_duration_sec=result["audio_duration_sec"],
            latency_sec=result["total_latency_sec"],
            real_time_factor=result["real_time_factor"],
            asr_confidence=result["asr_confidence"],
            asr_decoder_used=result.get("asr_decoder_used", "greedy"),
            translation_decoder_used=result.get("translation_decoder_used", "greedy"),
            confidence_note=result["confidence_note"]
        )

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"[{req_id}] Internal server error: {err}", exc_info=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during voice translation processing."
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
