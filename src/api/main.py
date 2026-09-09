import os
import io
import sys
import tempfile
import soundfile as sf
import torch
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.utils.logger import get_logger

logger = get_logger("api_server")

# Global singleton translator instance
global_translator: Optional[NepaliVoiceTranslator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load models once upon server startup."""
    global global_translator
    logger.info("Starting Nepali Voice Translator API server...")
    try:
        global_translator = NepaliVoiceTranslator()
        logger.info("Models loaded and pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize models during server startup: {e}")
        global_translator = None
    yield
    logger.info("Shutting down API server...")
    global_translator = None

app = FastAPI(
    title="Nepali Voice Translator API",
    description="End-to-End ASR and NMT Speech Translation API (Nepali Audio -> English Text)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for UI / external clients
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

class TranslateResponse(BaseModel):
    transcription: str
    translation: str
    audio_duration_sec: float
    latency_sec: float
    real_time_factor: float
    asr_confidence: float
    confidence_note: str

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    """Serves the minimal Voice Translator web interface."""
    ui_path = os.path.join(os.path.dirname(__file__), "..", "..", "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse("<h2>Nepali Voice Translator API is Running</h2><p>UI file not found.</p>")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
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

@app.post("/translate", response_model=TranslateResponse, tags=["Inference"])
async def translate_audio_endpoint(
    file: UploadFile = File(...)
):
    """
    Translates an uploaded Nepali audio file (.wav) into English text,
    returning both the Nepali transcription and the English translation.
    """
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

    # Validate file extension
    valid_exts = (".wav", ".flac", ".ogg", ".mp3")
    if not any(file.filename.lower().endswith(ext) for ext in valid_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Supported formats: {', '.join(valid_exts)}"
        )

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Error reading upload file stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded audio file."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio file is empty (0 bytes)."
        )

    # Write temporarily to disk for robust audio loading
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(content)
            temp_path = temp_audio.name

        # Validate audio content with soundfile
        try:
            info = sf.info(temp_path)
            if info.frames == 0 or info.duration < 0.1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Audio file contains no audible frames or is too short."
                )
        except Exception as sf_err:
            logger.error(f"Soundfile validation error: {sf_err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is corrupted or not a valid playable audio format."
            )

        # Execute translation pipeline
        result = global_translator.translate_audio(temp_path)

        return TranslateResponse(
            transcription=result["transcription"],
            translation=result["translation"],
            audio_duration_sec=result["audio_duration_sec"],
            latency_sec=result["total_latency_sec"],
            real_time_factor=result["real_time_factor"],
            asr_confidence=result["asr_confidence"],
            confidence_note=result["confidence_note"]
        )

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Inference execution error: {err}", exc_info=False)
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
