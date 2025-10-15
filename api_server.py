#!/usr/bin/env python3
"""
FastAPI Server for Insanely Fast Whisper on GPU 2
Provides REST API endpoints for audio transcription
"""

# IMPORTANT: Disable torchcodec to avoid FFmpeg dependency issues
# This must be set before importing transformers
import os
os.environ["TRANSFORMERS_NO_TORCHCODEC"] = "1"

import json
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import aiofiles

# Configuration
GPU_DEVICE = 2  # GPU 2
DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_BATCH_SIZE = 24
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Insanely Fast Whisper API",
    description="High-performance audio transcription API running on GPU 2",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline cache
_pipeline_cache = {}


# Response Models
class TranscriptionResponse(BaseModel):
    text: str
    chunks: Optional[List[dict]] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    model: str
    gpu: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str
    gpu_id: int
    flash_attention: bool
    model_loaded: bool


# Helper Functions
def get_pipeline(model_name: str = DEFAULT_MODEL, use_flash: bool = True):
    """Get or create a cached pipeline"""
    cache_key = f"{model_name}_{use_flash}"
    
    if cache_key not in _pipeline_cache:
        print(f"Loading model: {model_name} on GPU {GPU_DEVICE}")
        
        model_kwargs = {}
        if use_flash and is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        else:
            model_kwargs["attn_implementation"] = "sdpa"
            print("Using SDPA attention")
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device=f"cuda:{GPU_DEVICE}",
            model_kwargs=model_kwargs,
        )
        
        _pipeline_cache[cache_key] = pipe
        print(f"Model loaded successfully on GPU {GPU_DEVICE}")
    
    return _pipeline_cache[cache_key]


async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to disk"""
    file_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{upload_file.filename}"
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path


def cleanup_file(file_path: Path):
    """Delete file after processing"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insanely Fast Whisper API",
        "version": "1.0.0",
        "gpu": f"GPU {GPU_DEVICE}",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "transcribe_url": "/transcribe/url (POST)",
            "models": "/models"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(GPU_DEVICE) if gpu_available else "N/A"
    flash_available = is_flash_attn_2_available()
    model_loaded = len(_pipeline_cache) > 0
    
    return HealthResponse(
        status="healthy" if gpu_available else "degraded",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_id=GPU_DEVICE,
        flash_attention=flash_available,
        model_loaded=model_loaded
    )


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            "openai/whisper-large-v3",
            "openai/whisper-large-v2",
            "openai/whisper-medium",
            "openai/whisper-small",
            "openai/whisper-base",
            "distil-whisper/distil-large-v2",
            "distil-whisper/distil-medium.en",
            "distil-whisper/distil-small.en"
        ],
        "default": DEFAULT_MODEL,
        "loaded_models": list(_pipeline_cache.keys())
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(DEFAULT_MODEL, description="Model name"),
    batch_size: int = Form(DEFAULT_BATCH_SIZE, description="Batch size"),
    use_flash: bool = Form(True, description="Use Flash Attention 2"),
    return_timestamps: bool = Form(True, description="Return timestamps"),
    language: Optional[str] = Form(None, description="Language code (auto-detect if None)"),
):
    """
    Transcribe an uploaded audio file
    
    - **file**: Audio file (mp3, wav, m4a, etc.)
    - **model**: Whisper model to use
    - **batch_size**: Number of parallel batches (reduce if OOM)
    - **use_flash**: Enable Flash Attention 2 for speed
    - **return_timestamps**: Include timestamps in response
    - **language**: Force language (e.g., 'en', 'es', 'fr')
    """
    start_time = datetime.now()
    file_path = None
    
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        print(f"Processing file: {file_path}")
        
        # Get pipeline
        pipe = get_pipeline(model_name=model, use_flash=use_flash)
        
        # Prepare generation kwargs
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        
        # Run transcription
        outputs = pipe(
            str(file_path),
            chunk_length_s=30,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs if generate_kwargs else None,
        )
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, file_path)
        
        # Prepare response
        response = TranscriptionResponse(
            text=outputs["text"],
            chunks=outputs.get("chunks"),
            language=language,
            duration=duration,
            model=model,
            gpu=f"GPU {GPU_DEVICE} ({torch.cuda.get_device_name(GPU_DEVICE)})",
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        if file_path:
            cleanup_file(file_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe/url")
async def transcribe_url(
    url: str = Form(..., description="URL to audio file"),
    model: str = Form(DEFAULT_MODEL, description="Model name"),
    batch_size: int = Form(DEFAULT_BATCH_SIZE, description="Batch size"),
    use_flash: bool = Form(True, description="Use Flash Attention 2"),
    return_timestamps: bool = Form(True, description="Return timestamps"),
    language: Optional[str] = Form(None, description="Language code"),
):
    """
    Transcribe audio from a URL
    
    - **url**: Direct URL to audio file
    - Other parameters same as /transcribe endpoint
    """
    start_time = datetime.now()
    
    try:
        # Get pipeline
        pipe = get_pipeline(model_name=model, use_flash=use_flash)
        
        # Prepare generation kwargs
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        
        # Run transcription
        outputs = pipe(
            url,
            chunk_length_s=30,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            generate_kwargs=generate_kwargs if generate_kwargs else None,
        )
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = TranscriptionResponse(
            text=outputs["text"],
            chunks=outputs.get("chunks"),
            language=language,
            duration=duration,
            model=model,
            gpu=f"GPU {GPU_DEVICE} ({torch.cuda.get_device_name(GPU_DEVICE)})",
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/preload")
async def preload_model(
    model: str = Form(DEFAULT_MODEL),
    use_flash: bool = Form(True)
):
    """Preload a model into memory"""
    try:
        pipe = get_pipeline(model_name=model, use_flash=use_flash)
        return {
            "status": "success",
            "message": f"Model {model} loaded on GPU {GPU_DEVICE}",
            "flash_attention": use_flash and is_flash_attn_2_available()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/cache")
async def clear_cache():
    """Clear model cache"""
    global _pipeline_cache
    count = len(_pipeline_cache)
    _pipeline_cache.clear()
    torch.cuda.empty_cache()
    return {
        "status": "success",
        "message": f"Cleared {count} cached models",
        "gpu_memory_freed": True
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting Insanely Fast Whisper API Server")
    print("=" * 60)
    print(f"GPU: {GPU_DEVICE} - {torch.cuda.get_device_name(GPU_DEVICE)}")
    print(f"Flash Attention 2: {is_flash_attn_2_available()}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
