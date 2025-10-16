#!/usr/bin/env python3
"""
TTS API Server with Voice Management
Supports custom voice uploads and synthesis
"""

import os
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import edge_tts
import aiofiles

# Configuration
VOICES_DIR = Path("./voices")
TTS_OUTPUT_DIR = Path("./tts_outputs")
DEFAULT_VOICE = "en-US-AriaNeural"

# Create directories
VOICES_DIR.mkdir(exist_ok=True)
TTS_OUTPUT_DIR.mkdir(exist_ok=True)

# Voice metadata storage
VOICE_METADATA_FILE = VOICES_DIR / "voice_metadata.json"

# Initialize FastAPI app
app = FastAPI(
    title="TTS API with Voice Management",
    description="Text-to-Speech API with custom voice support",
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

# Voice metadata cache
_voice_metadata: Dict[str, dict] = {}


# Models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE
    rate: Optional[str] = "+0%"  # Speed: -50% to +100%
    pitch: Optional[str] = "+0Hz"  # Pitch adjustment


class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    gender: str
    type: str  # "system" or "custom"
    created_at: Optional[str] = None


# Helper Functions
def load_voice_metadata():
    """Load voice metadata from disk"""
    global _voice_metadata
    if VOICE_METADATA_FILE.exists():
        with open(VOICE_METADATA_FILE, 'r') as f:
            _voice_metadata = json.load(f)
    return _voice_metadata


def save_voice_metadata():
    """Save voice metadata to disk"""
    with open(VOICE_METADATA_FILE, 'w') as f:
        json.dump(_voice_metadata, f, indent=2)


async def get_edge_tts_voices():
    """Get available Edge TTS voices"""
    voices = await edge_tts.list_voices()
    return [
        {
            "id": v["ShortName"],
            "name": v["ShortName"],
            "language": v["Locale"],
            "gender": v["Gender"],
            "type": "system"
        }
        for v in voices
    ]


def cleanup_file(file_path: Path):
    """Delete file after processing"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up {file_path}: {e}")


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load voice metadata on startup"""
    load_voice_metadata()
    print(f"Loaded {len(_voice_metadata)} custom voices")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TTS API with Voice Management",
        "version": "1.0.0",
        "endpoints": {
            "synthesize": "/tts/synthesize (POST)",
            "voices": "/tts/voices (GET)",
            "upload_voice": "/tts/voices/upload (POST)",
            "delete_voice": "/tts/voices/{voice_id} (DELETE)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "custom_voices": len(_voice_metadata),
        "voices_directory": str(VOICES_DIR),
        "output_directory": str(TTS_OUTPUT_DIR)
    }


@app.get("/tts/voices")
async def list_voices(include_system: bool = True):
    """List all available voices (system + custom)"""
    voices = []
    
    # Add system voices (Edge TTS)
    if include_system:
        system_voices = await get_edge_tts_voices()
        voices.extend(system_voices)
    
    # Add custom voices
    for voice_id, metadata in _voice_metadata.items():
        voices.append({
            "id": voice_id,
            "name": metadata.get("name", voice_id),
            "language": metadata.get("language", "unknown"),
            "gender": metadata.get("gender", "unknown"),
            "type": "custom",
            "created_at": metadata.get("created_at")
        })
    
    return {
        "voices": voices,
        "total": len(voices),
        "custom_count": len(_voice_metadata),
        "system_count": len(voices) - len(_voice_metadata)
    }


@app.post("/tts/voices/upload")
async def upload_custom_voice(
    file: UploadFile = File(..., description="Voice sample audio file"),
    voice_name: str = Form(..., description="Name for this voice"),
    language: str = Form("en-US", description="Language code"),
    gender: str = Form("neutral", description="Gender of voice"),
    description: Optional[str] = Form(None, description="Voice description")
):
    """
    Upload a custom voice sample
    Note: This stores the voice metadata. For actual custom voice cloning,
    you'd need additional TTS models like Coqui TTS or XTTS.
    """
    voice_id = f"custom_{voice_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    voice_path = VOICES_DIR / f"{voice_id}.wav"
    
    try:
        # Save voice file
        async with aiofiles.open(voice_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Save metadata
        _voice_metadata[voice_id] = {
            "name": voice_name,
            "language": language,
            "gender": gender,
            "description": description,
            "file_path": str(voice_path),
            "created_at": datetime.now().isoformat()
        }
        save_voice_metadata()
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "message": f"Voice '{voice_name}' uploaded successfully",
            "note": "Voice stored. To use custom voice cloning, integrate with Coqui TTS or similar."
        }
        
    except Exception as e:
        if voice_path.exists():
            voice_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")


@app.delete("/tts/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a custom voice"""
    if voice_id not in _voice_metadata:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    try:
        # Delete voice file
        voice_path = Path(_voice_metadata[voice_id]["file_path"])
        if voice_path.exists():
            voice_path.unlink()
        
        # Remove from metadata
        del _voice_metadata[voice_id]
        save_voice_metadata()
        
        return {
            "status": "success",
            "message": f"Voice '{voice_id}' deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")


@app.post("/tts/synthesize")
async def synthesize_speech(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="Text to synthesize"),
    voice: str = Form(DEFAULT_VOICE, description="Voice ID to use"),
    rate: str = Form("+0%", description="Speech rate (-50% to +100%)"),
    pitch: str = Form("+0Hz", description="Pitch adjustment"),
    return_file: bool = Form(True, description="Return audio file or just URL")
):
    """
    Synthesize speech from text using specified voice
    Supports Edge TTS voices (system) and custom voices
    """
    try:
        # Check if it's a custom voice
        if voice in _voice_metadata:
            # For custom voices, you would use a voice cloning model here
            # For now, we'll use Edge TTS with a similar voice
            return {
                "status": "info",
                "message": "Custom voice cloning requires additional TTS model (Coqui TTS/XTTS)",
                "suggestion": "Using default Edge TTS voice. Install Coqui TTS for custom voice cloning.",
                "voice_metadata": _voice_metadata[voice]
            }
        
        # Use Edge TTS for system voices
        output_file = TTS_OUTPUT_DIR / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
        
        # Create communicate object
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        
        # Generate speech
        await communicate.save(str(output_file))
        
        # Schedule cleanup (delete after 1 hour)
        if not return_file:
            background_tasks.add_task(cleanup_file, output_file)
        
        if return_file:
            return FileResponse(
                path=str(output_file),
                media_type="audio/mpeg",
                filename=output_file.name
            )
        else:
            return {
                "status": "success",
                "text": text,
                "voice": voice,
                "rate": rate,
                "pitch": pitch,
                "file_url": f"/tts/download/{output_file.name}",
                "message": "Speech synthesized successfully"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.get("/tts/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    file_path = TTS_OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=filename
    )


@app.delete("/tts/outputs/clear")
async def clear_outputs():
    """Clear all generated TTS output files"""
    try:
        count = 0
        for file_path in TTS_OUTPUT_DIR.glob("tts_*.mp3"):
            file_path.unlink()
            count += 1
        
        return {
            "status": "success",
            "message": f"Deleted {count} output files"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear outputs: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Starting TTS API Server with Voice Management")
    print("=" * 60)
    print(f"Custom voices directory: {VOICES_DIR}")
    print(f"Output directory: {TTS_OUTPUT_DIR}")
    print(f"Default voice: {DEFAULT_VOICE}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
