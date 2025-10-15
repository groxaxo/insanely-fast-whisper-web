#!/usr/bin/env python3
"""
Gradio-based UI for Insanely Fast Whisper
Adapted from Whisper-Fast-Cpu-OpenVino with GPU acceleration and Flash Attention
"""

from __future__ import annotations

# IMPORTANT: Disable torchcodec to avoid FFmpeg dependency issues
# This must be set before importing transformers
import os
os.environ["TRANSFORMERS_NO_TORCHCODEC"] = "1"

import argparse
import gc
import copy
import logging
import socket
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

TARGET_SAMPLE_RATE = 16000
LOGGER = logging.getLogger("gradio_app")
PIPELINE_LOCK = Lock()
LAST_CALL_TIME = {"streaming": 0.0, "upload": 0.0}
MIN_CALL_INTERVAL = 1.5  # Minimum seconds between calls

# Global pipeline cache
_pipeline = None


def find_free_port(start_port: int = 7860, max_attempts: int = 100) -> int:
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio UI for Insanely Fast Whisper")
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3",
        help="Whisper model to use (default: openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--device",
        default="cuda:2",
        help="Target device for inference (default: cuda:2)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for the Gradio server bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the Gradio server (default: auto-detect free port starting from 7860)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Batch size for processing (default: 24)",
    )
    parser.add_argument(
        "--use-flash",
        action="store_true",
        default=True,
        help="Use Flash Attention 2 if available (default: True)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing tunnel",
    )
    return parser.parse_args()


def build_pipeline(model_name: str, device: str, use_flash: bool):
    """Build Whisper pipeline with GPU and Flash Attention"""
    global _pipeline
    
    LOGGER.info(f"Loading Whisper pipeline: {model_name}")
    LOGGER.info(f"Device: {device}")
    LOGGER.info(f"Flash Attention available: {is_flash_attn_2_available()}")
    
    model_kwargs = {}
    if use_flash and is_flash_attn_2_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"
        LOGGER.info("Using Flash Attention 2")
    else:
        model_kwargs["attn_implementation"] = "sdpa"
        LOGGER.info("Using SDPA attention")
    
    _pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16,
        device=device,
        model_kwargs=model_kwargs,
    )
    
    LOGGER.info("Pipeline loaded successfully!")
    return _pipeline


def ensure_mono(data: np.ndarray) -> np.ndarray:
    """Convert stereo to mono"""
    if data.ndim == 1:
        return data
    return data.mean(axis=1)


def resample_if_needed(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample audio to 16kHz if needed"""
    if sample_rate == TARGET_SAMPLE_RATE:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Normalize audio waveform"""
    max_abs = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if max_abs > 1.0:
        waveform = waveform / max_abs
    return waveform.astype(np.float32)


def prepare_audio(audio: Tuple[int, np.ndarray] | None) -> Optional[Tuple[np.ndarray, int]]:
    """Prepare audio for transcription"""
    if audio is None:
        LOGGER.warning("Received empty audio payload")
        return None
    
    sample_rate: Optional[int] = None
    data: Optional[np.ndarray] = None
    
    if isinstance(audio, (list, tuple)):
        if len(audio) == 2 and isinstance(audio[0], (int, float)):
            sample_rate = int(audio[0])
            data = audio[1]
        else:
            data = audio
    elif isinstance(audio, dict):
        sample_rate = audio.get("sample_rate") or audio.get("sampling_rate")
        data = audio.get("data") or audio.get("array")
    elif isinstance(audio, np.ndarray):
        data = audio
    else:
        LOGGER.error("Unsupported audio payload type: %s", type(audio).__name__)
        return None
    
    if data is None:
        LOGGER.warning("Audio payload missing data field")
        return None
    
    try:
        waveform = ensure_mono(np.asarray(data, dtype=np.float32))
        if not waveform.size:
            LOGGER.warning("Audio payload is empty after mono conversion")
            return None
        
        if sample_rate is None:
            sample_rate = TARGET_SAMPLE_RATE
        
        tensor = torch.from_numpy(waveform)
        tensor = resample_if_needed(tensor, sample_rate)
        processed = tensor.numpy()
        normalized = normalize_waveform(processed)
        
        return normalized, TARGET_SAMPLE_RATE
    except Exception as e:
        LOGGER.error(f"Error processing audio data: {str(e)}", exc_info=True)
        return None


def transcribe(
    audio: Tuple[int, np.ndarray] | None,
    language_code: str,
    task: str,
    return_timestamps: bool,
    batch_size: int,
) -> Tuple[str, List[Dict]]:
    """Transcribe audio using the pipeline"""
    prepared = prepare_audio(audio)
    if prepared is None:
        return "", []
    
    waveform, sample_rate = prepared
    duration = len(waveform) / float(sample_rate)
    
    # Skip very short audio chunks
    if duration < 1.0:
        LOGGER.debug("Skipping audio chunk too short: %.2fs", duration)
        return "", []
    
    LOGGER.info(
        "Transcribing audio duration=%.2fs language=%s task=%s",
        duration,
        language_code,
        task,
    )
    
    # Prepare generation kwargs
    generate_kwargs = {}
    if language_code and language_code != "auto":
        generate_kwargs["language"] = language_code
    if task and task != "transcribe":
        generate_kwargs["task"] = task
    
    try:
        # Check if pipeline is loaded
        if _pipeline is None:
            LOGGER.error("Pipeline not initialized")
            raise gr.Error("Model not loaded. Please wait for initialization to complete.")
        
        with PIPELINE_LOCK:
            # Run transcription - build kwargs dynamically
            pipeline_kwargs = {
                "chunk_length_s": 30,
                "batch_size": batch_size,
                "return_timestamps": return_timestamps,
            }
            
            # Only add generate_kwargs if not empty
            if generate_kwargs:
                pipeline_kwargs["generate_kwargs"] = generate_kwargs
            
            outputs = _pipeline(
                {"array": waveform, "sampling_rate": sample_rate},
                **pipeline_kwargs
            )
        
        # Validate outputs
        if outputs is None:
            LOGGER.error("Pipeline returned None")
            return "", []
        
        text = outputs.get("text", "").strip()
        chunks = outputs.get("chunks", []) if outputs else []
        
        # Format chunks for display
        segments = []
        if return_timestamps and chunks:
            for chunk in chunks:
                if chunk is None:
                    LOGGER.warning("Encountered None chunk, skipping")
                    continue
                
                # Handle None timestamps safely
                timestamp = chunk.get("timestamp", (0, duration))
                if timestamp is None or not isinstance(timestamp, (list, tuple)):
                    timestamp = (0, duration)
                
                # Safely extract start and end times
                try:
                    start_time = float(timestamp[0]) if len(timestamp) > 0 and timestamp[0] is not None else 0
                    end_time = float(timestamp[1]) if len(timestamp) > 1 and timestamp[1] is not None else duration
                except (TypeError, ValueError, IndexError) as e:
                    LOGGER.warning(f"Invalid timestamp format: {timestamp}, using defaults. Error: {e}")
                    start_time = 0
                    end_time = duration
                
                segments.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "text": chunk.get("text", "").strip() if isinstance(chunk, dict) else "",
                })
        
        LOGGER.info("Transcription output length=%d characters", len(text))
        return text, segments
        
    except Exception as exc:
        LOGGER.exception("Inference request failed")
        raise gr.Error(f"Transcription failed: {str(exc)}")


def create_interface(batch_size: int, model_name: str, device: str):
    """Create Gradio interface"""
    
    # Language options
    language_options = [
        "auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", 
        "ja", "ko", "zh", "ar", "hi", "tr", "vi", "th", "id", "ms"
    ]
    
    with gr.Blocks(
        title="Insanely Fast Whisper",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            f"""
            # 🚀 Insanely Fast Whisper - GPU Accelerated
            
            **Model:** `{model_name}` | **Device:** `{device}` | **Flash Attention:** {'✅' if is_flash_attn_2_available() else '❌'}
            
            **🎤 Streaming Mode:** Use microphone for real-time transcription with accumulation.  
            **📁 Upload Mode:** Upload audio file, trim to desired length, then click 'Process'.
            """
        )
        
        with gr.Row():
            with gr.Column():
                # Streaming audio input (microphone)
                streaming_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="🎤 Streaming Audio (Microphone)",
                    show_download_button=False,
                )
                
                # Upload audio input (file)
                upload_audio = gr.Audio(
                    sources=["upload"],
                    type="numpy",
                    streaming=False,
                    label="📁 Upload Audio (File)",
                    show_download_button=True,
                    editable=True,
                    waveform_options={
                        "show_recording_waveform": True,
                        "show_controls": True,
                    },
                )
                
                # Trim controls
                with gr.Row():
                    trim_start = gr.Number(
                        label="Trim Start (seconds)",
                        value=0,
                        minimum=0,
                        precision=2,
                        info="Start time in seconds"
                    )
                    trim_end = gr.Number(
                        label="Trim End (seconds)",
                        value=None,
                        minimum=0,
                        precision=2,
                        info="End time (leave empty for end of file)"
                    )
                
                process_btn = gr.Button(
                    "🎯 Process Uploaded Audio",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                language_dd = gr.Dropdown(
                    choices=language_options,
                    value="auto",
                    label="Language",
                    info="Select target language or auto-detect",
                )
                task_radio = gr.Radio(
                    choices=["transcribe", "translate"],
                    value="transcribe",
                    label="Task",
                    info="Transcribe keeps source language; translate converts to English",
                )
                timestamps_checkbox = gr.Checkbox(
                    value=True,
                    label="Return timestamps"
                )
                clear_btn = gr.Button(
                    "🗑️ Clear Transcript",
                    variant="secondary"
                )
        
        status_text = gr.Textbox(
            label="Status",
            value="Ready",
            lines=1,
            interactive=False
        )
        transcript_output = gr.Textbox(
            label="Transcript",
            lines=8,
            max_lines=20
        )
        segments_output = gr.Dataframe(
            headers=["start", "end", "text"],
            datatype=["number", "number", "str"],
            label="Segments",
        )
        
        # State to accumulate streaming transcripts
        accumulated_text = gr.State(value="")
        accumulated_segments = gr.State(value=[])
        
        def streaming_handler(
            audio, prev_text, prev_segments, language_choice, task_choice, ts_flag
        ):
            """Handle streaming audio - accumulates results"""
            try:
                if audio is None:
                    return prev_text, prev_segments, prev_text, prev_segments
                
                # Rate limiting
                current_time = time.time()
                if current_time - LAST_CALL_TIME["streaming"] < MIN_CALL_INTERVAL:
                    LOGGER.debug("Skipping call - too soon (rate limiting)")
                    return prev_text, prev_segments, prev_text, prev_segments
                
                LAST_CALL_TIME["streaming"] = current_time
                
                # Transcribe the new chunk
                new_text, new_segments = transcribe(
                    audio,
                    language_choice,
                    task_choice,
                    ts_flag,
                    batch_size,
                )
                
                # Accumulate results
                if new_text:
                    combined_text = (prev_text + " " + new_text).strip() if prev_text else new_text
                    prev_segs_copy = copy.deepcopy(prev_segments) if prev_segments else []
                    new_segs_copy = copy.deepcopy(new_segments) if new_segments else []
                    combined_segments = prev_segs_copy + new_segs_copy
                else:
                    combined_text = prev_text
                    combined_segments = copy.deepcopy(prev_segments) if prev_segments else []
                
                gc.collect()
                
                return combined_text, combined_segments, combined_text, combined_segments
                
            except Exception as e:
                LOGGER.exception("Error in streaming handler")
                error_msg = f"Error: {str(e)}"
                safe_text = str(prev_text) if prev_text else ""
                safe_segments = copy.deepcopy(prev_segments) if prev_segments else []
                gc.collect()
                return safe_text, safe_segments, error_msg, safe_segments
        
        def upload_handler(
            audio, trim_start_val, trim_end_val, language_choice, task_choice, ts_flag
        ):
            """Handle uploaded audio files"""
            try:
                LOGGER.info("=== Upload handler called ===")
                LOGGER.info(f"Audio type: {type(audio)}")
                LOGGER.info(f"Audio value: {audio if not isinstance(audio, (tuple, list)) else f'tuple/list with {len(audio)} elements'}")
                
                if audio is None:
                    LOGGER.warning("No audio provided")
                    return "⚠️ No audio file selected", "", [], "", []
                
                # Rate limiting
                current_time = time.time()
                if current_time - LAST_CALL_TIME["upload"] < MIN_CALL_INTERVAL:
                    LOGGER.warning("Upload called too soon, throttling")
                    time.sleep(MIN_CALL_INTERVAL - (current_time - LAST_CALL_TIME["upload"]))
                
                LAST_CALL_TIME["upload"] = current_time
                LOGGER.info("Rate limiting passed")
                
                # Apply trimming
                LOGGER.info("Processing audio data...")
                if isinstance(audio, tuple):
                    sample_rate, audio_data = audio
                    LOGGER.info(f"Audio is tuple: sample_rate={sample_rate}, data_shape={np.array(audio_data).shape if hasattr(audio_data, '__len__') else 'N/A'}")
                else:
                    audio_data = audio[1] if isinstance(audio, tuple) else audio
                    sample_rate = audio[0] if isinstance(audio, tuple) else TARGET_SAMPLE_RATE
                    LOGGER.info(f"Audio extracted: sample_rate={sample_rate}")
                
                total_duration = len(audio_data) / float(sample_rate)
                LOGGER.info(f"Total audio duration: {total_duration:.2f}s")
                
                # Apply trim
                start_sample = 0
                end_sample = len(audio_data)
                
                if trim_start_val is not None and trim_start_val > 0:
                    start_sample = int(trim_start_val * sample_rate)
                    start_sample = max(0, min(start_sample, len(audio_data)))
                
                if trim_end_val is not None and trim_end_val > 0:
                    end_sample = int(trim_end_val * sample_rate)
                    end_sample = max(start_sample, min(end_sample, len(audio_data)))
                
                # Extract trimmed portion
                trimmed_audio = audio_data[start_sample:end_sample]
                
                if len(trimmed_audio) == 0:
                    return "⚠️ Invalid trim range - no audio data", "", [], "", []
                
                trimmed_duration = len(trimmed_audio) / float(sample_rate)
                
                LOGGER.info(
                    f"Processing audio: total={total_duration:.2f}s, "
                    f"trimmed={trimmed_duration:.2f}s"
                )
                
                # Create tuple for transcribe function
                audio_to_process = (sample_rate, trimmed_audio)
                LOGGER.info(f"Calling transcribe with language={language_choice}, task={task_choice}, timestamps={ts_flag}")
                
                text, segments = transcribe(
                    audio_to_process,
                    language_choice,
                    task_choice,
                    ts_flag,
                    batch_size,
                )
                
                LOGGER.info(f"Transcription completed: {len(text) if text else 0} characters")
                
                clean_segments = copy.deepcopy(segments) if segments else []
                gc.collect()
                
                if trim_start_val or trim_end_val:
                    status_msg = (
                        f"✅ Processed {trimmed_duration:.2f}s "
                        f"(trimmed from {total_duration:.2f}s) - {len(text)} characters"
                    )
                else:
                    status_msg = (
                        f"✅ Processed {trimmed_duration:.2f}s of audio - "
                        f"{len(text)} characters"
                    )
                
                return status_msg, text, clean_segments, text, clean_segments
                
            except Exception as e:
                LOGGER.exception("Error in upload handler")
                gc.collect()
                error_msg = f"❌ Error: {str(e)}"
                return error_msg, f"Error: {str(e)}", [], f"Error: {str(e)}", []
        
        def clear_transcript():
            """Clear accumulated transcript"""
            return "🗑️ Cleared", "", [], "", []
        
        # Streaming: accumulate results as user speaks
        streaming_audio.stream(
            streaming_handler,
            inputs=[
                streaming_audio,
                accumulated_text,
                accumulated_segments,
                language_dd,
                task_radio,
                timestamps_checkbox,
            ],
            outputs=[
                accumulated_text,
                accumulated_segments,
                transcript_output,
                segments_output,
            ],
            stream_every=3.0,
        )
        
        # Upload: process when button is clicked
        process_btn.click(
            upload_handler,
            inputs=[
                upload_audio,
                trim_start,
                trim_end,
                language_dd,
                task_radio,
                timestamps_checkbox,
            ],
            outputs=[
                status_text,
                accumulated_text,
                accumulated_segments,
                transcript_output,
                segments_output,
            ],
        )
        
        # Clear button
        clear_btn.click(
            clear_transcript,
            outputs=[
                status_text,
                accumulated_text,
                accumulated_segments,
                transcript_output,
                segments_output,
            ],
        )
    
    return demo


def main() -> None:
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    
    # Find free port if not specified
    if args.port is None:
        args.port = find_free_port()
        LOGGER.info(f"Auto-detected free port: {args.port}")
    
    # Build pipeline
    build_pipeline(args.model, args.device, args.use_flash)
    
    # Create interface
    interface = create_interface(args.batch_size, args.model, args.device)
    
    # Launch
    LOGGER.info("=" * 60)
    LOGGER.info(f"🚀 Starting Gradio server on {args.host}:{args.port}")
    LOGGER.info(f"📊 Model: {args.model}")
    LOGGER.info(f"🎮 Device: {args.device}")
    LOGGER.info(f"📦 Batch size: {args.batch_size}")
    LOGGER.info(f"⚡ Flash Attention: {args.use_flash and is_flash_attn_2_available()}")
    LOGGER.info("")
    LOGGER.info(f"🌐 Access URL: http://localhost:{args.port}")
    LOGGER.info(f"🌐 Network URL: http://{args.host}:{args.port}")
    LOGGER.info("=" * 60)
    
    interface.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
