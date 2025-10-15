#!/bin/bash

# Start Gradio UI for Insanely Fast Whisper on GPU 2

echo "=========================================="
echo "Starting Insanely Fast Whisper Gradio UI"
echo "=========================================="

# Disable torchcodec to avoid FFmpeg dependency issues
export TRANSFORMERS_NO_TORCHCODEC=1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate insanely-fast-whisper

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'insanely-fast-whisper'"
    exit 1
fi

echo "✓ Conda environment activated"
echo "✓ Starting Gradio UI (auto-detecting free port...)"
echo ""
echo "Features:"
echo "  - 🎤 Microphone streaming transcription"
echo "  - 📁 File upload with trimming support"
echo "  - ⚡ GPU acceleration with Flash Attention"
echo "  - 🔤 Multi-language support"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the Gradio app (port auto-detected)
python gradio_app.py \
    --model openai/whisper-large-v3 \
    --device cuda:2 \
    --host 0.0.0.0 \
    --batch-size 24 \
    --use-flash
