#!/bin/bash

# Start FastAPI Server for Insanely Fast Whisper on GPU 2

echo "=========================================="
echo "Starting Insanely Fast Whisper API Server"
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
echo "✓ Starting server on http://localhost:8000"
echo ""
echo "API Documentation:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo "  - Web Client: Open web_client.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start the server
python api_server.py
