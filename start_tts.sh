#!/bin/bash
# Start TTS API Server

# Set environment variable to disable torchcodec
export TRANSFORMERS_NO_TORCHCODEC=1

echo "=========================================="
echo "Starting TTS API Server on Port 8001"
echo "=========================================="
echo ""
echo "Server will be available at:"
echo "  - http://localhost:8001"
echo "  - http://0.0.0.0:8001"
echo ""
echo "API Documentation:"
echo "  - http://localhost:8001/docs"
echo ""
echo "Web Client:"
echo "  - Open tts_client.html in your browser"
echo ""
echo "=========================================="

# Start the server
python3 tts_server.py
