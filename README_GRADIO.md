# 🎙️ Insanely Fast Whisper - Gradio UI

## 🚀 Quick Start

The Gradio UI is **now running** on your system!

### Access the Interface

**Local Access:**
- http://localhost:7860
- http://0.0.0.0:7860

**Network Access:**
- Replace `localhost` with your server's IP address
- Example: http://YOUR_SERVER_IP:7860

---

## ✨ Features

### 🎤 **Streaming Transcription**
- Real-time microphone input
- Automatic accumulation of transcripts
- Live processing with rate limiting

### 📁 **File Upload**
- Support for all audio formats (MP3, WAV, M4A, etc.)
- Built-in audio trimming controls
- Waveform visualization
- Precise time-based trimming (start/end seconds)

### ⚡ **GPU Acceleration**
- Running on **GPU 2 (NVIDIA GeForce RTX 3090)**
- **Flash Attention 2** for 3x faster inference
- Batch processing with size 24
- Optimized for long audio files

### 🌍 **Multi-Language Support**
- Auto-detect language
- Manual language selection (20+ languages)
- Translation to English
- Timestamp support

---

## 🎯 How to Use

### Method 1: Streaming (Microphone)

1. Click on the **🎤 Streaming Audio** section
2. Click "Allow" when prompted for microphone access
3. Start speaking
4. Transcription appears automatically with accumulation
5. Click **🗑️ Clear Transcript** to start fresh

### Method 2: File Upload

1. Click on the **📁 Upload Audio** section
2. Upload your audio file
3. (Optional) Use trim controls:
   - **Trim Start**: Enter start time in seconds (e.g., 5.5)
   - **Trim End**: Enter end time in seconds (leave empty for full file)
4. Select **Language** and **Task** (transcribe or translate)
5. Toggle **Return timestamps** if needed
6. Click **🎯 Process Uploaded Audio**
7. View results in the transcript and segments table

---

## 🛠️ Configuration

### Current Settings
- **Model**: openai/whisper-large-v3
- **Device**: GPU 2 (cuda:2)
- **Host**: 0.0.0.0 (accessible from network)
- **Port**: 7860
- **Batch Size**: 24
- **Flash Attention**: Enabled

### Customize Settings

Edit `start_gradio.sh` to change parameters:

```bash
python gradio_app.py \
    --model openai/whisper-large-v2 \    # Change model
    --device cuda:0 \                     # Change GPU
    --host 127.0.0.1 \                    # Localhost only
    --port 8080 \                         # Different port
    --batch-size 16 \                     # Smaller batch
    --use-flash                            # Enable Flash Attention
```

---

## 📋 Available Models

- `openai/whisper-large-v3` (default, best quality)
- `openai/whisper-large-v2`
- `openai/whisper-medium`
- `openai/whisper-small`
- `openai/whisper-base`
- `distil-whisper/distil-large-v2` (faster, smaller)

---

## 🔧 Management

### Start the Server

```bash
# Using the script
./start_gradio.sh

# Or manually
conda activate insanely-fast-whisper
python gradio_app.py --host 0.0.0.0 --port 7860
```

### Stop the Server

```bash
# Find the process
ps aux | grep gradio_app.py

# Kill it
kill -9 <PID>

# Or use pkill
pkill -f gradio_app.py
```

### Check Status

```bash
# Check if port is in use
lsof -i :7860

# Check GPU usage
nvidia-smi
```

---

## 🎨 Interface Overview

### Main Components

1. **Streaming Audio (Microphone)**
   - Real-time audio capture
   - Automatic transcription
   - Accumulates text across multiple chunks

2. **Upload Audio (File)**
   - Drag & drop or click to upload
   - Visual waveform editor
   - Precise trimming controls

3. **Settings Panel**
   - Language selection (auto or specific)
   - Task: Transcribe or Translate
   - Timestamp toggle

4. **Output Display**
   - Full transcript text
   - Timestamp segments table (start, end, text)
   - Status messages

---

## 🚨 Troubleshooting

### Server Won't Start

```bash
# Check if port is already in use
lsof -i :7860

# Kill existing process
kill -9 $(lsof -t -i:7860)

# Try again
./start_gradio.sh
```

### Out of Memory

```bash
# Edit start_gradio.sh and reduce batch size
--batch-size 12  # Instead of 24

# Or use a smaller model
--model openai/whisper-medium
```

### Microphone Not Working

- Check browser permissions (must be HTTPS or localhost)
- Try a different browser (Chrome/Firefox recommended)
- Check system microphone settings

### Can't Access from Network

- Verify firewall settings
- Ensure port 7860 is open
- Check if server is binding to 0.0.0.0 (not 127.0.0.1)

---

## 📊 Performance

With Flash Attention 2 on GPU 2 (RTX 3090):

- **~75x faster** than real-time
- **150 minutes** of audio → **~2 minutes** transcription
- **24GB VRAM** available for large batches

---

## 🔒 Security Notes

For production deployment:

1. Add authentication (basic auth, OAuth)
2. Use HTTPS (reverse proxy with nginx)
3. Rate limiting (already built-in)
4. Firewall rules
5. VPN access for sensitive data

---

## 🆚 Comparison with FastAPI Server

| Feature | Gradio UI | FastAPI Server |
|---------|-----------|----------------|
| Interface | Web UI | REST API |
| Microphone | ✅ Yes | ❌ No |
| File Upload | ✅ Yes | ✅ Yes |
| Trimming | ✅ Built-in | ❌ Manual |
| Live Preview | ✅ Yes | ❌ No |
| API Access | ❌ No | ✅ Yes |
| Integration | ❌ Limited | ✅ Full |

**Use Gradio UI for:**
- Interactive testing
- Quick transcriptions
- Microphone input
- Visual feedback

**Use FastAPI Server for:**
- Programmatic access
- Batch processing
- Integration with other apps
- Production APIs

---

## 📚 Additional Files

- `gradio_app.py` - Main application
- `start_gradio.sh` - Launch script
- `README_FASTAPI.md` - FastAPI server docs
- `README_GRADIO.md` - This file

---

## 🎉 You're All Set!

The Gradio UI is running at:
- **http://localhost:7860** (local)
- **http://0.0.0.0:7860** (network)

**Happy transcribing! 🎙️✨**

---

## 💡 Tips

1. **For long files**: Use the trimming feature to test small sections first
2. **For best quality**: Use `whisper-large-v3` with timestamps enabled
3. **For speed**: Use `distil-whisper/distil-large-v2` with smaller batch size
4. **For real-time**: Use streaming mode with auto language detection
5. **For debugging**: Check terminal output for detailed logs

---

## 🐛 Known Issues

1. **Streaming mode** may have ~1.5s delay between chunks (rate limiting)
2. **Very short audio** (<1s) is automatically skipped
3. **Large files** (>2 hours) may need batch size reduction
4. **Microphone streaming** accumulates text - use Clear button to reset

---

## 🔗 Links

- **Original Project**: [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- **Whisper Models**: [OpenAI Whisper](https://github.com/openai/whisper)
- **Gradio Docs**: [gradio.app](https://gradio.app)
- **Flash Attention**: [flash-attention](https://github.com/Dao-AILab/flash-attention)
