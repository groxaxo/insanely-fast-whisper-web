# 🚀 Gradio UI - Quick Start

## ✅ Server Status: RUNNING

**Access URL:** http://localhost:7860 or http://0.0.0.0:7860

---

## 📍 Quick Commands

### Start Server
```bash
./start_gradio.sh
```

### Stop Server
```bash
pkill -f gradio_app.py
```

### Check Status
```bash
lsof -i :7860
```

### View Logs
```bash
ps aux | grep gradio_app.py
```

---

## 🎯 Features at a Glance

| Feature | Available |
|---------|-----------|
| 🎤 Microphone Streaming | ✅ |
| 📁 File Upload | ✅ |
| ✂️ Audio Trimming | ✅ |
| 🌍 Multi-Language | ✅ |
| ⚡ Flash Attention | ✅ |
| 🎮 GPU Acceleration | ✅ (GPU 2) |
| ⏱️ Timestamps | ✅ |
| 🔄 Translation | ✅ |

---

## 🎮 Current Configuration

- **Model:** openai/whisper-large-v3
- **Device:** cuda:2 (RTX 3090)
- **Host:** 0.0.0.0
- **Port:** 7860
- **Batch Size:** 24
- **Flash Attention:** Enabled

---

## 📱 How to Use

### Microphone (Streaming)
1. Click **🎤 Streaming Audio**
2. Allow microphone access
3. Start speaking
4. Watch transcript accumulate

### File Upload
1. Click **📁 Upload Audio**
2. Select audio file
3. (Optional) Set trim start/end times
4. Click **🎯 Process Uploaded Audio**

---

## 🔗 Access Points

- **Gradio UI:** http://localhost:7860
- **FastAPI:** http://localhost:8000 (separate server)
- **API Docs:** http://localhost:8000/docs

---

## 📚 Documentation

- `README_GRADIO.md` - Full documentation
- `README_FASTAPI.md` - API documentation
- `README.md` - Original project docs

---

**Enjoy your GPU-accelerated Whisper transcription! 🎙️✨**
