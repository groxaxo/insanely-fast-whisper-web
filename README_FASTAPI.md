# 🚀 Insanely Fast Whisper with FastAPI - Complete Setup

## ✅ What's Installed

- **FastAPI Server** running on GPU 2 (NVIDIA GeForce RTX 3090)
- **Flash Attention 2** for 3x faster transcription
- **Web Client** for easy testing
- **REST API** for integration with any application

---

## 🎯 Quick Start

### Option 1: Start Server (Simple)

```bash
conda activate insanely-fast-whisper
python api_server.py
```

### Option 2: Use Start Script

```bash
./start_server.sh
```

The server will start on **http://localhost:8000**

---

## 🌐 Access Points

Once the server is running:

1. **Interactive API Docs (Swagger)**: http://localhost:8000/docs
   - Test all endpoints directly in your browser
   - See request/response schemas
   - Try out the API interactively

2. **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
   - Clean, readable API documentation

3. **Web Client**: Open `web_client.html` in your browser
   - Beautiful UI for testing
   - Upload files or use URLs
   - See results in real-time

4. **Health Check**: http://localhost:8000/health
   - Check server status
   - Verify GPU availability

---

## 📡 API Usage Examples

### 1. Python Client

```python
import requests

# Transcribe a local file
with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    data = {
        'model': 'openai/whisper-large-v3',
        'batch_size': 24,
        'use_flash': True
    }
    response = requests.post('http://localhost:8000/transcribe', 
                            files=files, data=data)
    result = response.json()
    print(result['text'])
```

### 2. cURL

```bash
# Upload file
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=openai/whisper-large-v3" \
  -F "use_flash=true"

# From URL
curl -X POST "http://localhost:8000/transcribe/url" \
  -F "url=https://example.com/audio.mp3" \
  -F "use_flash=true"
```

### 3. JavaScript/Node.js

```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('use_flash', 'true');

const response = await fetch('http://localhost:8000/transcribe', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result.text);
```

---

## 📁 Files Created

| File | Description |
|------|-------------|
| `api_server.py` | Main FastAPI server (runs on GPU 2) |
| `web_client.html` | Beautiful web interface for testing |
| `test_api.py` | Python test script |
| `start_server.sh` | Convenient startup script |
| `FASTAPI_GUIDE.md` | Comprehensive API documentation |

---

## 🎛️ Available Endpoints

### GET /health
Check server health and GPU status

### GET /models
List available Whisper models

### POST /transcribe
Upload and transcribe audio file

**Parameters:**
- `file`: Audio file (required)
- `model`: Model name (default: whisper-large-v3)
- `batch_size`: Batch size (default: 24)
- `use_flash`: Use Flash Attention 2 (default: true)
- `language`: Force language (optional)

### POST /transcribe/url
Transcribe audio from URL

**Parameters:**
- `url`: Audio URL (required)
- Same other parameters as /transcribe

### POST /preload
Preload model into GPU memory

### DELETE /cache
Clear model cache and free GPU memory

---

## ⚡ Performance

With Flash Attention 2 on GPU 2 (RTX 3090):
- **150 minutes** of audio → **~2 minutes** transcription time
- **~75x faster** than real-time
- **24GB VRAM** available for large batch sizes

---

## 🔧 Configuration

Edit `api_server.py` to customize:

```python
GPU_DEVICE = 2  # Change GPU
DEFAULT_MODEL = "openai/whisper-large-v3"  # Change model
DEFAULT_BATCH_SIZE = 24  # Adjust batch size
```

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :8000

# Kill process if needed
kill -9 $(lsof -t -i:8000)
```

### Out of Memory
- Reduce `batch_size` (try 12 or 16)
- Use smaller model: `distil-whisper/distil-large-v2`
- Clear cache: `curl -X DELETE http://localhost:8000/cache`

### Can't connect from browser
- Make sure server is running
- Check firewall settings
- Try http://127.0.0.1:8000 instead of localhost

---

## 🚀 Production Deployment

### Run in Background

```bash
conda activate insanely-fast-whisper
nohup python api_server.py > api.log 2>&1 &

# Check logs
tail -f api.log

# Stop server
pkill -f api_server.py
```

### With Gunicorn (Recommended)

```bash
conda activate insanely-fast-whisper
pip install gunicorn

gunicorn api_server:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

---

## 📊 Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Server Status
```bash
curl http://localhost:8000/health | jq
```

### View Logs
```bash
# If running with nohup
tail -f api.log

# If running with systemd
journalctl -u whisper-api -f
```

---

## 🔒 Security Notes

For production:
1. Add authentication (API keys, OAuth)
2. Use HTTPS (reverse proxy with nginx/caddy)
3. Rate limiting
4. Input validation
5. CORS configuration

---

## 📚 Additional Resources

- **Full API Guide**: See `FASTAPI_GUIDE.md`
- **CLI Usage**: See `GPU2_SETUP.md`
- **Test Script**: Run `python test_api.py`
- **Web Client**: Open `web_client.html`

---

## 🎉 You're All Set!

Start the server and begin transcribing:

```bash
./start_server.sh
```

Then open http://localhost:8000/docs in your browser!

**Happy transcribing! 🎙️✨**
