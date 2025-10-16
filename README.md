# 🎙️ Insanely Fast Whisper

<p align="center">
  <strong>Blazingly fast speech-to-text transcription powered by Whisper, Transformers & Flash Attention</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-web-ui--api">Web UI & API</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-acknowledgements">Credits</a>
</p>

---

## 🚀 TL;DR

Transcribe **150 minutes** (2.5 hours) of audio in less than **98 seconds** using [OpenAI's Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) with Flash Attention 2. Achieve **~75x real-time** transcription speed!

**Plus: Full-featured TTS (Text-to-Speech) with voice management!**

### Speech-to-Text (Transcription)

| Interface | Best For | Command | Port |
|-----------|----------|---------|------|
| 🎤 **Gradio Web UI** | Interactive testing, microphone | `./start_gradio.sh` | Auto-detect (7860+) |
| 🚀 **FastAPI Server** | Production APIs, integration | `./start_server.sh` | 8000 |
| 🌐 **Web Client** | Browser-based testing | Open `web_client.html` | 8000 (uses FastAPI) |
| ⚡ **CLI Tool** | Quick transcriptions, scripts | `insanely-fast-whisper --file-name audio.mp3` | N/A |

### Text-to-Speech (TTS)

| Interface | Best For | Command | Port |
|-----------|----------|---------|------|
| 🎙️ **TTS API Server** | Voice synthesis, custom voices | `./start_tts.sh` | 8001 |
| 🌐 **TTS Web Client** | Voice management UI | Open `tts_client.html` | 8001 (uses TTS API) |

## 🎉 What's New in This Fork

This enhanced version adds:

- **🎤 Gradio Web UI** - Interactive interface with:
  - Real-time microphone streaming with accumulation
  - File upload with visual waveform trimming
  - Support for 20+ languages
  - Automatic free port detection
  - GPU acceleration with Flash Attention 2
  
- **🚀 FastAPI REST API Server** with:
  - OpenAPI/Swagger documentation
  - File upload and URL transcription endpoints
  - Model preloading and cache management
  - CORS support for web integration
  - Health check and monitoring endpoints

- **🌐 Modern Web Client** with:
  - Beautiful gradient UI design
  - Dual input modes (file upload & URL)
  - Live server health monitoring
  - Real-time transcription results
  - GPU and Flash Attention status display

- **🎙️ TTS (Text-to-Speech) System** with:
  - Voice synthesis using Microsoft Edge TTS
  - 300+ system voices in 100+ languages
  - Custom voice upload and management
  - Voice browsing with filters (language, type)
  - Adjustable speech rate and pitch
  - REST API for programmatic access
  - Beautiful web interface for voice management

- **⚡ Enhanced Performance**:
  - Dynamic port allocation (no conflicts)
  - Optimized batch processing
  - GPU memory management
  - Rate limiting for streaming

## 📦 Installation

```bash
pipx install insanely-fast-whisper==0.0.15 --force
```

<p align="center">
<img src="https://huggingface.co/datasets/reach-vb/random-images/resolve/main/insanely-fast-whisper-img.png" width="615" height="308">
</p>

Not convinced? Here are some benchmarks we ran on a Nvidia A100 - 80GB 👇

| Optimisation type    | Time to Transcribe (150 mins of Audio) |
|------------------|------------------|
| large-v3 (Transformers) (`fp32`)             | ~31 (*31 min 1 sec*)             |
| large-v3 (Transformers) (`fp16` + `batching [24]` + `bettertransformer`) | ~5 (*5 min 2 sec*)            |
| **large-v3 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~2 (*1 min 38 sec*)**            |
| distil-large-v2 (Transformers) (`fp16` + `batching [24]` + `bettertransformer`) | ~3 (*3 min 16 sec*)            |
| **distil-large-v2 (Transformers) (`fp16` + `batching [24]` + `Flash Attention 2`)** | **~1 (*1 min 18 sec*)**           |
| large-v2 (Faster Whisper) (`fp16` + `beam_size [1]`) | ~9.23 (*9 min 23 sec*)            |
| large-v2 (Faster Whisper) (`8-bit` + `beam_size [1]`) | ~8 (*8 min 15 sec*)            |

*P.S. We also ran benchmarks on a [Google Colab T4 GPU](/notebooks/) instance!*

*P.P.S. This project originally started as a way to showcase benchmarks for Transformers but has since evolved into a full-featured transcription suite. This is purely community driven!*

---

## ✨ Features

- **⚡ Lightning Fast**: ~75x faster than real-time with Flash Attention 2
- **🎤 Multiple Interfaces**: Web UI, REST API, and CLI
- **🌍 Multi-Language**: Support for 20+ languages with auto-detection
- **🎯 Flexible Input**: Microphone streaming, file upload, or URLs
- **🔧 Highly Configurable**: Batch size, models, devices, and more
- **📊 Production Ready**: Health checks, monitoring, and API documentation
- **💾 Memory Efficient**: GPU cache management and optimization
- **🔄 Real-time Streaming**: Live transcription with accumulation

---

## 🚀 Quick Start

### Option 1: 🎤 Gradio Web UI (Recommended for Interactive Use)

Launch the beautiful web interface with microphone support:

```bash
./start_gradio.sh
```

**Features:**
- 🎤 **Real-time microphone streaming** with automatic accumulation
- 📁 **File upload** with visual trimming controls
- 🌍 **20+ languages** supported
- ⚡ **GPU accelerated** with Flash Attention 2
- 🔄 **Auto-detects free port** (starts from 7860)

The UI will automatically launch on a free port and show you the access URL.

### Option 2: 🚀 FastAPI REST API (For Integration)

Start the REST API server:

```bash
./start_server.sh
# or
python api_server.py
```

**API Endpoints:**
- 📚 **Interactive Docs**: http://localhost:8000/docs
- 💚 **Health Check**: http://localhost:8000/health
- 🎙️ **Transcribe**: POST http://localhost:8000/transcribe
- 🌐 **From URL**: POST http://localhost:8000/transcribe/url
- 📋 **List Models**: GET http://localhost:8000/models
- 🗑️ **Clear Cache**: DELETE http://localhost:8000/cache

**Quick Example:**
```python
import requests

with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    data = {'use_flash': True, 'batch_size': 24}
    response = requests.post('http://localhost:8000/transcribe', 
                            files=files, data=data)
    print(response.json()['text'])
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "use_flash=true" \
  -F "batch_size=24"
```

### Option 3: 🌐 Web Client (Browser UI for FastAPI)

A modern, beautiful web interface that connects to the FastAPI server:

**Setup:**
1. Start the FastAPI server: `./start_server.sh`
2. Open `web_client.html` in your browser

**Features:**
- 🎨 **Modern UI** - Beautiful gradient design with smooth animations
- 📤 **Dual Input** - Upload files or provide URLs
- ⚡ **Live Status** - Real-time server health and GPU monitoring
- 🎯 **Easy Configuration** - Choose models, batch sizes, Flash Attention
- 📊 **Results Display** - View transcription with metadata (duration, model, GPU)

The web client provides a user-friendly alternative to the API documentation interface.

---

## 🎙️ Text-to-Speech (TTS) System

### Quick Start

**1. Start the TTS Server:**
```bash
./start_tts.sh
```

**2. Open the Web Interface:**
Open `tts_client.html` in your browser

### Features

- **🎵 Voice Synthesis** - Convert text to speech with 300+ voices
- **📤 Custom Voices** - Upload your own voice samples
- **🔍 Voice Browser** - Search and filter voices by language, gender, type
- **⚙️ Voice Tuning** - Adjust speech rate (-50% to +100%) and pitch
- **🌍 Multi-Language** - Support for 100+ languages
- **🔌 REST API** - Programmatic access to all TTS features

### API Usage Examples

**Python:**
```python
import requests

# Synthesize speech
with open('output.mp3', 'wb') as f:
    response = requests.post('http://localhost:8001/tts/synthesize', 
        data={
            'text': 'Hello, this is a test!',
            'voice': 'en-US-AriaNeural',
            'rate': '+0%',
            'pitch': '+0Hz'
        })
    f.write(response.content)

# Upload custom voice
with open('my_voice.wav', 'rb') as f:
    files = {'file': f}
    data = {'voice_name': 'My Custom Voice', 'language': 'en-US'}
    response = requests.post('http://localhost:8001/tts/voices/upload',
                            files=files, data=data)
    print(response.json())

# List all voices
response = requests.get('http://localhost:8001/tts/voices')
voices = response.json()['voices']
```

**cURL:**
```bash
# Synthesize speech
curl -X POST "http://localhost:8001/tts/synthesize" \
  -F "text=Hello world" \
  -F "voice=en-US-AriaNeural" \
  --output speech.mp3

# Upload voice
curl -X POST "http://localhost:8001/tts/voices/upload" \
  -F "file=@my_voice.wav" \
  -F "voice_name=My Voice" \
  -F "language=en-US"

# List voices
curl http://localhost:8001/tts/voices
```

### TTS API Endpoints

- **POST /tts/synthesize** - Generate speech from text
- **GET /tts/voices** - List all available voices
- **POST /tts/voices/upload** - Upload custom voice sample
- **DELETE /tts/voices/{voice_id}** - Delete custom voice
- **GET /tts/download/{filename}** - Download generated audio
- **DELETE /tts/outputs/clear** - Clear all generated files

### Custom Voice Management

1. **Upload Voice Sample:**
   - Go to "Upload Voice" tab in `tts_client.html`
   - Select audio file (WAV, MP3, etc.)
   - Provide name, language, and gender
   - Click upload

2. **Use Custom Voice:**
   - Custom voices are stored in `./voices/` directory
   - Access via API or web interface
   - Note: For voice cloning, integrate Coqui TTS or XTTS

3. **Manage Voices:**
   - Browse all voices in "Browse Voices" tab
   - Filter by language, type (system/custom)
   - Test voices with preview button
   - Delete custom voices as needed

---

### Option 4: ⚡ Command Line Interface

Perfect for quick, one-off transcriptions:

We've added a CLI to enable fast transcriptions. Here's how you can use it:

Install `insanely-fast-whisper` with `pipx` (`pip install pipx` or `brew install pipx`):

```bash
pipx install insanely-fast-whisper
```

⚠️ If you have python 3.11.XX installed, `pipx` may parse the version incorrectly and install a very old version of `insanely-fast-whisper` without telling you (version `0.0.8`, which won't work anymore with the current `BetterTransformers`). In that case, you can install the latest version by passing `--ignore-requires-python` to `pip`:

```bash
pipx install insanely-fast-whisper --force --pip-args="--ignore-requires-python"
```

If you're installing with `pip`, you can pass the argument directly: `pip install insanely-fast-whisper --ignore-requires-python`.


Run inference from any path on your computer:

```bash
insanely-fast-whisper --file-name <filename or URL>
```
*Note: if you are running on macOS, you also need to add `--device-id mps` flag.*

🔥 You can run [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) w/ [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) from this CLI too:

```bash
insanely-fast-whisper --file-name <filename or URL> --flash True 
```

🌟 You can run [distil-whisper](https://huggingface.co/distil-whisper) directly from this CLI too:

```bash
insanely-fast-whisper --model-name distil-whisper/large-v2 --file-name <filename or URL> 
```

Don't want to install `insanely-fast-whisper`? Just use `pipx run`:

```bash
pipx run insanely-fast-whisper --file-name <filename or URL>
```

> [!NOTE]
> The CLI is highly opinionated and only works on NVIDIA GPUs & Mac. Make sure to check out the defaults and the list of options you can play around with to maximise your transcription throughput. Run `insanely-fast-whisper --help` or `pipx run insanely-fast-whisper --help` to get all the CLI arguments along with their defaults. 


## CLI Options

The `insanely-fast-whisper` repo provides an all round support for running Whisper in various settings. Note that as of today 26th Nov, `insanely-fast-whisper` works on both CUDA and mps (mac) enabled devices.
```
  -h, --help            show this help message and exit
  --file-name FILE_NAME
                        Path or URL to the audio file to be transcribed.
  --device-id DEVICE_ID
                        Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")
  --transcript-path TRANSCRIPT_PATH
                        Path to save the transcription output. (default: output.json)
  --model-name MODEL_NAME
                        Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)
  --task {transcribe,translate}
                        Task to perform: transcribe or translate to another language. (default: transcribe)
  --language LANGUAGE   
                        Language of the input audio. (default: "None" (Whisper auto-detects the language))
  --batch-size BATCH_SIZE
                        Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)
  --flash FLASH         
                        Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)
  --timestamp {chunk,word}
                        Whisper supports both chunked as well as word level timestamps. (default: chunk)
  --hf-token HF_TOKEN
                        Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips
  --diarization_model DIARIZATION_MODEL
                        Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)
  --num-speakers NUM_SPEAKERS
                        Specifies the exact number of speakers present in the audio file. Useful when the exact number of participants in the conversation is known. Must be at least 1. Cannot be used together with --min-speakers or --max-speakers. (default: None)
  --min-speakers MIN_SPEAKERS
                        Sets the minimum number of speakers that the system should consider during diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be less than or equal to --max-speakers if both are specified. (default: None)
  --max-speakers MAX_SPEAKERS
                        Defines the maximum number of speakers that the system should consider in diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be greater than or equal to --min-speakers if both are specified. (default: None)
```

## Frequently Asked Questions

**How to correctly install flash-attn to make it work with `insanely-fast-whisper`?**

Make sure to install it via `pipx runpip insanely-fast-whisper install flash-attn --no-build-isolation`. Massive kudos to @li-yifei for helping with this.

**How to solve an `AssertionError: Torch not compiled with CUDA enabled` error on Windows?**

The root cause of this problem is still unknown, however, you can resolve this by manually installing torch in the virtualenv like `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. Thanks to @pto2k for all tdebugging this.

**How to avoid Out-Of-Memory (OOM) exceptions on Mac?**

The *mps* backend isn't as optimised as CUDA, hence is way more memory hungry. Typically you can run with `--batch-size 4` without any issues (should use roughly 12GB GPU VRAM). Don't forget to set `--device-id mps`.

---

## 📚 Advanced Usage

### Using Whisper Programmatically

<details>
<summary>Python snippet for direct usage:</summary>

```bash
pip install --upgrade transformers optimum accelerate
```

```python
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0", # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

outputs = pipe(
    "<FILE_NAME>",
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=True,
)

print(outputs)
```
</details>

---

## 🙏 Acknowledgements

This project builds upon incredible work from the community:

### Core Technologies
1. **[OpenAI Whisper](https://github.com/openai/whisper)** - For open sourcing this brilliant speech recognition model
2. **[Hugging Face Transformers](https://github.com/huggingface/transformers)** team:
   - [Arthur Zucker](https://github.com/ArthurZucker)
   - [Patrick von Platen](https://github.com/patrickvonplaten)
   - [Sanchit Gandhi](https://github.com/sanchit-gandhi)
   - [Yoach Lacombe](https://github.com/ylacombe)
   
   For maintaining and improving Whisper in Transformers
   
3. **[Hugging Face Optimum](https://github.com/huggingface/optimum)** - For making BetterTransformer API easily accessible
4. **[Flash Attention](https://github.com/Dao-AILab/flash-attention)** team - For the incredible speedup
5. **[Gradio](https://gradio.app)** team - For the amazing web UI framework
6. **[FastAPI](https://fastapi.tiangolo.com/)** - For the elegant API framework

### Original Development
- **[VB (Vaibhav Srivastav)](https://github.com/Vaibhavs10)** - Original creator and maintainer
- **[Patrick Arminio](https://github.com/patrick91)** - For helping tremendously with the CLI

### Enhanced Features
- **Web UI & API Server** - Adapted and enhanced from community feedback and [Whisper-Fast-Cpu-OpenVino](https://github.com/) project patterns
- **Dynamic port detection** - Community-driven improvement
- **Streaming support** - Built for real-world use cases

### Special Thanks
- **[@li-yifei](https://github.com/li-yifei)** - For Flash Attention installation guidance
- **[@pto2k](https://github.com/pto2k)** - For Windows CUDA debugging
- All contributors and users providing feedback and improvements

## 🎛️ Server Management

### Gradio UI Commands

```bash
# Start (auto-detects free port)
./start_gradio.sh

# Stop
pkill -f gradio_app.py

# Custom configuration
python gradio_app.py --model openai/whisper-medium --device cuda:0 --batch-size 16

# Check status
lsof -i :7860  # or whatever port it's using
```

### FastAPI Server Commands

```bash
# Start
./start_server.sh
# or
python api_server.py

# Stop
pkill -f api_server.py

# Check status
curl http://localhost:8000/health
```

### Available Models

Both interfaces support all Whisper models:
- `openai/whisper-large-v3` (best quality, recommended)
- `openai/whisper-large-v2`
- `openai/whisper-medium`
- `openai/whisper-small`
- `openai/whisper-base`
- `distil-whisper/distil-large-v2` (faster, smaller)
- `distil-whisper/distil-medium.en`
- `distil-whisper/distil-small.en`

### Configuration

**Gradio UI Options:**
- `--model`: Model name (default: openai/whisper-large-v3)
- `--device`: GPU device (default: cuda:2)
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: auto-detect from 7860)
- `--batch-size`: Batch size (default: 24)
- `--use-flash`: Enable Flash Attention 2 (default: True)
- `--share`: Create Gradio public URL

**FastAPI Server:**
Edit `api_server.py` to change GPU device, model, or batch size defaults.

### Troubleshooting

**Port already in use:**
```bash
# Find and kill process
lsof -i :7860  # or :8000 for FastAPI
kill -9 <PID>
```

**Out of memory:**
- Reduce batch size: `--batch-size 12`
- Use smaller model: `--model openai/whisper-medium`
- Clear GPU cache via API: `curl -X DELETE http://localhost:8000/cache`

**Gradio not finding free port:**
The app automatically scans 100 ports starting from 7860. If all are occupied, specify manually: `--port 9000`

**TorchCodec / FFmpeg errors:**
```
Error: Could not load libtorchcodec...FFmpeg not installed
```
✅ **FIXED**: This is already handled! The environment variable `TRANSFORMERS_NO_TORCHCODEC=1` is set in all scripts and code. No FFmpeg installation needed.

## 📊 Performance Comparison

| Interface | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **Gradio UI** | Interactive testing, microphone input | Beautiful UI, real-time streaming, easy trimming | Web-only access |
| **FastAPI** | Production, batch processing, integration | REST API, scriptable, OpenAPI docs | Requires programming |
| **Web Client** | Browser-based testing | No coding, beautiful UI, easy to use | Requires FastAPI running |
| **CLI** | One-off transcriptions, scripting | Simple, no server needed | No streaming, no interactive features |

## 📁 Project Files

### Transcription (Speech-to-Text)
| File | Description |
|------|-------------|
| `gradio_app.py` | Gradio web UI server with microphone support |
| `api_server.py` | FastAPI REST API server (GPU 2) |
| `web_client.html` | Modern browser-based web client for FastAPI |
| `start_gradio.sh` | Launch script for Gradio UI |
| `start_server.sh` | Launch script for FastAPI server |
| `test_api.py` | Python test script for API endpoints |

### TTS (Text-to-Speech)
| File | Description |
|------|-------------|
| `tts_server.py` | TTS API server with voice management |
| `tts_client.html` | Web interface for TTS and voice management |
| `start_tts.sh` | Launch script for TTS server |

### Core
| File | Description |
|------|-------------|
| `src/insanely_fast_whisper/cli.py` | Original CLI tool |
| `README.md` | This documentation file |

---

## 🌟 Community Showcase

Amazing projects built by the community:

1. **[@ochen1](https://github.com/ochen1)** - [insanely-fast-whisper-cli](https://github.com/ochen1/insanely-fast-whisper-cli) - Brilliant MVP CLI wrapper
2. **[@arihanv](https://github.com/arihanv)** - [Shush](https://github.com/arihanv/Shush) - NextJS frontend with Modal backend
3. **[@kadirnar](https://github.com/kadirnar)** - [whisper-plus](https://github.com/kadirnar/whisper-plus) - Enhanced Python package with optimizations

*Want to add your project? Open a PR!*

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

Contributions are welcome! This is a community-driven project. Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Share your use cases and feedback
- Add to the community showcase

---

<p align="center">
  <strong>Made with ❤️ by the community</strong><br>
  <sub>Powered by 🤗 Transformers • OpenAI Whisper • Flash Attention</sub>
</p>

<p align="center">
  <a href="https://github.com/Vaibhavs10/insanely-fast-whisper">⭐ Star on GitHub</a> •
  <a href="https://github.com/Vaibhavs10/insanely-fast-whisper/issues">🐛 Report Bug</a> •
  <a href="https://github.com/Vaibhavs10/insanely-fast-whisper/issues">💡 Request Feature</a>
</p>
