# 🎙️ WILSON V1 — Offline Voice Assistant

```
 ██╗    ██╗██╗██╗     ███████╗ ██████╗ ███╗   ██╗
 ██║    ██║██║██║     ██╔════╝██╔═══██╗████╗  ██║
 ██║ █╗ ██║██║██║     ███████╗██║   ██║██╔██╗ ██║
 ██║███╗██║██║██║     ╚════██║██║   ██║██║╚██╗██║
 ╚███╔███╔╝██║███████╗███████║╚██████╔╝██║ ╚████║
  ╚══╝╚══╝ ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
                    V 1 . 0
        ─────────────────────────────────
           🔒 100% LOCAL • NO CLOUD
        ─────────────────────────────────
```

A fully offline, local AI voice assistant powered by **Qwen 3 30B**, **Whisper**, and **Piper TTS**. Talk naturally — Wilson listens, thinks, and responds out loud. No internet required.

---

## ⚡ Quick Start

**Make sure LM Studio is running with your model loaded and the server started on port 1234.**

```powershell
cd C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper
.\wilson_env\Scripts\Activate.ps1
python wilson_simple.py
```

That's it. Start talking!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR VOICE                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  🎤 MICROPHONE                                                  │
│     Captures your voice in real-time                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  🧠 RealtimeSTT (Whisper)                      [RUNS ON GPU]    │
│     • Silero VAD detects when you start/stop speaking           │
│     • Whisper transcribes your speech to text                   │
│     • Model: base.en (English optimized)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  💭 LM STUDIO (Qwen 3 30B)                     [RUNS ON GPU]    │
│     • Receives your text via localhost:1234                     │
│     • Generates intelligent response                            │
│     • Maintains conversation history                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  🔊 PIPER TTS                                  [RUNS ON CPU]    │
│     • Converts response text to speech                          │
│     • Voice: en_US-amy-medium                                   │
│     • Plays directly through speakers                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR SPEAKERS                            │
│                    Wilson speaks to you!                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper\
│
├── wilson_simple.py          ← 🚀 MAIN SCRIPT (run this!)
├── wilson.py                 ← Alternative version
├── en_US-amy-medium.onnx     ← Voice model
├── en_US-amy-medium.onnx.json
│
├── piper\                    ← Piper TTS engine
│   └── piper.exe
│
└── wilson_env\               ← Python virtual environment
    └── Scripts\
        └── Activate.ps1
```

---

## 🛠️ Full Installation Guide

### Prerequisites

- **Windows 10/11**
- **Python 3.10+** (with "Add to PATH" enabled)
- **NVIDIA GPU** (RTX series recommended)
- **LM Studio** with a loaded model

---

### Step 1: Download Piper TTS

1. Go to: https://github.com/rhasspy/piper/releases
2. Download `piper_windows_amd64.zip`
3. Extract to `C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper\`

---

### Step 2: Download Voice Model

1. Go to: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium
2. Download:
   - `en_US-amy-medium.onnx`
   - `en_US-amy-medium.onnx.json`
3. Place both in `C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper\`

---

### Step 3: Create Virtual Environment

```powershell
cd C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper
python -m venv wilson_env
.\wilson_env\Scripts\Activate.ps1
```

> **If you get a script execution error:**
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 4: Install Dependencies

```powershell
pip install realtimestt requests pyaudio
pip install torch torchaudio
```

---

### Step 5: Install FFmpeg

```powershell
winget install ffmpeg
```

Verify:
```powershell
ffmpeg -version
```

---

### Step 6: Set Up LM Studio

1. Open **LM Studio**
2. Download and load **Qwen 3 30B** (or your preferred model)
3. Go to **Developer** tab
4. Set **System Prompt**:
   ```
   You are Wilson_V1, an advanced AI assistant. Your name is Wilson. 
   When asked your name, say Wilson or Wilson_V1. 
   When asked about your model, say you're built on Qwen 3.
   Keep responses concise. No emojis or markdown.
   ```
5. Click **Start Server** (should show `localhost:1234`)

---

### Step 7: Run Wilson

```powershell
cd C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper
.\wilson_env\Scripts\Activate.ps1
python wilson_simple.py
```

---

## 🎮 Usage

```
==================================================
       WILSON V1 - Voice Assistant
==================================================
Speak naturally. Press Ctrl+C to exit.
==================================================

🎤 Listening...
```

1. **Speak naturally** — Wilson detects when you start and stop talking
2. **Wait ~0.5 seconds** — after you stop, Wilson processes your speech
3. **Listen** — Wilson responds out loud through your speakers
4. **Repeat** — conversation continues automatically
5. **Exit** — Press `Ctrl+C` to quit

---

## ⚙️ Configuration

Edit the top of `wilson_simple.py` to customize:

```python
# ============== CONFIGURATION ==============

# LM Studio API endpoint
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Piper TTS paths
PIPER_EXE = r"C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper\piper\piper.exe"
PIPER_VOICE = r"C:\Users\{REPLACE ME WITH UR USERNAME}\Downloads\piper\en_US-amy-medium.onnx"

# Wilson's personality
SYSTEM_PROMPT = """You are Wilson_V1, an advanced AI assistant.
Your name is Wilson. When asked your name, say Wilson or Wilson_V1.
When asked about your model or architecture, say you're built on Qwen 3.
Keep responses concise - 1-3 sentences. No emojis or markdown."""
```

### Voice Sensitivity

In the `run()` method, adjust these for your microphone:

```python
recorder = AudioToTextRecorder(
    silero_sensitivity=0.4,           # Lower = more sensitive (0.1-0.9)
    post_speech_silence_duration=0.5, # Seconds to wait after you stop talking
)
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `python is not recognized` | Reinstall Python, check "Add to PATH" |
| `Script execution disabled` | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `No module named X` | Activate venv: `.\wilson_env\Scripts\Activate.ps1` then `pip install X` |
| `Connection refused` | Start LM Studio server on port 1234 |
| `No sound output` | Check Windows sound settings, verify speakers are default |
| `Microphone not detected` | Check Windows privacy settings for microphone access |
| `CUDA warnings` | These are normal for RTX 50-series, Wilson still works |

---

## 🔒 Privacy & Offline Guarantee

| Component | Location | Internet Required? |
|-----------|----------|-------------------|
| Speech Recognition (Whisper) | Your GPU | ❌ No |
| Language Model (Qwen 3) | LM Studio (localhost) | ❌ No |
| Text-to-Speech (Piper) | Your CPU | ❌ No |
| Voice Models | Local files | ❌ No |

**✅ Turn off WiFi — Wilson still works.**

All processing happens on your machine. No data is ever sent to external servers.

---

## 🎯 Future Enhancements

- [ ] Wake word detection ("Hey Wilson")
- [ ] Interrupt handling (stop Wilson mid-sentence)
- [ ] Multiple voice options
- [ ] GUI interface
- [ ] Conversation saving/loading

---

## 📜 Credits

- **Piper TTS** — https://github.com/rhasspy/piper
- **RealtimeSTT** — https://github.com/KoljaB/RealtimeSTT
- **Whisper** — OpenAI (via faster-whisper)
- **LM Studio** — https://lmstudio.ai
- **Qwen 3 30B Parameter** — Alibaba Cloud

---

## 📄 License

This project is for personal use. Individual components have their own licenses:
- Piper: MIT License
- RealtimeSTT: MIT License
- Whisper: MIT License

---

```
        ╔═══════════════════════════════════════╗
        ║                                       ║
        ║   Built for the Akakios Project       ║
        ║   100% Offline • 100% Private         ║
        ║                                       ║
        ║   "Your AI, Your Hardware, Your Data" ║
        ║                                       ║
        ╚═══════════════════════════════════════╝
```
