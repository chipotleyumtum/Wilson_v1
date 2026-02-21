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

A fully offline AI voice assistant. Talk naturally — Wilson listens, thinks, and responds out loud. **No internet required.** Works on Windows, macOS, and Linux.

```
  Your Voice → 🎤 Microphone → 🧠 Whisper (Speech-to-Text)
                                        ↓
                               💭 LLM (thinks of a reply)
                                        ↓
                               🔊 Piper TTS (speaks it back) → Speakers
```

---

## 📋 What You Need (Overview)

| Component | What it does | Where it runs |
|-----------|-------------|---------------|
| **Python 3.10+** | Runs Wilson | Your computer |
| **faster-whisper** | Converts your voice → text | GPU (fast) or CPU (slower) |
| **LM Studio** or **Ollama** | The AI brain that generates replies | Your computer |
| **Piper TTS** | Converts text → speech (Wilson's voice) | Your computer |
| **A microphone** | So Wilson can hear you | Plugged into your computer |

> **GPU is optional.** If you don't have an NVIDIA GPU, Wilson will use your CPU instead. It's slower but works fine.

---

## 🚀 Quick Start (for people who just want to run it)

Once everything is installed (see full guides below):

**Windows (PowerShell):**
```powershell
cd path\to\Wilson_v1
.\venv\Scripts\Activate.ps1
python wilson.py
```

**macOS / Linux (Terminal):**
```bash
cd path/to/Wilson_v1
source venv/bin/activate
python wilson.py
```

Press **Space** or click the button to talk. Wilson auto-stops when you go silent.

---

---

# 🪟 Full Setup Guide — Windows

Follow every step in order. Copy and paste the commands.

---

### Step 1: Install Python

1. Go to **https://www.python.org/downloads/**
2. Download **Python 3.12** or newer
3. Run the installer
4. **⚠️ IMPORTANT: Check the box that says `Add Python to PATH`** at the bottom of the installer
5. Click **Install Now**

To verify it worked, open **PowerShell** (search "PowerShell" in the Start menu) and paste:

```powershell
python --version
```

You should see something like `Python 3.12.x`. If you see an error, restart your computer and try again.

---

### Step 2: Install Git (optional but recommended)

1. Go to **https://git-scm.com/downloads/win**
2. Download and install with default settings

Then clone the project:

```powershell
cd $HOME\Documents
git clone https://github.com/YOUR_USERNAME/Wilson_v1.git
cd Wilson_v1
```

Or if you downloaded the ZIP, extract it and open PowerShell in that folder.

---

### Step 3: Create a Virtual Environment

This keeps Wilson's packages separate from the rest of your system. Paste these one at a time:

```powershell
python -m venv venv
```

```powershell
.\venv\Scripts\Activate.ps1
```

> **If you get a red error about "script execution"**, paste this first, then try again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

You'll know it worked when you see `(venv)` at the beginning of your terminal line.

---

### Step 4: Install Python Packages

Paste this entire block:

```powershell
pip install numpy sounddevice soundfile requests faster-whisper
```

**If you have an NVIDIA GPU** (GTX 1060 or newer), also install GPU acceleration:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> If you **don't** have an NVIDIA GPU, skip the torch line. Wilson will use your CPU instead (just a bit slower).

---

### Step 5: Download Piper TTS (Wilson's Voice)

1. Go to: **https://github.com/rhasspy/piper/releases**
2. Under the latest release, download **`piper_windows_amd64.zip`**
3. Extract the ZIP
4. Copy the **`piper`** folder (the one containing `piper.exe`) into your Wilson_v1 folder

Your folder should now look like this:
```
Wilson_v1/
├── wilson.py
├── piper/
│   └── piper.exe      ← This must exist
└── ...
```

---

### Step 6: Download the Voice Model

1. Go to: **https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium**
2. Download these two files:
   - `en_US-amy-medium.onnx`
   - `en_US-amy-medium.onnx.json`
3. Place **both files** directly in the `Wilson_v1/` folder (next to `wilson.py`, **NOT** inside the piper folder)

Your folder should now look like this:
```
Wilson_v1/
├── wilson.py
├── en_US-amy-medium.onnx         ← Voice model
├── en_US-amy-medium.onnx.json    ← Voice config
├── piper/
│   └── piper.exe
└── ...
```

---

### Step 7: Install a Local LLM (the AI Brain)

Wilson needs a local language model running on your computer. The easiest option on Windows is **LM Studio**.

1. Go to: **https://lmstudio.ai/**
2. Download and install LM Studio
3. Open LM Studio
4. Search for and download a model (recommendations below)
5. Load the model
6. Go to the **Developer** tab on the left
7. Click **Start Server** — it should say `localhost:1234`

**Recommended models** (pick one based on your RAM):

| Your RAM | Model | Quality |
|----------|-------|---------|
| 8 GB | Qwen 2.5 7B Q4_K_M | Good |
| 16 GB | Qwen 2.5 14B Q4_K_M | Great |
| 32 GB+ | Qwen 3 30B A3B | Excellent |

> **Alternative:** You can also use **Ollama** (https://ollama.ai). If you do, set this environment variable before running Wilson:
> ```powershell
> $env:WILSON_LLM_URL = "http://localhost:11434/v1/chat/completions"
> ```

---

### Step 8: Run Wilson

Make sure LM Studio is running with a model loaded and the server started. Then:

```powershell
cd path\to\Wilson_v1
.\venv\Scripts\Activate.ps1
python wilson.py
```

A window will appear. Click the green button or press **Space** to talk!

---

### Step 9: Verify Everything (Optional)

Run the built-in diagnostics to check all components:

```powershell
python wilson.py --check
```

This will show ✓ or ✗ for each dependency.

---

---

# 🍎 Full Setup Guide — macOS

Follow every step in order. Copy and paste the commands into **Terminal** (search "Terminal" in Spotlight).

---

### Step 1: Install Homebrew (Package Manager)

Homebrew makes it easy to install everything else. Paste this into Terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. When it's done, **close and reopen Terminal**.

---

### Step 2: Install Python and Dependencies

```bash
brew install python git
```

Verify:
```bash
python3 --version
```

---

### Step 3: Clone the Project

```bash
cd ~/Documents
git clone https://github.com/YOUR_USERNAME/Wilson_v1.git
cd Wilson_v1
```

Or download the ZIP and extract it.

---

### Step 4: Create a Virtual Environment

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

You'll know it worked when you see `(venv)` at the beginning of your terminal line.

---

### Step 5: Install Python Packages

```bash
pip install numpy sounddevice soundfile requests faster-whisper
```

> **Note:** macOS does not support NVIDIA CUDA. Wilson will automatically use your CPU. On Apple Silicon (M1/M2/M3/M4), CPU mode is still very fast.

---

### Step 6: Download Piper TTS (Wilson's Voice)

1. Go to: **https://github.com/rhasspy/piper/releases**
2. Download **`piper_macos_x64.tar.gz`** (Intel Mac) or **`piper_macos_aarch64.tar.gz`** (Apple Silicon M1+)
3. Extract it and copy the `piper` folder into your Wilson_v1 folder

> **Not sure which Mac you have?** Click the Apple menu → **About This Mac**. If it says "Apple M1/M2/M3/M4", download `aarch64`. If it says "Intel", download `x64`.

If Piper doesn't have a build for your Mac, don't worry — Wilson will automatically use the built-in macOS `say` command as a fallback voice.

---

### Step 7: Download the Voice Model

1. Go to: **https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium**
2. Download:
   - `en_US-amy-medium.onnx`
   - `en_US-amy-medium.onnx.json`
3. Place both files in the `Wilson_v1/` folder (next to `wilson.py`)

---

### Step 8: Install Ollama (the AI Brain)

Ollama is the easiest way to run a local AI model on macOS.

```bash
brew install ollama
```

Start the Ollama server:
```bash
ollama serve &
```

Download a model (pick one based on your RAM):

| Your RAM | Command | Quality |
|----------|---------|---------|
| 8 GB | `ollama pull qwen2.5:7b` | Good |
| 16 GB | `ollama pull qwen2.5:14b` | Great |
| 32 GB+ | `ollama pull qwen3:30b-a3b` | Excellent |

Example:
```bash
ollama pull qwen2.5:7b
```

---

### Step 9: Run Wilson

Make sure Ollama is running (`ollama serve &`), then:

```bash
cd ~/Documents/Wilson_v1
source venv/bin/activate
python wilson.py
```

A window will appear. Click the green button or press **Space** to talk!

> **Headless mode** (no GUI, just terminal — useful over SSH):
> ```bash
> python wilson.py --headless
> ```

---

---

# 🐧 Full Setup Guide — Linux (Ubuntu/Debian)

Follow every step in order. Copy and paste the commands into your terminal.

---

### Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk git portaudio19-dev espeak-ng
```

---

### Step 2: Clone the Project

```bash
cd ~/Documents
git clone https://github.com/YOUR_USERNAME/Wilson_v1.git
cd Wilson_v1
```

---

### Step 3: Create a Virtual Environment

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

### Step 4: Install Python Packages

```bash
pip install numpy sounddevice soundfile requests faster-whisper
```

**If you have an NVIDIA GPU**, also install GPU acceleration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> If you **don't** have an NVIDIA GPU, skip the torch line.

---

### Step 5: Download Piper TTS

1. Go to: **https://github.com/rhasspy/piper/releases**
2. Download **`piper_linux_x86_64.tar.gz`** (or `aarch64` for ARM devices like Raspberry Pi / Jetson)
3. Extract and copy the `piper` folder into your Wilson_v1 folder:

```bash
tar -xzf piper_linux_x86_64.tar.gz
cp -r piper ~/Documents/Wilson_v1/
```

Make it executable:
```bash
chmod +x ~/Documents/Wilson_v1/piper/piper
```

> If Piper isn't available for your architecture, Wilson falls back to `espeak-ng` (installed in Step 1).

---

### Step 6: Download the Voice Model

```bash
cd ~/Documents/Wilson_v1
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

---

### Step 7: Install Ollama (the AI Brain)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Start the server:
```bash
ollama serve &
```

Download a model:

```bash
ollama pull qwen2.5:7b
```

(See the model size table in the macOS section for more options.)

---

### Step 8: Run Wilson

```bash
cd ~/Documents/Wilson_v1
source venv/bin/activate
python wilson.py
```

> **Headless mode** (no GUI, just terminal):
> ```bash
> python wilson.py --headless
> ```

---

---

# ⚠️ Known Issues & Limitations

Please read before opening an issue.

| Issue | Details |
|-------|---------|
| **Whisper model downloads on first run** | The first time you launch Wilson, it downloads the Whisper model (~500 MB for `small`). This requires internet **once**. After that it's cached locally and never downloaded again. |
| **`--check` shows `—` for espeak-ng** | This is normal. espeak-ng is an optional fallback TTS — it's **not required** if Piper is installed. The `—` symbol means "optional, not found" and does **not** count as a failure. |
| **No NVIDIA GPU? No problem** | Wilson auto-detects CUDA. If no GPU is found, it falls back to CPU. Everything works, just a bit slower for transcription. |
| **AMD / Intel GPUs are not supported for acceleration** | Only NVIDIA GPUs with CUDA 12+ are supported for GPU-accelerated Whisper. AMD ROCm and Intel OneAPI are not supported by faster-whisper. CPU mode is used automatically. |
| **Python 3.10+ required** | Wilson uses features from Python 3.10 and newer. Older versions may produce errors. We recommend Python 3.12. |
| **PortAudio hang on some Windows PCs** | On rare Windows/AMD audio driver configurations, importing sounddevice can freeze briefly during device enumeration. Wilson lazy-loads sounddevice to minimize this. If it hangs on startup, try plugging in a USB microphone. |
| **Piper TTS not available on all architectures** | Piper provides pre-built binaries for Windows (x64), Linux (x64, aarch64), and macOS (x64, aarch64). If no binary exists for your platform, Wilson falls back to espeak-ng (Linux) or macOS `say`. |
| **macOS has no GPU acceleration** | Apple Silicon Macs (M1–M4) don't support NVIDIA CUDA. Wilson runs Whisper on CPU with `float32`, which is still fast on Apple Silicon. |
| **LM Studio vs Ollama default** | Windows defaults to LM Studio (port 1234). macOS and Linux default to Ollama (port 11434). You can override with `WILSON_LLM_URL` if you use a different backend. |
| **First-time antivirus warnings (Windows)** | Some antivirus software may flag `piper.exe` because it's an unsigned binary. This is a false positive — Piper is open-source. You may need to allow it through your antivirus. |

---

# ⚙️ Configuration (All Platforms)

Wilson auto-detects your platform and hardware. You can override anything with environment variables.

### Environment Variables

Set these **before** running `python wilson.py`:

**Windows (PowerShell):**
```powershell
$env:WILSON_LLM_URL = "http://localhost:11434/v1/chat/completions"
$env:WILSON_WHISPER_MODEL = "base"
```

**macOS / Linux:**
```bash
export WILSON_LLM_URL="http://localhost:11434/v1/chat/completions"
export WILSON_WHISPER_MODEL="base"
```

### All Available Settings

| Variable | Default | What it does |
|----------|---------|-------------|
| `WILSON_LLM_URL` | `localhost:1234` (Win) / `localhost:11434` (Mac/Linux) | LLM API endpoint |
| `WILSON_LLM_MODEL` | Auto-detected | Force a specific model name |
| `WILSON_WHISPER_MODEL` | `small` (desktop) / `base` (Jetson) | Whisper size: `tiny`, `base`, `small`, `medium` |
| `WILSON_WHISPER_DEVICE` | `cuda` if GPU, else `cpu` | Force `cuda` or `cpu` |
| `WILSON_WHISPER_COMPUTE` | `float16` (GPU) / `int8` (CPU) | Compute type |
| `WILSON_MAX_TOKENS` | `1024` | Max response length |
| `WILSON_LLM_TIMEOUT` | `90` | Seconds before LLM timeout |
| `WILSON_SYSTEM_PROMPT` | Built-in prompt | Custom personality for Wilson |
| `WILSON_HEADLESS` | `0` | Set to `1` for terminal-only mode (no GUI window) |

---

# 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `python is not recognized` | Reinstall Python and check **"Add to PATH"** during install. Restart your terminal. |
| `Script execution disabled` (Windows) | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `No module named X` | Make sure your venv is activated (you should see `(venv)` in your terminal), then `pip install X` |
| `Connection refused` / `Cannot connect to LM Studio` | Make sure LM Studio or Ollama is running with a model loaded and the server started |
| `No sound output` | Check your system audio settings. Make sure speakers/headphones are set as default output. |
| `Microphone not detected` | Check system privacy settings for microphone access. Try plugging in an external USB mic. |
| `CUDA warnings but still works` | Normal! Wilson fell back to CPU. Everything still works, just a bit slower. |
| `Whisper download is slow` | The model downloads on first run. It's cached after that — only slow once. |
| GUI doesn't appear (Linux) | Install tkinter: `sudo apt install python3-tk` |
| GUI doesn't appear (macOS) | Install tkinter: `brew install python-tk` |

### Run Diagnostics

The built-in check validates every component:

```bash
python wilson.py --check
```

Look for ✓ (working) and ✗ (broken) next to each component.

---

# 🎮 How to Use Wilson

1. **Start Wilson** — a window appears with a green button
2. **Click the button** (or press **Space**) to start listening
3. **Speak naturally** — Wilson shows your volume level in real-time
4. **Stop talking** — Wilson auto-detects silence after ~2 seconds and stops recording
5. **Wait** — Wilson transcribes your speech, sends it to the LLM, and speaks the reply
6. **Repeat** — the button turns green again and you can talk again
7. **Exit** — close the window or press `Ctrl+C` in the terminal

---

# 🔒 Privacy

Everything runs on your machine. Nothing is ever sent to the internet.

| Component | Runs on | Internet? |
|-----------|---------|-----------|
| Whisper (speech-to-text) | Your CPU/GPU | ❌ No |
| LLM (the AI brain) | Your CPU/GPU | ❌ No |
| Piper (text-to-speech) | Your CPU | ❌ No |
| Voice models | Local files | ❌ No |

**You can turn off WiFi and Wilson still works.**

---

# 📜 Credits

- **Piper TTS** — https://github.com/rhasspy/piper
- **faster-whisper** — https://github.com/SYSTRAN/faster-whisper
- **Whisper** — OpenAI (via CTranslate2)
- **LM Studio** — https://lmstudio.ai
- **Ollama** — https://ollama.ai

---

# 📄 License

This project is for personal use. Individual components have their own licenses:
- Piper: MIT License
- faster-whisper: MIT License
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
