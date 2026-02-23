# 📘 WILSON V1 — Complete Technical Documentation

> **Audience:** This document is written so that even a complete beginner can understand how Wilson works. Every concept is explained from the ground up. If you're a seasoned developer, feel free to skip ahead to the sections you need.

---

## Table of Contents

1. [What Is Wilson?](#1-what-is-wilson)
2. [The Big Picture — How It Works](#2-the-big-picture--how-it-works)
3. [File Structure](#3-file-structure)
4. [How the Code Is Organized](#4-how-the-code-is-organized)
5. [Section-by-Section Walkthrough](#5-section-by-section-walkthrough)
   - [5.1 Imports](#51-imports)
   - [5.2 Platform Detection](#52-platform-detection)
   - [5.3 CUDA (GPU) Detection](#53-cuda-gpu-detection)
   - [5.4 Configuration](#54-configuration)
   - [5.5 Jetson Utilities](#55-jetson-utilities)
   - [5.6 AudioRecorder](#56-audiorecorder)
   - [5.7 Transcriber (Speech-to-Text)](#57-transcriber-speech-to-text)
   - [5.8 TTSEngine (Text-to-Speech)](#58-ttsengine-text-to-speech)
   - [5.9 LLMClient (The AI Brain)](#59-llmclient-the-ai-brain)
   - [5.10 WilsonGUI (Graphical Interface)](#510-wilsongui-graphical-interface)
   - [5.11 WilsonHeadless (Terminal Interface)](#511-wilsonheadless-terminal-interface)
   - [5.12 Diagnostics](#512-diagnostics)
   - [5.13 Main Entry Point](#513-main-entry-point)
6. [The Full Pipeline — Step by Step](#6-the-full-pipeline--step-by-step)
7. [Configuration Reference](#7-configuration-reference)
8. [Platform-Specific Behavior](#8-platform-specific-behavior)
9. [Error Handling & Fallbacks](#9-error-handling--fallbacks)
10. [Glossary](#10-glossary)

---

## 1. What Is Wilson?

Wilson is a **voice assistant** that runs entirely on your computer. You talk to it, it listens, thinks of a reply, and speaks it back to you — just like Siri or Alexa, **except nothing ever leaves your machine**. No cloud, no internet, no data collection.

It's made up of three core technologies chained together:

| Step | Technology | What it does |
|------|-----------|-------------|
| 1. Listen | **faster-whisper** | Converts your voice into text (Speech-to-Text) |
| 2. Think | **LLM** (LM Studio or Ollama) | Reads the text and generates a smart reply |
| 3. Speak | **Piper TTS** | Converts the reply text back into audio (Text-to-Speech) |

Wilson supports **Windows, macOS, Linux, and NVIDIA Jetson** devices. It auto-detects your hardware and configures itself accordingly.

---

## 2. The Big Picture — How It Works

Here's what happens every time you press the button and talk:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YOU PRESS THE BUTTON                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. RECORD AUDIO                                                    │
│     Your microphone captures your voice as raw audio data.          │
│     Wilson watches for silence — after 2 seconds of quiet, it       │
│     automatically stops recording.                                  │
│     Class: AudioRecorder                                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. TRANSCRIBE (Speech → Text)                                      │
│     The audio is fed into the Whisper AI model, which converts      │
│     spoken words into written text. This runs on your GPU (fast)    │
│     or CPU (slower but always works).                                │
│     Class: Transcriber                                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. QUERY THE LLM (Text → Reply)                                    │
│     The transcribed text is sent to a local AI model (like ChatGPT  │
│     but running on your computer). It reads your question and       │
│     generates a reply.                                              │
│     Class: LLMClient                                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. SPEAK THE REPLY (Text → Audio)                                  │
│     The reply text is fed into Piper TTS, which generates a         │
│     natural-sounding voice. The audio plays through your speakers.  │
│     Class: TTSEngine                                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     READY FOR NEXT QUESTION                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

Here's what each file in the project does:

```
Wilson_v1/
│
├── wilson.py                      ← The main application (everything is in here)
├── wilson_v1_backup.py            ← A backup of the original, simpler version
│
├── en_US-amy-medium.onnx          ← The voice model file (Amy's voice data)
├── en_US-amy-medium.onnx.json     ← Configuration for the voice model
│
├── piper/                         ← Piper TTS engine folder
│   ├── piper.exe (or piper)       ← The TTS program that generates speech
│   ├── espeak-ng-data/            ← Pronunciation dictionaries for many languages
│   └── ...                        ← Supporting libraries
│
├── setup_jetson.sh                ← Setup script for NVIDIA Jetson devices
├── README.md                      ← User-facing setup guide
├── WILSON_DOCUMENTATION.md        ← This file (developer documentation)
└── venv/                          ← Python virtual environment (your packages)
```

### Key files explained:

- **wilson.py** — This is the entire application. Everything — the GUI, the audio recording, the AI pipeline, the settings — lives in this one file (~1,287 lines). This makes it easy to deploy: just copy one file.

- **en_US-amy-medium.onnx** — This is the "voice" Wilson uses to speak. It's a neural network model trained on a voice called "Amy." The `.onnx` format means it can run on any platform without needing special hardware.

- **piper/** — This folder contains the Piper program and its language data. Piper is what actually turns text into spoken audio. It reads the `.onnx` voice model and generates `.wav` audio files.

---

## 4. How the Code Is Organized

The code in `wilson.py` is organized top-to-bottom in a logical order. Here's the layout:

```
Lines    1–20     Docstring (file header with description)
Lines   22–57     Imports (loading external libraries)
Lines   59–82     Platform Detection (what OS are we on?)
Lines   84–142    CUDA Detection (do we have a GPU?)
Lines  144–215    Configuration (all settings and defaults)
Lines  217–310    JetsonMonitor (hardware monitoring for Jetson)
Lines  312–340    set_jetson_power_mode() (Jetson power management)
Lines  342–420    AudioRecorder (microphone recording)
Lines  422–545    Transcriber (speech-to-text with Whisper)
Lines  547–655    TTSEngine (text-to-speech with Piper)
Lines  657–725    LLMClient (talking to the AI model)
Lines  727–930    WilsonGUI (the graphical window)
Lines  932–1060   WilsonHeadless (terminal-only mode)
Lines 1062–1210   run_diagnostics() (the --check command)
Lines 1212–1240   print_banner() (startup info display)
Lines 1242–1287   Main entry point (where execution begins)
```

Each section is clearly marked with decorated comment headers like:
```python
# ═══════════════════════════════════════════════════════════════
#                        SECTION NAME
# ═══════════════════════════════════════════════════════════════
```

---

## 5. Section-by-Section Walkthrough

### 5.1 Imports

```
Lines 22–57
```

Wilson uses these external libraries:

| Library | What it does | Required? |
|---------|-------------|-----------|
| `threading` | Runs tasks in parallel (e.g., recording while updating the GUI) | Built-in |
| `subprocess` | Runs external programs like Piper and espeak | Built-in |
| `requests` | Sends HTTP requests to the LLM server | `pip install` |
| `numpy` | Handles audio data as arrays of numbers | `pip install` |
| `sounddevice` | Records audio from the microphone | `pip install` |
| `soundfile` | Reads/writes `.wav` audio files | `pip install` |
| `tkinter` | Creates the GUI window (buttons, text, etc.) | Built-in* |
| `faster_whisper` | The Whisper speech-to-text engine | `pip install` |

**\*Note about tkinter:** It comes with Python on Windows and macOS. On Linux, you may need to install it separately (`sudo apt install python3-tk`). If it's not available, Wilson automatically runs in terminal-only mode.

**Lazy-loading sounddevice:**
```python
_sd = None

def _get_sd():
    global _sd
    if _sd is None:
        import sounddevice
        _sd = sounddevice
    return _sd
```
This is a common pattern called "lazy loading." Instead of importing sounddevice immediately (which triggers audio device scanning and can freeze on some systems), we wait until the first time it's actually needed. The `_get_sd()` function imports it once, then returns the cached module on every subsequent call.

---

### 5.2 Platform Detection

```
Lines 59–82
```

Wilson needs to know what operating system it's running on, because many things differ between platforms (file paths, available tools, GPU support, etc.).

```python
IS_LINUX   = sys.platform.startswith("linux")    # True on any Linux
IS_WINDOWS = sys.platform == "win32"              # True on Windows
IS_MACOS   = sys.platform == "darwin"             # True on macOS
IS_ARM64   = platform.machine() in ("aarch64", "arm64")  # ARM processor?
IS_JETSON  = IS_LINUX and IS_ARM64                # NVIDIA Jetson = Linux + ARM
```

**How this works:**
- `sys.platform` is a string Python sets automatically: `"win32"`, `"darwin"`, or `"linux"`
- `platform.machine()` returns the CPU architecture: `"AMD64"`, `"x86_64"`, `"aarch64"`, or `"arm64"`
- NVIDIA Jetson devices are ARM-based Linux boards, so `IS_JETSON` is `True` when both `IS_LINUX` and `IS_ARM64` are `True`

These boolean flags are used throughout the entire codebase to make decisions like:
- Which LLM server to default to
- Whether to check for CUDA
- Which fonts to use in the GUI
- What TTS fallback to try

**Jetson model detection:**
```python
def _detect_jetson_model():
    try:
        with open("/proc/device-tree/model", "r") as f:
            return f.read().strip().rstrip("\x00")
    except Exception:
        return "Jetson (unknown)" if IS_JETSON else None
```
On Jetson devices, Linux stores the hardware name in a special file. This reads it so Wilson can display "NVIDIA Jetson Orin Nano" instead of just "Linux aarch64."

---

### 5.3 CUDA (GPU) Detection

```
Lines 84–142
```

**What is CUDA?**
CUDA is NVIDIA's technology that lets programs use the GPU (graphics card) for general computing. Whisper (speech-to-text) runs **much faster** on a GPU than a CPU. Wilson checks if CUDA is available and uses it automatically.

The function `_cuda_is_usable()` tries multiple ways to detect a GPU, in order from most reliable to least:

```
Detection Order:
1. PyTorch (torch.cuda.is_available)     ← Most reliable, if installed
2. CTranslate2 CUDA compute types        ← The Whisper backend itself
3. nvidia-cublas Python package           ← Windows pip packages
4. CUDA_PATH environment variable         ← System CUDA toolkit
5. nvidia-smi command                     ← NVIDIA driver installed
6. libcuda via ldconfig                   ← Linux shared library
```

**Why so many checks?** Because CUDA can be installed in many different ways:
- As part of PyTorch (`pip install torch`)
- As a system-wide toolkit (NVIDIA CUDA Toolkit)
- As pip packages (`nvidia-cublas-cu12`)
- Just as the NVIDIA driver (which includes basic CUDA support)

Wilson tries them all, so it works regardless of how CUDA was installed.

**macOS early exit:**
```python
if IS_MACOS:
    return False
```
macOS never has CUDA (Apple uses their own GPU technology called Metal), so we skip all checks immediately.

The result is stored in a global constant:
```python
CUDA_AVAILABLE = _cuda_is_usable()   # True or False
```
This is used later to decide whether Whisper should run on GPU or CPU.

---

### 5.4 Configuration

```
Lines 144–215
```

This section defines all of Wilson's settings. Every setting has a sensible default but can be overridden with environment variables.

**Piper TTS paths:**
```python
PIPER_EXE   = os.path.join(WILSON_DIR, "piper", "piper.exe" if IS_WINDOWS else "piper")
PIPER_VOICE = os.path.join(WILSON_DIR, "en_US-amy-medium.onnx")
```
Wilson expects the Piper binary inside a `piper/` subfolder, and the voice model next to `wilson.py`. The binary name differs on Windows (`.exe`) vs Linux/macOS (no extension).

**LLM backend selection:**
```python
if IS_JETSON:
    LLM_URL = "http://localhost:11434/v1/chat/completions"   # Ollama
elif IS_MACOS or IS_LINUX:
    LLM_URL = "http://localhost:11434/v1/chat/completions"   # Ollama
else:   # Windows
    LLM_URL = "http://localhost:1234/v1/chat/completions"    # LM Studio
```
- **Windows** defaults to LM Studio (port 1234) — it has the best Windows UI
- **macOS, Linux, Jetson** default to Ollama (port 11434) — it's the most common on those platforms
- You can override this with `WILSON_LLM_URL`

**Whisper model settings:**
```python
WHISPER_MODEL   = "small"    # or "base" on Jetson (uses less RAM)
WHISPER_DEVICE  = "cuda"     # or "cpu" if no GPU
WHISPER_COMPUTE = "float16"  # or "int8" (CPU) or "float32" (Apple Silicon)
```

**What do the Whisper model sizes mean?**

| Size | Accuracy | Speed | RAM Usage |
|------|----------|-------|-----------|
| `tiny` | Basic | Very fast | ~150 MB |
| `base` | Good | Fast | ~300 MB |
| `small` | Great | Medium | ~500 MB |
| `medium` | Excellent | Slow | ~1.5 GB |

**What are compute types?**

| Type | Meaning | When to use |
|------|---------|------------|
| `float16` | Half-precision math | GPU (fastest, good accuracy) |
| `int8` | Integer-quantized math | CPU (fast, good accuracy) |
| `float32` | Full-precision math | Apple Silicon (most compatible) |

**Other settings:**
- `SAMPLE_RATE = 16000` — Audio is recorded at 16,000 samples per second (Whisper requires this)
- `MAX_TOKENS = 1024` — Maximum length of the LLM's response
- `LLM_TIMEOUT = 90` — Wait up to 90 seconds for the LLM to respond (120 on Jetson, which is slower)
- `SYSTEM_PROMPT` — The personality instruction given to the AI ("You are Wilson, a helpful AI assistant...")
- `HEADLESS` — If `True`, Wilson runs in the terminal without a GUI window
- `FONT_UI` / `FONT_MONO` — Platform-appropriate fonts (Segoe UI on Windows, Helvetica Neue on macOS, Ubuntu on Linux)

---

### 5.5 Jetson Utilities

```
Lines 217–340
```

These are specialized tools for NVIDIA Jetson devices (small, powerful ARM computers used for AI projects). **They silently do nothing on Windows, macOS, or regular Linux.**

#### JetsonMonitor

This class reads live hardware stats from a Jetson-specific command called `tegrastats`:

```
What it monitors:
- RAM usage (e.g., "3200/8192 MB")
- GPU utilization (e.g., "45%")
- CPU & GPU temperatures (e.g., "52°C / 48°C")
- Power draw (e.g., "12.5W")
```

**How it works:**
1. Spawns `tegrastats` as a background process
2. Reads its output line by line in a separate thread (so it doesn't block the main app)
3. Parses each line with regex patterns to extract numbers
4. Stores the values in properties that the GUI reads every 60ms

**Why is it needed?** Jetson devices have limited RAM (often 8 GB shared between CPU and GPU). Monitoring lets you see if Wilson is running out of resources.

#### set_jetson_power_mode()

Jetson devices have power modes (like a laptop's "performance" vs "battery saver"):
- **MAXN** — Full power, all CPU/GPU cores at max speed
- **15W** — Medium power, lower heat
- **7W** — Low power, for battery or fanless operation

This function runs `sudo nvpmodel` and `sudo jetson_clocks` to set the desired mode.

---

### 5.6 AudioRecorder

```
Lines 342–420
```

This class handles recording audio from the microphone.

#### Finding the best microphone

Wilson doesn't just use any random microphone — it has a priority system:

```
Priority 1: Known high-quality USB mics
             (Anker PowerConf, Blue Yeti, Rode, Jabra, etc.)
             
Priority 2: Any USB audio device
             (usually better than built-in mics)
             
Priority 3: System default microphone
             (whatever your OS has selected)
             
Priority 4: First available input device
             (absolute last resort)
```

This matters because built-in laptop microphones often pick up fan noise and keyboard sounds, while USB microphones are much cleaner for speech recognition.

#### Recording process

```python
def start(self):
    self.audio_data = []           # Empty list to collect audio chunks
    self.is_recording = True
    self.stream = sounddevice.InputStream(
        samplerate=16000,          # 16 kHz (Whisper's required format)
        channels=1,                # Mono (one channel)
        dtype="float32",           # Numbers between -1.0 and 1.0
        blocksize=1024,            # Process 1024 samples at a time
        callback=self._callback,   # Function called for each chunk
    )
    self.stream.start()
```

**What's a callback?** Instead of reading audio in a loop (which would block other code), we give sounddevice a function to call every time a new chunk of audio is ready. That function just appends the chunk to our list:

```python
def _callback(self, indata, frames, time_info, status):
    if self.is_recording:
        self.audio_data.append(indata.copy())
```

When recording stops, all chunks are combined into one big array:
```python
return np.concatenate(self.audio_data, axis=0).flatten()
```

#### Volume detection

```python
def get_volume(self):
    recent = self.audio_data[-1]                      # Last chunk
    return float(np.sqrt(np.mean(recent ** 2)))       # RMS volume
```

This computes the **RMS (Root Mean Square)** volume — a standard way to measure audio loudness. The GUI uses this to update the volume bar, and the silence watchdog uses it to detect when you stop talking.

---

### 5.7 Transcriber (Speech-to-Text)

```
Lines 422–545
```

This class converts spoken audio into written text using the **Whisper** AI model (created by OpenAI, running locally via the `faster-whisper` library).

#### The fallback chain

This is one of the most important design decisions in Wilson. Instead of crashing if the preferred configuration doesn't work, it tries progressively simpler setups:

```
Attempt 1: cuda / float16    ← Best: GPU with half-precision (fast + accurate)
Attempt 2: cuda / int8       ← Good: GPU with integer math (less VRAM)
Attempt 3: cpu / int8        ← OK: CPU with integer math (slower but works)
Attempt 4: cpu / float32     ← Last resort: CPU with full precision (always works)
```

**Why does this matter?** Different GPUs support different features. Some older GPUs don't support `float16`. Some systems have CUDA installed but not enough GPU memory. By trying each option and falling back, Wilson **always loads successfully** regardless of hardware.

#### Deduplication

```python
seen = set()
unique = []
for c in combos:
    if c not in seen:
        seen.add(c)
        unique.append(c)
```

If the user's preferred config is already `cpu/int8`, we don't want to try it twice. This removes duplicates while keeping the order.

#### Transcription

```python
segments, _ = self.model.transcribe(
    audio,
    language="en",
    beam_size=3 if IS_JETSON else 5,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    ),
)
```

- **`language="en"`** — Tells Whisper to expect English (faster than auto-detecting)
- **`beam_size`** — How many possibilities to consider simultaneously (3 on Jetson to save RAM, 5 elsewhere for better accuracy)
- **`vad_filter=True`** — Voice Activity Detection: skips silent parts of the audio
- **`min_silence_duration_ms=300`** — A pause of 300ms counts as silence
- **`speech_pad_ms=200`** — Keep 200ms of audio before/after speech (avoids cutting off words)

#### Runtime CUDA recovery

```python
except Exception as e:
    err = str(e).lower()
    if any(k in err for k in ("cublas", "cuda", "cusparse", "cudnn", "gpu")):
        self._reload_cpu()
        return self.transcribe(audio)   # Retry on CPU
```

Sometimes CUDA works during loading but crashes during transcription (due to memory pressure, driver issues, etc.). When that happens, Wilson catches the error, reloads the model on CPU, and retries the transcription — all automatically, without the user noticing.

---

### 5.8 TTSEngine (Text-to-Speech)

```
Lines 547–655
```

This class makes Wilson "speak" by converting text into audio.

#### TTS priority

```
Priority 1: Piper TTS        ← Neural voice, natural sounding
Priority 2: espeak-ng         ← Robotic but universally available on Linux
Priority 3: macOS 'say'       ← Built into every Mac
Priority 4: Nothing           ← Text response only (still functional)
```

#### How Piper works

```python
proc = subprocess.Popen(
    [PIPER_EXE, "--model", PIPER_VOICE, "--output_file", tmp],
    stdin=subprocess.PIPE,
)
proc.communicate(input=text.encode("utf-8"), timeout=30)
```

1. Wilson starts the Piper program as a separate process
2. Sends the text to Piper's standard input (like typing it in)
3. Piper generates a `.wav` audio file
4. Wilson reads the `.wav` file and plays it through the speakers using `sounddevice`
5. The temporary `.wav` file is deleted

**Why a subprocess?** Piper is written in C++, not Python. Running it as a subprocess keeps things simple and avoids complex C++ bindings.

#### Text cleanup

```python
text = re.sub(r"[*#`]", "", text).replace("\n", " ").strip()
```

LLMs sometimes include markdown formatting (`*bold*`, `# heading`, `` `code` ``). These characters sound weird when spoken aloud, so Wilson strips them before sending text to TTS.

---

### 5.9 LLMClient (The AI Brain)

```
Lines 657–725
```

This class sends your transcribed speech to a local AI model and gets a reply.

#### The OpenAI-compatible API

Both LM Studio and Ollama implement the same API format as OpenAI's ChatGPT. This means Wilson's code works with either one (or any compatible server) without changes:

```python
payload = {
    "messages": self.messages[-21:],   # Last 21 messages (rolling window)
    "stream": False,                    # Get the full response at once
    "max_tokens": MAX_TOKENS,           # Limit response length
    "temperature": 0.7,                 # Creativity (0=factual, 1=creative)
}
```

#### Conversation history

```python
self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
```

Wilson maintains a conversation history (like a chat log). Each message has a role:
- **`system`** — Instructions for the AI ("You are Wilson, be concise...")
- **`user`** — What you said
- **`assistant`** — What Wilson replied

The `[-21:]` slice keeps only the last 21 messages (system prompt + 10 exchanges). This prevents the conversation from getting too large for the model to handle.

#### Thinking model support

Some AI models (like DeepSeek R1, Qwen QwQ) include their reasoning process in `<think>...</think>` tags. This is useful for complex questions but sounds terrible when spoken aloud:

```python
@staticmethod
def _strip_thinking(text):
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)  # Complete blocks
    cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned)        # Unclosed blocks
    return cleaned.strip()
```

Wilson stores the full response (including thinking) in the conversation history (so the AI has context for follow-up questions), but only speaks the final answer.

#### Error handling

```python
except requests.exceptions.ConnectionError:
    return "Cannot connect to LM Studio. Start the server on port 1234."
except requests.exceptions.Timeout:
    return "LLM timed out. Try a smaller model."
```

Instead of crashing, Wilson returns a helpful error message that gets spoken aloud, so you know what went wrong even without looking at the screen.

---

### 5.10 WilsonGUI (Graphical Interface)

```
Lines 727–930
```

This is the visual window you see when you run Wilson on a desktop.

#### Window layout (top to bottom)

```
┌──────────────────────────────────────┐
│         ◈ WILSON V1 ◈               │  ← Title
│      Offline Voice Assistant         │  ← Subtitle
├──────────────────────────────────────┤
│  ● Initializing…                     │  ← Status dot + text
│  Volume: [████████░░░░░░░]           │  ← Live volume meter
├──────────────────────────────────────┤
│                                      │
│  Platform: Windows AMD64             │
│  Loading TTS engine…                 │  ← Chat log
│  TTS: Piper                          │    (scrollable)
│  You: What's the weather like?       │
│  Wilson: I don't have access to...   │
│                                      │
├──────────────────────────────────────┤
│  🎤 CLICK TO START LISTENING         │  ← Big green button
├──────────────────────────────────────┤
│  STT: Whisper small (cuda) · LLM    │  ← Footer info
│  Spacebar or click · Auto-stops      │
└──────────────────────────────────────┘
```

#### Threading model

GUI applications have a critical rule: **never do slow work on the main thread**. If you do, the window freezes (the "Not Responding" state).

Wilson uses this pattern:
- **Main thread** — Runs the GUI (tkinter's `mainloop()`)
- **Init thread** — Loads microphone, TTS, and Whisper model (takes several seconds)
- **Watchdog thread** — Monitors silence while recording
- **Processing thread** — Runs the STT → LLM → TTS pipeline

Communication between threads happens via a **message queue**:
```python
self.msg_queue = queue.Queue()

def _log(self, tag, message):
    self.msg_queue.put((tag, message))    # Background thread adds messages

def _check_queue(self):
    while True:
        tag, msg = self.msg_queue.get_nowait()   # Main thread reads them
        # Update the chat log
    self.root.after(60, self._check_queue)    # Check again in 60ms
```

**Why not update the GUI directly from a background thread?** Because tkinter (like most GUI frameworks) is not "thread-safe." Only the main thread can modify widgets. The queue pattern safely passes data between threads.

#### Silence watchdog

```python
def _silence_watchdog(self):
    silence = 0.0
    while self.is_listening:
        time.sleep(0.1)
        if self.recorder.get_volume() < 0.01:
            silence += 0.1
        else:
            silence = 0.0
        if silence > 2.0 and len(self.recorder.audio_data) > 20:
            self.root.after(0, self._stop_listening)
            break
```

This runs in a background thread and checks the microphone volume every 100ms:
- If volume drops below 0.01 (near-silence), increment a silence counter
- If silence exceeds 2 seconds AND we have enough audio (20+ chunks ≈ 1.3 seconds), stop recording
- The `len > 20` check prevents stopping too early if there's a quiet moment before you start speaking

#### Button states

The button changes appearance to show what Wilson is doing:

| State | Button Text | Color |
|-------|------------|-------|
| Ready | 🎤 CLICK TO START LISTENING | Green |
| Listening | 🔴 LISTENING… (click to stop) | Red |
| Processing | ⌛ PROCESSING… | Orange (disabled) |

---

### 5.11 WilsonHeadless (Terminal Interface)

```
Lines 932–1060
```

For systems without a display (like a Jetson connected over SSH, or a Linux server), Wilson can run entirely in the terminal.

```
[SYS] Platform: Linux aarch64
[SYS] Loading Whisper...
[SYS] Ready. Press ENTER to record, ENTER again to stop. 'q' to quit.

[ENTER to talk | q to quit]
[SYS] Listening… (press ENTER to stop)
                                          ← (you press ENTER)
[SYS] Transcribing…
[YOU] What is the capital of France?
[SYS] Thinking…
[WILSON] The capital of France is Paris.

[ENTER to talk | q to quit] q
[SYS] Goodbye.
```

It uses ANSI color codes for readability:
- Grey for system messages
- Blue for your speech
- Green for Wilson's replies

The recording flow is simpler than the GUI: press ENTER to start, press ENTER again to stop (no automatic silence detection).

---

### 5.12 Diagnostics

```
Lines 1062–1210
```

Running `python wilson.py --check` validates every component without starting the assistant.

```
==========================================================
  WILSON V1 — System Diagnostics
==========================================================

  Platform : Windows AMD64
  Python   : 3.12.0

  Dependencies:
  ✓ numpy  (2.4.0)
  ✓ sounddevice  (0.5.3)
  ✓ soundfile  (0.13.1)
  ✓ faster-whisper
  ✓ requests  (2.32.5)

  CUDA:
  ✓ torch.cuda  (NVIDIA GeForce RTX 3060)

  TTS:
  ✓ Piper binary  (/path/to/piper)
  ✓ Voice model  (/path/to/en_US-amy-medium.onnx)
  — espeak-ng fallback  (optional — not needed if Piper is installed)

  LLM endpoint:
  ✓ API reachable  (http://localhost:1234/v1/chat/completions)

  Audio:
  ✓ Microphone  (Blue Yeti)

==========================================================
  ALL CHECKS PASSED
==========================================================
```

**Symbol meanings:**
- ✓ (green) — Working correctly
- ✗ (red) — Failed — required component is missing
- — (yellow) — Optional component not found (doesn't affect pass/fail)

The function returns `True` if all required checks pass, `False` otherwise. The main entry point uses this as the exit code: `sys.exit(0 if run_diagnostics() else 1)`.

---

### 5.13 Main Entry Point

```
Lines 1242–1287
```

This is where Python starts executing when you run `python wilson.py`:

```python
if __name__ == "__main__":
    print("Wilson starting...", flush=True)

    # --check: run diagnostics and exit
    if "--check" in sys.argv:
        sys.exit(0 if run_diagnostics() else 1)

    print_banner()

    if HEADLESS or not HAS_GUI:
        if not HAS_GUI:
            print("[NOTE] tkinter not available — running in headless mode")
        WilsonHeadless().run()
    else:
        WilsonGUI().run()
```

**Decision tree:**

```
python wilson.py --check    →  Run diagnostics, print results, exit
python wilson.py --headless →  Run in terminal mode (no window)
python wilson.py            →  Run with GUI (or headless if tkinter is missing)
```

The `if __name__ == "__main__":` guard means this code only runs when you execute the file directly — not when importing it as a module.

---

## 6. The Full Pipeline — Step by Step

Here's the **exact sequence of function calls** when you click the button and ask a question:

```
1.  YOU click the button (or press Space)
2.  WilsonGUI._toggle_listening() is called
3.  → _start_listening()
4.    → AudioRecorder.start()        — opens microphone stream
5.    → _silence_watchdog()          — starts monitoring volume in background
6.  YOU speak: "What is the capital of France?"
7.  YOU stop speaking — silence detected for 2 seconds
8.  → _stop_listening()
9.    → AudioRecorder.stop()         — closes stream, returns audio array
10.   → _process(audio) in a new thread
11.     → Transcriber.transcribe(audio)
12.       → WhisperModel.transcribe()  — returns "What is the capital of France?"
13.     → LLMClient.query("What is the capital of France?")
14.       → POST http://localhost:1234/v1/chat/completions
15.       → Response: "The capital of France is Paris."
16.       → _strip_thinking() removes any <think> blocks
17.     → TTSEngine.speak("The capital of France is Paris.")
18.       → Piper generates wilson_tts.wav
19.       → sounddevice plays the wav file through speakers
20. → _reset() — button turns green again, ready for next question
```

---

## 7. Configuration Reference

Every setting can be overridden with an environment variable:

| Variable | Default (Windows) | Default (macOS/Linux) | Default (Jetson) | Description |
|----------|-------------------|----------------------|-------------------|-------------|
| `WILSON_LLM_URL` | `localhost:1234` | `localhost:11434` | `localhost:11434` | LLM API endpoint |
| `WILSON_LLM_MODEL` | Auto | Auto | `qwen2.5:7b-instruct-q4_K_M` | Force a specific model |
| `WILSON_WHISPER_MODEL` | `small` | `small` | `base` | Whisper size |
| `WILSON_WHISPER_DEVICE` | `cuda` or `cpu` | `cpu` | `cuda` | Force GPU or CPU |
| `WILSON_WHISPER_COMPUTE` | `float16` or `int8` | `float32` or `int8` | `float16` | Math precision |
| `WILSON_MAX_TOKENS` | `1024` | `1024` | `1024` | Max reply length |
| `WILSON_LLM_TIMEOUT` | `90s` | `90s` | `120s` | Request timeout |
| `WILSON_SYSTEM_PROMPT` | Built-in | Built-in | Built-in | AI personality |
| `WILSON_HEADLESS` | `0` | `0` | `0` | `1` for terminal mode |
| `WILSON_POWER_MODE` | N/A | N/A | `MAXN` | Jetson power mode |

---

## 8. Platform-Specific Behavior

| Feature | Windows | macOS | Linux | Jetson |
|---------|---------|-------|-------|--------|
| Default LLM | LM Studio (1234) | Ollama (11434) | Ollama (11434) | Ollama (11434) |
| GPU support | CUDA (NVIDIA) | None (CPU only) | CUDA (NVIDIA) | CUDA (built-in) |
| Primary TTS | Piper | Piper | Piper | Piper |
| Fallback TTS | — | macOS `say` | espeak-ng | espeak-ng |
| GUI fonts | Segoe UI / Consolas | Helvetica Neue / Menlo | Ubuntu / Ubuntu Mono | Ubuntu / Ubuntu Mono |
| Whisper default | small / float16 | small / float32 | small / float16 | base / float16 |
| Piper binary | `piper.exe` | `piper` | `piper` | `piper` |
| Hardware monitor | No | No | No | Yes (tegrastats) |

---

## 9. Error Handling & Fallbacks

Wilson is designed to **never crash** where it can recover instead. Here's every fallback in the system:

### Whisper loading
```
cuda/float16 → cuda/int8 → cpu/int8 → cpu/float32
```
If one fails, it tries the next. If all fail, Wilson reports an error but doesn't crash.

### Whisper runtime CUDA failure
```
CUDA error during transcription → reload model on CPU → retry transcription
```
If CUDA dies mid-use (out of memory, driver issue), Wilson silently switches to CPU.

### TTS
```
Piper → espeak-ng → macOS say → no audio (text-only response)
```
If Piper's binary isn't found or crashes, Wilson falls back. On macOS, the built-in `say` command is always available.

### LLM connection
```
Connection refused → "Cannot connect to [backend]. Start the server on port [port]."
Timeout → "LLM timed out. Try a smaller model."
Other error → "LLM error: [details]"
```
Instead of crashing, Wilson speaks the error message so you know what to fix.

### GUI unavailable
```
tkinter not installed → automatically switch to headless terminal mode
```

### Microphone not found
```
Preferred mic → USB mic → system default → first available → RuntimeError
```

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| **STT** | Speech-to-Text — converting audio to written words |
| **TTS** | Text-to-Speech — converting written words to audio |
| **LLM** | Large Language Model — the AI that generates replies (like ChatGPT) |
| **CUDA** | NVIDIA's GPU computing platform |
| **Whisper** | OpenAI's speech recognition model |
| **faster-whisper** | A faster, optimized version of Whisper using CTranslate2 |
| **CTranslate2** | An inference engine that runs AI models efficiently |
| **Piper** | An open-source neural text-to-speech engine |
| **ONNX** | Open Neural Network Exchange — a model file format |
| **Ollama** | A tool for running LLMs locally |
| **LM Studio** | A desktop app for running LLMs locally (Windows/Mac) |
| **Jetson** | NVIDIA's small ARM computers designed for AI |
| **tkinter** | Python's built-in library for creating GUI windows |
| **PortAudio** | The audio library that sounddevice uses internally |
| **float16** | Half-precision floating point (uses less memory, still accurate) |
| **int8** | 8-bit integer math (uses even less memory, slightly less accurate) |
| **float32** | Full-precision floating point (uses more memory, most compatible) |
| **RMS** | Root Mean Square — a way to calculate average audio volume |
| **VAD** | Voice Activity Detection — detecting when someone is speaking |
| **beam_size** | How many word sequences Whisper considers at once (more = more accurate) |
| **daemon thread** | A background thread that automatically dies when the main program exits |
| **lazy loading** | Delaying an import/initialization until it's actually needed |
| **quantization** | Compressing an AI model to use less memory (e.g., Q4_K_M) |
| **API** | Application Programming Interface — how programs talk to each other |
| **environment variable** | A system-level setting you can change without editing code |

---

> **This document was written for the Wilson V1 codebase. If the code changes significantly, sections may need updating.**
