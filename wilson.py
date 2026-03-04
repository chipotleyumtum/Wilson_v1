"""
WILSON V1 — Offline Voice Assistant

Supports : Windows, macOS (Intel & Apple Silicon), Linux, NVIDIA Jetson
GPU      : NVIDIA CUDA auto-detected — falls back to CPU if unavailable
Pipeline : Mic → faster-whisper (STT) → LLM (Ollama | LM Studio) → Piper TTS → Speaker

Environment-variable overrides (all optional):
  WILSON_LLM_URL           LLM API endpoint
  WILSON_LLM_MODEL         Model name sent in API payload
  WILSON_WHISPER_MODEL      Whisper size: tiny | base | small | medium
  WILSON_WHISPER_COMPUTE    Compute type: float16 | int8 | float32
  WILSON_WHISPER_DEVICE     Device: cuda | cpu
  WILSON_MAX_TOKENS         Max response tokens (default 200)
  WILSON_LLM_TIMEOUT        Request timeout seconds
  WILSON_SYSTEM_PROMPT      System prompt override
  WILSON_HEADLESS           "1" for terminal-only mode (no GUI)
  WILSON_POWER_MODE         Jetson power mode: MAXN | 15W | 7W
"""

# ═══════════════════════════════════════════════════════════════════════════════
#                                IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import threading
import subprocess
import requests
import os
import sys
import platform
import tempfile
import time
import queue
import gc
import re
import random
import shutil
import math
import numpy as np

# sounddevice is lazy-loaded because PortAudio can hang during device
# enumeration on some Windows/AMD audio configurations at import time.
_sd = None

def _get_sd():
    """Lazy-import sounddevice on first use."""
    global _sd
    if _sd is None:
        import sounddevice
        _sd = sounddevice
    return _sd

try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

# ═══════════════════════════════════════════════════════════════════════════════
#                           PLATFORM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

IS_LINUX   = sys.platform.startswith("linux")
IS_WINDOWS = sys.platform == "win32"
IS_MACOS   = sys.platform == "darwin"
IS_ARM64   = platform.machine() in ("aarch64", "arm64")
IS_JETSON  = IS_LINUX and IS_ARM64


def _detect_jetson_model():
    """Read SoC model string from the device-tree (Jetson / Tegra only)."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return f.read().strip().rstrip("\x00")
    except Exception:
        return "Jetson (unknown)" if IS_JETSON else None


JETSON_MODEL = _detect_jetson_model()

# On Jetson, make sure CUDA device is visible
if IS_JETSON:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ═══════════════════════════════════════════════════════════════════════════════
#                          CUDA AVAILABILITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def _cuda_is_usable():
    """Return True if CUDA is usable for CTranslate2 / Whisper."""
    # macOS has no CUDA support (Apple Silicon uses Metal, not supported by CT2)
    if IS_MACOS:
        return False

    # Quick gate: if torch is available, trust its CUDA probe (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[CUDA] {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass

    # Check if ctranslate2 itself reports CUDA support (pip wheels bundle CUDA)
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            return True
    except Exception:
        pass

    # No torch / CT2 — look for CUDA files on disk (fast, non-blocking checks)
    if IS_WINDOWS:
        # 1) Check for pip-installed nvidia-cublas package (any CUDA version)
        try:
            import nvidia.cublas
            return True
        except ImportError:
            pass
        # 2) Check CUDA_PATH env variable (system-installed CUDA toolkit)
        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path and os.path.isdir(os.path.join(cuda_path, "bin")):
            return True
        # 3) Check nvidia-smi exists (NVIDIA driver is installed)
        #    If the driver is present, CUDA *should* work — let the Whisper
        #    fallback chain handle it if it doesn't.
        if shutil.which("nvidia-smi") is not None:
            return True
        return False

    if IS_LINUX:
        # On Jetson / Linux, check for libcuda via ldconfig (safe)
        import ctypes.util
        if ctypes.util.find_library("cuda") is None:
            return False
        return True

    return False


CUDA_AVAILABLE = _cuda_is_usable()

# ═══════════════════════════════════════════════════════════════════════════════
#                             CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WILSON_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Piper TTS ─────────────────────────────────────────────────────────────────
PIPER_EXE   = os.path.join(WILSON_DIR, "piper", "piper.exe" if IS_WINDOWS else "piper")
PIPER_VOICE = os.path.join(WILSON_DIR, "en_US-amy-medium.onnx")

# ── LLM Backend ──────────────────────────────────────────────────────────────
#  Jetson       → Ollama (default http://localhost:11434)
#  macOS / Linux → Ollama (most common local LLM on Mac/Linux)
#  Windows      → LM Studio (default http://localhost:1234)
if IS_JETSON:
    LLM_URL   = os.environ.get("WILSON_LLM_URL",   "http://localhost:11434/v1/chat/completions")
    LLM_MODEL = os.environ.get("WILSON_LLM_MODEL",  "qwen2.5:7b-instruct-q4_K_M")
elif IS_MACOS or IS_LINUX:
    LLM_URL   = os.environ.get("WILSON_LLM_URL",   "http://localhost:11434/v1/chat/completions")
    LLM_MODEL = os.environ.get("WILSON_LLM_MODEL",  "")          # Ollama auto-selects
else:   # Windows
    LLM_URL   = os.environ.get("WILSON_LLM_URL",   "http://localhost:1234/v1/chat/completions")
    LLM_MODEL = os.environ.get("WILSON_LLM_MODEL",  "")          # LM Studio auto-selects

# ── Whisper STT ───────────────────────────────────────────────────────────────
#  Jetson  → base model, float16, CUDA  (~300 MB of shared 8 GB)
#  GPU PC  → small model, float16, CUDA
#  CPU/Mac → small model, int8 (or float32 on macOS ARM for compatibility)
_default_device  = "cuda" if CUDA_AVAILABLE else "cpu"
if CUDA_AVAILABLE:
    _default_compute = "float16"
elif IS_MACOS and IS_ARM64:
    _default_compute = "float32"      # int8 can be unreliable on Apple Silicon
else:
    _default_compute = "int8"

WHISPER_MODEL   = os.environ.get("WILSON_WHISPER_MODEL",   "base"    if IS_JETSON else "small")
WHISPER_COMPUTE = os.environ.get("WILSON_WHISPER_COMPUTE",  _default_compute)
WHISPER_DEVICE  = os.environ.get("WILSON_WHISPER_DEVICE",   _default_device)

# ── General ───────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
# R1 / reasoning models use tokens for <think> blocks, so budget extra
MAX_TOKENS   = int(os.environ.get("WILSON_MAX_TOKENS",   "1024"))
LLM_TIMEOUT  = int(os.environ.get("WILSON_LLM_TIMEOUT",  "120" if IS_JETSON else "90"))

SYSTEM_PROMPT = os.environ.get(
    "WILSON_SYSTEM_PROMPT",
    "You are Wilson, a helpful AI assistant. "
    "Keep responses concise (1-3 sentences). No emojis or markdown."
)

HEADLESS = (
    os.environ.get("WILSON_HEADLESS", "0") == "1"
    or "--headless" in sys.argv
)

# ── Fonts (cross-platform) ───────────────────────────────────────────────────
if IS_LINUX:
    FONT_UI, FONT_MONO = "Ubuntu", "Ubuntu Mono"
elif IS_MACOS:
    FONT_UI, FONT_MONO = "Helvetica Neue", "Menlo"
else:   # Windows
    FONT_UI, FONT_MONO = "Segoe UI", "Consolas"


# ═══════════════════════════════════════════════════════════════════════════════
#                           JETSON UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class JetsonMonitor:
    """
    Live system metrics via tegrastats for Orin Nano / Orin NX / AGX Orin.
    Becomes a silent no-op on non-Jetson platforms.
    """

    def __init__(self):
        self.gpu_util    = 0
        self.cpu_temp    = 0.0
        self.gpu_temp    = 0.0
        self.ram_used_mb = 0
        self.ram_total_mb = 0
        self.power_mw    = 0
        self._running    = False
        self._proc       = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        if not IS_JETSON:
            return
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    # ── internal ──────────────────────────────────────────────────────────────

    def _poll_loop(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", "2000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            while self._running:
                line = self._proc.stdout.readline()
                if not line:
                    break
                self._parse(line)
            self._proc.terminate()
        except FileNotFoundError:
            pass                    # tegrastats not installed
        except Exception:
            pass

    def _parse(self, line):
        """
        Parse one tegrastats line.  Format varies by JetPack version; we
        use lenient regexes so partial matches still populate what they can.
        """
        try:
            m = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            if m:
                self.ram_used_mb  = int(m.group(1))
                self.ram_total_mb = int(m.group(2))

            m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            if m:
                self.gpu_util = int(m.group(1))

            m = re.search(r"cpu@(\d+\.?\d*)C", line, re.IGNORECASE)
            if m:
                self.cpu_temp = float(m.group(1))

            m = re.search(r"gpu@(\d+\.?\d*)C", line, re.IGNORECASE)
            if m:
                self.gpu_temp = float(m.group(1))

            m = re.search(r"VDD_IN\s+(\d+)", line)
            if m:
                self.power_mw = int(m.group(1))
        except Exception:
            pass

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def summary(self):
        if not IS_JETSON or self.ram_total_mb == 0:
            return ""
        pct = self.ram_used_mb / self.ram_total_mb * 100
        return (
            f"RAM {self.ram_used_mb}/{self.ram_total_mb}MB ({pct:.0f}%)  "
            f"GPU {self.gpu_util}%  "
            f"Temp {self.cpu_temp:.0f}/{self.gpu_temp:.0f}\u00b0C  "
            f"{self.power_mw / 1000:.1f}W"
        )

    @property
    def ram_free_mb(self):
        return max(0, self.ram_total_mb - self.ram_used_mb)


def set_jetson_power_mode(mode="MAXN"):
    """Set nvpmodel power mode and maximize clocks.  Requires sudo."""
    if not IS_JETSON:
        return False
    modes = {"MAXN": 0, "25W": 0, "15W": 1, "7W": 2}
    mid = modes.get(mode.upper(), 0)
    try:
        subprocess.run(
            ["sudo", "nvpmodel", "-m", str(mid)],
            capture_output=True, timeout=10,
        )
        subprocess.run(
            ["sudo", "jetson_clocks"],
            capture_output=True, timeout=10,
        )
        print(f"[POWER] Mode {mode} (id {mid}), clocks maximized")
        return True
    except Exception as e:
        print(f"[POWER] Could not set mode: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#                            AUDIO RECORDER
# ═══════════════════════════════════════════════════════════════════════════════

class AudioRecorder:
    """
    Records audio from the best available microphone using sounddevice.
    Works with USB mics on both Windows and Jetson.
    """

    def __init__(self):
        self.is_recording = False
        self.audio_data   = []
        self.stream       = None
        self.device_id    = self._find_device()

    def _find_device(self):
        devices = _get_sd().query_devices()

        # Priority 1: known high-quality USB mics
        preferred = [
            "anker", "powerconf", "respeaker", "seeed",
            "jabra", "blue yeti", "rode", "fifine",
        ]
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                name = d["name"].lower()
                if any(k in name for k in preferred):
                    print(f"[MIC] Preferred: {d['name']}")
                    return i

        # Priority 2: any USB audio device
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and "usb" in d["name"].lower():
                print(f"[MIC] USB: {d['name']}")
                return i

        # Priority 3: system default
        default = _get_sd().default.device[0]
        if default is not None and default >= 0:
            print(f"[MIC] Default: {_get_sd().query_devices(default)['name']}")
            return default

        # Priority 4: first available input
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                print(f"[MIC] Fallback: {d['name']}")
                return i

        raise RuntimeError("No microphone found")

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    # ── public ────────────────────────────────────────────────────────────────

    def start(self):
        self.audio_data = []
        self.is_recording = True
        self.stream = _get_sd().InputStream(
            device=self.device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.audio_data:
            return None
        return np.concatenate(self.audio_data, axis=0).flatten()

    def get_volume(self):
        if not self.audio_data:
            return 0.0
        recent = self.audio_data[-1]
        return float(np.sqrt(np.mean(recent ** 2)))


# ═══════════════════════════════════════════════════════════════════════════════
#                         WHISPER TRANSCRIBER
# ═══════════════════════════════════════════════════════════════════════════════

class Transcriber:
    """
    Speech-to-text via faster-whisper (CTranslate2 backend).
    Falls through a chain of device/compute combos so it always loads.
    On Jetson Orin Nano: base + float16 + CUDA ≈ 300 MB shared RAM.
    """

    def __init__(self):
        self.model = None

    def load(self, callback=None):
        def log(msg):
            if callback:
                callback(msg)
            print(f"[WHISPER] {msg}")

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log("ERROR: faster-whisper not installed.  pip install faster-whisper")
            return False

        log(f"Loading '{WHISPER_MODEL}' (target: {WHISPER_DEVICE}/{WHISPER_COMPUTE}) ...")

        # Fallback chain — try progressively lighter configs
        combos = [
            (WHISPER_DEVICE, WHISPER_COMPUTE),
            ("cuda", "float16"),
            ("cuda", "int8"),
            ("cpu", "int8"),
            ("cpu", "float32"),
        ]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in combos:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        for device, compute in unique:
            try:
                self.model = WhisperModel(
                    WHISPER_MODEL,
                    device=device,
                    compute_type=compute,
                )
                log(f"Loaded on {device.upper()} ({compute})")
                self._post_load_cleanup()
                return True
            except Exception as e:
                log(f"  {device}/{compute} failed: {e}")

        log("ERROR: All Whisper backends failed")
        return False

    def transcribe(self, audio):
        if self.model is None:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
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
            text = " ".join(s.text for s in segments).strip()
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("cublas", "cuda", "cusparse", "cudnn", "gpu")):
                print(f"[WHISPER] CUDA failed at runtime: {e}")
                print("[WHISPER] Reloading model on CPU...")
                self._reload_cpu()
                return self.transcribe(audio)       # retry once on CPU
            raise

        # Jetson: free any cached CUDA allocations after transcription
        if IS_JETSON:
            gc.collect()

        return text

    def _reload_cpu(self):
        """Emergency fallback: reload model on CPU if CUDA dies at runtime."""
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                WHISPER_MODEL, device="cpu", compute_type="int8",
            )
            print("[WHISPER] Reloaded on CPU (int8)")
        except Exception as e2:
            print(f"[WHISPER] CPU reload also failed: {e2}")
            self.model = None

    @staticmethod
    def _post_load_cleanup():
        """Release unused allocations after model load (important for 8 GB unified)."""
        if IS_JETSON:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#                              TTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    """
    Text-to-speech via Piper (ONNX).
    Falls back to espeak-ng when Piper is unavailable (common on first
    Jetson boot before running setup_jetson.sh).
    """

    def __init__(self):
        self.has_piper  = os.path.isfile(PIPER_EXE)
        self.has_espeak = (
            shutil.which("espeak-ng") is not None
            or shutil.which("espeak") is not None
        )
        # macOS ships with the 'say' command — use it as a last-resort fallback
        self.has_say = IS_MACOS and shutil.which("say") is not None

        if self.has_piper:
            # Ensure +x on Linux / macOS
            if not IS_WINDOWS:
                try:
                    os.chmod(PIPER_EXE, 0o755)
                except OSError:
                    pass
            print(f"[TTS] Piper @ {PIPER_EXE}")
        elif self.has_espeak:
            print("[TTS] Piper not found — using espeak-ng fallback")
        elif self.has_say:
            print("[TTS] Piper not found — using macOS 'say' fallback")
        else:
            print("[TTS] WARNING: No TTS engine available")

    def speak(self, text, log_fn=None):
        text = re.sub(r"[*#`]", "", text).replace("\n", " ").strip()
        if not text:
            return
        if self.has_piper:
            self._piper(text, log_fn)
        elif self.has_espeak:
            self._espeak(text, log_fn)
        elif self.has_say:
            self._say(text, log_fn)
        else:
            if log_fn:
                log_fn("No TTS engine available")

    def generate_wav(self, text, log_fn=None):
        """Generate TTS audio and return (numpy_array, sample_rate) without playing.
        Returns (None, None) if generation fails or Piper is unavailable."""
        text = re.sub(r"[*#`]", "", text).replace("\n", " ").strip()
        if not text or not self.has_piper:
            return None, None
        tmp = os.path.join(tempfile.gettempdir(), "wilson_tts_gen.wav")
        try:
            proc = subprocess.Popen(
                [PIPER_EXE, "--model", PIPER_VOICE, "--output_file", tmp],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.communicate(input=text.encode("utf-8"), timeout=30)
            proc.wait()
            time.sleep(0.05)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                import soundfile as sf
                data, sr = sf.read(tmp)
                return data, sr
        except Exception as e:
            if log_fn:
                log_fn(f"TTS generate error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return None, None

    # ── Piper ─────────────────────────────────────────────────────────────────

    def _piper(self, text, log_fn=None):
        tmp = os.path.join(tempfile.gettempdir(), "wilson_tts.wav")
        try:
            proc = subprocess.Popen(
                [PIPER_EXE, "--model", PIPER_VOICE, "--output_file", tmp],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.communicate(input=text.encode("utf-8"), timeout=30)
            proc.wait()
            time.sleep(0.05)

            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                import soundfile as sf
                data, sr = sf.read(tmp)
                _get_sd().play(data, sr)
                _get_sd().wait()
        except subprocess.TimeoutExpired:
            if log_fn:
                log_fn("TTS timed out")
            try:
                proc.kill()
            except Exception:
                pass
        except FileNotFoundError:
            if log_fn:
                log_fn("Piper binary not found — switching to fallback TTS")
            self.has_piper = False
            if self.has_espeak:
                self._espeak(text, log_fn)
            elif self.has_say:
                self._say(text, log_fn)
        except Exception as e:
            if log_fn:
                log_fn(f"TTS error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ── espeak-ng fallback ────────────────────────────────────────────────────

    def _espeak(self, text, log_fn=None):
        cmd = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
        try:
            subprocess.run(
                [cmd, "-v", "en", "-s", "160", "--", text],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            if log_fn:
                log_fn(f"espeak error: {e}")

    # ── macOS 'say' fallback ──────────────────────────────────────────────────

    def _say(self, text, log_fn=None):
        try:
            subprocess.run(
                ["say", "-v", "Samantha", text],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            if log_fn:
                log_fn(f"macOS say error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#                              LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    Talks to any OpenAI-compatible chat completions API.
      Jetson  → Ollama   (localhost:11434)
      Windows → LM Studio (localhost:1234)
    """

    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @staticmethod
    def _strip_thinking(text):
        """Remove <think>...</think> blocks emitted by reasoning models
        (DeepSeek R1, QwQ, etc.) so only the final answer is spoken aloud."""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
        # Also handle unclosed <think> blocks (model still reasoning at max_tokens)
        cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned)
        return cleaned.strip()

    def query(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "messages": self.messages[-21:],       # rolling context window
            "stream": False,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }
        if LLM_MODEL:
            payload["model"] = LLM_MODEL

        try:
            resp = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            raw_reply = resp.json()["choices"][0]["message"]["content"]
            # Store full reply in history (thinking context helps future turns)
            self.messages.append({"role": "assistant", "content": raw_reply})
            # Return only the visible answer (strip reasoning blocks)
            return self._strip_thinking(raw_reply)
        except requests.exceptions.ConnectionError:
            backend = "LM Studio" if IS_WINDOWS else "Ollama"
            port    = "1234"      if IS_WINDOWS else "11434"
            return f"Cannot connect to {backend}. Start the server on port {port}."
        except requests.exceptions.Timeout:
            return (
                "LLM timed out. Model may be too large for available memory. "
                "Try a smaller quantisation (Q4_K_S) or reduce context length."
            )
        except Exception as e:
            return f"LLM error: {e}"

    def clear_history(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


# ═══════════════════════════════════════════════════════════════════════════════
#                          WILSON — GUI MODE
# ═══════════════════════════════════════════════════════════════════════════════

class WilsonGUI:
    """Tkinter-based graphical interface. Works on both HDMI-out Jetson and
    Windows desktop. Jetson-specific widgets show live system telemetry.
    Hardened for resilience, localized, and optimized for performance."""

    THEME = {
        "bg_main": "#1e1e2e",         # Modern dark (Catppuccin Macchiato based)
        "bg_panel": "#181825",
        "fg_title": "#89b4fa",        # Blue accent
        "fg_text": "#cad3f5",         # Main text
        "fg_sub": "#a5adcb",          # Subtext
        "accent_ready": "#a6da95",    # Green
        "accent_ready_hover": "#8bd5ca",
        "accent_listen": "#ed8796",   # Red
        "accent_process": "#f5a97f",  # Orange
        "color_user": "#8aa2f8",
        "color_wilson": "#a6da95",
        "color_system": "#5b6078",
    }

    I18N = {
        "title": "WILSON V1",
        "subtitle": "Offline Voice Assistant",
        "btn_ready": "🎤 CLICK TO START LISTENING",
        "btn_listen": "🔴 LISTENING… (click to stop)",
        "btn_process": "⌛ PROCESSING…",
        "status_init": "Initializing…",
        "status_ready": "Ready",
        "status_dictate": "Listening…",
        "status_process": "Processing…",
        "status_transcribe": "Transcribing…",
        "status_think": "Thinking…",
        "status_speak": "Speaking…",
        "status_error": "Error",
        "telemetry_load": "Telemetry loading…",
        "volume": "Volume:",
        "footer_hints": "Spacebar or click • Auto-stops on silence"
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.I18N["title"])
        
        # Transparent Circular Window setup
        self._transparent_key = "#000001"
        self.root.geometry("700x700")
        self.root.configure(bg=self._transparent_key)
        
        if IS_WINDOWS:
            self.root.attributes("-transparentcolor", self._transparent_key)
            self.root.overrideredirect(True) # Remove titlebar
            # Bind ESC to quit because native close button is gone
            self.root.bind("<Escape>", lambda e: self.shutdown())

        # Dragging variables
        self._drag_x = 0
        self._drag_y = 0

        self.is_listening  = False
        self.is_processing = False
        self.recorder      = None
        self.transcriber   = None
        self.tts           = None
        self.llm           = LLMClient()
        self.monitor       = JetsonMonitor()
        self.msg_queue     = queue.Queue()

        # Matrix text animation state
        self._matrix_busy = False

        # Mouth / face animation state
        self._mouth_active = False
        self._mouth_data = []
        self._mouth_openness = 0.0
        self._mouth_start = 0.0
        self._mouth_duration = 0.0
        self._eye_open = True

        # Cache formatting elements for optimization
        self._font_title = (FONT_UI, 24, "bold")
        self._font_sub = (FONT_UI, 10)
        self._font_mono = (FONT_MONO, 10)
        self._font_btn = (FONT_UI, 13, "bold")

        self._build_ui()
        self._check_queue()
        threading.Thread(target=self._initialize, daemon=True).start()

    # ── window dragging ───────────────────────────────────────────────────────
    
    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        x = self.root.winfo_pointerx() - self._drag_x
        y = self.root.winfo_pointery() - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        t = self.THEME
        i18n = self.I18N

        # Draw transparent background with circular shape
        self.bg_canvas = tk.Canvas(self.root, bg=self._transparent_key, highlightthickness=0)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create outer circle with breathing room
        self.bg_canvas.create_oval(25, 25, 675, 675, fill=t["bg_main"], outline=t["fg_title"], width=3)
        
        # Bind left-click and drag events to canvas for window movement
        self.bg_canvas.bind("<ButtonPress-1>", self._start_drag)
        self.bg_canvas.bind("<B1-Motion>", self._on_drag)

        # Central container restricted closely inside the circle to avoid cutoff
        self.main_container = tk.Frame(self.root, bg=t["bg_main"])
        self.main_container.place(relx=0.5, rely=0.5, anchor="center", width=480, height=570)

        subtitle = JETSON_MODEL if IS_JETSON else i18n["subtitle"]

        # Header with drag & close actions explicitly mapped
        top_frame = tk.Frame(self.main_container, bg=t["bg_main"])
        top_frame.pack(fill=tk.X, pady=(0, 5))
        
        close_btn = tk.Label(top_frame, text="✕", font=(FONT_UI, 14, "bold"), fg=t["accent_listen"], bg=t["bg_main"], cursor="hand2")
        close_btn.pack(side=tk.RIGHT, padx=5)
        close_btn.bind("<Button-1>", lambda e: self.shutdown())

        tk.Label(
            self.main_container, text=f"◈ {i18n['title']} ◈",
            font=self._font_title,
            fg=t["fg_title"], bg=t["bg_main"],
        ).pack(pady=(0, 5))

        tk.Label(
            self.main_container, text=subtitle,
            font=self._font_sub,
            fg=t["fg_sub"], bg=t["bg_main"],
        ).pack()

        # ── Animated Face (eyes + mouth) ──────────────────────────────────
        self.face_canvas = tk.Canvas(
            self.main_container, width=200, height=100,
            bg=t["bg_main"], highlightthickness=0,
        )
        self.face_canvas.pack(pady=(8, 2))
        self._draw_face()
        self._schedule_blink()

        # Jetson telemetry bar
        if IS_JETSON:
            self.stats_label = tk.Label(
                self.main_container, text=i18n["telemetry_load"],
                font=(FONT_MONO, 9),
                fg=t["fg_sub"], bg=t["bg_panel"],
                anchor="w", padx=10, pady=5,
            )
            self.stats_label.pack(fill=tk.X, padx=20, pady=(10, 0))

        # Status indicator
        sf = tk.Frame(self.main_container, bg=t["bg_main"])
        sf.pack(pady=10)

        self.status_dot = tk.Label(
            sf, text="●", font=(FONT_UI, 18),
            fg=t["accent_process"], bg=t["bg_main"],
        )
        self.status_dot.pack(side=tk.LEFT)

        self.status_text = tk.Label(
            sf, text=i18n["status_init"],
            font=(FONT_UI, 12), fg=t["fg_text"], bg=t["bg_main"],
        )
        self.status_text.pack(side=tk.LEFT, padx=(8, 0))

        # Volume meter
        vf = tk.Frame(self.main_container, bg=t["bg_main"])
        vf.pack(pady=(0, 10))

        tk.Label(
            vf, text=i18n["volume"], font=(FONT_UI, 10),
            fg=t["fg_sub"], bg=t["bg_main"],
        ).pack(side=tk.LEFT)

        self.volume_bar = ttk.Progressbar(
            vf, length=180, mode="determinate", maximum=100,
        )
        self.volume_bar.pack(side=tk.LEFT, padx=8)

        # Chat log (Robust text overflow handling via wrap)
        cf = tk.Frame(self.main_container, bg=t["bg_panel"], padx=3, pady=3)
        cf.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.chat = scrolledtext.ScrolledText(
            cf, font=self._font_mono,
            bg=t["bg_main"], fg=t["fg_text"],
            wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, padx=12, pady=12,
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.tag_configure("user",   foreground=t["color_user"])
        self.chat.tag_configure("wilson", foreground=t["color_wilson"], spacing1=4, spacing3=4)
        self.chat.tag_configure("system", foreground=t["color_system"], justify=tk.CENTER)

        # Listen button (Delightr: larger, more padding, distinct colors)
        self.btn = tk.Button(
            self.main_container,
            text=i18n["btn_ready"],
            font=self._font_btn,
            bg=t["bg_panel"], fg=t["accent_ready"],
            activebackground=t["bg_panel"], activeforeground=t["accent_ready_hover"],
            highlightthickness=1, highlightbackground=t["accent_ready"],
            relief=tk.FLAT, width=28, height=2,
            cursor="hand2",
            command=self._toggle_listening,
        )
        self.btn.pack(pady=10)
        self.root.bind("<space>", lambda e: self._toggle_listening())

        # Footer
        _llm_name = "LM Studio" if IS_WINDOWS else "Ollama"
        cfg = (
            f"STT: Whisper {WHISPER_MODEL} ({WHISPER_DEVICE})  ·  "
            f"LLM: {_llm_name}"
        )
        tk.Label(
            self.main_container, text=cfg,
            font=(FONT_UI, 8), fg=t["color_system"], bg=t["bg_main"],
        ).pack()

        tk.Label(
            self.main_container,
            text=i18n["footer_hints"] + " • ESC to quit",
            font=(FONT_UI, 9), fg=t["color_system"], bg=t["bg_main"],
        ).pack(pady=(2, 10))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text, color=None):
        if color is None: color = self.THEME["fg_text"]
        # Delightr / Harden: ensure updates are thread-safe and resilient
        try:
            self.status_dot.config(fg=color)
            self.status_text.config(text=text)
        except tk.TclError:
            pass  # safely ignore if window is destroyed

    def _fade_button_color(self, target_fg, target_bg=None):
        """Delightr: simple state application (can be expanded to real fades)"""
        try:
            bg = target_bg or self.THEME["bg_panel"]
            self.btn.config(fg=target_fg, bg=bg, activeforeground=target_fg)
        except tk.TclError:
            pass

    def _log(self, tag, message):
        self.msg_queue.put((tag, message))

    # ── Matrix text animation ─────────────────────────────────────────────

    def _check_queue(self):
        """Drain message queue via Matrix-style typing. Resilient polling."""
        # Only pop a new message if not currently typing one out
        if not self._matrix_busy:
            try:
                tag, msg = self.msg_queue.get_nowait()
                self._matrix_busy = True
                if tag == "user":
                    self._matrix_type("\nYou: ", msg, "user")
                elif tag == "wilson":
                    self._matrix_type("\nWilson: ", msg, "wilson")
                else:
                    self._matrix_type("\n", msg, "system")
            except queue.Empty:
                pass

        # Volume
        try:
            if self.is_listening and self.recorder:
                current = self.volume_bar["value"]
                target = min(100, self.recorder.get_volume() * 500)
                self.volume_bar["value"] = current * 0.3 + target * 0.7
            else:
                self.volume_bar["value"] = 0
        except tk.TclError:
            pass

        # Jetson telemetry
        try:
            if IS_JETSON and hasattr(self, "stats_label"):
                s = self.monitor.summary
                if s:
                    self.stats_label.config(text=s)
        except Exception:
            pass

        try:
            self.root.after(50, self._check_queue)
        except tk.TclError:
            pass

    def _matrix_type(self, prefix, text, tag, idx=0):
        """Type text character-by-character, Matrix rain style."""
        try:
            self.chat.config(state=tk.NORMAL)
            if idx == 0:
                self.chat.insert(tk.END, prefix, tag)
            if idx < len(text):
                self.chat.insert(tk.END, text[idx], tag)
                self.chat.see(tk.END)
                self.chat.config(state=tk.DISABLED)
                # Speed profile: wilson=dramatic, system=fast, user=medium
                ch = text[idx]
                if tag == "wilson":
                    delay = 6 if ch in ' \n' else 22
                elif tag == "user":
                    delay = 4 if ch in ' \n' else 12
                else:
                    delay = 1  # system messages appear near-instantly
                self.root.after(delay, self._matrix_type, prefix, text, tag, idx + 1)
            else:
                self.chat.insert(tk.END, "\n", tag)
                self.chat.config(state=tk.DISABLED)
                self.chat.see(tk.END)
                self._matrix_busy = False
        except tk.TclError:
            self._matrix_busy = False

    # ── Animated Face / Mouth ─────────────────────────────────────────────

    def _draw_face(self):
        """Draw eyes and mouth on the face canvas."""
        c = self.face_canvas
        t = self.THEME
        c.delete("face")

        cx, cy_eyes = 100, 28
        eye_sep = 35

        # ── Eyes ──
        if self._eye_open:
            # Open eyes (iris ring + dark pupil)
            for ex in (cx - eye_sep, cx + eye_sep):
                c.create_oval(ex - 11, cy_eyes - 11, ex + 11, cy_eyes + 11,
                              fill=t["fg_title"], outline=t["fg_title"], tags="face")
                c.create_oval(ex - 5, cy_eyes - 5, ex + 5, cy_eyes + 5,
                              fill=t["bg_main"], outline=t["bg_main"], tags="face")
        else:
            # Closed eyes (horizontal lines)
            for ex in (cx - eye_sep, cx + eye_sep):
                c.create_line(ex - 11, cy_eyes, ex + 11, cy_eyes,
                              fill=t["fg_title"], width=3, tags="face")

        # ── Mouth ──
        cx_m, cy_m = 100, 72
        mouth_w = 28
        openness = self._mouth_openness

        if openness < 0.05:
            # Closed: gentle smile arc
            c.create_arc(cx_m - mouth_w, cy_m - 10, cx_m + mouth_w, cy_m + 14,
                         start=200, extent=140, style=tk.ARC,
                         outline=t["accent_ready"], width=3, tags="face")
        else:
            # Open: ellipse grows taller with amplitude
            h = int(4 + openness * 22)
            c.create_oval(cx_m - mouth_w, cy_m - h, cx_m + mouth_w, cy_m + h,
                          fill="#1a0808", outline=t["accent_ready"], width=2, tags="face")
            # Tongue hint when mouth is wide
            if openness > 0.55:
                tw = int(10 + openness * 6)
                c.create_arc(cx_m - tw, cy_m + 2, cx_m + tw, cy_m + h + 4,
                             start=0, extent=180,
                             fill="#c05555", outline="", tags="face")

    def _schedule_blink(self):
        """Random eye blink loop."""
        if not self._mouth_active:
            self._eye_open = False
            self._draw_face()
            try:
                self.root.after(150, self._open_eyes)
            except tk.TclError:
                return
        delay = random.randint(2500, 5500)
        try:
            self.root.after(delay, self._schedule_blink)
        except tk.TclError:
            pass

    def _open_eyes(self):
        self._eye_open = True
        try:
            self._draw_face()
        except tk.TclError:
            pass

    def _compute_mouth_amplitudes(self, audio_data, sr, fps=30):
        """Pre-compute per-frame RMS amplitudes for mouth sync."""
        samples_per_frame = max(1, sr // fps)
        amplitudes = []
        for i in range(0, len(audio_data), samples_per_frame):
            chunk = audio_data[i:i + samples_per_frame]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            # Scale to 0-1 range; speech RMS ~0.02-0.15 → multiply to fill range
            amplitudes.append(min(1.0, rms * 10))
        return amplitudes

    def _start_mouth_anim(self, amplitudes, duration):
        """Begin mouth animation synced to TTS audio. Called on main thread."""
        self._mouth_data = amplitudes
        self._mouth_start = time.time()
        self._mouth_duration = duration
        self._mouth_active = True
        self._eye_open = True
        self._tick_mouth()

    def _tick_mouth(self):
        """30fps mouth update driven by pre-computed amplitude envelope."""
        if not self._mouth_active:
            self._mouth_openness = 0.0
            self._draw_face()
            return
        elapsed = time.time() - self._mouth_start
        if elapsed >= self._mouth_duration or not self._mouth_data:
            self._stop_mouth_anim()
            return
        # Map elapsed time to amplitude array index
        progress = elapsed / self._mouth_duration
        idx = int(progress * len(self._mouth_data))
        idx = min(idx, len(self._mouth_data) - 1)
        self._mouth_openness = self._mouth_data[idx]
        self._draw_face()
        try:
            self.root.after(33, self._tick_mouth)  # ~30 fps
        except tk.TclError:
            pass

    def _stop_mouth_anim(self):
        """Return mouth to idle smile."""
        self._mouth_active = False
        self._mouth_openness = 0.0
        try:
            self._draw_face()
        except tk.TclError:
            pass

    # ── initialization ────────────────────────────────────────────────────────

    def _initialize(self):
        try:
            if IS_JETSON:
                self._log("system", f"Platform: {JETSON_MODEL}")
                pm = os.environ.get("WILSON_POWER_MODE", "MAXN")
                set_jetson_power_mode(pm)
                self.monitor.start()
            else:
                self._log("system", f"Platform: {platform.system()} {platform.machine()}")

            self._log("system", "Initializing microphone…")
            self.recorder = AudioRecorder()

            self._log("system", "Loading TTS engine…")
            self.tts = TTSEngine()
            tts_name = "Piper" if self.tts.has_piper else ("espeak-ng" if self.tts.has_espeak else "NONE")
            self._log("system", f"TTS: {tts_name}")

            self.transcriber = Transcriber()
            self.transcriber.load(callback=lambda m: self._log("system", m))

            if IS_JETSON:
                gc.collect()

            self._log("system", "Wilson is ready. Click the button or press Space.")
            self.root.after(0, lambda: self._set_status(self.I18N["status_ready"], self.THEME["accent_ready"]))
            self.root.after(0, lambda: self._fade_button_color(self.THEME["accent_ready"]))

        except Exception as e:
            self._log("system", f"Init error: {e}")
            self.root.after(0, lambda: self._set_status(self.I18N["status_error"], self.THEME["accent_listen"]))

    # ── listening controls ────────────────────────────────────────────────────

    def _toggle_listening(self):
        if self.is_processing:
            return
        if self.transcriber is None or self.transcriber.model is None:
            self._log("system", "Still loading, please wait…")
            return
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        self.is_listening = True
        self.btn.config(text=self.I18N["btn_listen"])
        self._fade_button_color(self.THEME["accent_listen"])
        self._set_status(self.I18N["status_dictate"], self.THEME["accent_listen"])
        self.recorder.start()
        threading.Thread(target=self._silence_watchdog, daemon=True).start()

    def _stop_listening(self):
        if not self.is_listening:
            return
        self.is_listening = False
        self.is_processing = True
        self.btn.config(text=self.I18N["btn_process"], state=tk.DISABLED)
        self._fade_button_color(self.THEME["accent_process"])
        self._set_status(self.I18N["status_process"], self.THEME["accent_process"])
        audio = self.recorder.stop()
        if audio is not None and len(audio) > SAMPLE_RATE * 0.5:
            threading.Thread(target=self._process, args=(audio,), daemon=True).start()
        else:
            self._log("system", "Recording too short")
            self._reset()

    def _silence_watchdog(self):
        """Auto-stop after 2 s of continuous silence (once we have ≥20 chunks)."""
        silence = 0.0
        while self.is_listening:
            time.sleep(0.1)
            # Resilient check if window still running
            try:
                if self.recorder.get_volume() < 0.01:
                    silence += 0.1
                else:
                    silence = 0.0
                if silence > 2.0 and len(self.recorder.audio_data) > 20:
                    self.root.after(0, self._stop_listening)
                    break
            except Exception:
                break

    # ── processing pipeline ───────────────────────────────────────────────────

    def _process(self, audio):
        try:
            # STT
            self.root.after(0, lambda: self._set_status(self.I18N["status_transcribe"], self.THEME["accent_process"]))
            text = self.transcriber.transcribe(audio)

            if not text or len(text.strip()) < 2:
                self._log("system", "Couldn't understand. Speak louder / closer to the mic.")
                self.root.after(0, self._reset)
                return

            self._log("user", text.strip())

            # LLM
            self.root.after(0, lambda: self._set_status(self.I18N["status_think"], self.THEME["accent_process"]))
            response = self.llm.query(text)
            self._log("wilson", response)

            # TTS with mouth animation
            self.root.after(0, lambda: self._set_status(self.I18N["status_speak"], self.THEME["accent_ready"]))
            audio_data, sr = self.tts.generate_wav(response, log_fn=lambda m: self._log("system", m))
            if audio_data is not None and sr is not None:
                amps = self._compute_mouth_amplitudes(audio_data, sr)
                duration = len(audio_data) / sr
                self.root.after(0, lambda a=amps, d=duration: self._start_mouth_anim(a, d))
                _get_sd().play(audio_data, sr)
                _get_sd().wait()
                self.root.after(0, self._stop_mouth_anim)
            else:
                self.tts.speak(response, log_fn=lambda m: self._log("system", m))

        except Exception as e:
            self._log("system", f"Pipeline error: {e}")
        finally:
            self.root.after(0, self._reset)

    def _reset(self):
        self.is_processing = False
        try:
            self.btn.config(text=self.I18N["btn_ready"], state=tk.NORMAL)
            self._fade_button_color(self.THEME["accent_ready"])
            self._set_status(self.I18N["status_ready"], self.THEME["accent_ready"])
        except tk.TclError:
            pass

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        self.is_listening = False
        try:
            self.monitor.stop()
        except Exception:
            pass
            
        if self.recorder and self.recorder.stream:
            try:
                self.recorder.stream.close()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#                       WILSON — HEADLESS / CLI MODE
# ═══════════════════════════════════════════════════════════════════════════════

class WilsonHeadless:
    """
    Terminal-only interface for headless Jetson deployments (SSH, systemd, etc.).
    Uses ANSI colour codes for readability.
    """

    COLORS = {
        "system": "\033[90m",       # grey
        "user":   "\033[94m",       # blue
        "wilson": "\033[92m",       # green
        "reset":  "\033[0m",
    }

    def __init__(self):
        self.recorder    = None
        self.transcriber = None
        self.tts         = None
        self.llm         = LLMClient()
        self.monitor     = JetsonMonitor()

    def _print(self, tag, msg):
        c = self.COLORS.get(tag, "")
        r = self.COLORS["reset"]
        prefix = {"system": "[SYS]", "user": "[YOU]", "wilson": "[WILSON]"}.get(tag, "")
        print(f"{c}{prefix} {msg}{r}")

    # ── init ──────────────────────────────────────────────────────────────────

    def _initialize(self):
        if IS_JETSON:
            self._print("system", f"Platform: {JETSON_MODEL}")
            set_jetson_power_mode(os.environ.get("WILSON_POWER_MODE", "MAXN"))
            self.monitor.start()
        else:
            self._print("system", f"Platform: {platform.system()} {platform.machine()}")

        self._print("system", "Initializing microphone\u2026")
        self.recorder = AudioRecorder()

        self._print("system", "Loading TTS\u2026")
        self.tts = TTSEngine()

        self.transcriber = Transcriber()
        self.transcriber.load(callback=lambda m: self._print("system", m))

        if IS_JETSON:
            gc.collect()

        self._print("system", "Ready.  Press ENTER to record, ENTER again to stop.  'q' to quit.")

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._initialize()

        while True:
            try:
                cmd = input("\n[ENTER to talk | q to quit] ").strip().lower()
                if cmd in ("q", "quit", "exit"):
                    break

                self._print("system", "Listening\u2026 (press ENTER to stop)")
                self.recorder.start()
                input()                             # block until user hits ENTER
                audio = self.recorder.stop()

                if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                    self._print("system", "Too short — try again.")
                    continue

                self._print("system", "Transcribing\u2026")
                text = self.transcriber.transcribe(audio)
                if not text or len(text.strip()) < 2:
                    self._print("system", "Couldn't understand.")
                    continue

                self._print("user", text.strip())

                self._print("system", "Thinking\u2026")
                response = self.llm.query(text)
                self._print("wilson", response)

                self.tts.speak(response, log_fn=lambda m: self._print("system", m))

                if IS_JETSON:
                    gc.collect()
                    s = self.monitor.summary
                    if s:
                        self._print("system", s)

            except KeyboardInterrupt:
                print()
                break
            except EOFError:
                break

        self.monitor.stop()
        self._print("system", "Goodbye.")


# ═══════════════════════════════════════════════════════════════════════════════
#                          SYSTEM DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnostics():
    """Invoked with --check.  Validates every dependency without starting the
    assistant, so you can confirm the Jetson (or Windows) environment is sane."""

    ok = True

    def check(label, result, detail=""):
        nonlocal ok
        sym = "\033[92m\u2713\033[0m" if result else "\033[91m\u2717\033[0m"
        d = f"  ({detail})" if detail else ""
        print(f"  {sym} {label}{d}")
        if not result:
            ok = False

    def info(label, result, detail=""):
        """Like check(), but doesn't affect the overall pass/fail status."""
        sym = "\033[92m\u2713\033[0m" if result else "\033[93m\u2014\033[0m"
        d = f"  ({detail})" if detail else ""
        print(f"  {sym} {label}{d}")

    print(f"\n{'='*58}")
    print("  WILSON V1 — System Diagnostics")
    print(f"{'='*58}\n")

    # Platform
    hw = JETSON_MODEL if IS_JETSON else f"{platform.system()} {platform.machine()}"
    print(f"  Platform : {hw}")
    print(f"  Python   : {sys.version.split()[0]}")
    print()

    # Dependencies
    print("  Dependencies:")
    check("numpy", True, np.__version__)

    try:
        sd_mod = _get_sd()
        check("sounddevice", True, sd_mod.__version__)
    except Exception as e:
        check("sounddevice", False, f"import failed: {e}")

    try:
        import soundfile
        check("soundfile", True, soundfile.__version__)
    except ImportError:
        check("soundfile", False, "pip install soundfile")

    try:
        from faster_whisper import WhisperModel
        check("faster-whisper", True)
    except ImportError:
        check("faster-whisper", False, "pip install faster-whisper")

    try:
        import requests as _r
        check("requests", True, _r.__version__)
    except ImportError:
        check("requests", False, "pip install requests")

    # CUDA
    print("\n  CUDA:")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        check("torch.cuda", has_cuda,
              torch.cuda.get_device_name(0) if has_cuda else "no GPU detected")
    except ImportError:
        check("torch", False, "not installed — Whisper will fall back to CPU")

    # Piper
    print("\n  TTS:")
    check("Piper binary", os.path.isfile(PIPER_EXE), PIPER_EXE)
    check("Voice model",  os.path.isfile(PIPER_VOICE), PIPER_VOICE)
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    info("espeak-ng fallback", espeak is not None,
          espeak if espeak else "optional — not needed if Piper is installed")
    if IS_MACOS:
        mac_say = shutil.which("say")
        info("macOS say fallback", mac_say is not None,
              mac_say if mac_say else "optional — not needed if Piper is installed")

    # LLM
    print("\n  LLM endpoint:")
    try:
        r = requests.get(
            LLM_URL.replace("/chat/completions", "/models"),
            timeout=(2, 5),          # (connect, read) — fail fast if nothing is listening
        )
        check("API reachable", r.status_code == 200, LLM_URL)
    except Exception:
        check("API reachable", False, f"{LLM_URL} — is the server running?")

    # Microphone
    print("\n  Audio:")
    try:
        rec = AudioRecorder()
        check("Microphone", True, _get_sd().query_devices(rec.device_id)["name"])
    except Exception as e:
        check("Microphone", False, str(e))

    # Jetson specifics
    if IS_JETSON:
        print("\n  Jetson:")
        check("tegrastats", shutil.which("tegrastats") is not None)
        check("nvpmodel",   shutil.which("nvpmodel") is not None)

    print(f"\n{'='*58}")
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED — see above'}")
    print(f"{'='*58}\n")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
#                                 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    hw = JETSON_MODEL if IS_JETSON else f"{platform.system()} {platform.machine()}"
    lines = [
        f"  Hardware  : {hw}",
        f"  Whisper   : {WHISPER_MODEL} ({WHISPER_DEVICE}/{WHISPER_COMPUTE})",
        f"  LLM       : {LLM_URL}",
    ]
    if LLM_MODEL:
        lines.append(f"  Model     : {LLM_MODEL}")
    lines.append(f"  TTS       : Piper ({'found' if os.path.isfile(PIPER_EXE) else 'NOT FOUND'})")

    print(f"\n{'='*58}")
    print("  WILSON V1 \u2014 Offline Voice Assistant")
    print(f"{'='*58}")
    for l in lines:
        print(l)
    print(f"{'='*58}\n")


if __name__ == "__main__":
    print("Wilson starting...", flush=True)

    # --check: run diagnostics and exit
    if "--check" in sys.argv:
        sys.exit(0 if run_diagnostics() else 1)

    print_banner()

    if HEADLESS or not HAS_GUI:
        if not HAS_GUI:
            print("[NOTE] tkinter not available \u2014 running in headless mode")
        WilsonHeadless().run()
    else:
        WilsonGUI().run()
