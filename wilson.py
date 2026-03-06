"""
WILSON V1 — Offline Voice Assistant

Supports : Windows, macOS (Intel & Apple Silicon), Linux, NVIDIA Jetson
GPU      : NVIDIA CUDA auto-detected — falls back to CPU if unavailable
Pipeline : Mic → faster-whisper (STT) → LLM (embedded llama.cpp) → Piper TTS → Speaker
Standalone : Packable as a single .exe — no LM Studio / Ollama required.
SAM      : Optional EfficientViT-SAM image segmentation when model files are present.

Environment-variable overrides (all optional):
  WILSON_EMBEDDED_LLM       "1" (default) embedded LLM, "0" for remote API
  WILSON_MODEL_REPO         HuggingFace repo for GGUF model
  WILSON_MODEL_FILE         GGUF filename inside the repo
  WILSON_GPU_LAYERS         Number of model layers on GPU (-1 = all)
  WILSON_LLM_URL            Remote LLM API endpoint (fallback)
  WILSON_LLM_MODEL          Model name sent in remote API payload
  WILSON_WHISPER_MODEL      Whisper size: tiny | base | small | medium
  WILSON_WHISPER_COMPUTE    Compute type: float16 | int8 | float32
  WILSON_WHISPER_DEVICE     Device: cuda | cpu
  WILSON_MAX_TOKENS         Max response tokens (default 1024)
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
import json as _json
import numpy as np

# ── Pre-load CUDA DLLs on Windows ─────────────────────────────────────────────
# pip-installed nvidia packages bury DLLs in site-packages/nvidia/*/bin/
# which aren't on PATH.  Register them so CTranslate2 / Whisper find them.
if sys.platform == "win32":
    try:
        import importlib.util
        _nv_base = os.path.join(os.path.dirname(importlib.util.find_spec("nvidia").submodule_search_locations[0]), "nvidia")
        for _pkg in ("cublas", "cudnn", "cuda_runtime", "cufft", "curand", "cusolver", "cusparse", "nccl"):
            _dll_dir = os.path.join(_nv_base, _pkg, "bin")
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)
    except Exception:
        pass

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

# ── Embedded LLM (default — no server needed) ────────────────────────────────
#  Uses llama-cpp-python to load a GGUF model directly.
#  Model auto-downloads from HuggingFace on first run (~1 GB).
#  Chosen to be lightweight so it can coexist with EfficientViT-SAM on one GPU.
USE_EMBEDDED_LLM = os.environ.get("WILSON_EMBEDDED_LLM", "1") == "1"
MODELS_DIR       = os.path.join(WILSON_DIR, "models")

EMBEDDED_MODEL_REPO = os.environ.get(
    "WILSON_MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
EMBEDDED_MODEL_FILE = os.environ.get(
    "WILSON_MODEL_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
EMBEDDED_GPU_LAYERS = int(os.environ.get("WILSON_GPU_LAYERS", "-1"))   # -1 = all

# ── Remote LLM Backend (fallback if embedded unavailable) ────────────────────
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
    Prefers external / Bluetooth devices over built-in mics.
    Opens the stream at the device's native sample rate and resamples
    to 16 kHz on stop() so Whisper always gets what it expects.
    """

    # Keywords for preferred external mics (matched case-insensitively)
    PREFERRED_KEYWORDS = [
        "anker", "powerconf", "poweranc", "respeaker", "seeed",
        "jabra", "blue yeti", "rode", "fifine", "amd bluetooth",
    ]

    # Automatic gain: if the recorded peak is below this we boost the signal.
    # Bluetooth HFP mics on Windows often deliver -60 dB signals through
    # the MME Sound Mapper (PortAudio can't open the device directly).
    _AGC_TARGET_PEAK = 0.5
    _AGC_MIN_PEAK    = 0.002   # anything below this is treated as silence

    # Host-API preference on Windows — DirectSound handles Bluetooth resampling
    # better than WASAPI (which often locks to a single rate on BT headsets)
    _HOSTAPI_RANK = {"Windows DirectSound": 0, "Windows WASAPI": 1, "MME": 2, "Windows WDM-KS": 3}

    def __init__(self):
        self.is_recording = False
        self.audio_data   = []
        self.stream       = None
        self._native_sr   = SAMPLE_RATE    # will be set by _find_device
        self._all_candidates = []          # ordered list of (device_id, sr, label) to try
        self._live_gain   = 1.0            # real-time gain for get_volume(); set after start()
        self.device_id    = self._find_device()

    # ── device selection ──────────────────────────────────────────────────────

    def _find_device(self):
        sd = _get_sd()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        # Build a hostapi name lookup
        api_names = {i: h["name"] for i, h in enumerate(hostapis)}

        # ── Collect all input devices grouped by name ──
        # For each unique device name, pick the best host-API variant
        by_name = {}   # base_name → list of (hostapi_rank, device_index, device_info)
        for i, d in enumerate(devices):
            if d["max_input_channels"] <= 0:
                continue
            api = api_names.get(d["hostapi"], "")
            rank = self._HOSTAPI_RANK.get(api, 99)
            base = d["name"].strip()
            by_name.setdefault(base, []).append((rank, i, d))

        # Sort each group so best host-API comes first
        for name in by_name:
            by_name[name].sort(key=lambda t: t[0])

        # ── Build a prioritised candidate list ──
        # We'll try all of these in order in start() if the top pick fails.
        candidates = []
        seen_ids = set()

        def _add_candidates(base_name, label):
            for rank, idx, info in by_name.get(base_name, []):
                if idx in seen_ids:
                    continue
                seen_ids.add(idx)
                sr = self._pick_samplerate(idx, info.get("default_samplerate", SAMPLE_RATE))
                api = api_names.get(info["hostapi"], "?")
                candidates.append((idx, int(sr), f"{label}: {info['name']} (id {idx}, {api}, {int(sr)} Hz)"))

        # Priority 1: preferred external mics
        for base_name in by_name:
            if any(k in base_name.lower() for k in self.PREFERRED_KEYWORDS):
                _add_candidates(base_name, "Preferred")

        # Priority 2: Bluetooth devices
        for base_name in by_name:
            lower = base_name.lower()
            if "bluetooth" in lower or "bt " in lower or "headset" in lower:
                _add_candidates(base_name, "Bluetooth")

        # Priority 3: USB devices
        for base_name in by_name:
            if "usb" in base_name.lower():
                _add_candidates(base_name, "USB")

        # Priority 4: MME Sound Mapper (id 0) — on Windows this routes to
        # whatever the OS default input device is.  Crucial fallback for
        # Bluetooth mics that PortAudio can't open directly but Windows
        # can route through the mapper.
        try:
            mapper_info = sd.query_devices(0)
            mapper_api = api_names.get(mapper_info["hostapi"], "?")
            if (mapper_info["max_input_channels"] > 0
                    and "mme" in mapper_api.lower()
                    and 0 not in seen_ids):
                mapper_sr = 16000  # Sound Mapper handles resampling
                candidates.append((0, mapper_sr,
                    f"SoundMapper: {mapper_info['name']} (id 0, {mapper_api}, {mapper_sr} Hz)"))
                seen_ids.add(0)
        except Exception:
            pass

        # Priority 5: system default
        default = sd.default.device[0]
        if default is not None and default >= 0 and default not in seen_ids:
            info = sd.query_devices(default)
            sr = int(info.get("default_samplerate", SAMPLE_RATE))
            api = api_names.get(info["hostapi"], "?")
            candidates.append((default, sr, f"Default: {info['name']} (id {default}, {api}, {sr} Hz)"))
            seen_ids.add(default)

        # Priority 6: any remaining input device
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and i not in seen_ids:
                sr = int(d.get("default_samplerate", SAMPLE_RATE))
                api = api_names.get(d["hostapi"], "?")
                candidates.append((i, sr, f"Fallback: {d['name']} (id {i}, {api}, {sr} Hz)"))
                seen_ids.add(i)

        self._all_candidates = candidates
        if not candidates:
            raise RuntimeError("No microphone found")

        # Return the top candidate as default
        top_id, top_sr, top_label = candidates[0]
        self._native_sr = top_sr
        print(f"[MIC] {top_label}")
        return top_id

    def _pick_samplerate(self, device_idx, reported_sr):
        """Choose a workable sample rate.  Prefer 16000 Hz (Whisper's native
        rate) to avoid resampling.  Fall back to the device's reported rate."""
        reported_sr = int(reported_sr)
        # If the device natively supports 16k, use it (no resampling needed)
        if reported_sr == SAMPLE_RATE:
            return SAMPLE_RATE
        try:
            _get_sd().check_input_settings(device=device_idx, samplerate=SAMPLE_RATE, channels=1)
            return SAMPLE_RATE
        except Exception:
            pass
        # 16k not available — use the reported rate (we'll resample in stop())
        if reported_sr >= 8000:
            return reported_sr
        # Suspiciously low — probe common rates
        for sr in [44100, 48000, 22050, 8000]:
            try:
                _get_sd().check_input_settings(device=device_idx, samplerate=sr, channels=1)
                return sr
            except Exception:
                continue
        return reported_sr

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    # ── public ────────────────────────────────────────────────────────────────

    def start(self):
        self.audio_data = []
        self.is_recording = True
        self._live_gain = 1.0  # reset; will be updated if using Sound Mapper

        # Walk the prioritised candidate list — try each device + rate until
        # one actually opens.  This handles disconnected Bluetooth gracefully.
        last_err = None
        for dev_id, sr, label in self._all_candidates:
            # Try the candidate's preferred rate, then fallback rates
            for try_sr in dict.fromkeys([sr, 16000, 44100, 48000]):
                try:
                    self._open_stream(dev_id, try_sr)
                    if dev_id != self.device_id or try_sr != self._native_sr:
                        print(f"[MIC] Opened → {label}"
                              + (f" (at {try_sr} Hz)" if try_sr != sr else ""))
                    self.device_id = dev_id
                    self._native_sr = try_sr

                    # Sound Mapper routing a Bluetooth mic delivers ~-60 dB.
                    # Pre-set a high gain so get_volume() and auto-listen
                    # thresholds work correctly in real time.
                    if "soundmapper" in label.lower() or "sound mapper" in label.lower():
                        self._live_gain = 1000.0   # ~60 dB boost
                        print(f"[MIC] Sound Mapper detected — live gain ×{self._live_gain:.0f}")
                    return
                except Exception as e:
                    last_err = e
                    continue

        raise RuntimeError(f"Could not open any microphone. Last error: {last_err}")

    def _open_stream(self, target_device, samplerate):
        self.stream = _get_sd().InputStream(
            device=target_device,
            samplerate=samplerate,
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
        audio = np.concatenate(self.audio_data, axis=0).flatten()

        # Resample to 16 kHz if the mic's native rate differs (Whisper needs 16k)
        if self._native_sr != SAMPLE_RATE:
            try:
                # Use scipy if available (high quality)
                from scipy.signal import resample
                num_samples = int(len(audio) * SAMPLE_RATE / self._native_sr)
                audio = resample(audio, num_samples).astype(np.float32)
            except ImportError:
                # Simple linear interpolation fallback
                indices = np.linspace(0, len(audio) - 1,
                                      int(len(audio) * SAMPLE_RATE / self._native_sr))
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # ── Automatic gain normalisation ──────────────────────────────────
        # Bluetooth HFP mics routed through the Windows Sound Mapper often
        # deliver signals ~60 dB below normal.  Boost to a usable level so
        # Whisper can transcribe reliably.
        peak = float(np.max(np.abs(audio)))
        if peak < self._AGC_MIN_PEAK:
            print(f"[MIC] Audio peak {peak:.6f} — too quiet, likely silence")
        elif peak < self._AGC_TARGET_PEAK:
            gain = self._AGC_TARGET_PEAK / peak
            audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
            print(f"[MIC] Auto-gain: {20*np.log10(gain):.0f} dB boost "
                  f"(peak {peak:.6f} → {float(np.max(np.abs(audio))):.3f})")

        return audio

    def get_volume(self):
        if not self.audio_data:
            return 0.0
        recent = self.audio_data[-1]
        return float(np.sqrt(np.mean(recent ** 2))) * self._live_gain


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
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=150,
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
#                           MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_model(repo_id, filename, dest_dir, callback=None):
    """Download a GGUF model file from HuggingFace Hub.
    Supports resume if a partial download exists."""
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        return dest_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    def log(msg):
        if callback:
            callback(msg)
        print(f"[DOWNLOAD] {msg}")

    log(f"Downloading {filename} from {repo_id}…")
    log("One-time download — model will be cached in models/")

    tmp_path = dest_path + ".downloading"
    downloaded = 0
    if os.path.exists(tmp_path):
        downloaded = os.path.getsize(tmp_path)

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        log(f"Resuming from {downloaded / 1024 / 1024:.0f} MB")

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        if resp.status_code == 416:          # Range not satisfiable → already done
            os.rename(tmp_path, dest_path)
            return dest_path
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0)) + downloaded
        mode = "ab" if downloaded > 0 else "wb"

        with open(tmp_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=131072):  # 128 KB
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb  = downloaded / 1024 / 1024
                    tmb = total / 1024 / 1024
                    print(f"\r[DOWNLOAD] {mb:.0f}/{tmb:.0f} MB ({pct:.1f}%)",
                          end="", flush=True)
        print()

        os.rename(tmp_path, dest_path)
        log(f"Saved → {dest_path}")
        return dest_path

    except Exception as e:
        log(f"Download failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
#                              LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    Language model client with two modes:

      Embedded (default) — llama-cpp-python loads a GGUF model locally.
                           No LM Studio / Ollama install required.
      Remote  (fallback) — OpenAI-compatible HTTP API.

    The embedded model is chosen to be small enough (~1 GB) to coexist
    with EfficientViT-SAM on a single GPU.
    """

    def __init__(self, embedded=None):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if embedded is None:
            embedded = USE_EMBEDDED_LLM
        self._embedded = embedded
        self._model    = None
        self._loaded   = not embedded   # remote is instantly "ready"

    # ── model loading (call from init thread) ────────────────────────────

    def load(self, callback=None):
        """Load the embedded LLM.  No-op for remote mode.
        Returns True on success (including graceful fallback to remote)."""
        if not self._embedded:
            self._loaded = True
            return True
        return self._init_embedded(callback)

    def _init_embedded(self, callback=None):
        def log(msg):
            if callback:
                callback(msg)
            print(f"[LLM] {msg}")

        model_path = os.path.join(MODELS_DIR, EMBEDDED_MODEL_FILE)

        # ── Download if missing ──
        if not os.path.exists(model_path):
            try:
                download_model(EMBEDDED_MODEL_REPO, EMBEDDED_MODEL_FILE,
                               MODELS_DIR, callback=callback)
            except Exception as e:
                log(f"Download failed: {e}")
                log("Falling back to remote LLM (start LM Studio / Ollama)")
                self._embedded = False
                self._loaded   = True
                return True

        # ── Load with llama-cpp-python ──
        try:
            from llama_cpp import Llama
        except ImportError:
            log("llama-cpp-python not installed — pip install llama-cpp-python")
            log("Falling back to remote LLM")
            self._embedded = False
            self._loaded   = True
            return True

        try:
            n_gpu = EMBEDDED_GPU_LAYERS if CUDA_AVAILABLE else 0
            log(f"Loading {EMBEDDED_MODEL_FILE} …")
            self._model = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=512,
                n_gpu_layers=n_gpu,
                flash_attn=CUDA_AVAILABLE,
                verbose=False,
            )
            gpu_str = f"GPU, {n_gpu} layers" if n_gpu else "CPU only"
            log(f"Model ready ({gpu_str})")
            self._loaded = True
            return True
        except Exception as e:
            log(f"Failed to load model: {e}")
            log("Falling back to remote LLM")
            self._embedded = False
            self._loaded   = True
            return True

    # ── thinking-tag cleanup ─────────────────────────────────────────────

    @staticmethod
    def _strip_thinking(text):
        """Remove <think>…</think> blocks emitted by reasoning models
        (DeepSeek R1, QwQ, etc.) so only the final answer is spoken."""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
        cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned)
        return cleaned.strip()

    # ── query dispatch ───────────────────────────────────────────────────

    def query(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        if self._embedded and self._model is not None:
            return self._query_embedded()
        return self._query_remote()

    def _query_embedded(self):
        """Run inference locally via llama-cpp-python."""
        try:
            resp = self._model.create_chat_completion(
                messages=self.messages[-21:],
                max_tokens=MAX_TOKENS,
                temperature=0.7,
            )
            raw_reply = resp["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": raw_reply})
            return self._strip_thinking(raw_reply)
        except Exception as e:
            return f"LLM error: {e}"

    def _query_remote(self):
        """Call an OpenAI-compatible API (LM Studio / Ollama)."""
        payload = {
            "messages": self.messages[-21:],
            "stream":     False,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }
        if LLM_MODEL:
            payload["model"] = LLM_MODEL

        try:
            resp = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            raw_reply = resp.json()["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": raw_reply})
            return self._strip_thinking(raw_reply)
        except requests.exceptions.ConnectionError:
            return (
                "Cannot connect to LLM server.  Install llama-cpp-python for "
                "embedded mode, or start LM Studio / Ollama."
            )
        except requests.exceptions.Timeout:
            return (
                "LLM timed out. Model may be too large for available memory. "
                "Try a smaller quantisation (Q4_K_S) or reduce context length."
            )
        except Exception as e:
            return f"LLM error: {e}"

    def clear_history(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @property
    def mode(self):
        """Return 'embedded' or 'remote' depending on active backend."""
        return "embedded" if self._embedded else "remote"


# ═══════════════════════════════════════════════════════════════════════════════
#                      EFFICIENTVIT-SAM INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class EfficientViTSAMEngine:
    """
    Optional EfficientViT-SAM (MIT Han Lab) integration.

    EfficientViT-SAM is a hardware-efficient variant of Segment Anything
    that achieves real-time inference with ONNX Runtime.

    Provides real-time image segmentation when the ONNX model files are
    present in the models/ directory.  Wilson can use this to "see" and
    describe objects through a connected camera.

    Required files in models/:
      efficientvit_sam_encoder.onnx   — image encoder
      efficientvit_sam_decoder.onnx   — mask decoder

    Recommended models (place in models/ directory):
      - efficientvit_sam_l0  (~30 MB)  — fastest, lightweight
      - efficientvit_sam_l1  (~45 MB)  — balanced speed / quality
      - efficientvit_sam_l2  (~65 MB)  — higher quality
      - efficientvit_sam_xl0 (~120 MB) — best quality
      - efficientvit_sam_xl1 (~180 MB) — maximum quality

    Get models from: https://github.com/mit-han-lab/efficientvit

    Install:
      pip install onnxruntime-gpu opencv-python

    EfficientViT-SAM is entirely optional.  Wilson works fine without it.
    """

    ENCODER_FILE = "efficientvit_sam_encoder.onnx"
    DECODER_FILE = "efficientvit_sam_decoder.onnx"

    # EfficientViT-SAM encoder expects 1024×1024 input (same as SAM)
    INPUT_SIZE = 1024

    # ImageNet normalisation constants
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(self):
        self.available     = False
        self._encoder      = None
        self._decoder      = None
        self._camera       = None
        self._camera_lock  = threading.Lock()
        self._try_load()

    def _try_load(self):
        """Attempt to load EfficientViT-SAM models.  Silent no-op if unavailable."""
        enc_path = os.path.join(MODELS_DIR, self.ENCODER_FILE)
        dec_path = os.path.join(MODELS_DIR, self.DECODER_FILE)

        if not os.path.isfile(enc_path) or not os.path.isfile(dec_path):
            return

        try:
            import onnxruntime as ort
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if CUDA_AVAILABLE
                         else ["CPUExecutionProvider"])
            self._encoder = ort.InferenceSession(enc_path, providers=providers)
            self._decoder = ort.InferenceSession(dec_path, providers=providers)
            self.available = True
            print("[SAM] EfficientViT-SAM encoder + decoder loaded")
        except ImportError:
            print("[SAM] onnxruntime not installed — pip install onnxruntime-gpu")
        except Exception as e:
            print(f"[SAM] Load failed: {e}")

    # ── camera ────────────────────────────────────────────────────────────

    def open_camera(self, device_id=0):
        """Open a camera for capture.  Returns True on success."""
        try:
            import cv2
            with self._camera_lock:
                if self._camera is not None:
                    self._camera.release()
                self._camera = cv2.VideoCapture(device_id)
                ok = self._camera.isOpened()
                if ok:
                    print(f"[SAM] Camera {device_id} opened")
                return ok
        except ImportError:
            print("[SAM] opencv-python not installed")
            return False
        except Exception as e:
            print(f"[SAM] Camera error: {e}")
            return False

    def capture_frame(self):
        """Capture a single frame.  Returns numpy HWC BGR array or None."""
        with self._camera_lock:
            if self._camera is None or not self._camera.isOpened():
                return None
            ret, frame = self._camera.read()
            return frame if ret else None

    def close_camera(self):
        with self._camera_lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None

    # ── pre-processing ────────────────────────────────────────────────────

    def _preprocess(self, image_bgr):
        """Resize, normalise (ImageNet stats), return NCHW float32 + scale info."""
        import cv2
        h, w = image_bgr.shape[:2]
        # Resize longest side to INPUT_SIZE, pad to square
        scale = self.INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h))
        # Pad to INPUT_SIZE × INPUT_SIZE
        padded = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        # BGR → RGB, HWC → CHW, normalise
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[np.newaxis]     # 1×3×1024×1024
        chw = (chw - self._MEAN) / self._STD
        return chw.astype(np.float32), (h, w, new_h, new_w, scale)

    # ── inference ─────────────────────────────────────────────────────────

    def encode_image(self, image_bgr):
        """Run the image encoder.  Returns (embedding, size_info) or (None, None)."""
        if not self.available or self._encoder is None:
            return None, None
        try:
            tensor, size_info = self._preprocess(image_bgr)
            inp_name = self._encoder.get_inputs()[0].name
            embedding = self._encoder.run(None, {inp_name: tensor})[0]
            return embedding, size_info
        except Exception as e:
            print(f"[SAM] Encode error: {e}")
            return None, None

    def segment(self, image_embedding, size_info=None, point=None, box=None):
        """Run the mask decoder.  Returns binary mask (H×W bool array) or None.

        Args:
            image_embedding: output of encode_image()
            size_info:       (orig_h, orig_w, new_h, new_w, scale) from encode_image
            point: (x, y) pixel coordinates in the original image
            box:   (x1, y1, x2, y2) pixel coordinates bounding box
        """
        if not self.available or self._decoder is None or image_embedding is None:
            return None
        try:
            import cv2
            scale = size_info[4] if size_info else 1.0

            # Build prompt coordinates in resized image space
            if point is not None:
                px, py = float(point[0]) * scale, float(point[1]) * scale
                coords = np.array([[[px, py]]], dtype=np.float32)       # 1×1×2
                labels = np.array([[1]], dtype=np.float32)              # foreground
            elif box is not None:
                x1, y1 = float(box[0]) * scale, float(box[1]) * scale
                x2, y2 = float(box[2]) * scale, float(box[3]) * scale
                coords = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)  # 1×2×2
                labels = np.array([[2, 3]], dtype=np.float32)
            else:
                # Default: centre of the image
                cx = self.INPUT_SIZE / 2.0
                cy = self.INPUT_SIZE / 2.0
                coords = np.array([[[cx, cy]]], dtype=np.float32)
                labels = np.array([[1]], dtype=np.float32)

            # Map inputs by name
            inputs = {}
            for inp in self._decoder.get_inputs():
                name_lower = inp.name.lower()
                if "image" in name_lower or "embed" in name_lower:
                    inputs[inp.name] = image_embedding
                elif "coord" in name_lower or "point" in name_lower:
                    inputs[inp.name] = coords
                elif "label" in name_lower:
                    inputs[inp.name] = labels
                elif "mask" in name_lower and "input" in name_lower:
                    # Some decoders expect a mask input (zeros)
                    inputs[inp.name] = np.zeros((1, 1, 256, 256), dtype=np.float32)
                elif "has_mask" in name_lower or "has" in name_lower:
                    inputs[inp.name] = np.array([0], dtype=np.float32)
                elif "orig" in name_lower or "size" in name_lower:
                    orig_h = size_info[0] if size_info else self.INPUT_SIZE
                    orig_w = size_info[1] if size_info else self.INPUT_SIZE
                    inputs[inp.name] = np.array([orig_h, orig_w], dtype=np.float32)

            masks = self._decoder.run(None, inputs)
            if masks and len(masks) > 0:
                mask = masks[0]
                # Handle different output shapes
                if mask.ndim == 4:
                    mask = mask[0, 0]
                elif mask.ndim == 3:
                    mask = mask[0]
                binary = mask > 0.0

                # Resize mask back to original image dimensions if size_info given
                if size_info is not None:
                    orig_h, orig_w = size_info[0], size_info[1]
                    binary_u8 = binary.astype(np.uint8) * 255
                    binary_u8 = cv2.resize(binary_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    binary = binary_u8 > 127

                return binary
            return None
        except Exception as e:
            print(f"[SAM] Segment error: {e}")
            return None

    @property
    def status(self):
        if self.available:
            return "ready"
        enc = os.path.isfile(os.path.join(MODELS_DIR, self.ENCODER_FILE))
        dec = os.path.isfile(os.path.join(MODELS_DIR, self.DECODER_FILE))
        if not enc or not dec:
            return "model files missing"
        return "load failed"


# ═══════════════════════════════════════════════════════════════════════════════
#                        FACE RECOGNITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FaceRecognitionEngine:
    """
    Real-time face detection and recognition using OpenCV.

    Detection  : Haar cascade (ships with opencv-python — no extra files).
    Recognition: Spatial histogram descriptors (5×5 grid, 32 bins per cell).
                 Compared via histogram correlation.  No deep learning deps.

    Stores known face descriptors + names in faces/face_db.json.
    Face images are *not* saved — only compact numerical descriptors.
    """

    FACES_DIR   = os.path.join(WILSON_DIR, "faces")
    DB_FILE     = os.path.join(WILSON_DIR, "faces", "face_db.json")
    FACE_SIZE   = (100, 100)
    GRID        = 5
    BINS        = 32
    DESCRIPTOR_LEN = GRID * GRID * BINS          # 800 floats
    MATCH_THRESHOLD = 0.55                        # histogram correlation ≥ this → match
    ENROLL_SAMPLES  = 20                          # frames captured during enrollment

    def __init__(self):
        self.available    = False
        self._cascade     = None
        self._camera      = None
        self._camera_lock = threading.Lock()
        self._known_faces = {}          # {name: [descriptor, …]}
        self._init()

    # ── bootstrap ─────────────────────────────────────────────────────────

    def _init(self):
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                print("[FACE] Haar cascade failed to load")
                return
            os.makedirs(self.FACES_DIR, exist_ok=True)
            self._load_db()
            self.available = True
            print(f"[FACE] Ready — {len(self._known_faces)} known person(s)")
        except ImportError:
            print("[FACE] opencv-python not installed")
        except Exception as e:
            print(f"[FACE] Init error: {e}")

    # ── persistence ───────────────────────────────────────────────────────

    def _load_db(self):
        self._known_faces = {}
        if not os.path.isfile(self.DB_FILE):
            return
        try:
            with open(self.DB_FILE, "r") as f:
                db = _json.load(f)
            for name, descriptors in db.items():
                self._known_faces[name] = [
                    np.array(d, dtype=np.float32) for d in descriptors
                ]
        except Exception as e:
            print(f"[FACE] DB load error: {e}")

    def _save_db(self):
        db = {}
        for name, descriptors in self._known_faces.items():
            db[name] = [d.tolist() for d in descriptors]
        try:
            with open(self.DB_FILE, "w") as f:
                _json.dump(db, f)
        except Exception as e:
            print(f"[FACE] DB save error: {e}")

    # ── descriptor ────────────────────────────────────────────────────────

    def _face_descriptor(self, face_gray):
        """Return a spatial-histogram descriptor (800-dim) for a grayscale face."""
        import cv2
        face = cv2.resize(face_gray, self.FACE_SIZE)
        cell_h = self.FACE_SIZE[1] // self.GRID
        cell_w = self.FACE_SIZE[0] // self.GRID
        parts = []
        for r in range(self.GRID):
            for c in range(self.GRID):
                cell = face[r * cell_h:(r + 1) * cell_h,
                            c * cell_w:(c + 1) * cell_w]
                hist = cv2.calcHist([cell], [0], None, [self.BINS], [0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                parts.append(hist.flatten())
        return np.concatenate(parts).astype(np.float32)

    # ── camera ────────────────────────────────────────────────────────────

    def open_camera(self, device_id=0):
        import cv2
        with self._camera_lock:
            if self._camera is not None:
                self._camera.release()
            self._camera = cv2.VideoCapture(device_id)
            ok = self._camera.isOpened()
            if ok:
                # Lower resolution for speed
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[FACE] Camera {device_id} opened")
            return ok

    def close_camera(self):
        with self._camera_lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None

    def capture_frame(self):
        with self._camera_lock:
            if self._camera is None or not self._camera.isOpened():
                return None
            ret, frame = self._camera.read()
            return frame if ret else None

    # ── detection + recognition ───────────────────────────────────────────

    def detect_faces(self, frame):
        """Detect faces in a BGR frame.
        Returns (rects, gray) where rects is a list of (x, y, w, h)."""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60),
        )
        # Convert ndarray to list of tuples
        if len(rects) == 0:
            return [], gray
        return [tuple(r) for r in rects], gray

    def recognize(self, gray, x, y, w, h):
        """Match a face region against known faces.
        Returns (name, confidence) or (None, 0.0)."""
        import cv2
        face_crop = gray[y:y + h, x:x + w]
        desc = self._face_descriptor(face_crop)

        best_name = None
        best_score = 0.0

        for name, descriptors in self._known_faces.items():
            scores = []
            for stored in descriptors:
                score = cv2.compareHist(
                    desc.reshape(-1, 1),
                    stored.reshape(-1, 1),
                    cv2.HISTCMP_CORREL,
                )
                scores.append(score)
            # Average top-5 scores for robustness
            top = sorted(scores, reverse=True)[:5]
            avg = float(np.mean(top)) if top else 0.0
            if avg > best_score:
                best_score = avg
                best_name = name

        if best_score >= self.MATCH_THRESHOLD:
            return best_name, best_score
        return None, best_score

    def add_sample(self, name, gray, x, y, w, h):
        """Add one face descriptor sample for a person."""
        face_crop = gray[y:y + h, x:x + w]
        desc = self._face_descriptor(face_crop)
        if name not in self._known_faces:
            self._known_faces[name] = []
        self._known_faces[name].append(desc)

    def save(self):
        """Persist the current face database to disk."""
        self._save_db()

    def remove_person(self, name):
        if name in self._known_faces:
            del self._known_faces[name]
            self._save_db()

    @property
    def known_names(self):
        return list(self._known_faces.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#                          WILSON — GUI MODE
# ═══════════════════════════════════════════════════════════════════════════════

class WilsonGUI:
    """Tkinter-based graphical interface. Works on both HDMI-out Jetson and
    Windows desktop. Jetson-specific widgets show live system telemetry.
    Hardened for resilience, localized, and optimized for performance."""

    THEME = {
        "bg_main": "#050505",         # Off-black to prevent Windows transparency key bugs
        "bg_panel": "#0a0c0a",        # Near-black with faint green tint
        "fg_title": "#00ff41",        # Matrix phosphor green
        "fg_text": "#00ff41",         # Matrix green
        "fg_sub": "#00cc33",          # Dimmer Matrix green
        "accent_ready": "#00ff41",    # Matrix green
        "accent_ready_hover": "#33ff66",
        "accent_listen": "#ff0040",   # Red alert
        "accent_process": "#00cc33",  # Processing green
        "color_user": "#00ff41",      # Matrix green for user
        "color_wilson": "#00ff41",    # Matrix green for Wilson
        "color_system": "#004d1a",    # Dark green for system
    }

    I18N = {
        "title": "WILSON V1",
        "subtitle": "OFFLINE VOICE ASSISTANT",
        "btn_ready": "> INITIATE TRANSMISSION",
        "btn_listen": "> SIGNAL ACTIVE...",
        "btn_process": "> DECODING...",
        "status_init": "LOADING...",
        "status_ready": "AWAITING INPUT",
        "status_dictate": "RECEIVING...",
        "status_process": "PROCESSING...",
        "status_transcribe": "DECODING SIGNAL...",
        "status_think": "COMPUTING...",
        "status_speak": "TRANSMITTING...",
        "status_error": "ERROR",
        "telemetry_load": "TELEMETRY LOADING...",
        "volume": "SIG:",
        "footer_hints": "[SPACE] or click  |  auto-stops on silence"
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.I18N["title"])
        
        # Transparent Circular Window setup
        self._transparent_key = "#ab00cd"
        self.root.geometry("800x800")
        self.root.configure(bg=self._transparent_key)
        
        if IS_WINDOWS:
            self.root.attributes("-transparentcolor", self._transparent_key)
            self.root.overrideredirect(True)
            self.root.bind("<Escape>", lambda e: self.shutdown())
            # Create a tiny hidden proxy window that lives in the taskbar
            # and controls minimize/restore of the real frameless window
            self.root.after(50, self._create_taskbar_proxy)

        # Dragging variables
        self._drag_x = 0
        self._drag_y = 0

        self.is_listening  = False
        self.is_processing = False
        self._ready        = False   # True once init thread completes successfully
        self._auto_listen  = False   # hands-free voice-activated mode
        self.recorder      = None
        self.transcriber   = None
        self.tts           = None
        self.llm           = LLMClient()
        self.sam           = EfficientViTSAMEngine()
        self.face_engine   = FaceRecognitionEngine()
        self.monitor       = JetsonMonitor()
        self.msg_queue     = queue.Queue()

        # Camera / face recognition state
        self._camera_active       = False
        self._enrolling_name      = None     # set during face enrollment
        self._enroll_count        = 0
        self._last_greeted        = None     # avoid repeated greetings
        self._unknown_streak      = 0        # consecutive unknown detections
        self._enroll_dialog_open  = False    # prevent duplicate enroll dialogs
        self._enroll_cooldown     = 0.0      # time.time() when cooldown expires

        # Face management panel state
        self._faces_panel_open    = False

        # Matrix text animation state
        self._matrix_busy = False
        self._pending_wilson_text = None   # holds wilson text to sync with TTS
        self._matrix_rain_chars = []       # falling rain columns

        # Matrix audio visualizer state
        self._viz_bars = 24          # number of frequency bars
        self._viz_levels = [0.0] * 24
        self._viz_particles = []     # list of (x, y, speed, brightness)

        # Mouth / face animation state
        self._mouth_active = False
        self._mouth_data = []
        self._mouth_openness = 0.0
        self._mouth_start = 0.0
        self._mouth_duration = 0.0
        self._eye_open = True

        # Cache formatting elements for optimization
        self._font_title = (FONT_MONO, 22, "bold")
        self._font_sub = (FONT_MONO, 9)
        self._font_mono = (FONT_MONO, 10)
        self._font_btn = (FONT_MONO, 12, "bold")

        self._build_ui()
        self._check_queue()
        threading.Thread(target=self._initialize, daemon=True).start()

    # ── window dragging and taskbar ───────────────────────────────────────────

    def _create_taskbar_proxy(self):
        """Create a 0×0 hidden Toplevel that keeps Wilson visible in the
        Windows taskbar. When the user clicks the taskbar icon, we intercept
        the proxy's state changes to show/hide the real window."""
        try:
            import ctypes
            self._proxy = tk.Toplevel(self.root)
            self._proxy.title(self.I18N["title"])
            self._proxy.geometry("0x0+0+0")
            self._proxy.attributes("-alpha", 0.0)      # fully invisible
            self._proxy.transient(None)                 # NOT a child — own taskbar entry
            self.root.wm_attributes("-topmost", False)

            # Make the real (overrideredirect) window owned by the proxy
            # so the OS groups them, but the proxy is what appears in the taskbar
            hwnd_main = ctypes.windll.user32.GetParent(self.root.winfo_id())
            hwnd_proxy = ctypes.windll.user32.GetParent(self._proxy.winfo_id())

            if hwnd_main and hwnd_proxy:
                # Set proxy as APPWINDOW
                GWL_EXSTYLE = -20
                ex = ctypes.windll.user32.GetWindowLongW(hwnd_proxy, GWL_EXSTYLE)
                ex = ex & ~0x00000080   # ~WS_EX_TOOLWINDOW
                ex = ex | 0x00040000    # WS_EX_APPWINDOW
                ctypes.windll.user32.SetWindowLongW(hwnd_proxy, GWL_EXSTYLE, ex)
                # Owner relationship: clicking taskbar icon controls both
                ctypes.windll.user32.SetWindowLongW(hwnd_main, -8, hwnd_proxy)  # GWL_HWNDPARENT

            # Intercept minimize / restore on the proxy
            self._proxy.bind("<Unmap>", self._on_proxy_minimize)
            self._proxy.bind("<Map>", self._on_proxy_restore)
            self._proxy.protocol("WM_DELETE_WINDOW", self.shutdown)
        except Exception as e:
            print(f"[UI] Taskbar proxy failed: {e}")

    def _on_proxy_minimize(self, event=None):
        """When the proxy is minimized (taskbar click), hide the real window."""
        try:
            self.root.withdraw()
        except tk.TclError:
            pass

    def _on_proxy_restore(self, event=None):
        """When the proxy is restored (taskbar click), show the real window."""
        try:
            self.root.deiconify()
            self.root.lift()
        except tk.TclError:
            pass

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
        
        # Create outer circle — black hole event horizon
        self.bg_canvas.create_oval(10, 10, 790, 790, fill=t["bg_main"], outline=t["fg_title"], width=2)
        # Inner glow ring
        self.bg_canvas.create_oval(20, 20, 780, 780, fill="", outline="#003300", width=1)
        
        # Bind left-click and drag events to canvas for window movement
        self.bg_canvas.bind("<ButtonPress-1>", self._start_drag)
        self.bg_canvas.bind("<B1-Motion>", self._on_drag)

        # Central container — firmly inscribed inside circle to prevent corner spikes
        self.main_container = tk.Frame(self.root, bg=t["bg_main"])
        self.main_container.place(relx=0.5, rely=0.5, anchor="center", width=460, height=560)

        subtitle = JETSON_MODEL if IS_JETSON else i18n["subtitle"]

        # Header with drag & close actions explicitly mapped
        top_frame = tk.Frame(self.main_container, bg=t["bg_main"])
        top_frame.pack(fill=tk.X, pady=(0, 3))
        
        close_btn = tk.Label(top_frame, text="[X]", font=(FONT_MONO, 11, "bold"), fg=t["accent_listen"], bg=t["bg_main"], cursor="hand2")
        close_btn.pack(side=tk.RIGHT, padx=5)
        close_btn.bind("<Button-1>", lambda e: self.shutdown())

        tk.Label(
            self.main_container, text=f"[ {i18n['title']} ]",
            font=self._font_title,
            fg=t["fg_title"], bg=t["bg_main"],
        ).pack(pady=(0, 3))

        self._subtitle_label = tk.Label(
            self.main_container, text=subtitle,
            font=self._font_sub,
            fg=t["fg_sub"], bg=t["bg_main"],
        )
        self._subtitle_label.pack()

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
        sf.pack(pady=(6, 0))

        self.status_dot = tk.Label(
            sf, text="●", font=(FONT_MONO, 14),
            fg=t["accent_process"], bg=t["bg_main"],
        )
        self.status_dot.pack(side=tk.LEFT)

        self.status_text = tk.Label(
            sf, text=i18n["status_init"],
            font=(FONT_MONO, 11), fg=t["fg_text"], bg=t["bg_main"],
        )
        self.status_text.pack(side=tk.LEFT, padx=(8, 0))

        # ── Matrix Audio Visualizer (replaces boring progress bar) ────────
        self.viz_canvas = tk.Canvas(
            self.main_container, width=340, height=36,
            bg=t["bg_main"], highlightthickness=0,
        )
        self.viz_canvas.pack(pady=(4, 6))
        self._draw_viz_idle()

        # Chat log — Matrix terminal — fixed height so the button below stays visible
        cf = tk.Frame(self.main_container, bg=t["bg_main"], padx=0, pady=0, height=160)
        cf.pack(padx=10, pady=4, fill=tk.X)
        cf.pack_propagate(False)  # prevent children from resizing this frame

        self.chat = tk.Text(
            cf, font=self._font_mono,
            bg=t["bg_main"], fg=t["fg_text"],
            wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, padx=10, pady=8,
            borderwidth=0, highlightthickness=0,
            insertbackground=t["fg_title"],
            selectbackground="#003300",
            selectforeground=t["fg_title"],
            cursor="arrow",
        )
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Thin Matrix scroll indicator (canvas-drawn, no native scrollbar)
        self.scroll_indicator = tk.Canvas(
            cf, width=6, bg=t["bg_main"], highlightthickness=0,
        )
        self.scroll_indicator.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.config(yscrollcommand=self._update_scroll_indicator)
        # Enable mouse wheel scrolling
        self.chat.bind("<MouseWheel>", self._on_chat_scroll)
        self.chat.bind("<Button-4>", self._on_chat_scroll)
        self.chat.bind("<Button-5>", self._on_chat_scroll)

        self.chat.tag_configure("user",   foreground="#00cc33", font=(FONT_MONO, 10))
        self.chat.tag_configure("wilson", foreground="#00ff41", font=(FONT_MONO, 10, "bold"), spacing1=4, spacing3=4)
        self.chat.tag_configure("system", foreground="#004d1a", font=(FONT_MONO, 9), justify=tk.CENTER)
        self.chat.tag_configure("cursor_blink", foreground="#00ff41", font=(FONT_MONO, 10, "bold"))

        # ── Text chat input ───────────────────────────────────────────────
        chat_input_frame = tk.Frame(self.main_container, bg=t["bg_main"])
        chat_input_frame.pack(fill=tk.X, padx=10, pady=(2, 0))

        self.chat_input = tk.Entry(
            chat_input_frame,
            font=(FONT_MONO, 10),
            bg="#0a1a0a", fg="#00ff41",
            insertbackground="#00ff41",
            highlightthickness=1, highlightbackground="#003300",
            highlightcolor="#00ff41",
            relief=tk.FLAT, bd=1,
        )
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.chat_input.bind("<Return>", lambda e: self._send_text())
        # Prevent spacebar from triggering voice while typing
        self.chat_input.bind("<space>", lambda e: "break" if False else None)

        send_btn = tk.Button(
            chat_input_frame,
            text=">",
            font=(FONT_MONO, 12, "bold"),
            bg="#0a1a0a", fg="#00ff41",
            activebackground="#003300", activeforeground="#33ff66",
            highlightthickness=1, highlightbackground="#003300",
            relief=tk.SOLID, bd=1, width=3,
            cursor="hand2",
            command=self._send_text,
        )
        send_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # Button row — side by side
        btn_row = tk.Frame(self.main_container, bg=t["bg_main"])
        btn_row.pack(pady=(6, 6))

        # Listen button — Matrix style
        self.btn = tk.Button(
            btn_row,
            text=i18n["btn_ready"],
            font=self._font_btn,
            bg="#0a1a0a", fg=t["accent_ready"],
            activebackground="#003300", activeforeground=t["accent_ready_hover"],
            highlightthickness=1, highlightbackground="#00ff41",
            relief=tk.SOLID, bd=1, width=20, height=2,
            cursor="hand2",
            command=self._toggle_listening,
        )
        self.btn.pack(side=tk.LEFT, padx=(0, 4))
        # Only trigger voice on Space if the text input doesn't have focus
        self.root.bind("<space>", lambda e: self._toggle_listening() if e.widget is not self.chat_input else None)

        # Auto-listen toggle button
        self.auto_btn = tk.Button(
            btn_row,
            text="AUTO",
            font=(FONT_MONO, 10, "bold"),
            bg="#0a1a0a", fg="#004d1a",
            activebackground="#001a00", activeforeground="#00ff41",
            highlightthickness=1, highlightbackground="#003300",
            relief=tk.SOLID, bd=1, width=6, height=2,
            cursor="hand2",
            command=self._toggle_auto_listen,
        )
        self.auto_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Camera / face recognition toggle button
        self.cam_btn = tk.Button(
            btn_row,
            text="CAM",
            font=(FONT_MONO, 10, "bold"),
            bg="#0a1a0a", fg="#004d1a",
            activebackground="#001a00", activeforeground="#00ff41",
            highlightthickness=1, highlightbackground="#003300",
            relief=tk.SOLID, bd=1, width=6, height=2,
            cursor="hand2",
            command=self._toggle_camera,
        )
        self.cam_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Face management button
        self.faces_btn = tk.Button(
            btn_row,
            text="FACES",
            font=(FONT_MONO, 9, "bold"),
            bg="#0a1a0a", fg="#004d1a",
            activebackground="#001a00", activeforeground="#00ff41",
            highlightthickness=1, highlightbackground="#003300",
            relief=tk.SOLID, bd=1, width=6, height=2,
            cursor="hand2",
            command=self._toggle_faces_panel,
        )
        self.faces_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Footer — Matrix terminal info
        _llm_name = "Embedded" if USE_EMBEDDED_LLM else ("LM Studio" if IS_WINDOWS else "Ollama")
        _sam_tag  = " | SAM:ready" if os.path.isfile(os.path.join(MODELS_DIR, EfficientViTSAMEngine.ENCODER_FILE)) else ""
        cfg = (
            f"STT:{WHISPER_MODEL}({WHISPER_DEVICE}) | "
            f"LLM:{_llm_name}{_sam_tag}"
        )
        tk.Label(
            self.main_container, text=cfg,
            font=(FONT_MONO, 7), fg="#003300", bg=t["bg_main"],
        ).pack()

        tk.Label(
            self.main_container,
            text=i18n["footer_hints"] + "  |  type to chat  |  [ESC] quit",
            font=(FONT_MONO, 7), fg="#004d1a", bg=t["bg_main"],
        ).pack(pady=(1, 4))

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
        """Matrix-style state transition for button."""
        try:
            bg = target_bg or "#0a1a0a"
            self.btn.config(fg=target_fg, bg=bg, activeforeground=target_fg)
        except tk.TclError:
            pass

    def _log(self, tag, message):
        self.msg_queue.put((tag, message))

    # ── Matrix Audio Visualizer ───────────────────────────────────────────

    def _draw_viz_idle(self):
        """Draw a dormant equalizer — dim static bars."""
        c = self.viz_canvas
        c.delete("all")
        w = 340
        n = self._viz_bars
        bar_w = max(2, (w - (n - 1) * 3) // n)
        for i in range(n):
            x = i * (bar_w + 3) + 4
            h = 2
            c.create_rectangle(x, 34 - h, x + bar_w, 34,
                               fill="#002200", outline="", tags="bars")

    def _draw_viz_active(self):
        """Draw the live Matrix equalizer with glowing bars + falling particles."""
        c = self.viz_canvas
        c.delete("all")
        w = 340
        n = self._viz_bars
        bar_w = max(2, (w - (n - 1) * 3) // n)

        for i in range(n):
            x = i * (bar_w + 3) + 4
            level = self._viz_levels[i]
            h = max(2, int(level * 30))

            # Multi-shade bar: brighter at the top
            if h > 15:
                # Hot section
                c.create_rectangle(x, 34 - h, x + bar_w, 34 - h + 4,
                                   fill="#00ff41", outline="")
                c.create_rectangle(x, 34 - h + 4, x + bar_w, 34,
                                   fill="#00aa2a", outline="")
            elif h > 6:
                c.create_rectangle(x, 34 - h, x + bar_w, 34,
                                   fill="#00882a", outline="")
            else:
                c.create_rectangle(x, 34 - h, x + bar_w, 34,
                                   fill="#004d1a", outline="")

            # Peak dot floating above bar
            if level > 0.3:
                peak_y = 34 - h - 3
                c.create_rectangle(x, peak_y, x + bar_w, peak_y + 2,
                                   fill="#00ff41", outline="")

        # Draw falling particles
        new_particles = []
        for px, py, spd, br in self._viz_particles:
            py += spd
            br -= 0.02
            if br > 0 and py < 36:
                g = int(br * 255)
                color = f"#00{g:02x}{g // 4:02x}"
                c.create_text(px, py, text=random.choice("01"), fill=color,
                              font=(FONT_MONO, 7), anchor="nw")
                new_particles.append((px, py, spd, br))
        self._viz_particles = new_particles

    def _update_viz_from_volume(self):
        """Update visualizer levels from current mic volume. Called from _check_queue."""
        if self.is_listening and self.recorder:
            vol = min(1.0, self.recorder.get_volume() * 12)
            # Generate frequency-like distribution from mono volume
            for i in range(self._viz_bars):
                # Each bar gets a slightly randomized version of the volume
                target = vol * random.uniform(0.3, 1.0)
                # Smooth towards target
                self._viz_levels[i] = self._viz_levels[i] * 0.3 + target * 0.7

            # Spawn particles from the loudest bars
            if vol > 0.15:
                n = self._viz_bars
                bar_w = max(2, (340 - (n - 1) * 3) // n)
                for i in range(n):
                    if self._viz_levels[i] > 0.5 and random.random() < 0.3:
                        x = i * (bar_w + 3) + 4
                        h = int(self._viz_levels[i] * 30)
                        self._viz_particles.append(
                            (x, 34 - h - 5, random.uniform(0.8, 2.0), random.uniform(0.5, 1.0))
                        )
                # Cap particles
                if len(self._viz_particles) > 60:
                    self._viz_particles = self._viz_particles[-40:]

            self._draw_viz_active()
        else:
            # Decay bars smoothly to zero
            any_active = False
            for i in range(self._viz_bars):
                self._viz_levels[i] *= 0.85
                if self._viz_levels[i] > 0.01:
                    any_active = True
            if any_active:
                self._draw_viz_active()
            else:
                self._draw_viz_idle()

    # ── Custom Scroll Indicator ───────────────────────────────────────────

    def _update_scroll_indicator(self, first, last):
        """Draw a thin Matrix-green scroll position indicator."""
        c = self.scroll_indicator
        c.delete("all")
        first_f = float(first)
        last_f = float(last)
        if last_f - first_f >= 0.999:
            # All content visible, no indicator needed
            return
        h = c.winfo_height()
        if h < 10:
            h = 200  # fallback before widget is rendered
        y1 = int(first_f * h)
        y2 = int(last_f * h)
        y2 = max(y2, y1 + 12)  # minimum thumb size
        # Glow track line
        c.create_line(3, 0, 3, h, fill="#001a00", width=1)
        # Thumb
        c.create_rectangle(1, y1, 5, y2, fill="#00ff41", outline="#003300")

    def _on_chat_scroll(self, event):
        """Handle mouse wheel scrolling in the chat."""
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.chat.yview_scroll(-3, "units")
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.chat.yview_scroll(3, "units")

    def _insert_separator(self):
        """Insert a dim separator line to visually mark new conversation turns."""
        try:
            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, "\n" + "─" * 44 + "\n", "system")
            self.chat.config(state=tk.DISABLED)
            self.chat.see(tk.END)
        except tk.TclError:
            pass

    # ── Matrix text animation ─────────────────────────────────────────────

    def _check_queue(self):
        """Drain message queue via Matrix-style typing. Resilient polling."""
        # Only pop a new message if not currently typing one out
        if not self._matrix_busy:
            try:
                tag, msg = self.msg_queue.get_nowait()
                self._matrix_busy = True
                if tag == "user":
                    # New conversation turn — add separator, then user text
                    self._insert_separator()
                    self._matrix_type("\n> ", msg, "user")
                elif tag == "wilson":
                    self._matrix_type("\n[WILSON] ", msg, "wilson")
                else:
                    self._matrix_type("\n  ", msg, "system")
            except queue.Empty:
                pass

        # Matrix audio visualizer update
        try:
            self._update_viz_from_volume()
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
        """Type text character-by-character, Matrix rain style with
        randomized glitch characters that resolve into final text."""
        try:
            self.chat.config(state=tk.NORMAL)
            if idx == 0:
                self.chat.insert(tk.END, prefix, tag)
            if idx < len(text):
                ch = text[idx]
                # Matrix glitch: briefly show a random character before the real one
                if tag == "wilson" and ch not in ' \n' and random.random() < 0.4:
                    glitch_chars = "01アイウエオカキクケコサシスセソ$#@&%"
                    glitch = random.choice(glitch_chars)
                    self.chat.insert(tk.END, glitch, tag)
                    self.chat.see(tk.END)
                    self.chat.config(state=tk.DISABLED)
                    # Schedule replacement of glitch with real char
                    self.root.after(40, self._resolve_glitch, text, tag, idx)
                    return
                self.chat.insert(tk.END, ch, tag)
                self.chat.see(tk.END)
                self.chat.config(state=tk.DISABLED)
                # Speed profile: wilson=dramatic, system=fast, user=medium
                if tag == "wilson":
                    delay = 4 if ch in ' \n' else 18
                elif tag == "user":
                    delay = 3 if ch in ' \n' else 10
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

    def _resolve_glitch(self, text, tag, idx):
        """Replace glitch character with the real character."""
        try:
            self.chat.config(state=tk.NORMAL)
            # Delete the last character (the glitch) and insert the real one
            self.chat.delete("end-2c", "end-1c")
            self.chat.insert("end-1c", text[idx], tag)
            self.chat.see(tk.END)
            self.chat.config(state=tk.DISABLED)
            ch = text[idx]
            if tag == "wilson":
                delay = 4 if ch in ' \n' else 18
            elif tag == "user":
                delay = 3 if ch in ' \n' else 10
            else:
                delay = 1
            self.root.after(delay, self._matrix_type, "", text, tag, idx + 1)
        except tk.TclError:
            self._matrix_busy = False

    # ── Animated Face / Mouth ─────────────────────────────────────────────

    def _draw_face(self):
        """Draw eyes and mouth on the face canvas — Matrix / black hole style."""
        c = self.face_canvas
        t = self.THEME
        c.delete("face")

        cx, cy_eyes = 100, 28
        eye_sep = 35

        # ── Eyes ── (Matrix green, hollow)
        if self._eye_open:
            for ex in (cx - eye_sep, cx + eye_sep):
                c.create_oval(ex - 11, cy_eyes - 11, ex + 11, cy_eyes + 11,
                              fill="#00ff41", outline="#00ff41", tags="face")
                c.create_oval(ex - 5, cy_eyes - 5, ex + 5, cy_eyes + 5,
                              fill="#000000", outline="#000000", tags="face")
        else:
            for ex in (cx - eye_sep, cx + eye_sep):
                c.create_line(ex - 11, cy_eyes, ex + 11, cy_eyes,
                              fill="#00ff41", width=3, tags="face")

        # ── Mouth ── (Matrix green outline)
        cx_m, cy_m = 100, 72
        mouth_w = 28
        openness = self._mouth_openness

        if openness < 0.05:
            c.create_arc(cx_m - mouth_w, cy_m - 10, cx_m + mouth_w, cy_m + 14,
                         start=200, extent=140, style=tk.ARC,
                         outline="#00ff41", width=3, tags="face")
        else:
            h = int(4 + openness * 22)
            c.create_oval(cx_m - mouth_w, cy_m - h, cx_m + mouth_w, cy_m + h,
                          fill="#000800", outline="#00ff41", width=2, tags="face")
            if openness > 0.55:
                tw = int(10 + openness * 6)
                c.create_arc(cx_m - tw, cy_m + 2, cx_m + tw, cy_m + h + 4,
                             start=0, extent=180,
                             fill="#003300", outline="", tags="face")

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

            self._log("system", "Loading LLM…")
            self.llm.load(callback=lambda m: self._log("system", m))
            self._log("system", f"LLM mode: {self.llm.mode}")

            if self.sam.available:
                self._log("system", "EfficientViT-SAM: ready")

            self.transcriber = Transcriber()
            loaded = self.transcriber.load(callback=lambda m: self._log("system", m))
            print(f"[DEBUG] Whisper load returned: {loaded}, model is None: {self.transcriber.model is None}")

            if IS_JETSON:
                gc.collect()

            if not loaded or self.transcriber.model is None:
                self._log("system", "WARNING: Speech recognition failed to load. Check faster-whisper install.")
                self.root.after(0, lambda: self._set_status("STT FAILED", self.THEME["accent_listen"]))
                return

            self._ready = True
            print("[DEBUG] _ready set to True")
            self._log("system", "Wilson is ready. Click the button or press Space.")
            self.root.after(0, lambda: self._set_status(self.I18N["status_ready"], self.THEME["accent_ready"]))
            self.root.after(0, lambda: self._fade_button_color(self.THEME["accent_ready"]))

        except Exception as e:
            self._log("system", f"Init error: {e}")
            self.root.after(0, lambda: self._set_status(self.I18N["status_error"], self.THEME["accent_listen"]))

    # ── listening controls ────────────────────────────────────────────────────

    def _toggle_auto_listen(self):
        """Toggle hands-free voice-activated mode."""
        if not self._ready:
            self._log("system", "Still loading, please wait\u2026")
            return
        self._auto_listen = not self._auto_listen
        t = self.THEME
        if self._auto_listen:
            self.auto_btn.config(fg="#00ff41", bg="#002200", highlightbackground="#00ff41")
            self._log("system", "Auto-listen ON \u2014 speak anytime, Wilson will detect your voice.")
            self._set_status("AUTO \u2014 LISTENING", t["accent_ready"])
            threading.Thread(target=self._auto_listen_loop, daemon=True).start()
        else:
            self.auto_btn.config(fg="#004d1a", bg="#0a1a0a", highlightbackground="#003300")
            self._log("system", "Auto-listen OFF")
            self._set_status(self.I18N["status_ready"], t["accent_ready"])

    def _auto_listen_loop(self):
        """Background loop: wait for voice activity, record, then process.
        Runs while _auto_listen is True and we're not already busy."""
        NOISE_FLOOR = 0.012       # volume threshold to detect speech
        CONFIRM_SECS = 0.3        # speech must persist this long to trigger
        CHECK_INTERVAL = 0.05     # polling interval in seconds

        while self._auto_listen:
            # Wait until not busy
            if self.is_listening or self.is_processing:
                time.sleep(0.2)
                continue

            # Start a monitoring stream to watch for voice activity
            try:
                self.recorder.start()
            except Exception:
                time.sleep(1)
                continue

            # Phase 1: Wait for voice onset
            speech_time = 0.0
            detected = False
            while self._auto_listen and not self.is_processing:
                time.sleep(CHECK_INTERVAL)
                try:
                    vol = self.recorder.get_volume()
                except Exception:
                    break
                if vol >= NOISE_FLOOR:
                    speech_time += CHECK_INTERVAL
                    if speech_time >= CONFIRM_SECS:
                        detected = True
                        break
                else:
                    speech_time = 0.0

            if not detected or not self._auto_listen:
                try:
                    self.recorder.stop()
                except Exception:
                    pass
                continue

            # Speech detected — switch to full listening mode on the UI thread
            self.root.after(0, self._auto_start_capture)

            # Phase 2: Wait for processing to finish before looping back
            while self._auto_listen and (self.is_listening or self.is_processing):
                time.sleep(0.2)

            # Small cooldown before re-arming so Wilson's own TTS doesn't re-trigger
            if self._auto_listen:
                time.sleep(1.0)

    def _auto_start_capture(self):
        """Called on the UI thread when auto-listen detects voice."""
        if self.is_processing or self.is_listening:
            return
        # The recorder is already running from the detection phase —
        # just set the UI to listening state and arm the silence watchdog
        self.is_listening = True
        self.btn.config(text=self.I18N["btn_listen"])
        self._fade_button_color(self.THEME["accent_listen"])
        self._set_status(self.I18N["status_dictate"], self.THEME["accent_listen"])
        threading.Thread(target=self._silence_watchdog, daemon=True).start()

    def _toggle_listening(self):
        if self.is_processing:
            return
        if not self._ready:
            self._log("system", "Still loading, please wait\u2026")
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

    # ── camera / face recognition ─────────────────────────────────────────────

    def _toggle_camera(self):
        """Toggle the camera feed for face recognition (inline in main window)."""
        if self._camera_active:
            self._close_camera()
        else:
            # Close faces panel if open (they share the same area)
            if self._faces_panel_open:
                self._close_faces_panel()
            self._open_camera()

    def _open_camera(self):
        if not self.face_engine.available:
            self._log("system", "Face recognition unavailable (opencv not installed)")
            return
        if not self.face_engine.open_camera():
            self._log("system", "Could not open camera")
            return

        self._camera_active = True
        self._last_greeted = None
        self._unknown_streak = 0
        self._enroll_dialog_open = False
        self.cam_btn.config(fg="#00ff41", bg="#002200", highlightbackground="#00ff41")

        # Hide the animated face canvas; show camera canvas in its place
        t = self.THEME
        self.face_canvas.pack_forget()

        # Camera container (replaces face_canvas slot)
        self._cam_frame = tk.Frame(self.main_container, bg=t["bg_main"])
        # Insert camera frame where face_canvas was (before status row)
        # Pack it right after the subtitle — find the right spot
        self._cam_frame.pack(after=self._subtitle_label, pady=(4, 2))

        self._cam_canvas = tk.Canvas(
            self._cam_frame, width=200, height=150,
            bg="#000000", highlightthickness=1, highlightbackground="#003300",
        )
        self._cam_canvas.pack()

        self._cam_label = tk.Label(
            self._cam_frame, text="Scanning…",
            font=(FONT_MONO, 8), fg="#00ff41", bg=t["bg_main"],
        )
        self._cam_label.pack()

        self._cam_photo = None  # prevent GC
        self._camera_tick()

    def _close_camera(self):
        self._camera_active = False
        self.face_engine.close_camera()
        self.cam_btn.config(fg="#004d1a", bg="#0a1a0a", highlightbackground="#003300")

        # Remove camera frame, restore animated face canvas
        try:
            if hasattr(self, '_cam_frame') and self._cam_frame is not None:
                self._cam_frame.destroy()
                self._cam_frame = None
        except tk.TclError:
            pass

        # Re-show the animated face canvas where it was
        try:
            self.face_canvas.pack(after=self._subtitle_label, pady=(8, 2))
            self._draw_face()
        except tk.TclError:
            pass

    def _camera_tick(self):
        """Grab a frame, detect/recognise faces, update preview.  ~15 FPS."""
        if not self._camera_active:
            return
        frame = self.face_engine.capture_frame()
        if frame is None:
            try:
                self.root.after(66, self._camera_tick)
            except tk.TclError:
                pass
            return

        import cv2

        rects, gray = self.face_engine.detect_faces(frame)

        # ── Enrollment mode: collecting samples ──
        if self._enrolling_name is not None:
            for (x, y, w, h) in rects:
                self.face_engine.add_sample(self._enrolling_name, gray, x, y, w, h)
                self._enroll_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f"Enrolling… {self._enroll_count}/{self.face_engine.ENROLL_SAMPLES}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if self._enroll_count >= self.face_engine.ENROLL_SAMPLES:
                self.face_engine.save()
                name = self._enrolling_name
                self._enrolling_name = None
                self._enroll_count = 0
                self._log("system", f"Enrolled '{name}' ({len(self.face_engine._known_faces.get(name, []))} samples)")
                self._last_greeted = name
                # Greet the newly enrolled person
                threading.Thread(target=self._greet_person, args=(name, True), daemon=True).start()

        else:
            # ── Normal recognition mode ──
            for (x, y, w, h) in rects:
                name, conf = self.face_engine.recognize(gray, x, y, w, h)
                if name is not None:
                    color = (0, 255, 65)   # green
                    label = f"{name} ({conf:.0%})"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    self._unknown_streak = 0
                    if name != self._last_greeted:
                        self._last_greeted = name
                        threading.Thread(target=self._greet_person, args=(name, False), daemon=True).start()
                else:
                    color = (0, 0, 255)    # red
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, "Unknown", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    self._unknown_streak += 1
                    # After 30 consecutive unknown frames (~2 sec), prompt to enroll
                    if self._unknown_streak >= 30 and not self._enroll_dialog_open and time.time() >= self._enroll_cooldown:
                        self._unknown_streak = 0
                        self.root.after(0, self._prompt_enroll)

        # ── Render frame to tkinter canvas (PPM method — no PIL needed) ──
        try:
            preview = cv2.resize(frame, (200, 150))
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            ppm = f"P6\n200 150\n255\n".encode() + rgb.tobytes()
            self._cam_photo = tk.PhotoImage(data=ppm, format="PPM")
            self._cam_canvas.create_image(0, 0, anchor=tk.NW, image=self._cam_photo)
        except Exception:
            pass

        # Update status label
        try:
            if self._enrolling_name:
                self._cam_label.config(text=f"Enrolling: {self._enrolling_name}")
            elif rects:
                self._cam_label.config(text=f"Face{'s' if len(rects) > 1 else ''} detected: {len(rects)}")
            else:
                self._cam_label.config(text="Scanning…")
        except tk.TclError:
            pass

        try:
            self.root.after(66, self._camera_tick)   # ~15 FPS
        except tk.TclError:
            pass

    def _prompt_enroll(self):
        """Show a dialog asking the user to name an unknown face."""
        # Guard: skip if already enrolling, dialog already open, or cooldown active
        if self._enrolling_name is not None:
            return
        if self._enroll_dialog_open:
            return
        if time.time() < self._enroll_cooldown:
            return

        self._enroll_dialog_open = True
        try:
            from tkinter import simpledialog
            name = simpledialog.askstring(
                "New Face Detected",
                "Enter the person's name (or Cancel to skip):",
                parent=self.root,
            )
            if name and name.strip():
                name = name.strip()
                self._enrolling_name = name
                self._enroll_count = 0
                self._log("system", f"Look at the camera — enrolling '{name}'…")
            else:
                # User cancelled — enforce 10-second cooldown before asking again
                self._enroll_cooldown = time.time() + 10.0
        except Exception:
            pass
        finally:
            self._enroll_dialog_open = False

    def _greet_person(self, name, is_new):
        """Wilson greets a recognised (or newly enrolled) person via TTS."""
        if is_new:
            greeting = f"Nice to meet you, {name}! I'll remember your face."
        else:
            greeting = f"Hello, {name}!"
        self._log("wilson", greeting)
        if self.tts:
            self.tts.speak(greeting, log_fn=lambda m: self._log("system", m))

    # ── face management panel ─────────────────────────────────────────────────

    def _toggle_faces_panel(self):
        """Toggle an inline face-management panel (replaces face canvas area)."""
        if self._faces_panel_open:
            self._close_faces_panel()
        else:
            self._open_faces_panel()

    def _open_faces_panel(self):
        """Show a panel listing all known faces with delete buttons, inline."""
        # Close camera first if active (they share the same display area)
        if self._camera_active:
            self._close_camera()

        self._faces_panel_open = True
        self.faces_btn.config(fg="#00ff41", bg="#002200", highlightbackground="#00ff41")

        t = self.THEME
        self.face_canvas.pack_forget()

        self._faces_frame = tk.Frame(self.main_container, bg=t["bg_main"])
        self._faces_frame.pack(after=self._subtitle_label, pady=(4, 2))

        tk.Label(
            self._faces_frame, text="[ FACE MANAGEMENT ]",
            font=(FONT_MONO, 10, "bold"), fg="#00ff41", bg=t["bg_main"],
        ).pack(pady=(2, 4))

        names = self.face_engine.known_names
        if not names:
            tk.Label(
                self._faces_frame, text="No faces enrolled yet.",
                font=(FONT_MONO, 9), fg="#004d1a", bg=t["bg_main"],
            ).pack(pady=4)
        else:
            # Scrollable list frame (max height ~100px)
            list_frame = tk.Frame(self._faces_frame, bg=t["bg_main"])
            list_frame.pack(fill=tk.X, padx=10)

            for name in sorted(names):
                row = tk.Frame(list_frame, bg=t["bg_main"])
                row.pack(fill=tk.X, pady=1)

                samples = len(self.face_engine._known_faces.get(name, []))
                tk.Label(
                    row, text=f"  {name} ({samples} samples)",
                    font=(FONT_MONO, 9), fg="#00cc33", bg=t["bg_main"],
                    anchor="w",
                ).pack(side=tk.LEFT, fill=tk.X, expand=True)

                del_btn = tk.Button(
                    row, text="DEL",
                    font=(FONT_MONO, 8, "bold"),
                    bg="#1a0000", fg="#ff0040",
                    activebackground="#330000", activeforeground="#ff3366",
                    highlightthickness=0, relief=tk.FLAT, bd=0,
                    cursor="hand2", width=4,
                    command=lambda n=name: self._delete_face(n),
                )
                del_btn.pack(side=tk.RIGHT, padx=2)

        # Add "Enroll New" button at the bottom
        enroll_btn = tk.Button(
            self._faces_frame, text="+ ENROLL NEW",
            font=(FONT_MONO, 9, "bold"),
            bg="#001a00", fg="#00ff41",
            activebackground="#003300", activeforeground="#33ff66",
            highlightthickness=1, highlightbackground="#003300",
            relief=tk.SOLID, bd=1, cursor="hand2",
            command=self._enroll_from_panel,
        )
        enroll_btn.pack(pady=(6, 2))

    def _close_faces_panel(self):
        """Close inline faces panel and restore animated face canvas."""
        self._faces_panel_open = False
        self.faces_btn.config(fg="#004d1a", bg="#0a1a0a", highlightbackground="#003300")
        try:
            if hasattr(self, '_faces_frame') and self._faces_frame is not None:
                self._faces_frame.destroy()
                self._faces_frame = None
        except tk.TclError:
            pass
        try:
            self.face_canvas.pack(after=self._subtitle_label, pady=(8, 2))
            self._draw_face()
        except tk.TclError:
            pass

    def _delete_face(self, name):
        """Delete a known face and refresh the panel."""
        self.face_engine.remove_person(name)
        self._log("system", f"Removed '{name}' from face database")
        # Refresh the panel
        self._close_faces_panel()
        self._open_faces_panel()

    def _enroll_from_panel(self):
        """Open the camera and start enrollment from the management panel."""
        self._close_faces_panel()
        # Open camera — it will handle enrollment once a face is detected as unknown
        self._open_camera()
        self._log("system", "Camera opened for enrollment. Show your face…")

    # ── text chat ─────────────────────────────────────────────────────────────

    def _send_text(self):
        """Send typed text through the LLM pipeline (skips STT)."""
        text = self.chat_input.get().strip()
        if not text or self.is_processing:
            return
        if not self._ready:
            self._log("system", "Still loading, please wait\u2026")
            return
        self.chat_input.delete(0, tk.END)
        self.is_processing = True
        self.btn.config(text=self.I18N["btn_process"], state=tk.DISABLED)
        self._fade_button_color(self.THEME["accent_process"])
        self._set_status(self.I18N["status_process"], self.THEME["accent_process"])
        threading.Thread(target=self._process_text, args=(text,), daemon=True).start()

    def _process_text(self, text):
        """Process typed text: LLM → TTS (no STT needed)."""
        try:
            self._log("user", text)

            # LLM
            self.root.after(0, lambda: self._set_status(self.I18N["status_think"], self.THEME["accent_process"]))
            response = self.llm.query(text)

            # TTS with mouth animation
            self.root.after(0, lambda: self._set_status(self.I18N["status_speak"], self.THEME["accent_ready"]))
            audio_data, sr = self.tts.generate_wav(response, log_fn=lambda m: self._log("system", m))
            if audio_data is not None and sr is not None:
                amps = self._compute_mouth_amplitudes(audio_data, sr)
                duration = len(audio_data) / sr
                self._log("wilson", response)
                self.root.after(0, lambda a=amps, d=duration: self._start_mouth_anim(a, d))
                _get_sd().play(audio_data, sr)
                _get_sd().wait()
                self.root.after(0, self._stop_mouth_anim)
            else:
                self._log("wilson", response)
                self.tts.speak(response, log_fn=lambda m: self._log("system", m))

        except Exception as e:
            self._log("system", f"Pipeline error: {e}")
        finally:
            self.root.after(0, self._reset)

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

            # TTS with mouth animation — text appears in sync with speech
            self.root.after(0, lambda: self._set_status(self.I18N["status_speak"], self.THEME["accent_ready"]))
            audio_data, sr = self.tts.generate_wav(response, log_fn=lambda m: self._log("system", m))
            if audio_data is not None and sr is not None:
                amps = self._compute_mouth_amplitudes(audio_data, sr)
                duration = len(audio_data) / sr
                # Queue the wilson text to appear AS speech plays (synced with audio)
                self._log("wilson", response)
                self.root.after(0, lambda a=amps, d=duration: self._start_mouth_anim(a, d))
                _get_sd().play(audio_data, sr)
                _get_sd().wait()
                self.root.after(0, self._stop_mouth_anim)
            else:
                self._log("wilson", response)
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
            if self._auto_listen:
                self._set_status("AUTO \u2014 LISTENING", self.THEME["accent_ready"])
            else:
                self._set_status(self.I18N["status_ready"], self.THEME["accent_ready"])
        except tk.TclError:
            pass

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self):
        self.is_listening = False
        self._auto_listen = False
        self._camera_active = False
        try:
            self.face_engine.close_camera()
        except Exception:
            pass
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
            if hasattr(self, "_proxy"):
                self._proxy.destroy()
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

        self._print("system", "Loading LLM\u2026")
        self.llm.load(callback=lambda m: self._print("system", m))
        self._print("system", f"LLM mode: {self.llm.mode}")

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
    print("\n  LLM:")
    if USE_EMBEDDED_LLM:
        model_path = os.path.join(MODELS_DIR, EMBEDDED_MODEL_FILE)
        check("Embedded model file", os.path.isfile(model_path),
              EMBEDDED_MODEL_FILE if os.path.isfile(model_path) else "will be downloaded on first run")
        try:
            from llama_cpp import Llama
            check("llama-cpp-python", True)
        except ImportError:
            check("llama-cpp-python", False, "pip install llama-cpp-python")
    else:
        print("  (Remote mode — checking API endpoint)")
        try:
            r = requests.get(
                LLM_URL.replace("/chat/completions", "/models"),
                timeout=(2, 5),
            )
            check("API reachable", r.status_code == 200, LLM_URL)
        except Exception:
            check("API reachable", False, f"{LLM_URL} — is the server running?")

    # EfficientViT-SAM
    print("\n  EfficientViT-SAM (optional):")
    enc_path = os.path.join(MODELS_DIR, EfficientViTSAMEngine.ENCODER_FILE)
    dec_path = os.path.join(MODELS_DIR, EfficientViTSAMEngine.DECODER_FILE)
    info("Image encoder", os.path.isfile(enc_path),
         enc_path if os.path.isfile(enc_path) else "place in models/")
    info("Mask decoder", os.path.isfile(dec_path),
         dec_path if os.path.isfile(dec_path) else "place in models/")
    try:
        import onnxruntime
        info("onnxruntime", True, onnxruntime.__version__)
    except ImportError:
        info("onnxruntime", False, "pip install onnxruntime-gpu (optional)")

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
    ]
    if USE_EMBEDDED_LLM:
        lines.append(f"  LLM       : Embedded ({EMBEDDED_MODEL_FILE})")
    else:
        lines.append(f"  LLM       : {LLM_URL}")
    if LLM_MODEL:
        lines.append(f"  Model     : {LLM_MODEL}")
    lines.append(f"  TTS       : Piper ({'found' if os.path.isfile(PIPER_EXE) else 'NOT FOUND'})")
    sam_enc = os.path.join(MODELS_DIR, EfficientViTSAMEngine.ENCODER_FILE)
    lines.append(f"  SAM       : {'found' if os.path.isfile(sam_enc) else 'not installed (optional)'}")
    lines.append(f"  Faces     : {len(FaceRecognitionEngine().known_names)} known person(s)")

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
