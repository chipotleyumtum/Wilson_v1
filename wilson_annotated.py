"""
Annotated study copy of wilson.py.

Purpose:
- Preserve the original runtime file unchanged.
- Provide a readable, heavily commented version you can study and edit safely.
- Keep the executable logic functionally equivalent to the original file.

How to use this file:
- Read it side-by-side with wilson.py.
- Make experimental changes here first.
- Once you understand a change, copy the same change into wilson.py.
"""

# Standard library imports used throughout the app.
import threading          # Runs background work without freezing the GUI.
import subprocess         # Launches external programs like Piper, espeak, tegrastats.
import requests           # Sends HTTP requests to the local LLM server.
import os                 # Reads environment variables and builds file paths.
import sys                # Detects platform info and command-line arguments.
import platform           # Detects CPU architecture and OS details.
import tempfile           # Creates temporary paths for generated WAV files.
import time               # Sleeps, timestamps animations, and watchdog polling.
import queue              # Safely passes messages from worker threads to the GUI thread.
import gc                 # Forces garbage collection on memory-constrained devices.
import re                 # Parses tegrastats output and strips formatting from text.
import random             # Drives UI glitches, particles, and blinking animations.
import shutil             # Checks if external commands exist in PATH.
import math               # Reserved for math-heavy UI work and parity with the original file.
import numpy as np        # Stores and transforms audio as numeric arrays.

# sounddevice is intentionally lazy-loaded.
# Reason: importing it can trigger device enumeration, which sometimes hangs
# on certain Windows audio stacks before the app is even ready.
_sd = None


def _get_sd():
    """Import and cache sounddevice the first time audio is needed."""
    global _sd                            # Reuse the same module object after first import.
    if _sd is None:                       # Only import once.
        import sounddevice                # Import here instead of at module load time.
        _sd = sounddevice                 # Cache the imported module globally.
    return _sd                            # Return the cached module every time.


# Try to load tkinter so the app can offer a GUI.
try:
    import tkinter as tk                 # Main GUI toolkit bundled with Python on most platforms.
    from tkinter import scrolledtext, ttk  # Imported for parity; ttk/scrolledtext are not central here.
    HAS_GUI = True                       # Flag that the graphical interface is available.
except ImportError:
    HAS_GUI = False                      # If tkinter is missing, the app falls back to terminal mode.


# Platform flags drive most cross-platform behavior in the rest of the app.
IS_LINUX = sys.platform.startswith("linux")   # True on Linux, including Jetson boards.
IS_WINDOWS = sys.platform == "win32"          # True on Windows.
IS_MACOS = sys.platform == "darwin"           # True on macOS.
IS_ARM64 = platform.machine() in ("aarch64", "arm64")  # True on ARM CPUs like Apple Silicon or Jetson.
IS_JETSON = IS_LINUX and IS_ARM64              # Heuristic: Linux + ARM in this project means Jetson.


def _detect_jetson_model():
    """Read the hardware model string from the Linux device tree on Jetson."""
    try:
        # Jetson boards expose the model name in this special file.
        with open("/proc/device-tree/model", "r") as file_handle:
            return file_handle.read().strip().rstrip("\x00")
    except Exception:
        # If the file is missing or unreadable, still return a useful fallback on Jetson.
        return "Jetson (unknown)" if IS_JETSON else None


JETSON_MODEL = _detect_jetson_model()   # Cache the model once at startup.

# Jetson systems often need CUDA device visibility set explicitly.
if IS_JETSON:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _cuda_is_usable():
    """Return True when the machine appears able to run Whisper on CUDA."""
    # macOS does not support CUDA in this app's Whisper backend.
    if IS_MACOS:
        return False

    # Most reliable path: ask PyTorch if a CUDA device is available.
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[CUDA] {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass

    # Second-best path: ask CTranslate2 directly whether CUDA compute types exist.
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            return True
    except Exception:
        pass

    # Windows-specific fallback checks.
    if IS_WINDOWS:
        try:
            import nvidia.cublas        # Presence often means CUDA runtime pieces are installed.
            return True
        except ImportError:
            pass

        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path and os.path.isdir(os.path.join(cuda_path, "bin")):
            return True

        if shutil.which("nvidia-smi") is not None:
            return True
        return False

    # Linux-specific fallback check.
    if IS_LINUX:
        import ctypes.util
        if ctypes.util.find_library("cuda") is None:
            return False
        return True

    return False


CUDA_AVAILABLE = _cuda_is_usable()      # Resolve once so the rest of config can depend on it.
WILSON_DIR = os.path.dirname(os.path.abspath(__file__))  # Folder containing this file.

# Piper text-to-speech binary path changes by platform.
PIPER_EXE = os.path.join(WILSON_DIR, "piper", "piper.exe" if IS_WINDOWS else "piper")

# Voice model lives at the project root next to wilson.py.
PIPER_VOICE = os.path.join(WILSON_DIR, "en_US-amy-medium.onnx")

# Choose a default LLM backend URL based on platform conventions.
if IS_JETSON:
    LLM_URL = os.environ.get("WILSON_LLM_URL", "http://localhost:11434/v1/chat/completions")
    LLM_MODEL = os.environ.get("WILSON_LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
elif IS_MACOS or IS_LINUX:
    LLM_URL = os.environ.get("WILSON_LLM_URL", "http://localhost:11434/v1/chat/completions")
    LLM_MODEL = os.environ.get("WILSON_LLM_MODEL", "")
else:
    LLM_URL = os.environ.get("WILSON_LLM_URL", "http://localhost:1234/v1/chat/completions")
    LLM_MODEL = os.environ.get("WILSON_LLM_MODEL", "")

# Decide which Whisper device/computation settings are safest by default.
_default_device = "cuda" if CUDA_AVAILABLE else "cpu"
if CUDA_AVAILABLE:
    _default_compute = "float16"
elif IS_MACOS and IS_ARM64:
    _default_compute = "float32"
else:
    _default_compute = "int8"

# Allow environment variables to override Whisper defaults.
WHISPER_MODEL = os.environ.get("WILSON_WHISPER_MODEL", "base" if IS_JETSON else "small")
WHISPER_COMPUTE = os.environ.get("WILSON_WHISPER_COMPUTE", _default_compute)
WHISPER_DEVICE = os.environ.get("WILSON_WHISPER_DEVICE", _default_device)

# Audio sample rate Whisper expects.
SAMPLE_RATE = 16000

# Token budget for the LLM response.
MAX_TOKENS = int(os.environ.get("WILSON_MAX_TOKENS", "1024"))

# Local models can be slow, so timeouts are intentionally generous.
LLM_TIMEOUT = int(os.environ.get("WILSON_LLM_TIMEOUT", "120" if IS_JETSON else "90"))

# System prompt defines Wilson's baseline personality and response style.
SYSTEM_PROMPT = os.environ.get(
    "WILSON_SYSTEM_PROMPT",
    "You are Wilson, a helpful AI assistant. "
    "Keep responses concise (1-3 sentences). No emojis or markdown.",
)

# Headless mode can be enabled either by environment variable or CLI switch.
HEADLESS = (
    os.environ.get("WILSON_HEADLESS", "0") == "1"
    or "--headless" in sys.argv
)

# Cross-platform font choices keep the UI native-looking on each OS.
if IS_LINUX:
    FONT_UI, FONT_MONO = "Ubuntu", "Ubuntu Mono"
elif IS_MACOS:
    FONT_UI, FONT_MONO = "Helvetica Neue", "Menlo"
else:
    FONT_UI, FONT_MONO = "Segoe UI", "Consolas"


class JetsonMonitor:
    """Read Jetson telemetry from tegrastats and expose simple properties."""

    def __init__(self):
        self.gpu_util = 0             # GPU utilization percentage.
        self.cpu_temp = 0.0           # CPU temperature in Celsius.
        self.gpu_temp = 0.0           # GPU temperature in Celsius.
        self.ram_used_mb = 0          # RAM currently used in megabytes.
        self.ram_total_mb = 0         # Total RAM in megabytes.
        self.power_mw = 0             # Input power draw in milliwatts.
        self._running = False         # Controls whether the background poll loop keeps running.
        self._proc = None             # Holds the tegrastats subprocess when active.

    def start(self):
        """Start telemetry polling on Jetson; do nothing elsewhere."""
        if not IS_JETSON:
            return
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def stop(self):
        """Stop polling and terminate the tegrastats process if it exists."""
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _poll_loop(self):
        """Launch tegrastats and continuously parse each output line."""
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
            pass
        except Exception:
            pass

    def _parse(self, line):
        """Extract RAM, GPU, temperature, and power values from one tegrastats line."""
        try:
            match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            if match:
                self.ram_used_mb = int(match.group(1))
                self.ram_total_mb = int(match.group(2))

            match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            if match:
                self.gpu_util = int(match.group(1))

            match = re.search(r"cpu@(\d+\.?\d*)C", line, re.IGNORECASE)
            if match:
                self.cpu_temp = float(match.group(1))

            match = re.search(r"gpu@(\d+\.?\d*)C", line, re.IGNORECASE)
            if match:
                self.gpu_temp = float(match.group(1))

            match = re.search(r"VDD_IN\s+(\d+)", line)
            if match:
                self.power_mw = int(match.group(1))
        except Exception:
            pass

    @property
    def summary(self):
        """Return a single formatted telemetry string for the GUI footer."""
        if not IS_JETSON or self.ram_total_mb == 0:
            return ""
        ram_percent = self.ram_used_mb / self.ram_total_mb * 100
        return (
            f"RAM {self.ram_used_mb}/{self.ram_total_mb}MB ({ram_percent:.0f}%)  "
            f"GPU {self.gpu_util}%  "
            f"Temp {self.cpu_temp:.0f}/{self.gpu_temp:.0f}°C  "
            f"{self.power_mw / 1000:.1f}W"
        )

    @property
    def ram_free_mb(self):
        """Expose remaining RAM so the rest of the app does not re-compute it."""
        return max(0, self.ram_total_mb - self.ram_used_mb)


def set_jetson_power_mode(mode="MAXN"):
    """Attempt to raise Jetson power/performance settings. Requires sudo."""
    if not IS_JETSON:
        return False

    modes = {"MAXN": 0, "25W": 0, "15W": 1, "7W": 2}
    mode_id = modes.get(mode.upper(), 0)
    try:
        subprocess.run(["sudo", "nvpmodel", "-m", str(mode_id)], capture_output=True, timeout=10)
        subprocess.run(["sudo", "jetson_clocks"], capture_output=True, timeout=10)
        print(f"[POWER] Mode {mode} (id {mode_id}), clocks maximized")
        return True
    except Exception as error:
        print(f"[POWER] Could not set mode: {error}")
        return False


class AudioRecorder:
    """Handle microphone discovery, live capture, resampling, and gain correction."""

    # Device names containing these words are treated as good external microphones.
    PREFERRED_KEYWORDS = [
        "anker", "powerconf", "poweranc", "respeaker", "seeed",
        "jabra", "blue yeti", "rode", "fifine", "amd bluetooth",
    ]

    # Automatic gain control targets.
    _AGC_TARGET_PEAK = 0.5
    _AGC_MIN_PEAK = 0.002

    # On Windows, host APIs are ranked because some are more reliable for Bluetooth mics.
    _HOSTAPI_RANK = {"Windows DirectSound": 0, "Windows WASAPI": 1, "MME": 2, "Windows WDM-KS": 3}

    def __init__(self):
        self.is_recording = False            # True while InputStream should keep appending chunks.
        self.audio_data = []                 # List of chunk arrays captured by the audio callback.
        self.stream = None                   # Active sounddevice input stream.
        self._native_sr = SAMPLE_RATE        # Sample rate used by the actual microphone stream.
        self._all_candidates = []            # Ordered microphone candidates to try opening.
        self._live_gain = 1.0                # UI-only multiplier for live volume metering.
        self.device_id = self._find_device() # Select the best initial microphone candidate.

    def _find_device(self):
        """Rank all input devices and return the best candidate's device id."""
        sd = _get_sd()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        api_names = {index: host_api["name"] for index, host_api in enumerate(hostapis)}
        by_name = {}

        for index, device in enumerate(devices):
            if device["max_input_channels"] <= 0:
                continue
            api_name = api_names.get(device["hostapi"], "")
            rank = self._HOSTAPI_RANK.get(api_name, 99)
            base_name = device["name"].strip()
            by_name.setdefault(base_name, []).append((rank, index, device))

        for name in by_name:
            by_name[name].sort(key=lambda item: item[0])

        candidates = []
        seen_ids = set()

        def _add_candidates(base_name, label):
            for rank, device_index, device_info in by_name.get(base_name, []):
                if device_index in seen_ids:
                    continue
                seen_ids.add(device_index)
                sample_rate = self._pick_samplerate(device_index, device_info.get("default_samplerate", SAMPLE_RATE))
                api_name = api_names.get(device_info["hostapi"], "?")
                candidates.append(
                    (device_index, int(sample_rate), f"{label}: {device_info['name']} (id {device_index}, {api_name}, {int(sample_rate)} Hz)")
                )

        for base_name in by_name:
            if any(keyword in base_name.lower() for keyword in self.PREFERRED_KEYWORDS):
                _add_candidates(base_name, "Preferred")

        for base_name in by_name:
            lowered = base_name.lower()
            if "bluetooth" in lowered or "bt " in lowered or "headset" in lowered:
                _add_candidates(base_name, "Bluetooth")

        for base_name in by_name:
            if "usb" in base_name.lower():
                _add_candidates(base_name, "USB")

        try:
            mapper_info = sd.query_devices(0)
            mapper_api = api_names.get(mapper_info["hostapi"], "?")
            if mapper_info["max_input_channels"] > 0 and "mme" in mapper_api.lower() and 0 not in seen_ids:
                mapper_sample_rate = 16000
                candidates.append((0, mapper_sample_rate, f"SoundMapper: {mapper_info['name']} (id 0, {mapper_api}, {mapper_sample_rate} Hz)"))
                seen_ids.add(0)
        except Exception:
            pass

        default_device = sd.default.device[0]
        if default_device is not None and default_device >= 0 and default_device not in seen_ids:
            device_info = sd.query_devices(default_device)
            sample_rate = int(device_info.get("default_samplerate", SAMPLE_RATE))
            api_name = api_names.get(device_info["hostapi"], "?")
            candidates.append((default_device, sample_rate, f"Default: {device_info['name']} (id {default_device}, {api_name}, {sample_rate} Hz)"))
            seen_ids.add(default_device)

        for index, device in enumerate(devices):
            if device["max_input_channels"] > 0 and index not in seen_ids:
                sample_rate = int(device.get("default_samplerate", SAMPLE_RATE))
                api_name = api_names.get(device["hostapi"], "?")
                candidates.append((index, sample_rate, f"Fallback: {device['name']} (id {index}, {api_name}, {sample_rate} Hz)"))
                seen_ids.add(index)

        self._all_candidates = candidates
        if not candidates:
            raise RuntimeError("No microphone found")

        top_id, top_sr, top_label = candidates[0]
        self._native_sr = top_sr
        print(f"[MIC] {top_label}")
        return top_id

    def _pick_samplerate(self, device_idx, reported_sr):
        """Prefer 16 kHz if the device supports it; otherwise use a workable native rate."""
        reported_sr = int(reported_sr)
        if reported_sr == SAMPLE_RATE:
            return SAMPLE_RATE
        try:
            _get_sd().check_input_settings(device=device_idx, samplerate=SAMPLE_RATE, channels=1)
            return SAMPLE_RATE
        except Exception:
            pass
        if reported_sr >= 8000:
            return reported_sr
        for sample_rate in [44100, 48000, 22050, 8000]:
            try:
                _get_sd().check_input_settings(device=device_idx, samplerate=sample_rate, channels=1)
                return sample_rate
            except Exception:
                continue
        return reported_sr

    def _callback(self, indata, frames, time_info, status):
        """Receive one audio chunk from sounddevice and store a copy of it."""
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def _open_stream(self, target_device, samplerate):
        """Open a sounddevice input stream for a chosen device and sample rate."""
        self.stream = _get_sd().InputStream(
            device=target_device,
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self.stream.start()

    def start(self):
        """Start recording by trying candidate microphones until one opens successfully."""
        self.audio_data = []
        self.is_recording = True
        self._live_gain = 1.0
        last_error = None

        for device_id, sample_rate, label in self._all_candidates:
            for try_sample_rate in dict.fromkeys([sample_rate, 16000, 44100, 48000]):
                try:
                    self._open_stream(device_id, try_sample_rate)
                    if device_id != self.device_id or try_sample_rate != self._native_sr:
                        print(f"[MIC] Opened → {label}" + (f" (at {try_sample_rate} Hz)" if try_sample_rate != sample_rate else ""))
                    self.device_id = device_id
                    self._native_sr = try_sample_rate
                    if "soundmapper" in label.lower() or "sound mapper" in label.lower():
                        self._live_gain = 1000.0
                        print(f"[MIC] Sound Mapper detected — live gain ×{self._live_gain:.0f}")
                    return
                except Exception as error:
                    last_error = error
                    continue

        raise RuntimeError(f"Could not open any microphone. Last error: {last_error}")

    def stop(self):
        """Stop recording, merge chunks, resample to 16 kHz, and normalize quiet audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.audio_data:
            return None

        audio = np.concatenate(self.audio_data, axis=0).flatten()

        if self._native_sr != SAMPLE_RATE:
            try:
                from scipy.signal import resample
                output_samples = int(len(audio) * SAMPLE_RATE / self._native_sr)
                audio = resample(audio, output_samples).astype(np.float32)
            except ImportError:
                indices = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / self._native_sr))
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        peak = float(np.max(np.abs(audio)))
        if peak < self._AGC_MIN_PEAK:
            print(f"[MIC] Audio peak {peak:.6f} — too quiet, likely silence")
        elif peak < self._AGC_TARGET_PEAK:
            gain = self._AGC_TARGET_PEAK / peak
            audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
            print(f"[MIC] Auto-gain: {20 * np.log10(gain):.0f} dB boost (peak {peak:.6f} → {float(np.max(np.abs(audio))):.3f})")

        return audio

    def get_volume(self):
        """Compute RMS loudness from the most recent chunk for meters and silence detection."""
        if not self.audio_data:
            return 0.0
        recent = self.audio_data[-1]
        return float(np.sqrt(np.mean(recent ** 2))) * self._live_gain


class Transcriber:
    """Load faster-whisper and convert recorded audio arrays into text."""

    def __init__(self):
        self.model = None

    def load(self, callback=None):
        """Load Whisper with a fallback chain of device/compute configurations."""

        def log(message):
            if callback:
                callback(message)
            print(f"[WHISPER] {message}")

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log("ERROR: faster-whisper not installed.  pip install faster-whisper")
            return False

        log(f"Loading '{WHISPER_MODEL}' (target: {WHISPER_DEVICE}/{WHISPER_COMPUTE}) ...")

        combos = [
            (WHISPER_DEVICE, WHISPER_COMPUTE),
            ("cuda", "float16"),
            ("cuda", "int8"),
            ("cpu", "int8"),
            ("cpu", "float32"),
        ]

        seen = set()
        unique = []
        for combo in combos:
            if combo not in seen:
                seen.add(combo)
                unique.append(combo)

        for device, compute in unique:
            try:
                self.model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
                log(f"Loaded on {device.upper()} ({compute})")
                self._post_load_cleanup()
                return True
            except Exception as error:
                log(f"  {device}/{compute} failed: {error}")

        log("ERROR: All Whisper backends failed")
        return False

    def transcribe(self, audio):
        """Run Whisper on a float32 waveform and return the joined transcript text."""
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
                vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
            )
            text = " ".join(segment.text for segment in segments).strip()
        except Exception as error:
            lowered = str(error).lower()
            if any(keyword in lowered for keyword in ("cublas", "cuda", "cusparse", "cudnn", "gpu")):
                print(f"[WHISPER] CUDA failed at runtime: {error}")
                print("[WHISPER] Reloading model on CPU...")
                self._reload_cpu()
                return self.transcribe(audio)
            raise

        if IS_JETSON:
            gc.collect()

        return text

    def _reload_cpu(self):
        """Emergency runtime fallback when the CUDA backend dies after loading."""
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            print("[WHISPER] Reloaded on CPU (int8)")
        except Exception as error:
            print(f"[WHISPER] CPU reload also failed: {error}")
            self.model = None

    @staticmethod
    def _post_load_cleanup():
        """Free temporary allocations after model load, especially useful on Jetson."""
        if IS_JETSON:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class TTSEngine:
    """Turn Wilson's reply text into speech, preferring Piper over simpler fallbacks."""

    def __init__(self):
        self.has_piper = os.path.isfile(PIPER_EXE)
        self.has_espeak = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None
        self.has_say = IS_MACOS and shutil.which("say") is not None

        if self.has_piper:
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
        """Choose the best TTS backend and speak the cleaned text immediately."""
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
        """Generate TTS audio data without playing it, used by the animated GUI path."""
        text = re.sub(r"[*#`]", "", text).replace("\n", " ").strip()
        if not text or not self.has_piper:
            return None, None
        tmp_path = os.path.join(tempfile.gettempdir(), "wilson_tts_gen.wav")
        try:
            process = subprocess.Popen(
                [PIPER_EXE, "--model", PIPER_VOICE, "--output_file", tmp_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.communicate(input=text.encode("utf-8"), timeout=30)
            process.wait()
            time.sleep(0.05)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                import soundfile as sf
                data, sample_rate = sf.read(tmp_path)
                return data, sample_rate
        except Exception as error:
            if log_fn:
                log_fn(f"TTS generate error: {error}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return None, None

    def _piper(self, text, log_fn=None):
        """Run the Piper executable, read the generated WAV, and play it."""
        tmp_path = os.path.join(tempfile.gettempdir(), "wilson_tts.wav")
        try:
            process = subprocess.Popen(
                [PIPER_EXE, "--model", PIPER_VOICE, "--output_file", tmp_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.communicate(input=text.encode("utf-8"), timeout=30)
            process.wait()
            time.sleep(0.05)

            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                import soundfile as sf
                data, sample_rate = sf.read(tmp_path)
                _get_sd().play(data, sample_rate)
                _get_sd().wait()
        except subprocess.TimeoutExpired:
            if log_fn:
                log_fn("TTS timed out")
            try:
                process.kill()
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
        except Exception as error:
            if log_fn:
                log_fn(f"TTS error: {error}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _espeak(self, text, log_fn=None):
        """Fallback TTS path for Linux systems where Piper is missing."""
        command = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
        try:
            subprocess.run([command, "-v", "en", "-s", "160", "--", text], capture_output=True, timeout=30)
        except Exception as error:
            if log_fn:
                log_fn(f"espeak error: {error}")

    def _say(self, text, log_fn=None):
        """Fallback TTS path for macOS using the built-in say command."""
        try:
            subprocess.run(["say", "-v", "Samantha", text], capture_output=True, timeout=30)
        except Exception as error:
            if log_fn:
                log_fn(f"macOS say error: {error}")


class LLMClient:
    """Maintain chat history and query a local OpenAI-compatible chat endpoint."""

    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    @staticmethod
    def _strip_thinking(text):
        """Remove hidden reasoning blocks so Wilson only speaks the final answer."""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
        cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned)
        return cleaned.strip()

    def query(self, user_input):
        """Append the user's text to history, call the model, store full reply, return visible reply."""
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "messages": self.messages[-21:],
            "stream": False,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }
        if LLM_MODEL:
            payload["model"] = LLM_MODEL

        try:
            response = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            raw_reply = response.json()["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": raw_reply})
            return self._strip_thinking(raw_reply)
        except requests.exceptions.ConnectionError:
            backend = "LM Studio" if IS_WINDOWS else "Ollama"
            port = "1234" if IS_WINDOWS else "11434"
            return f"Cannot connect to {backend}. Start the server on port {port}."
        except requests.exceptions.Timeout:
            return (
                "LLM timed out. Model may be too large for available memory. "
                "Try a smaller quantisation (Q4_K_S) or reduce context length."
            )
        except Exception as error:
            return f"LLM error: {error}"

    def clear_history(self):
        """Reset the conversation while preserving the system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


class WilsonGUI:
    """Matrix-themed tkinter interface wrapped around the same audio and LLM pipeline."""

    THEME = {
        "bg_main": "#050505",
        "bg_panel": "#0a0c0a",
        "fg_title": "#00ff41",
        "fg_text": "#00ff41",
        "fg_sub": "#00cc33",
        "accent_ready": "#00ff41",
        "accent_ready_hover": "#33ff66",
        "accent_listen": "#ff0040",
        "accent_process": "#00cc33",
        "color_user": "#00ff41",
        "color_wilson": "#00ff41",
        "color_system": "#004d1a",
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
        "footer_hints": "[SPACE] or click  |  auto-stops on silence",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.I18N["title"])

        self._transparent_key = "#ab00cd"
        self.root.geometry("800x800")
        self.root.configure(bg=self._transparent_key)

        if IS_WINDOWS:
            self.root.attributes("-transparentcolor", self._transparent_key)
            self.root.overrideredirect(True)
            self.root.bind("<Escape>", lambda event: self.shutdown())
            self.root.after(50, self._create_taskbar_proxy)

        self._drag_x = 0
        self._drag_y = 0

        self.is_listening = False
        self.is_processing = False
        self._ready = False
        self._auto_listen = False
        self.recorder = None
        self.transcriber = None
        self.tts = None
        self.llm = LLMClient()
        self.monitor = JetsonMonitor()
        self.msg_queue = queue.Queue()

        self._matrix_busy = False
        self._pending_wilson_text = None
        self._matrix_rain_chars = []

        self._viz_bars = 24
        self._viz_levels = [0.0] * 24
        self._viz_particles = []

        self._mouth_active = False
        self._mouth_data = []
        self._mouth_openness = 0.0
        self._mouth_start = 0.0
        self._mouth_duration = 0.0
        self._eye_open = True

        self._font_title = (FONT_MONO, 22, "bold")
        self._font_sub = (FONT_MONO, 9)
        self._font_mono = (FONT_MONO, 10)
        self._font_btn = (FONT_MONO, 12, "bold")

        self._build_ui()
        self._check_queue()
        threading.Thread(target=self._initialize, daemon=True).start()

    def _create_taskbar_proxy(self):
        """Create a hidden proxy window so a frameless window still gets a taskbar icon on Windows."""
        try:
            import ctypes

            self._proxy = tk.Toplevel(self.root)
            self._proxy.title(self.I18N["title"])
            self._proxy.geometry("0x0+0+0")
            self._proxy.attributes("-alpha", 0.0)
            self._proxy.transient(None)
            self.root.wm_attributes("-topmost", False)

            hwnd_main = ctypes.windll.user32.GetParent(self.root.winfo_id())
            hwnd_proxy = ctypes.windll.user32.GetParent(self._proxy.winfo_id())

            if hwnd_main and hwnd_proxy:
                GWL_EXSTYLE = -20
                exstyle = ctypes.windll.user32.GetWindowLongW(hwnd_proxy, GWL_EXSTYLE)
                exstyle = exstyle & ~0x00000080
                exstyle = exstyle | 0x00040000
                ctypes.windll.user32.SetWindowLongW(hwnd_proxy, GWL_EXSTYLE, exstyle)
                ctypes.windll.user32.SetWindowLongW(hwnd_main, -8, hwnd_proxy)

            self._proxy.bind("<Unmap>", self._on_proxy_minimize)
            self._proxy.bind("<Map>", self._on_proxy_restore)
            self._proxy.protocol("WM_DELETE_WINDOW", self.shutdown)
        except Exception as error:
            print(f"[UI] Taskbar proxy failed: {error}")

    def _on_proxy_minimize(self, event=None):
        """Hide the real frameless window when the proxy is minimized."""
        try:
            self.root.withdraw()
        except tk.TclError:
            pass

    def _on_proxy_restore(self, event=None):
        """Show the main window again when the proxy is restored."""
        try:
            self.root.deiconify()
            self.root.lift()
        except tk.TclError:
            pass

    def _start_drag(self, event):
        """Store click offset so the circular frameless window can be dragged."""
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        """Move the window by comparing current pointer location to the stored click offset."""
        x = self.root.winfo_pointerx() - self._drag_x
        y = self.root.winfo_pointery() - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _build_ui(self):
        """Construct the entire GUI: shell, face, log, controls, and footer."""
        theme = self.THEME
        labels = self.I18N

        self.bg_canvas = tk.Canvas(self.root, bg=self._transparent_key, highlightthickness=0)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)
        self.bg_canvas.create_oval(10, 10, 790, 790, fill=theme["bg_main"], outline=theme["fg_title"], width=2)
        self.bg_canvas.create_oval(20, 20, 780, 780, fill="", outline="#003300", width=1)
        self.bg_canvas.bind("<ButtonPress-1>", self._start_drag)
        self.bg_canvas.bind("<B1-Motion>", self._on_drag)

        self.main_container = tk.Frame(self.root, bg=theme["bg_main"])
        self.main_container.place(relx=0.5, rely=0.5, anchor="center", width=460, height=560)

        subtitle = JETSON_MODEL if IS_JETSON else labels["subtitle"]

        top_frame = tk.Frame(self.main_container, bg=theme["bg_main"])
        top_frame.pack(fill=tk.X, pady=(0, 3))

        close_btn = tk.Label(top_frame, text="[X]", font=(FONT_MONO, 11, "bold"), fg=theme["accent_listen"], bg=theme["bg_main"], cursor="hand2")
        close_btn.pack(side=tk.RIGHT, padx=5)
        close_btn.bind("<Button-1>", lambda event: self.shutdown())

        tk.Label(self.main_container, text=f"[ {labels['title']} ]", font=self._font_title, fg=theme["fg_title"], bg=theme["bg_main"]).pack(pady=(0, 3))
        tk.Label(self.main_container, text=subtitle, font=self._font_sub, fg=theme["fg_sub"], bg=theme["bg_main"]).pack()

        self.face_canvas = tk.Canvas(self.main_container, width=200, height=100, bg=theme["bg_main"], highlightthickness=0)
        self.face_canvas.pack(pady=(8, 2))
        self._draw_face()
        self._schedule_blink()

        if IS_JETSON:
            self.stats_label = tk.Label(self.main_container, text=labels["telemetry_load"], font=(FONT_MONO, 9), fg=theme["fg_sub"], bg=theme["bg_panel"], anchor="w", padx=10, pady=5)
            self.stats_label.pack(fill=tk.X, padx=20, pady=(10, 0))

        status_frame = tk.Frame(self.main_container, bg=theme["bg_main"])
        status_frame.pack(pady=(6, 0))

        self.status_dot = tk.Label(status_frame, text="●", font=(FONT_MONO, 14), fg=theme["accent_process"], bg=theme["bg_main"])
        self.status_dot.pack(side=tk.LEFT)

        self.status_text = tk.Label(status_frame, text=labels["status_init"], font=(FONT_MONO, 11), fg=theme["fg_text"], bg=theme["bg_main"])
        self.status_text.pack(side=tk.LEFT, padx=(8, 0))

        self.viz_canvas = tk.Canvas(self.main_container, width=340, height=36, bg=theme["bg_main"], highlightthickness=0)
        self.viz_canvas.pack(pady=(4, 6))
        self._draw_viz_idle()

        chat_frame = tk.Frame(self.main_container, bg=theme["bg_main"], padx=0, pady=0, height=160)
        chat_frame.pack(padx=10, pady=4, fill=tk.X)
        chat_frame.pack_propagate(False)

        self.chat = tk.Text(
            chat_frame,
            font=self._font_mono,
            bg=theme["bg_main"],
            fg=theme["fg_text"],
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
            insertbackground=theme["fg_title"],
            selectbackground="#003300",
            selectforeground=theme["fg_title"],
            cursor="arrow",
        )
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_indicator = tk.Canvas(chat_frame, width=6, bg=theme["bg_main"], highlightthickness=0)
        self.scroll_indicator.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.config(yscrollcommand=self._update_scroll_indicator)
        self.chat.bind("<MouseWheel>", self._on_chat_scroll)
        self.chat.bind("<Button-4>", self._on_chat_scroll)
        self.chat.bind("<Button-5>", self._on_chat_scroll)

        self.chat.tag_configure("user", foreground="#00cc33", font=(FONT_MONO, 10))
        self.chat.tag_configure("wilson", foreground="#00ff41", font=(FONT_MONO, 10, "bold"), spacing1=4, spacing3=4)
        self.chat.tag_configure("system", foreground="#004d1a", font=(FONT_MONO, 9), justify=tk.CENTER)
        self.chat.tag_configure("cursor_blink", foreground="#00ff41", font=(FONT_MONO, 10, "bold"))

        button_row = tk.Frame(self.main_container, bg=theme["bg_main"])
        button_row.pack(pady=(8, 6))

        self.btn = tk.Button(
            button_row,
            text=labels["btn_ready"],
            font=self._font_btn,
            bg="#0a1a0a",
            fg=theme["accent_ready"],
            activebackground="#003300",
            activeforeground=theme["accent_ready_hover"],
            highlightthickness=1,
            highlightbackground="#00ff41",
            relief=tk.SOLID,
            bd=1,
            width=20,
            height=2,
            cursor="hand2",
            command=self._toggle_listening,
        )
        self.btn.pack(side=tk.LEFT, padx=(0, 4))
        self.root.bind("<space>", lambda event: self._toggle_listening())

        self.auto_btn = tk.Button(
            button_row,
            text="AUTO",
            font=(FONT_MONO, 10, "bold"),
            bg="#0a1a0a",
            fg="#004d1a",
            activebackground="#001a00",
            activeforeground="#00ff41",
            highlightthickness=1,
            highlightbackground="#003300",
            relief=tk.SOLID,
            bd=1,
            width=6,
            height=2,
            cursor="hand2",
            command=self._toggle_auto_listen,
        )
        self.auto_btn.pack(side=tk.LEFT, padx=(4, 0))

        llm_name = "LM Studio" if IS_WINDOWS else "Ollama"
        config_text = f"STT:{WHISPER_MODEL}({WHISPER_DEVICE}) | LLM:{llm_name}"
        tk.Label(self.main_container, text=config_text, font=(FONT_MONO, 7), fg="#003300", bg=theme["bg_main"]).pack()
        tk.Label(self.main_container, text=labels["footer_hints"] + "  |  [ESC] quit", font=(FONT_MONO, 7), fg="#004d1a", bg=theme["bg_main"]).pack(pady=(1, 4))

    def _set_status(self, text, color=None):
        """Safely update the status light and status text from the GUI thread."""
        if color is None:
            color = self.THEME["fg_text"]
        try:
            self.status_dot.config(fg=color)
            self.status_text.config(text=text)
        except tk.TclError:
            pass

    def _fade_button_color(self, target_fg, target_bg=None):
        """Update button colors to reflect listening/processing/ready states."""
        try:
            background = target_bg or "#0a1a0a"
            self.btn.config(fg=target_fg, bg=background, activeforeground=target_fg)
        except tk.TclError:
            pass

    def _log(self, tag, message):
        """Push a message onto the inter-thread queue for later GUI rendering."""
        self.msg_queue.put((tag, message))

    def _draw_viz_idle(self):
        """Render a quiet equalizer when Wilson is not listening."""
        canvas = self.viz_canvas
        canvas.delete("all")
        width = 340
        bar_count = self._viz_bars
        bar_width = max(2, (width - (bar_count - 1) * 3) // bar_count)
        for index in range(bar_count):
            x = index * (bar_width + 3) + 4
            canvas.create_rectangle(x, 32, x + bar_width, 34, fill="#002200", outline="", tags="bars")

    def _draw_viz_active(self):
        """Render the live equalizer using current mic levels and falling particles."""
        canvas = self.viz_canvas
        canvas.delete("all")
        width = 340
        bar_count = self._viz_bars
        bar_width = max(2, (width - (bar_count - 1) * 3) // bar_count)

        for index in range(bar_count):
            x = index * (bar_width + 3) + 4
            level = self._viz_levels[index]
            height = max(2, int(level * 30))
            if height > 15:
                canvas.create_rectangle(x, 34 - height, x + bar_width, 34 - height + 4, fill="#00ff41", outline="")
                canvas.create_rectangle(x, 34 - height + 4, x + bar_width, 34, fill="#00aa2a", outline="")
            elif height > 6:
                canvas.create_rectangle(x, 34 - height, x + bar_width, 34, fill="#00882a", outline="")
            else:
                canvas.create_rectangle(x, 34 - height, x + bar_width, 34, fill="#004d1a", outline="")

            if level > 0.3:
                peak_y = 34 - height - 3
                canvas.create_rectangle(x, peak_y, x + bar_width, peak_y + 2, fill="#00ff41", outline="")

        new_particles = []
        for particle_x, particle_y, speed, brightness in self._viz_particles:
            particle_y += speed
            brightness -= 0.02
            if brightness > 0 and particle_y < 36:
                green = int(brightness * 255)
                color = f"#00{green:02x}{green // 4:02x}"
                canvas.create_text(particle_x, particle_y, text=random.choice("01"), fill=color, font=(FONT_MONO, 7), anchor="nw")
                new_particles.append((particle_x, particle_y, speed, brightness))
        self._viz_particles = new_particles

    def _update_viz_from_volume(self):
        """Convert the latest RMS value into animated bar heights and particles."""
        if self.is_listening and self.recorder:
            volume = min(1.0, self.recorder.get_volume() * 12)
            for index in range(self._viz_bars):
                target = volume * random.uniform(0.3, 1.0)
                self._viz_levels[index] = self._viz_levels[index] * 0.3 + target * 0.7

            if volume > 0.15:
                bar_count = self._viz_bars
                bar_width = max(2, (340 - (bar_count - 1) * 3) // bar_count)
                for index in range(bar_count):
                    if self._viz_levels[index] > 0.5 and random.random() < 0.3:
                        x = index * (bar_width + 3) + 4
                        height = int(self._viz_levels[index] * 30)
                        self._viz_particles.append((x, 34 - height - 5, random.uniform(0.8, 2.0), random.uniform(0.5, 1.0)))
                if len(self._viz_particles) > 60:
                    self._viz_particles = self._viz_particles[-40:]

            self._draw_viz_active()
        else:
            any_active = False
            for index in range(self._viz_bars):
                self._viz_levels[index] *= 0.85
                if self._viz_levels[index] > 0.01:
                    any_active = True
            if any_active:
                self._draw_viz_active()
            else:
                self._draw_viz_idle()

    def _update_scroll_indicator(self, first, last):
        """Draw a simple custom scrollbar track and thumb beside the chat box."""
        canvas = self.scroll_indicator
        canvas.delete("all")
        first_float = float(first)
        last_float = float(last)
        if last_float - first_float >= 0.999:
            return
        height = canvas.winfo_height()
        if height < 10:
            height = 200
        y1 = int(first_float * height)
        y2 = max(int(last_float * height), y1 + 12)
        canvas.create_line(3, 0, 3, height, fill="#001a00", width=1)
        canvas.create_rectangle(1, y1, 5, y2, fill="#00ff41", outline="#003300")

    def _on_chat_scroll(self, event):
        """Handle mouse-wheel scrolling on Windows and Linux."""
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self.chat.yview_scroll(-3, "units")
        elif event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
            self.chat.yview_scroll(3, "units")

    def _insert_separator(self):
        """Insert a visible separator line between conversation turns."""
        try:
            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, "\n" + "─" * 44 + "\n", "system")
            self.chat.config(state=tk.DISABLED)
            self.chat.see(tk.END)
        except tk.TclError:
            pass

    def _check_queue(self):
        """Drain one queued log message per tick and keep visuals responsive."""
        if not self._matrix_busy:
            try:
                tag, message = self.msg_queue.get_nowait()
                self._matrix_busy = True
                if tag == "user":
                    self._insert_separator()
                    self._matrix_type("\n> ", message, "user")
                elif tag == "wilson":
                    self._matrix_type("\n[WILSON] ", message, "wilson")
                else:
                    self._matrix_type("\n  ", message, "system")
            except queue.Empty:
                pass

        try:
            self._update_viz_from_volume()
        except tk.TclError:
            pass

        try:
            if IS_JETSON and hasattr(self, "stats_label"):
                summary = self.monitor.summary
                if summary:
                    self.stats_label.config(text=summary)
        except Exception:
            pass

        try:
            self.root.after(50, self._check_queue)
        except tk.TclError:
            pass

    def _matrix_type(self, prefix, text, tag, idx=0):
        """Animate text into the chat one character at a time with occasional glitch frames."""
        try:
            self.chat.config(state=tk.NORMAL)
            if idx == 0:
                self.chat.insert(tk.END, prefix, tag)
            if idx < len(text):
                character = text[idx]
                if tag == "wilson" and character not in " \n" and random.random() < 0.4:
                    glitch_chars = "01アイウエオカキクケコサシスセソ$#@&%"
                    glitch = random.choice(glitch_chars)
                    self.chat.insert(tk.END, glitch, tag)
                    self.chat.see(tk.END)
                    self.chat.config(state=tk.DISABLED)
                    self.root.after(40, self._resolve_glitch, text, tag, idx)
                    return
                self.chat.insert(tk.END, character, tag)
                self.chat.see(tk.END)
                self.chat.config(state=tk.DISABLED)
                if tag == "wilson":
                    delay = 4 if character in " \n" else 18
                elif tag == "user":
                    delay = 3 if character in " \n" else 10
                else:
                    delay = 1
                self.root.after(delay, self._matrix_type, prefix, text, tag, idx + 1)
            else:
                self.chat.insert(tk.END, "\n", tag)
                self.chat.config(state=tk.DISABLED)
                self.chat.see(tk.END)
                self._matrix_busy = False
        except tk.TclError:
            self._matrix_busy = False

    def _resolve_glitch(self, text, tag, idx):
        """Replace the temporary glitch character with the real intended character."""
        try:
            self.chat.config(state=tk.NORMAL)
            self.chat.delete("end-2c", "end-1c")
            self.chat.insert("end-1c", text[idx], tag)
            self.chat.see(tk.END)
            self.chat.config(state=tk.DISABLED)
            character = text[idx]
            if tag == "wilson":
                delay = 4 if character in " \n" else 18
            elif tag == "user":
                delay = 3 if character in " \n" else 10
            else:
                delay = 1
            self.root.after(delay, self._matrix_type, "", text, tag, idx + 1)
        except tk.TclError:
            self._matrix_busy = False

    def _draw_face(self):
        """Draw Wilson's eyes and mouth based on blink and speech animation state."""
        canvas = self.face_canvas
        canvas.delete("face")

        center_x = 100
        eyes_y = 28
        eye_separation = 35

        if self._eye_open:
            for eye_x in (center_x - eye_separation, center_x + eye_separation):
                canvas.create_oval(eye_x - 11, eyes_y - 11, eye_x + 11, eyes_y + 11, fill="#00ff41", outline="#00ff41", tags="face")
                canvas.create_oval(eye_x - 5, eyes_y - 5, eye_x + 5, eyes_y + 5, fill="#000000", outline="#000000", tags="face")
        else:
            for eye_x in (center_x - eye_separation, center_x + eye_separation):
                canvas.create_line(eye_x - 11, eyes_y, eye_x + 11, eyes_y, fill="#00ff41", width=3, tags="face")

        mouth_center_x = 100
        mouth_center_y = 72
        mouth_width = 28
        openness = self._mouth_openness

        if openness < 0.05:
            canvas.create_arc(mouth_center_x - mouth_width, mouth_center_y - 10, mouth_center_x + mouth_width, mouth_center_y + 14, start=200, extent=140, style=tk.ARC, outline="#00ff41", width=3, tags="face")
        else:
            height = int(4 + openness * 22)
            canvas.create_oval(mouth_center_x - mouth_width, mouth_center_y - height, mouth_center_x + mouth_width, mouth_center_y + height, fill="#000800", outline="#00ff41", width=2, tags="face")
            if openness > 0.55:
                tongue_width = int(10 + openness * 6)
                canvas.create_arc(mouth_center_x - tongue_width, mouth_center_y + 2, mouth_center_x + tongue_width, mouth_center_y + height + 4, start=0, extent=180, fill="#003300", outline="", tags="face")

    def _schedule_blink(self):
        """Loop forever, occasionally closing the eyes briefly when not speaking."""
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
        """Re-open Wilson's eyes after a blink frame."""
        self._eye_open = True
        try:
            self._draw_face()
        except tk.TclError:
            pass

    def _compute_mouth_amplitudes(self, audio_data, sample_rate, fps=30):
        """Turn spoken reply audio into a per-frame mouth openness envelope."""
        samples_per_frame = max(1, sample_rate // fps)
        amplitudes = []
        for index in range(0, len(audio_data), samples_per_frame):
            chunk = audio_data[index:index + samples_per_frame]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            amplitudes.append(min(1.0, rms * 10))
        return amplitudes

    def _start_mouth_anim(self, amplitudes, duration):
        """Initialize mouth animation state for the upcoming TTS playback."""
        self._mouth_data = amplitudes
        self._mouth_start = time.time()
        self._mouth_duration = duration
        self._mouth_active = True
        self._eye_open = True
        self._tick_mouth()

    def _tick_mouth(self):
        """Advance mouth animation approximately 30 times per second."""
        if not self._mouth_active:
            self._mouth_openness = 0.0
            self._draw_face()
            return
        elapsed = time.time() - self._mouth_start
        if elapsed >= self._mouth_duration or not self._mouth_data:
            self._stop_mouth_anim()
            return
        progress = elapsed / self._mouth_duration
        index = min(int(progress * len(self._mouth_data)), len(self._mouth_data) - 1)
        self._mouth_openness = self._mouth_data[index]
        self._draw_face()
        try:
            self.root.after(33, self._tick_mouth)
        except tk.TclError:
            pass

    def _stop_mouth_anim(self):
        """Return the mouth to its resting expression."""
        self._mouth_active = False
        self._mouth_openness = 0.0
        try:
            self._draw_face()
        except tk.TclError:
            pass

    def _initialize(self):
        """Create runtime components in a background thread so the window stays responsive."""
        try:
            if IS_JETSON:
                self._log("system", f"Platform: {JETSON_MODEL}")
                power_mode = os.environ.get("WILSON_POWER_MODE", "MAXN")
                set_jetson_power_mode(power_mode)
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
            loaded = self.transcriber.load(callback=lambda message: self._log("system", message))
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
        except Exception as error:
            self._log("system", f"Init error: {error}")
            self.root.after(0, lambda: self._set_status(self.I18N["status_error"], self.THEME["accent_listen"]))

    def _toggle_auto_listen(self):
        """Enable or disable voice-activated capture mode."""
        if not self._ready:
            self._log("system", "Still loading, please wait…")
            return
        self._auto_listen = not self._auto_listen
        theme = self.THEME
        if self._auto_listen:
            self.auto_btn.config(fg="#00ff41", bg="#002200", highlightbackground="#00ff41")
            self._log("system", "Auto-listen ON — speak anytime, Wilson will detect your voice.")
            self._set_status("AUTO — LISTENING", theme["accent_ready"])
            threading.Thread(target=self._auto_listen_loop, daemon=True).start()
        else:
            self.auto_btn.config(fg="#004d1a", bg="#0a1a0a", highlightbackground="#003300")
            self._log("system", "Auto-listen OFF")
            self._set_status(self.I18N["status_ready"], theme["accent_ready"])

    def _auto_listen_loop(self):
        """Monitor the mic for voice onset, then hand off to the normal listening workflow."""
        noise_floor = 0.012
        confirm_secs = 0.3
        check_interval = 0.05

        while self._auto_listen:
            if self.is_listening or self.is_processing:
                time.sleep(0.2)
                continue
            try:
                self.recorder.start()
            except Exception:
                time.sleep(1)
                continue

            speech_time = 0.0
            detected = False
            while self._auto_listen and not self.is_processing:
                time.sleep(check_interval)
                try:
                    volume = self.recorder.get_volume()
                except Exception:
                    break
                if volume >= noise_floor:
                    speech_time += check_interval
                    if speech_time >= confirm_secs:
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

            self.root.after(0, self._auto_start_capture)

            while self._auto_listen and (self.is_listening or self.is_processing):
                time.sleep(0.2)

            if self._auto_listen:
                time.sleep(1.0)

    def _auto_start_capture(self):
        """Switch the UI into active listening once voice onset has been confirmed."""
        if self.is_processing or self.is_listening:
            return
        self.is_listening = True
        self.btn.config(text=self.I18N["btn_listen"])
        self._fade_button_color(self.THEME["accent_listen"])
        self._set_status(self.I18N["status_dictate"], self.THEME["accent_listen"])
        threading.Thread(target=self._silence_watchdog, daemon=True).start()

    def _toggle_listening(self):
        """Start or stop capture when the user clicks the main button or presses Space."""
        if self.is_processing:
            return
        if not self._ready:
            self._log("system", "Still loading, please wait…")
            return
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        """Put the app into listening mode and start recording immediately."""
        self.is_listening = True
        self.btn.config(text=self.I18N["btn_listen"])
        self._fade_button_color(self.THEME["accent_listen"])
        self._set_status(self.I18N["status_dictate"], self.THEME["accent_listen"])
        self.recorder.start()
        threading.Thread(target=self._silence_watchdog, daemon=True).start()

    def _stop_listening(self):
        """End capture, collect audio, and start pipeline processing if enough audio was recorded."""
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
        """Auto-stop recording after about two seconds of continuous silence."""
        silence = 0.0
        while self.is_listening:
            time.sleep(0.1)
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

    def _process(self, audio):
        """Run the full pipeline: STT, LLM, then TTS and UI animation."""
        try:
            self.root.after(0, lambda: self._set_status(self.I18N["status_transcribe"], self.THEME["accent_process"]))
            text = self.transcriber.transcribe(audio)
            if not text or len(text.strip()) < 2:
                self._log("system", "Couldn't understand. Speak louder / closer to the mic.")
                self.root.after(0, self._reset)
                return

            self._log("user", text.strip())

            self.root.after(0, lambda: self._set_status(self.I18N["status_think"], self.THEME["accent_process"]))
            response = self.llm.query(text)

            self.root.after(0, lambda: self._set_status(self.I18N["status_speak"], self.THEME["accent_ready"]))
            audio_data, sample_rate = self.tts.generate_wav(response, log_fn=lambda message: self._log("system", message))
            if audio_data is not None and sample_rate is not None:
                amplitudes = self._compute_mouth_amplitudes(audio_data, sample_rate)
                duration = len(audio_data) / sample_rate
                self._log("wilson", response)
                self.root.after(0, lambda a=amplitudes, d=duration: self._start_mouth_anim(a, d))
                _get_sd().play(audio_data, sample_rate)
                _get_sd().wait()
                self.root.after(0, self._stop_mouth_anim)
            else:
                self._log("wilson", response)
                self.tts.speak(response, log_fn=lambda message: self._log("system", message))
        except Exception as error:
            self._log("system", f"Pipeline error: {error}")
        finally:
            self.root.after(0, self._reset)

    def _reset(self):
        """Return the button and status area to the appropriate ready state."""
        self.is_processing = False
        try:
            self.btn.config(text=self.I18N["btn_ready"], state=tk.NORMAL)
            self._fade_button_color(self.THEME["accent_ready"])
            if self._auto_listen:
                self._set_status("AUTO — LISTENING", self.THEME["accent_ready"])
            else:
                self._set_status(self.I18N["status_ready"], self.THEME["accent_ready"])
        except tk.TclError:
            pass

    def shutdown(self):
        """Cleanly stop background activity and destroy GUI windows."""
        self.is_listening = False
        self._auto_listen = False
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
        """Enter the tkinter main event loop."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


class WilsonHeadless:
    """Terminal-only interface that reuses the same recorder, Whisper, LLM, and TTS components."""

    COLORS = {
        "system": "\033[90m",
        "user": "\033[94m",
        "wilson": "\033[92m",
        "reset": "\033[0m",
    }

    def __init__(self):
        self.recorder = None
        self.transcriber = None
        self.tts = None
        self.llm = LLMClient()
        self.monitor = JetsonMonitor()

    def _print(self, tag, message):
        """Print a colored, prefixed log line in terminal mode."""
        color = self.COLORS.get(tag, "")
        reset = self.COLORS["reset"]
        prefix = {"system": "[SYS]", "user": "[YOU]", "wilson": "[WILSON]"}.get(tag, "")
        print(f"{color}{prefix} {message}{reset}")

    def _initialize(self):
        """Set up all shared components before entering the terminal interaction loop."""
        if IS_JETSON:
            self._print("system", f"Platform: {JETSON_MODEL}")
            set_jetson_power_mode(os.environ.get("WILSON_POWER_MODE", "MAXN"))
            self.monitor.start()
        else:
            self._print("system", f"Platform: {platform.system()} {platform.machine()}")

        self._print("system", "Initializing microphone…")
        self.recorder = AudioRecorder()

        self._print("system", "Loading TTS…")
        self.tts = TTSEngine()

        self.transcriber = Transcriber()
        self.transcriber.load(callback=lambda message: self._print("system", message))

        if IS_JETSON:
            gc.collect()

        self._print("system", "Ready.  Press ENTER to record, ENTER again to stop.  'q' to quit.")

    def run(self):
        """Run the interactive terminal conversation loop until the user quits."""
        self._initialize()
        while True:
            try:
                command = input("\n[ENTER to talk | q to quit] ").strip().lower()
                if command in ("q", "quit", "exit"):
                    break

                self._print("system", "Listening… (press ENTER to stop)")
                self.recorder.start()
                input()
                audio = self.recorder.stop()

                if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                    self._print("system", "Too short — try again.")
                    continue

                self._print("system", "Transcribing…")
                text = self.transcriber.transcribe(audio)
                if not text or len(text.strip()) < 2:
                    self._print("system", "Couldn't understand.")
                    continue

                self._print("user", text.strip())

                self._print("system", "Thinking…")
                response = self.llm.query(text)
                self._print("wilson", response)

                self.tts.speak(response, log_fn=lambda message: self._print("system", message))

                if IS_JETSON:
                    gc.collect()
                    summary = self.monitor.summary
                    if summary:
                        self._print("system", summary)
            except KeyboardInterrupt:
                print()
                break
            except EOFError:
                break

        self.monitor.stop()
        self._print("system", "Goodbye.")


def run_diagnostics():
    """Validate dependencies and local services without starting the full assistant."""
    ok = True

    def check(label, result, detail=""):
        nonlocal ok
        symbol = "\033[92m✓\033[0m" if result else "\033[91m✗\033[0m"
        detail_text = f"  ({detail})" if detail else ""
        print(f"  {symbol} {label}{detail_text}")
        if not result:
            ok = False

    def info(label, result, detail=""):
        symbol = "\033[92m✓\033[0m" if result else "\033[93m—\033[0m"
        detail_text = f"  ({detail})" if detail else ""
        print(f"  {symbol} {label}{detail_text}")

    print(f"\n{'=' * 58}")
    print("  WILSON V1 — System Diagnostics")
    print(f"{'=' * 58}\n")

    hardware = JETSON_MODEL if IS_JETSON else f"{platform.system()} {platform.machine()}"
    print(f"  Platform : {hardware}")
    print(f"  Python   : {sys.version.split()[0]}")
    print()

    print("  Dependencies:")
    check("numpy", True, np.__version__)

    try:
        sounddevice_module = _get_sd()
        check("sounddevice", True, sounddevice_module.__version__)
    except Exception as error:
        check("sounddevice", False, f"import failed: {error}")

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
        import requests as requests_module
        check("requests", True, requests_module.__version__)
    except ImportError:
        check("requests", False, "pip install requests")

    print("\n  CUDA:")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        check("torch.cuda", has_cuda, torch.cuda.get_device_name(0) if has_cuda else "no GPU detected")
    except ImportError:
        check("torch", False, "not installed — Whisper will fall back to CPU")

    print("\n  TTS:")
    check("Piper binary", os.path.isfile(PIPER_EXE), PIPER_EXE)
    check("Voice model", os.path.isfile(PIPER_VOICE), PIPER_VOICE)
    espeak_path = shutil.which("espeak-ng") or shutil.which("espeak")
    info("espeak-ng fallback", espeak_path is not None, espeak_path if espeak_path else "optional — not needed if Piper is installed")
    if IS_MACOS:
        say_path = shutil.which("say")
        info("macOS say fallback", say_path is not None, say_path if say_path else "optional — not needed if Piper is installed")

    print("\n  LLM endpoint:")
    try:
        response = requests.get(LLM_URL.replace("/chat/completions", "/models"), timeout=(2, 5))
        check("API reachable", response.status_code == 200, LLM_URL)
    except Exception:
        check("API reachable", False, f"{LLM_URL} — is the server running?")

    print("\n  Audio:")
    try:
        recorder = AudioRecorder()
        check("Microphone", True, _get_sd().query_devices(recorder.device_id)["name"])
    except Exception as error:
        check("Microphone", False, str(error))

    if IS_JETSON:
        print("\n  Jetson:")
        check("tegrastats", shutil.which("tegrastats") is not None)
        check("nvpmodel", shutil.which("nvpmodel") is not None)

    print(f"\n{'=' * 58}")
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED — see above'}")
    print(f"{'=' * 58}\n")
    return ok


def print_banner():
    """Print a concise startup summary before Wilson enters GUI or headless mode."""
    hardware = JETSON_MODEL if IS_JETSON else f"{platform.system()} {platform.machine()}"
    lines = [
        f"  Hardware  : {hardware}",
        f"  Whisper   : {WHISPER_MODEL} ({WHISPER_DEVICE}/{WHISPER_COMPUTE})",
        f"  LLM       : {LLM_URL}",
    ]
    if LLM_MODEL:
        lines.append(f"  Model     : {LLM_MODEL}")
    lines.append(f"  TTS       : Piper ({'found' if os.path.isfile(PIPER_EXE) else 'NOT FOUND'})")

    print(f"\n{'=' * 58}")
    print("  WILSON V1 — Offline Voice Assistant")
    print(f"{'=' * 58}")
    for line in lines:
        print(line)
    print(f"{'=' * 58}\n")


if __name__ == "__main__":
    # Startup breadcrumb so the user knows the process really launched.
    print("Wilson annotated study copy starting...", flush=True)

    # If the user only wants diagnostics, run them and exit with pass/fail status.
    if "--check" in sys.argv:
        sys.exit(0 if run_diagnostics() else 1)

    # Show the runtime summary before opening the interface.
    print_banner()

    # Use terminal mode when requested or when tkinter is unavailable.
    if HEADLESS or not HAS_GUI:
        if not HAS_GUI:
            print("[NOTE] tkinter not available — running in headless mode")
        WilsonHeadless().run()
    else:
        WilsonGUI().run()