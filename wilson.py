"""
WILSON V2 - Offline Voice Assistant (GPU Enabled)
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import subprocess
import requests
import os
import tempfile
import time
import queue
import numpy as np
import sounddevice as sd

# ═══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WILSON_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_EXE = os.path.join(WILSON_DIR, "piper", "piper.exe")
PIPER_VOICE = os.path.join(WILSON_DIR, "en_US-amy-medium.onnx")
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
WHISPER_MODEL = "small"
SAMPLE_RATE = 16000

SYSTEM_PROMPT = """You are Wilson, a helpful AI assistant. Keep responses concise (1-3 sentences). No emojis or markdown."""

# ═══════════════════════════════════════════════════════════════════════════════

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.device_id = self._find_best_device()
        
    def _find_best_device(self):
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                name = device['name'].lower()
                if 'anker' in name or 'powerconf' in name:
                    print(f"[MIC] Found Anker: {device['name']}")
                    return i
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                name = device['name'].lower()
                if 'usb' in name:
                    print(f"[MIC] Found USB: {device['name']}")
                    return i
        default = sd.default.device[0]
        if default is not None:
            device_name = sd.query_devices(default)['name']
            print(f"[MIC] Using default: {device_name}")
            return default
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[MIC] Using: {device['name']}")
                return i
        raise RuntimeError("No microphone found!")
    
    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())
    
    def start(self):
        self.audio_data = []
        self.is_recording = True
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=1024,
            callback=self._audio_callback
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
            return 0
        recent = self.audio_data[-1] if self.audio_data else np.zeros(1)
        return float(np.sqrt(np.mean(recent ** 2)))


class Transcriber:
    def __init__(self):
        self.model = None
        
    def load(self, callback=None):
        try:
            from faster_whisper import WhisperModel
            if callback:
                callback(f"Loading Whisper '{WHISPER_MODEL}' model...")
            
            # Try GPU first, fall back to CPU
            try:
                self.model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
                if callback:
                    callback("Whisper loaded on GPU")
            except Exception as e:
                if callback:
                    callback(f"GPU failed: {e}")
                    callback("Falling back to CPU...")
                self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
                if callback:
                    callback("Whisper loaded on CPU")
            return True
        except ImportError:
            if callback:
                callback("ERROR: faster-whisper not installed")
            return False
        except Exception as e:
            if callback:
                callback(f"ERROR: {e}")
            return False
    
    def transcribe(self, audio):
        if self.model is None:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200)
        )
        return " ".join([s.text for s in segments]).strip()


class Wilson:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WILSON V2")
        self.root.geometry("500x750")
        self.root.configure(bg="#0d1117")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        
        self.is_listening = False
        self.is_processing = False
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.recorder = None
        self.transcriber = None
        self.msg_queue = queue.Queue()
        
        self._build_ui()
        self._check_queue()
        threading.Thread(target=self._initialize, daemon=True).start()

    def _build_ui(self):
        # Header
        tk.Label(
            self.root, text="◈ WILSON V2 ◈",
            font=("Segoe UI", 22, "bold"),
            fg="#58a6ff", bg="#0d1117"
        ).pack(pady=(15, 0))
        
        tk.Label(
            self.root, text="Offline Voice Assistant",
            font=("Segoe UI", 10),
            fg="#484f58", bg="#0d1117"
        ).pack()
        
        # Status
        status_frame = tk.Frame(self.root, bg="#0d1117")
        status_frame.pack(pady=10)
        
        self.status_dot = tk.Label(
            status_frame, text="●",
            font=("Segoe UI", 16),
            fg="#f0883e", bg="#0d1117"
        )
        self.status_dot.pack(side=tk.LEFT)
        
        self.status_text = tk.Label(
            status_frame, text="Initializing...",
            font=("Segoe UI", 12),
            fg="#c9d1d9", bg="#0d1117"
        )
        self.status_text.pack(side=tk.LEFT, padx=(8, 0))
        
        # Volume
        vol_frame = tk.Frame(self.root, bg="#0d1117")
        vol_frame.pack(pady=(0, 8))
        
        tk.Label(
            vol_frame, text="Volume:",
            font=("Segoe UI", 9),
            fg="#484f58", bg="#0d1117"
        ).pack(side=tk.LEFT)
        
        self.volume_bar = ttk.Progressbar(vol_frame, length=180, mode='determinate', maximum=100)
        self.volume_bar.pack(side=tk.LEFT, padx=5)
        
        # Chat
        chat_frame = tk.Frame(self.root, bg="#161b22", padx=2, pady=2)
        chat_frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)
        
        self.chat = scrolledtext.ScrolledText(
            chat_frame, font=("Consolas", 10),
            bg="#0d1117", fg="#c9d1d9",
            wrap=tk.WORD, state=tk.DISABLED,
            relief=tk.FLAT, padx=10, pady=10
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.tag_configure("user", foreground="#58a6ff")
        self.chat.tag_configure("wilson", foreground="#7ee787")
        self.chat.tag_configure("system", foreground="#484f58")
        
        # BUTTON
        self.btn = tk.Button(
            self.root,
            text="🎤  CLICK TO START LISTENING",
            font=("Segoe UI", 13, "bold"),
            bg="#238636",
            fg="#ffffff",
            activebackground="#2ea043",
            relief=tk.FLAT,
            width=28,
            height=2,
            cursor="hand2",
            command=self._toggle_listening
        )
        self.btn.pack(pady=15)
        
        self.root.bind("<space>", lambda e: self._toggle_listening())
        
        tk.Label(
            self.root,
            text="Click button or SPACEBAR • Auto-stops after silence",
            font=("Segoe UI", 9),
            fg="#30363d", bg="#0d1117"
        ).pack(pady=(0, 15))

    def _set_status(self, text, color="#c9d1d9"):
        self.status_dot.config(fg=color)
        self.status_text.config(text=text)

    def _log(self, tag, message):
        self.msg_queue.put((tag, message))

    def _check_queue(self):
        try:
            while True:
                tag, message = self.msg_queue.get_nowait()
                self.chat.config(state=tk.NORMAL)
                if tag == "user":
                    self.chat.insert(tk.END, f"\nYou: {message}\n", "user")
                elif tag == "wilson":
                    self.chat.insert(tk.END, f"\nWilson: {message}\n", "wilson")
                else:
                    self.chat.insert(tk.END, f"\n{message}\n", "system")
                self.chat.config(state=tk.DISABLED)
                self.chat.see(tk.END)
        except queue.Empty:
            pass
        
        if self.is_listening and self.recorder:
            vol = self.recorder.get_volume() * 500
            self.volume_bar['value'] = min(100, vol)
        else:
            self.volume_bar['value'] = 0
        
        self.root.after(50, self._check_queue)

    def _initialize(self):
        try:
            self._log("system", "Initializing microphone...")
            self.recorder = AudioRecorder()
            self.transcriber = Transcriber()
            self.transcriber.load(callback=lambda m: self._log("system", m))
            self._log("system", "Wilson is ready! Click the button to start.")
            self.root.after(0, lambda: self._set_status("Ready", "#238636"))
        except Exception as e:
            self._log("system", f"Error: {e}")
            self.root.after(0, lambda: self._set_status("Error", "#f85149"))

    def _toggle_listening(self):
        if self.is_processing:
            return
        if self.transcriber is None or self.transcriber.model is None:
            self._log("system", "Still loading, please wait...")
            return
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        self.is_listening = True
        self.btn.config(text="🔴  LISTENING... (click to stop)", bg="#f85149")
        self._set_status("Listening...", "#f85149")
        self.recorder.start()
        threading.Thread(target=self._monitor_silence, daemon=True).start()

    def _stop_listening(self):
        if not self.is_listening:
            return
        self.is_listening = False
        self.is_processing = True
        self.btn.config(text="⏳  PROCESSING...", bg="#f0883e", state=tk.DISABLED)
        self._set_status("Processing...", "#f0883e")
        audio = self.recorder.stop()
        if audio is not None and len(audio) > SAMPLE_RATE * 0.5:
            threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()
        else:
            self._log("system", "Recording too short")
            self._reset()

    def _monitor_silence(self):
        silence_duration = 0
        while self.is_listening:
            time.sleep(0.1)
            vol = self.recorder.get_volume()
            if vol < 0.01:
                silence_duration += 0.1
            else:
                silence_duration = 0
            if silence_duration > 2.0 and len(self.recorder.audio_data) > 20:
                self.root.after(0, self._stop_listening)
                break

    def _process_audio(self, audio):
        try:
            self.root.after(0, lambda: self._set_status("Transcribing...", "#f0883e"))
            text = self.transcriber.transcribe(audio)
            if not text or len(text.strip()) < 2:
                self._log("system", "Couldn't understand. Speak louder/clearer.")
                self.root.after(0, self._reset)
                return
            self._log("user", text.strip())
            self.root.after(0, lambda: self._set_status("Thinking...", "#f0883e"))
            response = self._query_llm(text)
            self._log("wilson", response)
            self.root.after(0, lambda: self._set_status("Speaking...", "#7ee787"))
            self._speak(response)
        except Exception as e:
            self._log("system", f"Error: {e}")
        finally:
            self.root.after(0, self._reset)

    def _query_llm(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        try:
            response = requests.post(
                LM_STUDIO_URL,
                json={
                    "messages": self.messages[-21:],
                    "stream": False,
                    "max_tokens": 200,
                    "temperature": 0.7
                },
                timeout=60
            )
            reply = response.json()["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except requests.exceptions.ConnectionError:
            return "Can't connect to LM Studio. Start server on port 1234."
        except Exception as e:
            return f"Error: {e}"

    def _speak(self, text):
        text = text.replace("*", "").replace("#", "").replace("`", "").replace("\n", " ").strip()
        if not text:
            return
        tmp_file = os.path.join(tempfile.gettempdir(), "wilson_output.wav")
        try:
            process = subprocess.Popen(
                [PIPER_EXE, "--model", PIPER_VOICE, "--output_file", tmp_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            process.communicate(input=text.encode('utf-8'))
            process.wait()
            time.sleep(0.1)
            if os.path.exists(tmp_file):
                import soundfile as sf
                audio_data, sample_rate = sf.read(tmp_file)
                sd.play(audio_data, sample_rate)
                sd.wait()
        except FileNotFoundError:
            self._log("system", "ERROR: Piper not found")
        except Exception as e:
            pass
        finally:
            try:
                os.unlink(tmp_file)
            except:
                pass

    def _reset(self):
        self.is_processing = False
        self.btn.config(text="🎤  CLICK TO START LISTENING", bg="#238636", state=tk.NORMAL)
        self._set_status("Ready", "#238636")

    def shutdown(self):
        self.is_listening = False
        if self.recorder and self.recorder.stream:
            self.recorder.stream.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  WILSON V2 - Starting...")
    print("="*50 + "\n")
    Wilson().run()
