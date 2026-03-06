"""
WILSON V1 — PyInstaller Build Script

Creates a standalone .exe that bundles:
  • Wilson application code
  • Piper TTS binary + voice model + espeak-ng data
  • Whisper model (downloaded on first run if not cached)
  • llama-cpp-python (embedded LLM runtime)

The GGUF model file is NOT bundled inside the .exe (too large).
It lives in a 'models/' folder next to the .exe and is auto-downloaded
on first launch (~1 GB one-time download).

Usage:
  1. pip install pyinstaller
  2. python build_exe.py
  3. The .exe will be in dist/Wilson/

To create a single-file .exe (slower startup):
  python build_exe.py --onefile
"""

import PyInstaller.__main__
import os
import sys
import shutil

WILSON_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths to bundle ──────────────────────────────────────────────────────────

PIPER_DIR = os.path.join(WILSON_DIR, "piper")
VOICE_MODEL = os.path.join(WILSON_DIR, "en_US-amy-medium.onnx")
VOICE_JSON = os.path.join(WILSON_DIR, "en_US-amy-medium.onnx.json")
MODELS_DIR = os.path.join(WILSON_DIR, "models")

# ── Build arguments ──────────────────────────────────────────────────────────

onefile = "--onefile" in sys.argv

args = [
    os.path.join(WILSON_DIR, "wilson.py"),
    "--name", "Wilson",
    "--noconsole" if "--noconsole" in sys.argv else "--console",
    "--icon", "NONE",
    # Don't prompt to overwrite
    "--noconfirm",
    # Clean build directory
    "--clean",
]

if onefile:
    args.append("--onefile")
else:
    args.append("--onedir")

# ── Data files to include ────────────────────────────────────────────────────

# Piper TTS binary + espeak-ng data
if os.path.isdir(PIPER_DIR):
    args.extend(["--add-data", f"{PIPER_DIR}{os.pathsep}piper"])

# Voice model files
if os.path.isfile(VOICE_MODEL):
    args.extend(["--add-data", f"{VOICE_MODEL}{os.pathsep}."])
if os.path.isfile(VOICE_JSON):
    args.extend(["--add-data", f"{VOICE_JSON}{os.pathsep}."])

# ── Hidden imports (dynamic imports that PyInstaller can't detect) ───────────

hidden_imports = [
    "faster_whisper",
    "ctranslate2",
    "llama_cpp",
    "sounddevice",
    "soundfile",
    "numpy",
    "requests",
    "tkinter",
    "tkinter.scrolledtext",
    "tkinter.ttk",
    # llama-cpp-python internals
    "llama_cpp.llama",
    "llama_cpp.llama_types",
    "llama_cpp.llama_grammar",
    "llama_cpp.llama_chat_format",
    # faster-whisper / ctranslate2 internals
    "ctranslate2.converters",
    "ctranslate2.specs",
]

for imp in hidden_imports:
    args.extend(["--hidden-import", imp])

# ── Exclude unnecessary large packages ───────────────────────────────────────

excludes = [
    "matplotlib",
    "pandas",
    "scipy.spatial",
    "scipy.optimize",
    "PIL",
    "torch",           # Not needed at runtime (Whisper uses ctranslate2)
    "tensorflow",
    "jupyter",
    "notebook",
    "IPython",
]

for exc in excludes:
    args.extend(["--exclude-module", exc])

# ── Run PyInstaller ──────────────────────────────────────────────────────────

print("=" * 60)
print("  WILSON V1 — Building Standalone Executable")
print("=" * 60)
print(f"  Mode    : {'Single file' if onefile else 'Directory'}")
print(f"  Piper   : {'included' if os.path.isdir(PIPER_DIR) else 'NOT FOUND'}")
print(f"  Voice   : {'included' if os.path.isfile(VOICE_MODEL) else 'NOT FOUND'}")
print(f"  Models/ : {'exists' if os.path.isdir(MODELS_DIR) else 'will be created on first run'}")
print("=" * 60)
print()

PyInstaller.__main__.run(args)

# ── Post-build: create models directory in output ────────────────────────────

if not onefile:
    dist_models = os.path.join(WILSON_DIR, "dist", "Wilson", "models")
    os.makedirs(dist_models, exist_ok=True)

    # Copy GGUF model if it exists locally
    gguf_src = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    if os.path.isfile(gguf_src):
        gguf_dst = os.path.join(dist_models, os.path.basename(gguf_src))
        if not os.path.isfile(gguf_dst):
            print(f"\nCopying GGUF model to dist ({os.path.getsize(gguf_src) / 1024 / 1024:.0f} MB)...")
            shutil.copy2(gguf_src, gguf_dst)
    else:
        print(f"\nNote: GGUF model not found at {gguf_src}")
        print("The model will be auto-downloaded on first run of the .exe")

    # Copy NanoSAM models if present
    for sam_file in ["nanosam_image_encoder.onnx", "nanosam_mask_decoder.onnx"]:
        src = os.path.join(MODELS_DIR, sam_file)
        if os.path.isfile(src):
            dst = os.path.join(dist_models, sam_file)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
                print(f"Copied {sam_file}")

print("\n" + "=" * 60)
print("  BUILD COMPLETE")
if onefile:
    print(f"  Output: dist/Wilson.exe")
else:
    print(f"  Output: dist/Wilson/Wilson.exe")
print(f"\n  The models/ folder must be next to the .exe.")
print(f"  The GGUF model (~1 GB) auto-downloads on first launch.")
print("=" * 60)
