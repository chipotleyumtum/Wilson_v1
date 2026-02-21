#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  WILSON V2 — Jetson Orin Nano Super Setup Script
#
#  Usage:
#    chmod +x setup_jetson.sh
#    sudo ./setup_jetson.sh
#
#  What this does:
#    1. Installs system packages (PortAudio, sndfile, espeak-ng, python-tk)
#    2. Sets MAXN power mode (25 W) and maximizes clocks
#    3. Creates a Python virtual environment with system-site-packages
#    4. Installs Python dependencies (faster-whisper, sounddevice, etc.)
#    5. Downloads Piper TTS ARM64 binary
#    6. Installs Ollama and pulls a default LLM model
#
#  Requirements:
#    - NVIDIA Jetson Orin Nano Super with JetPack 5.x or 6.x
#    - Internet connection (for package downloads)
#    - sudo / root access
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

WILSON_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${WILSON_DIR}/wilson_env"
PIPER_VERSION="2023.11.14-2"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  WILSON V2 — Jetson Orin Nano Super Setup"
JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "Unknown Jetson")
echo "  Target: ${JETSON_MODEL}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

ARCH=$(uname -m)
if [ "${ARCH}" != "aarch64" ]; then
    fail "This script is for ARM64 / aarch64 only.  Detected: ${ARCH}"
fi

if [ "$(id -u)" -ne 0 ]; then
    fail "Run with sudo:  sudo ./setup_jetson.sh"
fi

# ── 1. System packages ───────────────────────────────────────────────────────
info "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3-pip python3-venv python3-dev python3-tk \
    portaudio19-dev libportaudio2 libsndfile1 libsndfile1-dev \
    espeak-ng espeak-ng-data \
    curl wget git \
    alsa-utils \
    > /dev/null 2>&1

info "System packages installed."

# ── 2. Power mode ────────────────────────────────────────────────────────────
info "[2/7] Setting MAXN power mode (25 W)..."
if command -v nvpmodel &>/dev/null; then
    nvpmodel -m 0 2>/dev/null && info "nvpmodel → MAXN" || warn "nvpmodel failed (non-fatal)"
else
    warn "nvpmodel not found — skipping"
fi

if command -v jetson_clocks &>/dev/null; then
    jetson_clocks 2>/dev/null && info "Clocks maximized" || warn "jetson_clocks failed (non-fatal)"
else
    warn "jetson_clocks not found — skipping"
fi

# ── 3. Python virtual environment ────────────────────────────────────────────
info "[3/7] Creating Python virtual environment at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}" --system-site-packages
    info "Virtual environment created."
else
    info "Virtual environment already exists."
fi

# Activate (for this script session)
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── 4. Python packages ───────────────────────────────────────────────────────
info "[4/7] Installing Python packages..."
pip install -q \
    numpy \
    sounddevice \
    soundfile \
    requests

# faster-whisper / CTranslate2
# On JetPack 6.x, pip wheels for ctranslate2 aarch64+CUDA are available.
# If pip install fails, we fall back to CPU-only or point to NVIDIA containers.
info "Installing faster-whisper (this may take a few minutes on first install)..."
if pip install faster-whisper -q 2>/dev/null; then
    info "faster-whisper installed via pip."
else
    warn "pip install faster-whisper failed."
    warn "Trying ctranslate2 CPU-only wheel..."
    if pip install ctranslate2 faster-whisper --no-deps -q 2>/dev/null; then
        info "Installed (CPU fallback)."
    else
        warn "Could not install faster-whisper automatically."
        warn "You may need to build CTranslate2 from source for CUDA on this JetPack version."
        warn "  See: https://github.com/OpenNMT/CTranslate2/tree/master#installation"
        warn "  Or use the NVIDIA L4T ML container which includes it."
    fi
fi

# ── 5. Piper TTS (ARM64) ─────────────────────────────────────────────────────
PIPER_DIR="${WILSON_DIR}/piper"
PIPER_BIN="${PIPER_DIR}/piper"

info "[5/7] Setting up Piper TTS..."
if [ -f "${PIPER_BIN}" ]; then
    info "Piper binary already exists at ${PIPER_BIN}"
else
    info "Downloading Piper ${PIPER_VERSION} for linux_aarch64..."
    TEMP_TAR=$(mktemp /tmp/piper_XXXXXX.tar.gz)
    
    if wget -q --show-progress -O "${TEMP_TAR}" "${PIPER_URL}"; then
        # Extract to temp, then copy binary + libs without overwriting existing data
        TEMP_EXTRACT=$(mktemp -d /tmp/piper_extract_XXXXXX)
        tar xzf "${TEMP_TAR}" -C "${TEMP_EXTRACT}"

        # The archive extracts to piper/ subfolder
        EXTRACTED="${TEMP_EXTRACT}/piper"

        # Copy binary
        cp -f "${EXTRACTED}/piper" "${PIPER_BIN}" 2>/dev/null || true

        # Copy shared libraries (libonnxruntime, libespeak-ng, etc.)
        cp -f "${EXTRACTED}/"lib*.so* "${PIPER_DIR}/" 2>/dev/null || true
        if [ -d "${EXTRACTED}/lib" ]; then
            cp -rf "${EXTRACTED}/lib" "${PIPER_DIR}/" 2>/dev/null || true
        fi

        # Copy espeak-ng-data only if not already present
        if [ ! -d "${PIPER_DIR}/espeak-ng-data" ] && [ -d "${EXTRACTED}/espeak-ng-data" ]; then
            cp -rf "${EXTRACTED}/espeak-ng-data" "${PIPER_DIR}/"
        fi

        chmod +x "${PIPER_BIN}"
        rm -rf "${TEMP_TAR}" "${TEMP_EXTRACT}"
        info "Piper installed at ${PIPER_BIN}"
    else
        warn "Failed to download Piper.  espeak-ng will be used as fallback TTS."
        rm -f "${TEMP_TAR}"
    fi
fi

# Verify piper runs
if [ -f "${PIPER_BIN}" ]; then
    if "${PIPER_BIN}" --help &>/dev/null; then
        info "Piper binary verified."
    else
        warn "Piper binary exists but failed to run.  Check library dependencies:"
        warn "  ldd ${PIPER_BIN}"
    fi
fi

# ── 6. Ollama ─────────────────────────────────────────────────────────────────
info "[6/7] Installing Ollama..."
if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
    info "Ollama already installed: ${OLLAMA_VER}"
else
    info "Downloading and installing Ollama..."
    if curl -fsSL https://ollama.com/install.sh | sh; then
        info "Ollama installed."
    else
        warn "Ollama installation failed.  You can install manually:"
        warn "  curl -fsSL https://ollama.com/install.sh | sh"
    fi
fi

# Ensure Ollama service is running
if command -v systemctl &>/dev/null; then
    systemctl enable ollama 2>/dev/null || true
    systemctl start  ollama 2>/dev/null || true
    sleep 3
fi

# ── 7. Pull default LLM model ────────────────────────────────────────────────
MODEL="qwen2.5:7b-instruct-q4_K_M"
info "[7/7] Pulling LLM model: ${MODEL}..."

if command -v ollama &>/dev/null; then
    if ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
        info "Model already available."
    else
        info "Pulling ${MODEL} (~4.5 GB) — this will take a while on first run..."
        if ollama pull "${MODEL}"; then
            info "Model pulled successfully."
        else
            warn "Failed to pull model.  You can pull manually:"
            warn "  ollama pull ${MODEL}"
        fi
    fi
else
    warn "Ollama not available — skipping model pull."
fi

# ── Create convenience run script ────────────────────────────────────────────
RUN_SCRIPT="${WILSON_DIR}/run_wilson.sh"
cat > "${RUN_SCRIPT}" << 'RUNEOF'
#!/usr/bin/env bash
# Quick-start script for Wilson on Jetson
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/wilson_env/bin/activate"

# Set power mode (adjust: MAXN, 15W, 7W)
export WILSON_POWER_MODE="MAXN"

# Pass through any args (e.g., --headless, --check)
python3 "${SCRIPT_DIR}/wilson.py" "$@"
RUNEOF
chmod +x "${RUN_SCRIPT}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  To run Wilson (GUI mode):"
echo "    ${RUN_SCRIPT}"
echo ""
echo "  Headless mode (SSH / no display):"
echo "    ${RUN_SCRIPT} --headless"
echo ""
echo "  System diagnostics:"
echo "    ${RUN_SCRIPT} --check"
echo ""
echo "  Environment overrides:"
echo "    WILSON_LLM_MODEL=qwen2.5:7b-instruct-q4_K_M"
echo "    WILSON_WHISPER_MODEL=base"
echo "    WILSON_POWER_MODE=MAXN"
echo ""
echo "  Memory budget (8 GB shared):"
echo "    Whisper base FP16 ≈ 300 MB"
echo "    Ollama 7B Q4      ≈ 4.5 GB"
echo "    Piper + OS + GUI  ≈ 1.5 GB"
echo "    Free headroom     ≈ 1.7 GB"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
