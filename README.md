## Piper + “Akakios” (LM Studio voice output)

This folder is a small Windows-focused setup that combines:

- **LM Studio** (OpenAI-compatible local API) for chat responses
- **Piper** (`piper.exe`) for text-to-speech (TTS)
- A simple Python loop in `akakios.py` that:
	1) sends your prompt to LM Studio,
	2) speaks the model’s response using Piper,
	3) plays the generated `.wav`.

The included voice model in this workspace is `en_US-amy-medium.onnx` (located one directory above this folder).

---

## What’s in here

- `akakios.py` — CLI chat loop + TTS playback.
- `piper.exe` + `*.dll` — Piper binary and runtime libraries.
- `espeak-ng-data/` + `espeak-ng.dll` — phonemizer data used by Piper.
- `response.wav` — output file written by `akakios.py` (overwritten each response).
- `test.wav` — a sample wav file (if present).
- `libtashkeel_model.ort` — included runtime model file (used by some Piper builds for Arabic; harmless to keep).

One directory above (sibling of this folder):

- `../en_US-amy-medium.onnx` — the voice model.
- `../en_US-amy-medium.onnx.json` — model config (sample rate, eSpeak voice, inference params).

---

## Requirements

- Windows (this bundle uses `piper.exe` and `os.startfile` for playback)
- Python 3.x
- LM Studio running locally with the OpenAI-compatible server enabled

Python dependency:

- `requests`

Install it:

```bash
pip install requests
```

---

## LM Studio setup

`akakios.py` calls this endpoint:

- `http://localhost:1234/v1/chat/completions`

In LM Studio:

1. Download/load a chat model.
2. Start the local server (OpenAI-compatible).
3. Ensure it listens on `localhost:1234`.

If your server uses a different host/port, update `LM_STUDIO_URL` in `akakios.py`.

---

## Run

From this folder:

```bash
python akakios.py
```

Then type messages at the `You:` prompt. Type `quit` or `exit` to stop.

On each turn:

- The assistant response prints to the console.
- Piper generates audio into `response.wav`.
- Windows opens the wav using your default associated app.

---

## Configuration (important)

`akakios.py` currently uses **absolute paths** matching one specific machine:

- `PIPER_EXE = C:\Users\icanm\Downloads\piper\piper\piper.exe`
- `PIPER_MODEL = C:\Users\icanm\Downloads\piper\en_US-amy-medium.onnx`
- `OUTPUT_WAV = C:\Users\icanm\Downloads\piper\piper\response.wav`

If your folder location is different, edit those constants near the top of `akakios.py`.

To change voices, point `PIPER_MODEL` at a different `*.onnx` model.

---

## Troubleshooting

- **No audio plays**: `os.startfile` depends on a default program for `.wav`. Confirm Windows has a default media player association.
- **LM Studio errors / connection refused**: make sure the server is running and the URL/port matches `LM_STUDIO_URL`.
- **Piper fails silently**: stderr is suppressed (`stderr=subprocess.DEVNULL`). Temporarily remove that line to see Piper error output.
- **Output file in use**: close the media player that has `response.wav` open, or change `OUTPUT_WAV`.

---

## Notes

- The text is lightly “cleaned” before TTS (removes `*`, `#`, and backticks) to reduce awkward spoken punctuation.
- `response.wav` is overwritten each response.

