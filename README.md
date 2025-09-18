# Live Avatar Engine: MuseTalk 1.5 + FishAudio LIVE

This project scaffolds a live avatar system that drives a portrait (few photos) with MuseTalk 1.5 while streaming speech audio from FishAudio's LIVE engine.

It provides:
- Installation scripts to fetch MuseTalk 1.5
- A Python skeleton to connect to FishAudio LIVE (WebSocket) and feed audio into MuseTalk
- A simple CLI to test live or batch mode

Note: This repo is a thin integration layer. The actual MuseTalk model code is pulled from its official GitHub, and FishAudio requires your API key.

## Prerequisites
- OS: Linux or macOS recommended
- GPU: NVIDIA GPU with recent CUDA for real-time performance
- Python: 3.10+ (match MuseTalk’s requirements)
- FFmpeg: for audio/video handling (`ffmpeg` on PATH)
- Git: to clone MuseTalk
- FishAudio credentials: `FISH_AUDIO_API_KEY`

## Quick Start

1) Clone MuseTalk 1.5

```
./scripts/setup_musetalk.sh
```
This clones `TMElyralab/MuseTalk` into `external/MuseTalk` and checks out the latest 1.5 branch/tag if available.

2) Create environment and install deps

Option A: Conda (recommended)
```
conda create -n live-avatar python=3.10 -y
conda activate live-avatar
pip install -r requirements.txt
# Then follow MuseTalk/README to install its dependencies (torch, etc.)
```

Option B: venv
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Then follow MuseTalk/README to install its dependencies (torch, etc.)
```

3) Set API key

Create a `.env` file (copy from `.env.example`) and set:
```
FISH_AUDIO_API_KEY=sk-...
FISH_AUDIO_VOICE_ID=your-default-voice-id
```

4) Prepare avatar image(s)

Place a clear, front-facing portrait (or a few) under `assets/avatars/`. Use PNG or JPG. MuseTalk expects a single reference image; for multiple photos, pick the best or extend the runner to support identity building.

5) Run a batch demo (text → audio → talking video)

```
python -m src.live_avatar.main \
  --mode batch \
  --image assets/avatars/example.jpg \
  --text "こんにちは、ライブアバターのテストです。" \
  --out out/demo.mp4
```
By default this uses OpenAI (gpt-4o) to generate the response text from your `--text` prompt, then requests TTS audio from FishAudio (HTTP stub by default), and renders video with MuseTalk. To bypass OpenAI, add `--no-use-openai`.

6) Run a live demo (experimental)

```
python -m src.live_avatar.main \
  --mode live \
  --image assets/avatars/example.jpg
```
This connects to FishAudio LIVE (SDK by default) and streams audio chunks into MuseTalk to render frames progressively. It also uses OpenAI by default to generate responses. Use `--no-use-openai` and/or `--no-use-fishaudio-sdk` to disable.

### Run with Docker (GPU)

Prereqs: Docker + NVIDIA Container Toolkit (GPU), network access.

Build image:
```
docker compose build
```

Run LIVE (OpenAI + FishAudio SDK enabled by default):
```
docker compose run --rm --gpus all live-avatar \
  python -m src.live_avatar.main \
  --mode live \
  --image assets/avatars/avatar.png
```

Run BATCH:
```
docker compose run --rm --gpus all live-avatar \
  python -m src.live_avatar.main \
  --mode batch \
  --image assets/avatars/avatar.png \
  --text "自己紹介してや" \
  --out out/demo.mp4
```

Note: The container clones `external/MuseTalk` and downloads weights on build. Ensure `.env` contains your API keys before running.

Default persona: Japanese Kansai-dialect “veteran comedian-like” assistant. It speaks brightly and concisely, adds light jokes and interjections in Kansai dialect, while avoiding direct imitation of any specific real person’s unique catchphrases.

### Install FishAudio SDK (recommended, enabled by default)

If you use FishAudio’s official SDK for real-time streaming:

```
# Using uv
uv pip install fish-audio-sdk

# Or plain pip
pip install fish-audio-sdk
```

LIVE uses the SDK by default. You can explicitly pass it as well:

```
python -m src.live_avatar.main \
  --mode live \
  --image assets/avatars/example.jpg
```

## MuseTalk Runner Integration

The file `src/live_avatar/musetalk_runner.py` wraps MuseTalk. Because MuseTalk’s CLI/API may change and v1.5 specifics evolve, this is implemented as a thin adapter with two options:
- Subprocess mode: call MuseTalk’s official inference script (recommended for stability)
- Python API mode: import MuseTalk modules and call them directly

By default, the adapter contains TODOs you should finalize once MuseTalk is installed. See in-file comments to point it to the correct script and arguments in your checked-out `external/MuseTalk`.

## FishAudio LIVE Integration

Recommended: use the official FishAudio SDK.

- SDK client: `src/live_avatar/fishaudio_sdk_client.py`
- Enable via CLI: pass `--use-fishaudio-sdk`
- Configure via `.env`: `FISH_AUDIO_API_KEY`, `FISH_AUDIO_VOICE_ID`

If the SDK is unavailable, a direct WebSocket client is provided:
`src/live_avatar/fishaudio_client.py` contains an async client for FishAudio LIVE over WebSocket. Confirm the exact LIVE endpoints and message schema from FishAudio’s docs. The code includes placeholders for:
- Auth header / token query
- Start/stop synthesis messages
- Audio chunk parsing (e.g., PCM or opus)

Adjust the endpoint and payload as per FishAudio’s latest API. For offline/batch mode, there’s a simple HTTP TTS fallback stub; wire in FishAudio’s non-live TTS API if preferred.

## Project Layout

- `scripts/setup_musetalk.sh` – clone MuseTalk under `external/MuseTalk`
- `src/live_avatar/main.py` – CLI entrypoint (batch/live)
- `src/live_avatar/fishaudio_client.py` – FishAudio LIVE client (WebSocket)
- `src/live_avatar/musetalk_runner.py` – MuseTalk 1.5 adapter
- `requirements.txt` – minimal Python deps for integration code
- `assets/avatars/` – put your portrait image(s) here

## Notes
- Real-time performance depends heavily on GPU and module choices. Enable half-precision/optimized models in MuseTalk where available.
- Audio formats: adjust sample rate and channels to match MuseTalk expectations (often 16k mono PCM).
- If lipsync quality isn’t ideal, refine the reference image and consider face-alignment per MuseTalk guidance.

## Next Steps
- Confirm FishAudio LIVE endpoint + schema, update the client.
- Finalize the MuseTalk subprocess command line for v1.5 inference.
- Optionally add a small web UI for camera and chat input.
### OpenAI (gpt-4o) is enabled by default

Set in `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

Disable it with `--no-use-openai`.
