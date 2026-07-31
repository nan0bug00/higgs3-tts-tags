# higgs3-tts-tags — Higgs Audio v3 TTS with control tags (Vast.ai)

Higgs Audio v3 TTS 4B (Boson AI) served through
[audio.cpp](https://github.com/0xShug0/audio.cpp) (pure C++/ggml, no torch),
wrapped in the fixed Zonos Gradio API SkyrimNet speaks. This image is
**TTS-only**: no whisper.cpp. Speech-to-text stays on the game PC
(SkyrimNet’s local whisper, or SimpleParakeet, etc.).

Based on [Elbios/higgs3-whisper-server](https://github.com/Elbios/higgs3-whisper-server),
with ALL-CAPS / bracket performance-tag rewriting adapted from
[cleanestpoison/higgs3-tts-skyrimnet](https://github.com/cleanestpoison/higgs3-tts-skyrimnet).

> **License note (operator decision before public/commercial use).**
> Higgs Audio v3 weights are released under Boson AI's
> **research/non-commercial license** with a Creator Use Grant; commercial or
> hosted use needs a separate license from Boson. This image bakes those
> weights into a Docker image — whether that is acceptable is the operator's
> call.

## What this adds over stock elbios

SkyrimNet’s Zonos config page has no UI for Higgs performance tags, and the
mod often strips characters like `<` and `|` before the line reaches the
container. Prompt the LLM to emit tags such as `[EMOTION-FEAR]` or
`EMOTION-FEAR`; the wrapper rewrites them into real Higgs tokens
(`<|emotion:fear|>`, …) before calling the engine. Sound tags get the
model-card onomatopoeia injected; junk tags are dropped so NPCs do not read
markup aloud.

Stay on the **Zonos** provider URL. A sample SkyrimNet prompt fragment is in
[`0650_audio_tags.prompt`](0650_audio_tags.prompt) — drop it into your
SkyrimNet prompts; the container does not need that file to run.

## Ports

| Port | What | Published |
|---|---|---|
| 7860 | Zonos-compatible Gradio API (`generate_audio`, 29 inputs) | yes |
| 8081 | `audiocpp_server` (engine) | **no — 127.0.0.1 only** |

7860 is not bound until the model is loaded and a full dummy generation has
run; `GET /health` on 7860 reports `{"status":"ok", ...}` (no whisper field).

## Environment

| Var | Default | Meaning |
|---|---|---|
| `HIGGS_TOKEN_BUDGET` | `0` | `1` sends a text-derived `max_tokens`; off by default — a cap hit is a failed request (no audio), never a truncation |
| `HIGGS_TOKENS_PER_SECOND` / `HIGGS_CHARS_PER_SECOND` | `25` / `15` (measured) | length-budget model (only with `HIGGS_TOKEN_BUDGET=1`) |
| `HIGGS_MAX_TOKEN_SLACK` / `HIGGS_MAX_TOKEN_FLOOR` / `HIGGS_MAX_TOKEN_CEIL` | `4.0` / `150` / `2048` | `max_tokens = min(ceil, max(floor, slack·expected + floor))` |
| `HIGGS_TRIM_SILENCE` | `0` | trim trailing near-silence from outputs (off: see NOTES Q3) |
| `HIGGS_REF_MAX_SECONDS` | `60` | reference clips longer than this are truncated (logged) |
| `HIGGS_REF_CACHE_SLOTS` | `8` | engine-side reference-encode cache slots |
| `MAX_IDLE_SECONDS` | `1800` | idle watchdog self-stops the vast instance |
| `CONTAINER_API_KEY` / `CONTAINER_ID` | — | needed by the watchdog (vast sets these) |
| `START_RETRIES` / `START_RETRY_DELAY` | `6` / `10` | boot retries for transient CUDA failures |

Optional overrides: `HIGGS_TOP_K`, `HIGGS_TEMPERATURE` (engine sampling).

## Build (on a machine with Docker — e.g. Vast)

```bash
docker build -t higgs3-tts-tags:q8 .
```

Weights are baked at pinned revisions (`audio-cpp/audio.cpp-gguf` @
`4afa5086`, Q8_0). Nothing downloads at runtime. All ggml binaries are built
with every AVX-family flag off and verified by disassembly (`check-no-avx.sh`)
at build time and in the final image.

## Acceptance

```bash
python3 measure.py --host http://<box>:<port7860> --ref-audio refs/ref_audio.wav
python3 verify_cloning.py --host http://<box>:<port7860> \
    --ref-a refs/ref_audio.wav --ref-b refs/ref_audio_2.wav
```

See `MEASUREMENTS.md` for historical numbers from the upstream elbios image
(with whisper) and `NOTES.md` for engine decisions.

## Cloning

Voice cloning is **transcript-free** only. Elbios measured no meaningful gain
from feeding a whisper transcript of the reference, and this fork does not
ship STT at all.
