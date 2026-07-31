# NOTES — higgs3-tts-tags (Higgs Audio v3 TTS 4B via audio.cpp)

Fork of elbios/higgs3-whisper-server: TTS-only (no whisper.cpp), plus ALL-CAPS
control-tag rewriting for SkyrimNet’s Zonos path. What broke *here*, engine
decisions, open questions. Numbers in MEASUREMENTS.md (historical, from the
upstream image that still bundled whisper).

## Engine decisions

### 1. audio.cpp primary path worked — no PyTorch fallback needed

`0xShug0/audio.cpp` @ `a343fb6` (2026-07-27 HEAD; v0.4 added Higgs v3 TTS).
Bare-proved on the box before any Dockerfile: CUDA build, official Q8_0 GGUF
(`audio-cpp/audio.cpp-gguf` @ `4afa5086`, 4.8 GB, standalone — embedded model
spec, no sidecar files), CLI and server both generate correct speech.

- Family name is **`higgs_audio_tts`** (the brief's research guessed
  `higgs_tts`).
- Output is 24 kHz / 16-bit / mono from the codec (`kCodecSampleRate = 24000`)
  — exactly what the mod consumes; no resampling in the wrapper.
- The engine builds **static** by default (`ENGINE_ENABLE_CPU_ALL_VARIANTS=OFF`
  ⇒ `BUILD_SHARED_LIBS=OFF`, `GGML_BACKEND_DL=OFF`): one big binary each for
  `audiocpp_cli` / `audiocpp_server`, no `.so` soup, no DL-backend AVX
  reintroduction risk. `ENGINE_ENABLE_NATIVE_CPU=OFF` maps to `GGML_NATIVE`;
  the full `GGML_AVX*=OFF` family is passed anyway (belt and braces) and
  verified by disassembly.

### 2. Transcript-free cloning only — no whisper in this image

`--reference-text` is optional. Measured bare (same text, same seed, ref b.wav
with its true transcript vs none): both 4.7–4.9 s, no EOS collapse either way,
and the two-voice specificity test ranks **OK on both F0 and LTAS without any
transcript**. A transcript adds a small F0 nudge toward the reference but
changes no ranking. So: no whisper transcription on the TTS path.

This fork goes further than elbios: whisper.cpp is **not shipped** at all.
STT for SkyrimNet belongs on the game PC (local whisper or SimpleParakeet).
There is no `HIGGS_USE_REFERENCE_TEXT` / `WHISPER_ENABLED` switch here.

**RESOLVED by ear test (2026-07-28, upstream).** The user listened to the
labelled A/B pairs in `samples/ab_transcript/` (same ref, line and seed;
transcript vs none) and judged both good, with a mild preference for
**transcript-free** most of the time.

### 3. No short-line runaway at stock settings — measured, not assumed

16-cell bare grid ("Yes?" / "Halt!" × 4 seeds × 2 refs, engine defaults, NO
max_tokens bound): every output landed at 0.72–0.96 s. audio.cpp's narrower
sampling defaults (top_k 30 / top_p 0.8) plus Higgs's EOS behaviour appear
inherently stable where MOSS ran away for minutes.

**A cap hit is a hard error, not a flush.** Measured over the server API:
`max_tokens` too small ⇒ HTTP error `"Higgs TTS generation reached max_tokens
before EOC"` and *no audio*.

**Decision (user's call): no wrapper cap by default.** The MOSS cap existed to
truncate that model's hallucinations; Higgs doesn't hallucinate length, and
because audio.cpp's cap fails the request rather than trimming it, a budget
can never save a good line — only turn a slow-spoken one into an error. The
engine's own per-chunk default (2048 steps ≈ 80 s) remains the pathological
bound. `HIGGS_TOKEN_BUDGET=1` re-enables the measured text-derived budget
(25 tok/s, slack 4.0, floor 150), with a one-shot retry at the ceiling on the
cap error so even then nothing is lost but latency. The retry is provoked in
testing via a deliberately tiny budget.

### 4. Engine-side reference cache is content-keyed — path churn is harmless

`HiggsTTSSession` caches codec-encoded references keyed on
`(sample_rate, channels, sample_count, FNV-hash-of-samples, reference_text)` —
not the path. Gradio's fresh-temp-path-per-upload therefore cannot miss it.
The wrapper's SHA-256 refcache still exists to (a) normalise MP3-in-WAV
(fmt 85) refs to PCM once, (b) pin identical bytes so the engine hash always
matches, (c) give the engine a stable server-local `voice_ref` path.
Default cache is **1 slot**; raised to 8 via
`session_options["higgs_audio_tts.reference_cache_slots"]` (a playthrough
alternates actor voices). There is also a single-slot AR prefix cache
(`ReferencePrefixCache`) that alternating voices will invalidate — that is
prefill cost only, not correctness. Analysed in full in §7.

### 5. Server architecture: resident `audiocpp_server`, wrapper as HTTP client

`POST /v1/audio/speech` accepts request-level `voice_ref` (server-local path),
`reference_text`, `seed`, `top_p`, `top_k`, `temperature`, `max_tokens` —
everything the wrapper needs; no CLI-worker hack, no patches required. Engine
port set to 8081 bound to 127.0.0.1 in the generated `server.json` (its
documented example defaults to 8080, which whisper owns — the collision the
brief warned about is real but config-solved). `lazy_load: false` so the GGUF
maps at boot; the wrapper's full dummy generation then warms AR graphs +
codec encode/decode before 7860 binds.

### 6. Quant: single q8 tag

Official GGUFs are Q8_0 and BF16 only; nobody has published a Q4 (the
community mirror has only the STT model). 4.8 GB of weights leaves no VRAM
pressure on a 16 GB card even with whisper resident, so a self-made Q4 buys
nothing the fleet needs — and audio-code prediction is exactly where
quantization error becomes audible (guide §quantization). Not pursued;
revisit only if a smaller-VRAM tier ever matters.

### 7. Three caches, and why we are NOT patching the prefill one

Investigated properly after the port shipped (source read at
`0xShug0/audio.cpp` @ `a343fb6`). Conclusion first: **leave the engine
alone.** The reasoning is below so nobody re-derives it.

**The three caches, in the order a request touches them:**

| # | cache | where | slots | keyed on | protects |
|---|---|---|---|---|---|
| 1 | normalised WAV files | wrapper, disk `/opt/refcache` | 64 (`HIGGS_REF_CACHE_SIZE`) | SHA-256 of upload | an ffmpeg run |
| 2 | codec reference codes | engine, `HiggsTTSSession` | 8 (`HIGGS_REF_CACHE_SLOTS`) | FNV hash of samples + `reference_text` | `codec_->encode_reference()` |
| 3 | AR prefix / KV state | engine, `HiggsGenerator` | **1, hardcoded** | element-wise compare of `reference_codes`, `reference_text`, `prefix_tokens` | the AR **prefill** over the reference |

**What is actually cached in #3.** `encode_prompt()` lays the prompt out as:

```
[tts] [ref_text][…transcript…] [ref_audio][…reference audio tokens…] │ [text][…the line…] [audio]
└──────────────── cached prefix (prefix_steps) ─────────────────────┘ └── varies per line ──┘
```

`prefix_steps = reference_positions.back() + 1`, i.e. it ends right after the
reference audio tokens, so **the spoken line is not part of the key** — same
actor, any dialogue, still a hit. On a hit the generator sets
`prefill_start_step = prefix_steps` and reuses the existing KV; after
generating it calls `retain_prefix(prefix_steps)` to truncate the KV back to
the reference, deliberately preserving it for next time. `ReferencePrefixCache`
holds only *inputs* (a fingerprint); the state itself lives in the single
`ar_kv_cache_`, guarded by `reference_kv_ready_`.

**Why one slot is pathological here, in principle.** Any conversation between
two or more NPCs evicts the slot every line, so the hit rate is **0%**, not
"sometimes misses". Four actors rotating A,B,C,D never hit; four slots would
hit ~100%.

**But the measured cost is small at our reference lengths.** Normalising the
probe-2 timings by seconds-of-audio generated (the two prefix-HIT runs agree
to 0.3%, which validates the linear model):

| run | wall | audio | predicted | overhead |
|---|---|---|---|---|
| same ref, prefix HIT | 0.58–0.66 s | 3.28/3.72 s | — | baseline 0.1771 s per audio-second |
| back to ref A, prefix MISS | 0.81 s | 3.92 s | 0.69 s | **+0.12 s** |
| new ref B, codes + prefix MISS | 0.93 s | 4.52 s | 0.80 s | +0.13 s |
| first request ever | 1.18 s | 3.84 s | 0.68 s | +0.50 s (one-time graph/CUDA warmup) |

Two things fall out. The **codes cache (#2) is worth ~0.01 s** — the
codes+prefix miss costs the same as the prefix miss alone — so raising its 8
slots buys essentially nothing, and the wrapper's #1 is worth less still.
And the whole alternating-voice penalty is #3.

**Prefill scales with reference length, not line length.** That 0.12 s is for
a **13.1 s** reference (~325 prefix steps at ~25 tokens/audio-second). It is
constant with respect to the *spoken line*, which is why a 4-character "Yes?"
pays the same as a sentence — but it is proportional to the *reference*:

| reference | prefill miss |
|---|---|
| 4 s | ~0.04 s |
| 10 s | ~0.09 s |
| 13.1 s (measured) | 0.12 s |
| 30 s | ~0.27 s |
| 60 s (`REF_MAX_SECONDS` cap) | ~0.54 s (extrapolated) |

**The decision.** Skyrim reference clips are typically **4–10 s**, rarely
longer (user's own data), so the real penalty is **~0.04–0.09 s per line** —
under the ~0.1 s threshold at which a delay reads as non-instant. For scale,
`measure.py` on a short line ("What is it? I'm busy.", 1.49 s audio, 0.43 s
wall) implies ~0.26 s of generation and therefore **~0.17 s of Gradio
upload/SSE overhead** — two to four times the prefill miss, and not
addressable because the mod's API is frozen. Patching upstream C++, then
rebasing that patch on every audio.cpp bump, is not worth 40–90 ms.

**What would flip this decision:** moving to long reference clips (a 30 s
reference triples the penalty), a mod change that makes latency dominate, or
audio.cpp adding multi-slot prefix caching upstream (then it is just a config
line, no patch).

**If it is ever built,** the shape is known and modest: `export_state()` /
`import_state()` already exist on `HiggsARKVCache`, so it is "keep a map of
reference digest → exported KV state, import on hit", plus a
`higgs_audio_tts.prefix_cache_slots` session option and a
`HIGGS_KV_CACHE_SLOTS` env var mirroring `HIGGS_REF_CACHE_SLOTS`. Size it to
~4 slots, not 64: unlike #1 and #2 (kilobytes), a KV state is
`layers × steps × kv_heads × head_dim` floats — the one cache here that is
genuinely expensive in VRAM.

### 8. Control tags in the dialogue text (plus optional `language` channel)

Higgs honours inline control tokens written on the line —
`<|emotion:elation|>`, `<|style:whispering|>`, `<|sfx:cough|>`, and they
compose (`<|style:whispering|><|emotion:arousal|>`). The Zonos API has no
field for that, and SkyrimNet often strips `<` / `|` before the line arrives.

**Primary path:** the LLM is prompted to emit ALL-CAPS / bracket tags
(`[EMOTION-FEAR]`, `SFX-LAUGHTER`, …). After `preprocess_text`,
`apply_control_tags` rewrites them into real Higgs tokens, moves
sentence-level tags to the front of their sentence, injects model-card
onomatopoeia after SFX tokens, and drops unrecognised markup. Tag rewrite
logic adapted from cleanestpoison/higgs3-tts-skyrimnet; always on (no env
toggle).

**Optional `language` channel (param 2):** still accepted for a sentence-level
token or caps tag (`<|emotion:anger|>` or `EMOTION-ANGER`). Plain `en-us` is
ignored. Useful as a back door; the intended SkyrimNet path is tags in the
dialogue text via a prompt such as `0650_audio_tags.prompt`.

Spoken length maths use `strip_control_tokens(...)` so token budget and
short-output checks measure speech, not markup. If the engine hits its
max_tokens ceiling before finishing (CapHit), the retry may drop control
tokens and speak the plain line rather than repeating an identical request.

Upstream ear check (elbios): 28 clips in `samples/emotion_set/` covered all
21 emotions plus styles and sfx at a fixed seed. Whether each tag *sounds*
right remains an ear question on this fork after rebuild.

This does not reopen Q4 below: that dead end was the engine's `instructions`
field, which `higgs_audio_tts` never reads. Inline text tokens are a
different mechanism entirely.

## Open quality questions (untested — ranked by likely impact)

Nothing here is known to be wrong; these are knobs we shipped defaults for
without A/B-ing them. Listed so they are not rediscovered from scratch.

### Q1. Q8 vs BF16 — never compared. Biggest untested risk.

We shipped `q8_0` because it is the official default GGUF, and never tested
it against `higgs-audio-v3-tts-4b-bf16.gguf` (same HF repo, same revision).
`guides/ggml.md` warns specifically that a backbone predicting **audio
codebook tokens** is not like a text model: quantization error becomes an
audible artifact rather than a synonym, and must be A/B'd objectively plus
given to the user as labelled samples.

VRAM is not the obstacle it would be on MOSS — we use 7,963 MiB of 24 GB, and
BF16 (~9.6 GB of weights vs 4.8) should land ~13 GB total, still inside a
16 GB card. Needs a rebuild (`HIGGS_GGUF` build arg) plus box time.

### Q2. Sampling is deliberately narrower than Boson's reference client.

audio.cpp's `higgs_audio_tts` defaults (`session.cpp`) are
`temperature 0.8, top_p 0.8, top_k 30`. Its own docs state the reference
Python client uses **`top_k` 50** and `top_p` 1.0, and that the narrower
default was chosen because it is "less prone to premature EOC" — i.e.
expressiveness traded for stability. Narrow sampling tends toward flatter,
more monotone delivery, and we never tested the reference values.

Note the effective runtime config is a mix: `top_p` comes from the mod
(Zonos input [20], 0.9), while `temperature`/`top_k` sit at audio.cpp's
narrowed defaults.

**Cheapest test in this list — no rebuild:**
`-e HIGGS_TOP_K=50 -e HIGGS_TEMPERATURE=1.0` and listen. Try this first.

### Q3 (closed). The silence trim is inherited from MOSS and defaulted off.

`TRIM_SILENCE` used to default on, with `TRIM_THRESHOLD=0.02` of peak and a
0.15 s tail pad. It exists because MOSS emitted seconds of pad frames after
finishing a line (up to 5.8 s after a 0.5 s "Yes?"). Higgs shows no such
pathology in any grid we ran, so the trim was either doing nothing or — the
real risk — shaving natural breath and decaying consonants.

**Decision: default flipped to `0`.** An unverified transform that can only
subtract from the audio should not be on by default for an engine that never
exhibited the problem it was written for. The code stays and is unchanged;
`-e HIGGS_TRIM_SILENCE=1` restores the old behaviour if a pad-frame pathology
ever shows up.

### Q4 (closed, dead end). The mood *sliders* cannot be wired up.

> Superseded in part by #8 above: expressiveness IS reachable, via inline
> control tokens in the text. What follows remains true of the eight Zonos
> emotion sliders and the engine's `instructions` field.

The Zonos API accepts eight emotion sliders and the mod does send a mood.
`audiocpp_server` accepts an `instructions` field and maps it to
`request.options["instruct"]` — but `higgs_audio_tts` **never reads it**
(`grep -rn instruct src/models/higgs_audio_tts/` returns nothing). So there is
no expressiveness knob to connect, and discarding the sliders costs nothing.
Do not re-investigate.

## Known minor (deliberately not changed after the image was verified)

With the budget off (the default), a `CapHitError` can only mean the engine's
own 2048-step default was hit — a genuine runaway. The wrapper still performs
its one retry at 2048, which is the same value, so that retry cannot help: it
fails again and the request correctly returns `None`. Cost is one wasted
generation on a case never observed in any test (16-cell bare grid, 16-cell
wrapper grid, all of measure.py). Left as-is so the shipped image and this
source stay byte-identical; worth a two-line guard (`skip the retry when the
budget is None`) on the next rebuild.

## What broke here

- `git clone --depth 1` of audio.cpp is fine — external deps (ggml, cJSON,
  libyaml, sentencepiece, llama_tokenizer) are **vendored in-tree**, not
  submodules (unlike openmoss).
- The devel container lacks `/usr/bin/time` and `bc` — probe scripts, not the
  port. (Noted so the next port's probes don't repeat it.)
- License: Higgs v3 weights are research/non-commercial (Creator Use Grant;
  commercial use needs Boson's separate license). Flagged in README.md —
  user's call, not a build blocker.
