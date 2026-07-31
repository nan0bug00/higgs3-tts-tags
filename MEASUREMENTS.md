# MEASUREMENTS — higgs3-tts-tags

Historical numbers below were taken against **elbios/higgs3-whisper:q8** (the
upstream image that still bundled whisper.cpp). This fork is TTS-only: expect
lower image size and ~2 GB less VRAM than the “whisper + engine, warm” row.
Re-measure on Vast after you build this image.

Box: vast.ai RTX 4090 24 GB (instance 46038423), CUDA 12.8, 30-core host.
"Bare" = audio.cpp built in a devel container, no wrapper. Container numbers
follow after image verification. Numbers are measured; anything not measured
says so.

## Bare engine (audio.cpp @ a343fb6, Q8_0 GGUF @ 4afa5086, arch 89)

### Output format

24 000 Hz, 16-bit PCM, mono — every probe output. Matches the mod; the
wrapper does no resampling.

### Resident server latency (68-char line, `/v1/audio/speech`, RTX 4090)

| request | wall | note |
|---|---|---|
| 1st, new reference | 1.18 s | includes codec encode of 13 s reference |
| 2nd, same ref, same seed | 0.66 s | engine reference-cache hit |
| 3rd, same ref, new seed | 0.58 s | ~3.3–3.8 s audio ⇒ RTF ≈ 5–6× |
| switch to ref B | 0.93 s | new ref encode |
| back to ref A | 0.81 s | cache slot survived (slots=8) |

CLI-per-request (model reload every call) was used only for the grid probes;
not representative — the image runs the resident server.

**Read these normalised, not raw.** Wall time scales with how much audio the
seed happened to produce, so comparing raw numbers overstates cache costs.
Dividing by seconds-of-audio, the two prefix-cache-hit runs agree to 0.3%
(0.1768 vs 0.1774 s per audio-second), giving a baseline of **0.1771 s per
audio-second** and this decomposition:

| run | wall | audio | predicted | overhead |
|---|---|---|---|---|
| same ref, both caches hit | 0.58 / 0.66 s | 3.28 / 3.72 s | — | baseline |
| back to ref A (codes hit, prefix miss) | 0.81 s | 3.92 s | 0.69 s | **+0.12 s** |
| new ref B (codes + prefix miss) | 0.93 s | 4.52 s | 0.80 s | +0.13 s |
| first request ever | 1.18 s | 3.84 s | 0.68 s | +0.50 s one-time |

So: reference **re-encode ≈ 0.01 s**, reference **re-prefill ≈ 0.12 s** (at a
13.1 s reference; it scales with reference length, ~0.04–0.09 s for the 4–10 s
clips the mod actually sends), and ~0.50 s of one-time graph/CUDA warmup that
the boot-time dummy generation absorbs before 7860 binds. See NOTES.md §7 for
the cache architecture and why the prefill cache was left unpatched.

### Token rate (bisection, server API)

| text | audio | max_tokens OK | max_tokens error |
|---|---|---|---|
| 68 chars | 3.76 s | 112 | 96 |
| "Yes?" | 0.80 s | 30 | 24 |

⇒ ~25 AR tokens per audio-second + a few overhead tokens. Wrapper budget:
`min(2048, max(75, 25·(chars/15)·2.5 + 75))`.
**Cap behaviour: hard HTTP error ("reached max_tokens before EOC"), no
audio** — wrapper retries once at the 2048 ceiling on that error.

### Short-line grid, bare (engine defaults, no bound)

"Yes?" / "Halt!" × 4 seeds × 2 refs = 16 cells: **all outputs 0.72–0.96 s.**
No runaway, no trailing-silence pathology at stock top_k=30/top_p=0.8.

### Cloning specificity, bare (transcript-free, seed 1234)

Two references (A = ref_audio 13.1 s, B = ref_audio_2 10.9 s), F0-quantile
distance (semitones) and LTAS cosine, per verify_cloning.py method:

| output | F0 dist A / B | LTAS cos A / B | verdict |
|---|---|---|---|
| cloneA (ref A) | **1.26** / 6.31 | **+0.998** / +0.990 | nearest own ref on both |
| cloneB (ref B) | 5.36 / **1.13** | +0.993 / **+0.998** | nearest own ref on both |

With vs without `--reference-text` (ref b, its true transcript, same seed):
4.88 s vs 4.68 s, both rank max-LTAS to own reference; transcript changes no
ranking ⇒ transcript-free default.

### VRAM, bare

Engine resident (server, model loaded, after requests): **5 528–5 702 MiB**
total on the card (≈ 5.2–5.4 GiB engine; 309 MiB was pre-existing).

## Container (elbios/higgs3-whisper:q8, 11.9 GB on disk)

### Boot

`docker run` → `/health` 200 (model loaded, whisper answering a real
transcription, full dummy clone completed): **16 s** on the dev box.
Caveat: image layers were hot in the host page cache (98 GB RAM, image built
minutes earlier); a first-ever boot on a fresh instance also pays disk reads
for ~6.5 GB of weights — budget 30–60 s depending on the host. 7860 is not
bound before readiness, so this number *is* the race-safe metric.

### Latency through the full Zonos flow (measure.py on the box, RTX 4090)

| | wall/request | note |
|---|---|---|
| cold reference (never seen) | 0.78 s | includes ffmpeg normalise + codec encode |
| warm, mean of 10 | **0.56 s** (median 0.56, min 0.42, max 0.68) | 1.5–3.4 s audio ⇒ client RTF ≈ 4.7 |
| `test-higgs-zonos.py` from the Mac over an SSH tunnel | 1.9 s end-to-end | upload + generate + download |

### Short-line grid through the wrapper

"Yes?" / "Halt!" × 2 refs × 4 random seeds = 16 cells: **0.57–1.08 s**, all
bounded, silence-trim active. Matches the bare grid.

### Cloning (formal, through the real API)

`verify_cloning.py`, 2 references, both measures: **PASS — output is speech
and cloning is voice-specific.** Each output nearest its own reference on
LTAS *and* on F0.

### VRAM

| configuration | total |
|---|---|
| whisper (large-v3-turbo) + engine, warm | **7,963 MiB** |
| `WHISPER_ENABLED=0` | **≈5,700 MiB** (engine process 5,376 MiB) |

Fits an 8 GB card in principle, 12 GB comfortably; the mod's 16 GB filter has
large margin. This is the headline difference vs moss (17.3 GB for :q8).

### Fallbacks provoked (each one fired; none shipped as hope)

| fallback | provocation | observed |
|---|---|---|
| CapHit → retry at ceiling | `HIGGS_TOKEN_BUDGET=1`, floor 8, slack 0.05 | every request logged "max_tokens (9–12) hit before EOC; retrying once at ceiling 2048" and still returned audio |
| whisper dies at runtime | `pkill -x whisper-server` in container | container stayed up, TTS kept generating, `/health` flipped to `"whisper":"down"` |
| runtime `WHISPER_MODEL` switch | `-e WHISPER_MODEL=base.en` | "not baked in; downloading" → STT served with base.en (**after** fixing the baked-ENV bug this provocation caught) |
| whisper download fails | `-e WHISPER_MODEL=no-such-model-xyz` | "download failed; disabling STT (TTS unaffected)" → `/health` `"whisper":"disabled"`, TTS unaffected |
| transient CUDA at boot | happened live (another container releasing the GPU) | supervise retried 6 × 10 s, then ended the container — exactly the contract |

### Registry

| tag | manifest digest | compressed pull | on disk |
|---|---|---|---|
| `elbios/higgs3-whisper:q8` | `sha256:ac1a56df6be43cf52d5891def8b8b930d94f65126dab2228ec1f9b1980165219` | 9.39 GB (30 layers) | 11.9 GB |

Digest verified from the registry after push. `test-higgs-zonos.py` then ran
**unmodified** against a container booted from
`elbios/higgs3-whisper@sha256:ac1a56df…` — the exact pushed artifact — and
passed: 3.60 s of 24 kHz speech (`samples/pushed_image_acceptance.wav`).

## Reference transcript A/B (for the ear test)

Same ref, line and seed; only `reference_text` differs. Distance to the
output's own reference — F0-quantile (lower closer) and LTAS cosine (higher
closer). Tally: F0 favours transcript-free 5–1, LTAS splits 3–3; every output
still ranks nearest its own reference either way.

| sample | none F0 / LTAS | stt F0 / LTAS |
|---|---|---|
| ref_audio_line1 | 0.64 / +0.9978 | 1.33 / +0.9982 |
| ref_audio_line2 | 0.86 / +0.9980 | 4.86 / +0.9911 |
| ref_audio_line3 | 2.85 / +0.9963 | 2.58 / +0.9969 |
| ref_audio_2_line1 | 1.07 / +0.9953 | 2.63 / +0.9960 |
| ref_audio_2_line2 | 2.41 / +0.9966 | 4.42 / +0.9949 |
| ref_audio_2_line3 | 3.13 / +0.9964 | 6.10 / +0.9958 |

**Not a controlled result:** the transcript changes the prompt, so a same-seed
pair diverges anyway — the spread above is the size of ordinary seed variance,
and one seed per cell cannot separate the two. Durations differ by 4–17%, so
the transcript does change pacing. Samples in `samples/ab_transcript/`;
the ear test decides, not this table.

## Piggyback: moss-whisper rebuild (same box, same session)

- Rollback aliases created before overwrite:
  `q8-v1` = manifest `sha256:9f0e6bb5…`, `q4-v1` = manifest `sha256:537b4cc5…`
  (wrapping the previously published digests `9090b824…` / `020768bf…`).
- `:q8` rebuilt with the staged WHISPER_ENABLED changes; smoke:
  `WHISPER_ENABLED=0` → `/health` 200, `"whisper":"disabled"`, warmup clone
  ran, **15 098 MiB total VRAM** ⇒ the "q8 fits a 16 GB card with STT off"
  README claim holds (measured, 1 286 MiB headroom).
- `:q4` rebuilt and pushed (`sha256:9574542748909…`), with the
  baked-`WHISPER_MODEL_FILE` fix that the higgs3 provocations uncovered.
- `:q8` was first pushed (`sha256:fbf1355886f69…`) *before* that bug was
  found, so it was rebuilt clean and re-pushed — see
  moss-whisper-server/MEASUREMENTS.md for the final digest.
