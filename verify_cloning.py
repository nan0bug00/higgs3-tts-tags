#!/usr/bin/env python3
"""Objectively verify that MOSS-TTS cloning produces speech and is voice-*specific*.

"The output is audio" is easy to pass with a 440 Hz tone; "the output is speech"
is easy to pass with a generic voice. This drives the running container over the
real Zonos API with two different reference clips and then checks, without ever
asking a human to listen:

  1. Each output is *speech-like*    -- duration consistent with the text,
                                        varying envelope, plausible ZCR.
  2. Each output is *voice-specific* -- nearest its own reference under two
                                        independent measures.

The Qwen3 version of this script used that model's own speaker encoder for
measure (2). openmoss has no such thing -- it is a C++ binary with no exposed
embedding API -- so both measures here are computed from the waveform alone:

  * median F0 over voiced frames (autocorrelation)
  * long-term average spectrum (LTAS) cosine similarity

They are independent: F0 is the excitation (vocal fold rate), LTAS is the
filter (vocal tract / timbre). A voice can match on one by accident; matching on
both is evidence.

Per the porting guide, the verdict is a *ranking* -- each output must be nearer
its own reference than the other one -- not a comparison against an absolute
threshold, which is meaningless in an unvalidated space.

    python3 verify_cloning.py --host http://127.0.0.1:7860 \
        --ref-a refs/pcm_ref_audio.wav --ref-b refs/pcm_ref_audio_2.wav
"""

import argparse
import json
import os
import random
import time

import numpy as np
import soundfile as sf

import requests

TEXT = ("Halt! You have committed crimes against Skyrim and her people. "
        "What say you in your defense?")


# --------------------------------------------------------------------------
# Driving the API (identical flow to test-higgs-zonos.py and the mod)
# --------------------------------------------------------------------------


def upload(session, host, path):
    with open(path, "rb") as f:
        r = session.post(f"{host}/gradio_api/upload",
                         files={"files": (os.path.basename(path), f, "audio/wav")},
                         timeout=60)
    r.raise_for_status()
    return r.json()[0]


def generate(session, host, text, ref_server_path, out_path, seed=None):
    if seed is None:
        seed = random.randint(1, 2 ** 32 - 1)
    data = ["Zyphra/Zonos-v0.1-hybrid", text, "en-us",
            {"meta": {"_type": "gradio.FileData"}, "path": ref_server_path}, None,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2,
            0.7, 24000.0, 45.0, 14.6, 4.0, True, 3.0, 0.9, 1, 0.2,
            False, 0.7, False, seed, False, []]
    t0 = time.time()
    r = session.post(f"{host}/gradio_api/call/generate_audio", json={"data": data}, timeout=60)
    r.raise_for_status()
    eid = r.json()["event_id"]

    result = None
    with session.get(f"{host}/gradio_api/call/generate_audio/{eid}",
                     stream=True, timeout=300) as resp:
        for line in resp.iter_lines():
            if line and line.decode().startswith("data:"):
                body = line.decode()[5:].strip()
                try:
                    d = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, list) and d and isinstance(d[0], dict) and "path" in d[0]:
                    result = d
                    break
    if result is None:
        raise RuntimeError("no result")

    audio = session.get(f"{host}/gradio_api/file={result[0]['path']}", timeout=120)
    audio.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(audio.content)
    return time.time() - t0


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------


def _load(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def _voiced_frames(x, sr, win_s=0.04, hop_s=0.02):
    """Frame the signal and return (frames, energies, threshold)."""
    win, hop = int(win_s * sr), int(hop_s * sr)
    starts = list(range(0, max(1, len(x) - win), hop))
    energies = np.array([np.sqrt(np.mean(x[i:i + win] ** 2)) for i in starts]) \
        if starts else np.array([])
    return starts, win, energies


F0_QUANTILES = np.linspace(10, 90, 9)


def f0_track(path, fmin=55.0, fmax=400.0):
    """Per-frame fundamental frequency over voiced frames, via autocorrelation.

    Deliberately dependency-free (numpy only) and independent of any spectral
    measure, so F0 and LTAS are genuinely separate lines of evidence.
    """
    x, sr = _load(path)
    starts, win, energies = _voiced_frames(x, sr)
    if energies.size == 0:
        return np.array([])
    thresh = 0.25 * energies.max()
    lo, hi = int(sr / fmax), int(sr / fmin)

    pitches = []
    for idx, i in enumerate(starts):
        if energies[idx] < thresh:
            continue  # unvoiced / silence
        frame = x[i:i + win] - x[i:i + win].mean()
        ac = np.correlate(frame, frame, mode="full")[win - 1:]
        if ac[0] <= 0 or hi >= len(ac):
            continue
        seg = ac[lo:hi]
        if seg.size == 0:
            continue
        lag = lo + int(np.argmax(seg))
        # Require a reasonably periodic frame, else it is noise not pitch.
        if ac[lag] / ac[0] > 0.3:
            pitches.append(sr / lag)
    return np.array(pitches)


def f0_profile(path):
    """Quantiles of the log-F0 track, in semitones.

    A single median is fragile here: autocorrelation octave errors and creaky
    phonation make some real speakers come out bimodal, and the median then
    lands wherever the two modes happen to balance. Comparing the whole
    quantile vector is far more stable, and semitones are the right units
    because pitch difference is perceptually multiplicative.
    """
    v = f0_track(path)
    if v.size < 5:
        return None
    return np.percentile(12.0 * np.log2(v), F0_QUANTILES)


def f0_distance(a, b):
    """Mean absolute quantile difference, in semitones. Smaller = closer."""
    if a is None or b is None:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def ltas(path, n_bands=48, fmin=80.0, fmax=8000.0):
    """Long-term average spectrum over voiced frames, log-scaled and normalised.

    This is the *filter* side of source-filter: it characterises the speaker's
    vocal-tract resonances and general timbre, and is close to pitch-invariant,
    which is what makes it independent evidence from median F0.
    """
    x, sr = _load(path)
    starts, win, energies = _voiced_frames(x, sr)
    if energies.size == 0:
        return np.zeros(n_bands)
    thresh = 0.25 * energies.max()

    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    # Log-spaced band edges: mimics how hearing (and formant structure) works.
    edges = np.geomspace(fmin, min(fmax, sr / 2 - 1), n_bands + 1)
    idx = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0]
           for b in range(n_bands)]

    acc = np.zeros(n_bands)
    n = 0
    for j, i in enumerate(starts):
        if energies[j] < thresh:
            continue
        spec = np.abs(np.fft.rfft((x[i:i + win] - x[i:i + win].mean()) * window)) ** 2
        acc += np.array([spec[k].mean() if k.size else 0.0 for k in idx])
        n += 1
    if n == 0:
        return np.zeros(n_bands)
    v = np.log10(acc / n + 1e-12)
    v = v - v.mean()                      # remove overall loudness
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def speech_report(path, text):
    """Duration / envelope / ZCR sanity: is this speech, or a tone or silence?"""
    x, sr = _load(path)
    duration = len(x) / sr
    starts, win, energies = _voiced_frames(x, sr)
    peak = float(np.abs(x).max()) if x.size else 0.0
    # Speech has a strongly varying envelope; a steady tone does not.
    env_cv = float(energies.std() / energies.mean()) if energies.size and energies.mean() > 0 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if x.size > 1 else 0.0
    expected = len(text) / 15.0    # English runs ~15 characters/second
    ok = (peak > 0.02 and env_cv > 0.35 and 0.005 < zcr < 0.35
          and 0.35 * expected < duration < 3.0 * expected)
    return ok, dict(duration=duration, expected=expected, peak=peak,
                    env_cv=env_cv, zcr=zcr)


def multi_trial(args):
    """Repeat the A/B over several fixed seeds and score the two measures.

    MOSS generation is stochastic and its pitch-register fidelity varies from
    seed to seed (NOTES.md). A single trial is therefore an anecdote in both
    directions -- it can pass by luck as easily as it can fail by luck. This
    runs N of them and reports the hit rate, which is the number worth quoting.
    """
    seeds = [1000 + 7919 * i for i in range(args.trials)]
    rows = []
    with requests.Session() as s:
        up_a = upload(s, args.host, args.ref_a)
        up_b = upload(s, args.host, args.ref_b)
        pa, pb = f0_profile(args.ref_a), f0_profile(args.ref_b)
        la, lb = ltas(args.ref_a), ltas(args.ref_b)
        for seed in seeds:
            oa, ob = f"/tmp/vc_a_{seed}.wav", f"/tmp/vc_b_{seed}.wav"
            generate(s, args.host, TEXT, up_a, oa, seed=seed)
            generate(s, args.host, TEXT, up_b, ob, seed=seed)
            qa, qb = f0_profile(oa), f0_profile(ob)
            ka, kb = ltas(oa), ltas(ob)
            f0_hit = (f0_distance(qa, pa) < f0_distance(qa, pb)
                      and f0_distance(qb, pb) < f0_distance(qb, pa))
            ltas_hit = (float(np.dot(ka, la)) > float(np.dot(ka, lb))
                        and float(np.dot(kb, lb)) > float(np.dot(kb, la)))
            spa = speech_report(oa, TEXT)[0]
            spb = speech_report(ob, TEXT)[0]
            rows.append((seed, spa and spb, f0_hit, ltas_hit))
            print(f"  seed {seed:6d}: speech {'ok ' if spa and spb else 'BAD'}   "
                  f"F0 {'hit ' if f0_hit else 'miss'}   LTAS {'hit ' if ltas_hit else 'miss'}")

    n = len(rows)
    speech = sum(r[1] for r in rows)
    f0 = sum(r[2] for r in rows)
    lt = sum(r[3] for r in rows)
    either = sum(r[2] or r[3] for r in rows)
    print(f"\n  speech-like:            {speech}/{n}")
    print(f"  F0 nearest own ref:     {f0}/{n}")
    print(f"  LTAS nearest own ref:   {lt}/{n}")
    print(f"  at least one measure:   {either}/{n}")
    # Chance is 1/4 per trial for "both outputs ranked correctly", so a measure
    # that hits most of the time is carrying real signal.
    passed = speech == n and lt > n / 2 and either == n
    print(f"\n  {'PASS - cloning is voice-specific' if passed else 'FAIL - review above'}")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:7860")
    ap.add_argument("--ref-a", required=True)
    ap.add_argument("--ref-b", required=True)
    ap.add_argument("--out-a", default="/tmp/out_a.wav")
    ap.add_argument("--out-b", default="/tmp/out_b.wav")
    ap.add_argument("--skip-generate", action="store_true",
                    help="analyse existing --out-a/--out-b instead of calling the API")
    ap.add_argument("--trials", type=int, default=1,
                    help="repeat the whole A/B with different fixed seeds and "
                         "report how often each measure ranks correctly. "
                         "Generation is stochastic; one sample is an anecdote.")
    args = ap.parse_args()

    if args.trials > 1:
        return multi_trial(args)

    if not args.skip_generate:
        with requests.Session() as s:
            print("Generating with reference A ...")
            ta = generate(s, args.host, TEXT, upload(s, args.host, args.ref_a), args.out_a)
            print(f"  {ta:.2f}s")
            print("Generating with reference B ...")
            tb = generate(s, args.host, TEXT, upload(s, args.host, args.ref_b), args.out_b)
            print(f"  {tb:.2f}s")

    names = {"ref_A": args.ref_a, "ref_B": args.ref_b,
             "out_A": args.out_a, "out_B": args.out_b}

    print("\n=== is it speech at all? (outputs only) ===")
    speech_ok = True
    for k in ("out_A", "out_B"):
        ok, m = speech_report(names[k], TEXT)
        speech_ok &= ok
        print(f"  {k}: {m['duration']:.2f}s (expected ~{m['expected']:.1f}s)  "
              f"peak {m['peak']:.3f}  envelope CV {m['env_cv']:.2f}  "
              f"ZCR {m['zcr']:.3f}   [{'ok' if ok else 'SUSPECT'}]")

    print("\n=== F0 profile distance, semitones (autocorrelation, voiced frames) ===")
    prof = {k: f0_profile(v) for k, v in names.items()}
    for k in ("ref_A", "out_A", "ref_B", "out_B"):
        med = float(np.median(f0_track(names[k]))) if prof[k] is not None else float("nan")
        print(f"  {k}: median {med:6.1f} Hz")
    da_own = f0_distance(prof["out_A"], prof["ref_A"])
    da_other = f0_distance(prof["out_A"], prof["ref_B"])
    db_own = f0_distance(prof["out_B"], prof["ref_B"])
    db_other = f0_distance(prof["out_B"], prof["ref_A"])
    print(f"  out_A: {da_own:.2f} st from its own ref vs {da_other:.2f} st from the other")
    print(f"  out_B: {db_own:.2f} st from its own ref vs {db_other:.2f} st from the other")
    pitch_ok = da_own < da_other and db_own < db_other
    # A measure can only discriminate voices it can tell apart in the first
    # place. If the two references sit close together in pitch, F0 says nothing
    # either way and must not be allowed to fail the run -- report it as
    # uninformative rather than pretending it is evidence.
    ref_f0_gap = f0_distance(prof["ref_A"], prof["ref_B"])
    pitch_informative = ref_f0_gap >= 3.0
    print(f"  the two references are {ref_f0_gap:.2f} st apart"
          f" -> F0 is {'discriminative here' if pitch_informative else 'NOT discriminative here'}")

    print("\n=== long-term average spectrum, cosine similarity ===")
    vecs = {k: ltas(v) for k, v in names.items()}

    def cos(a, b):
        return float(np.dot(vecs[a], vecs[b]))   # already unit-normalised

    matched = [("out_A", "ref_A"), ("out_B", "ref_B")]
    crossed = [("out_A", "ref_B"), ("out_B", "ref_A")]
    for a, b in matched + crossed:
        tag = "MATCHED" if (a, b) in matched else "crossed"
        print(f"  {a} vs {b}: {cos(a, b):+.4f}   [{tag}]")
    baseline = cos("ref_A", "ref_B")
    print(f"  ref_A vs ref_B (the two voices themselves): {baseline:+.4f}")

    # Ranking, not an absolute threshold: see PORTING-GUIDE.md §4.
    ltas_ok = (cos("out_A", "ref_A") > cos("out_A", "ref_B")
               and cos("out_B", "ref_B") > cos("out_B", "ref_A"))
    print(f"\n  each output nearest its own reference (LTAS): {ltas_ok}")
    print(f"  each output nearest its own reference (F0):   {pitch_ok}"
          f"{'' if pitch_informative else '  (uninformative, not counted)'}")

    passed = speech_ok and ltas_ok and (pitch_ok or not pitch_informative)
    print(f"\n  {'PASS - output is speech and cloning is voice-specific' if passed else 'FAIL - review above'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
