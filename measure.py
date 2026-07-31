#!/usr/bin/env python3
"""Latency harness for the Qwen3-TTS Zonos wrapper.

Drives exactly the same /gradio_api flow as test-higgs-zonos.py (and therefore
the Skyrim mod), timing each request end to end from the client's point of view.

    python3 measure.py --host http://localhost:7860 --ref-audio ref_audio.wav
"""

import argparse
import json
import os
import random
import statistics
import time
import wave

import requests

SKYRIM_LINES = [
    "I used to be an adventurer like you, then I took an arrow in the knee.",
    "You'll never see the inside of a jail cell, I promise you that.",
    "The Jarl will see you now. Mind your manners.",
    "What is it? I'm busy.",
    "By the gods, that's a lot of gold. Where did you get it?",
]


def upload(session, host, path):
    with open(path, "rb") as f:
        r = session.post(
            f"{host}/gradio_api/upload",
            files={"files": (os.path.basename(path), f, "audio/wav")},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()[0]


def payload_for(text, ref_server_path, seed):
    return {
        "data": [
            "Zyphra/Zonos-v0.1-hybrid", text, "en-us",
            {"meta": {"_type": "gradio.FileData"}, "path": ref_server_path},
            None,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2,
            0.7, 24000.0, 45.0, 14.6, 4.0, True, 3.0, 0.9, 1, 0.2,
            False, 0.7, False, seed, False, [],
        ]
    }


def one_request(session, host, text, ref_server_path):
    seed = random.randint(1, 2 ** 32 - 1)
    t0 = time.time()
    r = session.post(
        f"{host}/gradio_api/call/generate_audio",
        json=payload_for(text, ref_server_path, seed),
        timeout=60,
    )
    r.raise_for_status()
    event_id = r.json()["event_id"]

    result = None
    with session.get(
        f"{host}/gradio_api/call/generate_audio/{event_id}", stream=True, timeout=300
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:"):
                continue
            body = decoded[5:].strip()
            if not body:
                continue
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            if (isinstance(data, list) and data and isinstance(data[0], dict)
                    and "path" in data[0]):
                result = data
                break

    if result is None:
        raise RuntimeError("no result payload")

    audio = session.get(f"{host}/gradio_api/file={result[0]['path']}", timeout=120)
    audio.raise_for_status()
    elapsed = time.time() - t0

    out = "/tmp/qwen3_measure_out.wav"
    with open(out, "wb") as f:
        f.write(audio.content)
    with wave.open(out) as w:
        duration = w.getnframes() / w.getframerate()
    return elapsed, duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:7860")
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    with requests.Session() as s:
        print("== cold cache: fresh upload of a reference the server has never seen ==")
        ref = upload(s, args.host, args.ref_audio)
        cold_elapsed, cold_dur = one_request(s, args.host, SKYRIM_LINES[0], ref)
        print(f"cold  : {cold_elapsed:6.2f}s wall -> {cold_dur:5.2f}s audio "
              f"(RTF {cold_dur / cold_elapsed:.2f})")

        print("\n== warm cache: same reference clip, re-uploaded each time "
              "(exactly what the mod does) ==")
        times, rtfs = [], []
        for rnd in range(args.rounds):
            for i, line in enumerate(SKYRIM_LINES):
                ref = upload(s, args.host, args.ref_audio)  # fresh temp path each time
                el, dur = one_request(s, args.host, line, ref)
                times.append(el)
                rtfs.append(dur / el)
                print(f"warm r{rnd} #{i}: {el:6.2f}s wall -> {dur:5.2f}s audio "
                      f"(RTF {dur / el:.2f})  \"{line[:45]}\"")

        print(f"\nwarm mean {statistics.mean(times):.2f}s  "
              f"median {statistics.median(times):.2f}s  "
              f"min {min(times):.2f}s  max {max(times):.2f}s")
        print(f"client-side RTF mean {statistics.mean(rtfs):.2f}")


if __name__ == "__main__":
    main()
