#!/usr/bin/env python3
"""Quick smoke test: plain line + tagged line through the Zonos Gradio API."""

import json
import os
import sys
import time
import wave

import requests

HOST = os.environ.get("HIGGS_HOST", "http://127.0.0.1:7860")
REF = os.environ.get("HIGGS_REF", "refs/ref_audio.wav")


def wait_health(timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{HOST}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("health ok:", r.json())
                return
            print("health:", r.status_code, r.text[:200])
        except Exception as exc:  # noqa: BLE001
            print(f"waiting for health ({type(exc).__name__})...")
        time.sleep(5)
    raise SystemExit("health timeout")


def upload(session, path):
    with open(path, "rb") as f:
        r = session.post(
            f"{HOST}/gradio_api/upload",
            files={"files": (os.path.basename(path), f, "audio/wav")},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()[0]


def generate(session, text, ref_path, out_path, language="en-us", seed=1234):
    data = [
        "Zyphra/Zonos-v0.1-hybrid", text, language,
        {"meta": {"_type": "gradio.FileData"}, "path": ref_path}, None,
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2,
        0.7, 24000.0, 45.0, 14.6, 4.0, True, 3.0, 0.9, 1, 0.2,
        False, 0.7, False, seed, False, [],
    ]
    t0 = time.time()
    r = session.post(f"{HOST}/gradio_api/call/generate_audio", json={"data": data}, timeout=60)
    r.raise_for_status()
    eid = r.json()["event_id"]
    result = None
    with session.get(f"{HOST}/gradio_api/call/generate_audio/{eid}", stream=True, timeout=600) as resp:
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
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "path" in payload[0]:
                result = payload
                break
    if result is None:
        raise RuntimeError(f"no audio result for text={text!r}")
    audio = session.get(f"{HOST}/gradio_api/file={result[0]['path']}", timeout=120)
    audio.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(audio.content)
    with wave.open(out_path) as w:
        duration = w.getnframes() / float(w.getframerate())
    elapsed = time.time() - t0
    print(f"OK {out_path}: {duration:.2f}s audio in {elapsed:.2f}s | text={text[:80]!r}")
    if duration < 0.3:
        raise RuntimeError(f"audio too short: {duration:.2f}s")
    return duration


def main():
    wait_health()
    session = requests.Session()
    ref = upload(session, REF)
    os.makedirs("smoke_out", exist_ok=True)
    generate(session, "The Jarl will see you now.", ref, "smoke_out/plain.wav")
    generate(
        session,
        "[EMOTION-FEAR] I'm not going back down into that barrow.",
        ref,
        "smoke_out/fear.wav",
    )
    generate(
        session,
        "You call that a sword? [SFX-LAUGHTER]",
        ref,
        "smoke_out/laugh.wav",
    )
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
