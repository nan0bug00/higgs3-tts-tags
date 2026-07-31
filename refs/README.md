# refs/

`ref_audio.wav` — the reference clip the commands in the top-level README use.
It is a byte-identical copy of the sample published by
[`andimarafioti/faster-qwen3-tts`](https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main/ref_audio.wav),
which the `Dockerfile` also fetches from that URL to warm the engine at boot.
It is vendored here only so the documented commands run without a download.

Any 16-bit WAV of clean speech works as a substitute. `verify_cloning.py` needs
a **second**, clearly different voice for `--ref-b`; supply your own, since the
point of that check is that the two outputs rank nearer their own references.
