# syntax=docker/dockerfile:1.7
#
# Higgs Audio v3 TTS 4B (audio.cpp) behind the Zonos Gradio API.
#
# Two processes at runtime:
#   7860  gradio wrapper (the Zonos API the Skyrim mod speaks)   -- published
#   8081  audiocpp_server (the engine)                           -- internal only
#
# Single tag (q8): the official Q8_0 GGUF is 4.8 GB, so the box sits far below
# a 16 GB card -- there is no VRAM pressure to justify a second quant (see
# MEASUREMENTS.md).
#
# Multi-stage on purpose: the CUDA *devel* image (~6 GB) is only needed to
# compile audio.cpp, so it never reaches the runtime layer.
# No torch anywhere -- the engine is C++ and the wrapper is an HTTP client.
#
# Layer order is deliberate -- expensive and stable first, volatile last:
#   audio.cpp binaries -> weights -> python deps -> wrapper
# so that editing the wrapper rebuilds one small layer.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=ubuntu24.04
ARG HIGGS_HF_REPO=audio-cpp/audio.cpp-gguf
ARG HIGGS_HF_REV=4afa5086d2411a3b9e8ec96f1a1c4dd5312b50cd
ARG HIGGS_GGUF=Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf

# ===========================================================================
# Stage 1: build audio.cpp (audiocpp_server + audiocpp_cli)
#
# audio.cpp is ggml too (vendored under external/ggml), so the AVX trap
# applies in full. Its own switch is ENGINE_ENABLE_NATIVE_CPU, which maps to
# GGML_NATIVE; the explicit GGML_AVX* family below belt-and-braces the rest,
# and the default ENGINE_ENABLE_CPU_ALL_VARIANTS=OFF keeps the build static
# (GGML_BACKEND_DL=OFF), so no per-microarch variant can dlopen AVX back in.
# ===========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU_VERSION} AS audiocpp-builder

ARG AUDIOCPP_REF=a343fb6a2c8e61f7f2dcca9912b7bf50a612a252
ARG CUDA_ARCHITECTURES="80;86;89;90;120"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Vendored deps (external/ggml etc.) live in-tree -- no submodules needed.
RUN git init -q . \
    && git remote add origin https://github.com/0xShug0/audio.cpp.git \
    && git fetch -q --depth 1 origin ${AUDIOCPP_REF} \
    && git checkout -q FETCH_HEAD

ENV CUDA_STUB_DIR=/usr/local/cuda/lib64/stubs
RUN ln -sf ${CUDA_STUB_DIR}/libcuda.so ${CUDA_STUB_DIR}/libcuda.so.1
ENV LIBRARY_PATH=${CUDA_STUB_DIR}

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-L${CUDA_STUB_DIR} -Wl,-rpath-link,${CUDA_STUB_DIR}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L${CUDA_STUB_DIR} -Wl,-rpath-link,${CUDA_STUB_DIR}" \
        -DENGINE_ENABLE_CUDA=ON \
        -DENGINE_ENABLE_NATIVE_CPU=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX=OFF \
        -DGGML_AVX2=OFF \
        -DGGML_AVX512=OFF \
        -DGGML_AVX_VNNI=OFF \
        -DGGML_AVX512_VBMI=OFF \
        -DGGML_AVX512_VNNI=OFF \
        -DGGML_AVX512_BF16=OFF \
        -DGGML_FMA=OFF \
        -DGGML_F16C=OFF \
        -DGGML_BMI2=OFF \
        -DGGML_AMX_TILE=OFF \
        -DGGML_AMX_INT8=OFF \
        -DGGML_AMX_BF16=OFF \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build build -j "$(nproc)" --target audiocpp_cli --target audiocpp_server

COPY check-no-avx.sh /usr/local/bin/check-no-avx.sh
# sed strips Windows CRLF if the build context was sent from a Windows checkout.
RUN sed -i 's/\r$//' /usr/local/bin/check-no-avx.sh \
    && chmod +x /usr/local/bin/check-no-avx.sh \
    && mkdir -p /out/bin /out/lib /out/model_specs \
    && cp build/bin/audiocpp_server build/bin/audiocpp_cli /out/bin/ \
    && { find build -maxdepth 3 -name '*.so*' -exec cp -a {} /out/lib/ \; || true; } \
    && cp -a model_specs/. /out/model_specs/ \
    && bash /usr/local/bin/check-no-avx.sh /out/bin/audiocpp_server /out/bin/audiocpp_cli /out/lib

# ===========================================================================
# Stage 2-base: the downloader. hf/Xet: measured ~85 MB/s vs curl's 2-3 MB/s.
# ===========================================================================
FROM python:3.12-slim AS hf-downloader

RUN pip install --no-cache-dir "huggingface_hub[cli]" hf_xet

# ===========================================================================
# Stage 2: weights, pinned
# ===========================================================================
FROM hf-downloader AS weights

ARG HIGGS_HF_REPO
ARG HIGGS_HF_REV
ARG HIGGS_GGUF
RUN hf download ${HIGGS_HF_REPO} --revision ${HIGGS_HF_REV} \
        "${HIGGS_GGUF}" --local-dir /dl \
    && mkdir -p /models/higgs \
    && mv "/dl/${HIGGS_GGUF}" /models/higgs/ \
    && rm -rf /dl && ls -l /models/higgs

# A reference clip baked into the image, used to warm every lazy code path at
# boot (same clip every previous port warms with).
ADD https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main/ref_audio.wav /models/warmup_ref_raw.wav

# ===========================================================================
# Stage 3: runtime. Plain `runtime`, not `cudnn-runtime`: nothing uses cuDNN.
# ===========================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates curl ffmpeg libsndfile1 \
        libgomp1 binutils procps \
    && rm -rf /var/lib/apt/lists/*

# --- python deps -----------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy

# No torch, no transformers: the engine is a separate C++ process, so the
# wrapper needs nothing but an HTTP client and the Gradio surface.
# The gradio/gradio-client/fastapi/starlette quartet is the known-good stack
# verified against the mod's /gradio_api SSE flow. Do not float it.
RUN uv venv --python 3.12 /opt/venv \
    && uv pip install --no-cache \
        "numpy<2.3" \
        soundfile \
        loguru \
        requests \
        gradio==5.38.2 \
        gradio-client==1.11.0 \
        fastapi==0.115.14 \
        starlette==0.45.3 \
        "pydantic>=2.0,<3" \
    && uv pip install --no-cache vastai

# --- audio.cpp -------------------------------------------------------------
# Static binaries (BUILD_SHARED_LIBS=OFF), but /out/lib is kept in case a
# future ref goes shared.
COPY --from=audiocpp-builder /out/bin/audiocpp_server /usr/local/bin/audiocpp_server
COPY --from=audiocpp-builder /out/bin/audiocpp_cli /usr/local/bin/audiocpp_cli
COPY --from=audiocpp-builder /out/lib/ /usr/local/lib/audiocpp/
# Fallback model-spec catalog: normal (non-deployment) builds resolve
# model_specs/<family>.json from disk when a GGUF has no embedded spec.
COPY --from=audiocpp-builder /out/model_specs/ /opt/higgs/model_specs/
ENV LD_LIBRARY_PATH=/usr/local/lib/audiocpp:${LD_LIBRARY_PATH}

COPY check-no-avx.sh /usr/local/bin/check-no-avx.sh
# Re-verify in the final image: this is what actually ships.
# sed strips Windows CRLF if the build context was sent from a Windows checkout.
RUN sed -i 's/\r$//' /usr/local/bin/check-no-avx.sh \
    && chmod +x /usr/local/bin/check-no-avx.sh \
    && bash /usr/local/bin/check-no-avx.sh \
        /usr/local/bin/audiocpp_server /usr/local/bin/audiocpp_cli /usr/local/lib/audiocpp

# --- models (large, stable) ------------------------------------------------
COPY --from=weights /models/higgs /opt/models/higgs

# The warmup reference is MP3-in-a-WAV-container (fmt tag 85); normalise it
# once at build time to the codec's 24 kHz PCM.
COPY --from=weights /models/warmup_ref_raw.wav /tmp/warmup_ref_raw.wav
RUN mkdir -p /opt/higgs \
    && ffmpeg -hide_banner -loglevel error -y -i /tmp/warmup_ref_raw.wav \
        -t 30 -ac 1 -ar 24000 -c:a pcm_s16le /opt/higgs/warmup_ref.wav \
    && rm -f /tmp/warmup_ref_raw.wav \
    && ls -l /opt/higgs/warmup_ref.wav

# --- application (volatile -- keep last) -----------------------------------
ENV HIGGS_QUANT=q8 \
    HIGGS_MODEL_FILE=/opt/models/higgs/higgs-audio-v3-tts-4b-q8_0.gguf \
    HIGGS_ENGINE_PORT=8081 \
    HIGGS_ENGINE_URL=http://127.0.0.1:8081 \
    HIGGS_REF_CACHE_DIR=/opt/refcache \
    HIGGS_PORT=7860 \
    HIGGS_WARMUP_REF=/opt/higgs/warmup_ref.wav \
    MAX_IDLE_SECONDS=1800

WORKDIR /opt/higgs
RUN mkdir -p /opt/refcache
COPY higgs3_zonos_wrapper.py /opt/higgs/higgs3_zonos_wrapper.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# 8081 is deliberately absent: audiocpp_server binds 127.0.0.1 and nothing
# outside the container should reach the engine directly.
EXPOSE 7860

# 7860 only answers once the engine is loaded and a full dummy generation has
# run, so this is a true readiness probe.
HEALTHCHECK --interval=15s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
