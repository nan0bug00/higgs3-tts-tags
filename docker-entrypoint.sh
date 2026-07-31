#!/usr/bin/env bash
set -uo pipefail

HIGGS_MODEL_FILE="${HIGGS_MODEL_FILE:-/opt/models/higgs/higgs-audio-v3-tts-4b-q8_0.gguf}"
HIGGS_ENGINE_PORT="${HIGGS_ENGINE_PORT:-8081}"
HIGGS_ENGINE_THREADS="${HIGGS_ENGINE_THREADS:-4}"
HIGGS_REF_CACHE_SLOTS="${HIGGS_REF_CACHE_SLOTS:-8}"
HIGGS_BUSY_TIMEOUT_MS="${HIGGS_BUSY_TIMEOUT_MS:-300000}"
LOG=/var/log/inference.log
touch "$LOG"

# Transient CUDA failures at boot are a fact of life on rented hardware: a
# previous container releasing the GPU, another tenant's process still tearing
# down, or the driver briefly reporting "device busy or unavailable". Retry a
# bounded number of times so a two-second race doesn't cost a whole instance.
START_RETRIES="${START_RETRIES:-6}"
START_RETRY_DELAY="${START_RETRY_DELAY:-10}"

if [[ ! -f "$HIGGS_MODEL_FILE" ]]; then
    echo "FATAL: Higgs GGUF not found at $HIGGS_MODEL_FILE" | tee -a "$LOG"
    exit 1
fi

# audiocpp_server takes its whole model roster from a JSON config. Generated
# here rather than baked so the env vars above stay live. lazy_load=false:
# the GGUF maps at boot, and the wrapper's dummy generation then warms the
# graphs before 7860 is ever bound.
cat > /opt/higgs/server.json <<JSON
{
  "host": "127.0.0.1",
  "port": ${HIGGS_ENGINE_PORT},
  "backend": "cuda",
  "device": 0,
  "threads": ${HIGGS_ENGINE_THREADS},
  "lazy_load": false,
  "busy_timeout_ms": ${HIGGS_BUSY_TIMEOUT_MS},
  "models": [
    {
      "id": "${HIGGS_MODEL_ID:-higgs}",
      "family": "higgs_audio_tts",
      "path": "${HIGGS_MODEL_FILE}",
      "task": "tts",
      "mode": "offline",
      "session_options": {
        "higgs_audio_tts.reference_cache_slots": ${HIGGS_REF_CACHE_SLOTS}
      }
    }
  ]
}
JSON

# supervise <name> <command...>
# Restarts the command if it exits within $START_RETRY_DELAY*2 seconds (i.e. it
# failed to come up); a later exit is treated as a real death and ends the
# container so vast.ai can restart it cleanly.
supervise() {
    local name="$1"; shift
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        local started elapsed rc
        started=$(date +%s)
        "$@" 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
        elapsed=$(( $(date +%s) - started ))

        if (( elapsed > START_RETRY_DELAY * 2 )); then
            echo "[boot] $name exited after ${elapsed}s (code $rc) - not a startup failure" | tee -a "$LOG"
            return "$rc"
        fi
        if (( attempt >= START_RETRIES )); then
            echo "[boot] $name failed to start ${attempt}x (last code $rc); giving up" | tee -a "$LOG"
            return "$rc"
        fi
        echo "[boot] $name died after ${elapsed}s (code $rc); retry ${attempt}/${START_RETRIES} in ${START_RETRY_DELAY}s" | tee -a "$LOG"
        sleep "$START_RETRY_DELAY"
    done
}

# The engine binds 127.0.0.1 only: nothing outside the container should reach it.
echo "[boot] $(date -Is) starting audiocpp_server on 127.0.0.1:${HIGGS_ENGINE_PORT} (${HIGGS_MODEL_FILE})" | tee -a "$LOG"
supervise audiocpp-server \
    audiocpp_server --config /opt/higgs/server.json &
HIGGS_PID=$!

echo "[boot] $(date -Is) starting Higgs3 wrapper (binds 7860 only when warm)" | tee -a "$LOG"
supervise higgs-wrapper python -u /opt/higgs/higgs3_zonos_wrapper.py &
TTS_PID=$!

# --- idle watchdog: self-stop the vast.ai instance after log silence --------
echo "[watchdog] MAX_IDLE_SECONDS set to ${MAX_IDLE_SECONDS:-1800}" | tee -a "$LOG"
(
    MAX_IDLE=${MAX_IDLE_SECONDS:-1800}
    while true; do
        sleep 60
        last_change=$(stat -c %Y "$LOG")
        now=$(date +%s)
        idle=$(( now - last_change ))
        if (( idle > MAX_IDLE )); then
            printf "[watchdog] No log activity for %dm%02ds - stopping instance\n" \
                "$(( idle / 60 ))" "$(( idle % 60 ))"
            if [[ -n "${CONTAINER_API_KEY:-}" && -n "${CONTAINER_ID:-}" ]]; then
                vastai set api-key "$CONTAINER_API_KEY"
                vastai stop instance "$CONTAINER_ID"
            else
                echo "[watchdog] CONTAINER_API_KEY/CONTAINER_ID unset; cannot self-stop"
            fi
        fi
    done
) &
WATCHDOG_PID=$!

term() {
    echo "[boot] shutting down" | tee -a "$LOG"
    kill "$HIGGS_PID" "$TTS_PID" "$WATCHDOG_PID" 2>/dev/null || true
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
}
trap term SIGTERM SIGINT

# If a TTS-path server gives up for good, bring the container down rather than
# leave a half-dead box the mod might still race and win with.
wait -n "$HIGGS_PID" "$TTS_PID"
EXIT_CODE=$?
echo "[boot] a server exited for good (code $EXIT_CODE); stopping container" | tee -a "$LOG"
term
exit "$EXIT_CODE"
