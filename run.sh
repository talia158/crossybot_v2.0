#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
AVD_NAME="${AVD_NAME:-Small_Phone}"    # Name from: emulator -list-avds
SERIAL="${SERIAL:-}"                   # Leave empty to use the first ADB device
FPS="${FPS:-60}"                       # Target FPS
CROP="${CROP:-}"                       # Optional: "X Y W H" e.g. "0 100 1080 1700"
IMAGE_NAME="${IMAGE_NAME:-crossybot}" # Docker image name

# Android Studio SDK paths (macOS default)
ANDROID_SDK="${ANDROID_SDK:-$HOME/Library/Android/sdk}"
EMULATOR_BIN="${EMULATOR_BIN:-$ANDROID_SDK/emulator/emulator}"
ADB="${ADB:-$ANDROID_SDK/platform-tools/adb}"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FRAMES="${LOG_FRAMES:-$SCRIPT_DIR/logs/session-$(date +%Y%m%d-%H%M%S).jsonl}"

# 1. Start the emulator if no device is already connected
if "$ADB" devices | grep -qE "emulator|device$"; then
    echo "[run.sh] ADB device already connected, skipping emulator launch."
else
    echo "[run.sh] Starting emulator: $AVD_NAME"
    "$EMULATOR_BIN" -avd "$AVD_NAME" -no-audio -no-boot-anim -no-snapshot-load &
    EMULATOR_PID=$!
    echo "[run.sh] Emulator PID: $EMULATOR_PID"
fi

# 2. Wait for the device to be ready
echo "[run.sh] Waiting for ADB device..."
"$ADB" wait-for-device

echo "[run.sh] Waiting for boot to complete..."
until "$ADB" ${SERIAL:+-s "$SERIAL"} shell getprop sys.boot_completed 2>/dev/null | grep -q "^1$"; do
    sleep 2
done
echo "[run.sh] Device ready."

# 3. Set display resolution
echo "[run.sh] Setting resolution to 270x480..."
"$ADB" ${SERIAL:+-s "$SERIAL"} shell wm size 270x480
"$ADB" ${SERIAL:+-s "$SERIAL"} shell wm density 160

# 4. Restart ADB server listening on all interfaces (so Docker can reach it)
echo "[run.sh] Restarting ADB server on 0.0.0.0:5037..."
"$ADB" kill-server
"$ADB" -a -P 5037 start-server
# Re-wait for device after server restart
"$ADB" wait-for-device

# 5. Build the Docker image
echo "[run.sh] Building Docker image: $IMAGE_NAME"
NO_CACHE_FLAG=""
[[ "${NO_CACHE:-0}" == "1" ]] && NO_CACHE_FLAG="--no-cache"
docker build ${NO_CACHE_FLAG} -t "$IMAGE_NAME" "$SCRIPT_DIR"

# 6. Configure XQuartz for Docker: listen on TCP, disable auth
NEED_XQUARTZ_RESTART=0
if [[ "$(defaults read org.xquartz.X11 nolisten_tcp 2>/dev/null || echo 1)" != "0" ]]; then
    defaults write org.xquartz.X11 nolisten_tcp -bool false
    NEED_XQUARTZ_RESTART=1
fi
if [[ "$(defaults read org.xquartz.X11 no_auth 2>/dev/null || echo 0)" != "1" ]]; then
    defaults write org.xquartz.X11 no_auth -bool true
    NEED_XQUARTZ_RESTART=1
fi

# Restart XQuartz if prefs changed or start it if not running
if [[ "$NEED_XQUARTZ_RESTART" == "1" ]] && pgrep -qf Xquartz; then
    echo "[run.sh] XQuartz prefs changed — restarting XQuartz..."
    osascript -e 'quit app "XQuartz"' 2>/dev/null || true
    pkill -f Xquartz 2>/dev/null || true
    sleep 2
fi
if ! pgrep -qf Xquartz; then
    echo "[run.sh] Starting XQuartz..."
    open -a XQuartz
fi

echo "[run.sh] Waiting for XQuartz socket..."
for _ in {1..30}; do
    [[ -S /tmp/.X11-unix/X0 ]] && break
    sleep 1
done
if [[ ! -S /tmp/.X11-unix/X0 ]]; then
    echo "[run.sh] ERROR: XQuartz socket /tmp/.X11-unix/X0 not available after 30s" >&2
    exit 1
fi

# 7. Ensure X11 is reachable on TCP 6000 (XQuartz listens itself when nolisten_tcp=false;
#    fall back to socat bridge if it isn't)
if lsof -iTCP:6000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[run.sh] XQuartz already listening on TCP 6000, skipping socat."
    SOCAT_PID=""
else
    echo "[run.sh] Starting socat X11 bridge on TCP 6000..."
    socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:/tmp/.X11-unix/X0 &
    SOCAT_PID=$!
    echo "[run.sh] X11 bridge PID: $SOCAT_PID"
fi

# Disable X11 access control so Docker connections aren't rejected
DISPLAY=:0 /opt/X11/bin/xhost + 2>/dev/null || true

# Clean up socat when script exits
cleanup() { [[ -n "$SOCAT_PID" ]] && kill "$SOCAT_PID" 2>/dev/null; }
trap cleanup EXIT

# 8. Build the Docker run command
DOCKER_ARGS=(
    docker run --rm -it
    -e "DISPLAY=host.docker.internal:0"
    -e "ADB_SERVER_SOCKET=tcp:host.docker.internal:5037"
    -e "QT_X11_NO_MITSHM=1"
    -e "PYTHONUNBUFFERED=1"
    -e "OPENCV_FFMPEG_CAPTURE_OPTIONS=probesize;32|analyzeduration;0"
)

BOT_ARGS=(--fps "$FPS")
[[ -n "$SERIAL" ]] && BOT_ARGS+=(--serial "$SERIAL")
[[ -n "$CROP"   ]] && BOT_ARGS+=(--crop $CROP)

# Mount a host directory for frame logs if LOG_FRAMES is set
if [[ -n "$LOG_FRAMES" ]]; then
    LOG_DIR_RAW="$(dirname "$LOG_FRAMES")"
    mkdir -p "$LOG_DIR_RAW"
    LOG_HOST_DIR="$(cd "$LOG_DIR_RAW" && pwd)"
    LOG_BASENAME="$(basename "$LOG_FRAMES")"
    echo "[run.sh] Logging frames to host path: $LOG_HOST_DIR/$LOG_BASENAME"
    DOCKER_ARGS+=(-v "$LOG_HOST_DIR:/logs")
    BOT_ARGS+=(--log-frames "/logs/$LOG_BASENAME")
fi

# 9. Launch the bot in Docker
echo "[run.sh] Starting bot in Docker..."
"${DOCKER_ARGS[@]}" "$IMAGE_NAME" "${BOT_ARGS[@]}"
