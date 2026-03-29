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
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

# 6. Bridge XQuartz Unix socket to TCP so Docker can reach it
pkill -f "socat TCP-LISTEN:6000" 2>/dev/null || true
sleep 0.5
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:/tmp/.X11-unix/X0 &
SOCAT_PID=$!
echo "[run.sh] X11 bridge PID: $SOCAT_PID"

# Disable X11 access control so Docker connections aren't rejected
DISPLAY=:0 /opt/X11/bin/xhost + 2>/dev/null || true

# Clean up socat when script exits
cleanup() { kill "$SOCAT_PID" 2>/dev/null; }
trap cleanup EXIT

# 7. Build the Docker run command
DOCKER_ARGS=(
    docker run --rm -it
    -e "DISPLAY=host.docker.internal:0"
    -e "ADB_SERVER_SOCKET=tcp:host.docker.internal:5037"
    -e "QT_X11_NO_MITSHM=1"
)

BOT_ARGS=(--fps "$FPS")
[[ -n "$SERIAL" ]] && BOT_ARGS+=(--serial "$SERIAL")
[[ -n "$CROP"   ]] && BOT_ARGS+=(--crop $CROP)

# 8. Launch the bot in Docker
echo "[run.sh] Starting bot in Docker..."
"${DOCKER_ARGS[@]}" "$IMAGE_NAME" "${BOT_ARGS[@]}"
