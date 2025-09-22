#!/usr/bin/env bash

set -Eeuo pipefail

CONTAINER_NAME=${CONTAINER_NAME:-stt-mcp-server-linux}
TMUX_SESSION=${TMUX_SESSION:-claude}
TMUX_TMPDIR=${TMUX_TMPDIR:-"$HOME"/.tmux}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or not accessible" >&2
    exit 1
fi

# Stop and remove container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" || true
    docker rm "$CONTAINER_NAME" || true
else
    echo "Container $CONTAINER_NAME does not exist, proceeding with creation"
fi

# Create required directories
mkdir -p "$HOME/.whisper"
mkdir -p "$TMUX_TMPDIR"

# Get input group ID for device access
INPUT_GID=$(getent group input | cut -d: -f3)

# Build Docker command using array for better handling of arguments
DOCKER_ARGS=(
    "run"
    "--rm"
    "--interactive"
    "--name" "$CONTAINER_NAME"
    "--device" "/dev/input"
    "--group-add" "$INPUT_GID"
)

# Add sound device if available
if [ -d "/dev/snd" ]; then
    DOCKER_ARGS+=("--device" "/dev/snd")
fi

# Add volume mounts
DOCKER_ARGS+=(
    "--volume" "$HOME/.whisper:/.whisper"
    "--volume" "$TMUX_TMPDIR:/.tmux"
    "--volume" "`pwd`/tests:/app/tests"
)

# Check if Docker image exists
if ! docker image inspect stt-mcp-server-linux >/dev/null 2>&1; then
    echo "Error: Docker image 'stt-mcp-server-linux' not found" >&2
    echo "Please build the image first" >&2
    exit 1
fi

# Add image and command
DOCKER_ARGS+=(
    "stt-mcp-server-linux"
    "/home/nonroot/venv/bin/python"
    "/app/stt_mcp_server_linux.py"
    "--session" "$TMUX_SESSION"
)

echo "Starting container with command:"
echo "docker ${DOCKER_ARGS[*]}"

# Execute Docker command
docker "${DOCKER_ARGS[@]}"
