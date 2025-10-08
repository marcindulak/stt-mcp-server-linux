#!/usr/bin/env bash

set -Eeuo pipefail

CONTAINER_NAME=${CONTAINER_NAME:-stt-mcp-server-linux}
TMUX_SESSION=${TMUX_SESSION:-claude}
TMUX_TMPDIR=${TMUX_TMPDIR:-~/.stt-mcp-server-linux/tmux}

docker stop "$CONTAINER_NAME" || true
docker rm "$CONTAINER_NAME" || true

DOCKER_CMD="docker run --rm --interactive --name $CONTAINER_NAME"
DOCKER_CMD="$DOCKER_CMD --device /dev/input"
if [ -d "/dev/snd" ]; then
    DOCKER_CMD="$DOCKER_CMD --device /dev/snd"
fi
# The /dev/input group owner ID may differ outside/inside the container.
# For the keyboard detection to work inside of the container,
# the user inside of the container must be the member
# of the /dev/input group ID present outside of the container.
INPUT_GID=$(getent group input | cut -d: -f3)
DOCKER_CMD="$DOCKER_CMD --group-add $INPUT_GID"
DOCKER_CMD="$DOCKER_CMD --volume ~/.stt-mcp-server-linux/whisper:/.whisper"
DOCKER_CMD="$DOCKER_CMD --volume $TMUX_TMPDIR:/.tmux"
DOCKER_CMD="$DOCKER_CMD --volume ./tests:/app/tests"
DOCKER_CMD="$DOCKER_CMD stt-mcp-server-linux"
DOCKER_CMD="$DOCKER_CMD /home/nonroot/venv/bin/python /app/stt_mcp_server_linux.py"
DOCKER_CMD="$DOCKER_CMD --session $TMUX_SESSION"

eval $DOCKER_CMD
