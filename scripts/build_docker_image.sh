#!/usr/bin/env bash

set -Eeuo pipefail

export BUILDKIT_PROGRESS=plain
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg INPUT_GID=$(getent group input | cut -d: -f3) --tag stt-mcp-server-linux .
