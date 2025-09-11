#!/usr/bin/env bash

set -Eeuo pipefail

export BUILDKIT_PROGRESS=plain
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --tag stt-mcp-server-linux .
