#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p ~/.stt-mcp-server-linux/whisper && chmod 0700 ~/.stt-mcp-server-linux/whisper
docker run --rm --tty --name stt-mcp-server-linux-download \
       --volume ~/.stt-mcp-server-linux/whisper:/.whisper \
       --volume $(pwd)/scripts:/app/scripts \
       stt-mcp-server-linux bash -ci "python scripts/download_whisper_model.py"
