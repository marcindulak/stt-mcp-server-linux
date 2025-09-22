#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p ~/.whisper && chmod 0700 ~/.whisper
docker run --rm --tty --name stt-mcp-server-linux-download \
       --volume ~/.whisper:/.whisper \
       --volume `pwd`/scripts:/app/scripts \
       stt-mcp-server-linux bash -ci "python scripts/download_whisper_model.py"
