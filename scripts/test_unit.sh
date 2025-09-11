#!/usr/bin/env bash

set -Eeuo pipefail

docker run --rm --tty --name stt-mcp-server-linux-tests \
       --volume ~/.whisper:/.whisper \
       --volume ./pytest.ini:/app/pytest.ini \
       --volume ./tests:/app/tests \
       stt-mcp-server-linux bash -ci "python -m pytest --verbose"
