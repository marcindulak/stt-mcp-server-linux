#!/usr/bin/env bash

set -Eeuo pipefail

bash scripts/build_docker_image.sh

docker run --rm --tty --name stt-mcp-server-linux-mypy \
       --volume $(pwd)/stt_mcp_server_linux.py:/app/stt_mcp_server_linux.py \
       --volume $(pwd)/tests:/app/tests \
       stt-mcp-server-linux bash -ci "python -m mypy ."
