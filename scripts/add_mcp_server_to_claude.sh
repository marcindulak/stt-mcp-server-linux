#!/usr/bin/env bash

set -Eeuo pipefail

STT_MCP_SERVER_LINUX_PATH=${STT_MCP_SERVER_LINUX_PATH:-.}

claude mcp remove stt-mcp-server-linux || true
claude mcp add stt-mcp-server-linux -- bash -c \
       "mkdir -p ~/.tmux && chmod 0700 ~/.tmux && \
       mkdir -p ~/.whisper && chmod 0700 ~/.whisper && \
       bash $STT_MCP_SERVER_LINUX_PATH/scripts/restart_mcp_server.sh"
