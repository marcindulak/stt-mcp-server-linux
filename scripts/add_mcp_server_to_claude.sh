#!/usr/bin/env bash

set -Eeuo pipefail

STT_MCP_SERVER_LINUX_PATH=${STT_MCP_SERVER_LINUX_PATH:-.}

claude mcp remove stt-mcp-server-linux || true
claude mcp add --scope user stt-mcp-server-linux -- bash -c \
       "mkdir -p ~/.stt-mcp-server-linux/tmux && chmod 0700 ~/.stt-mcp-server-linux/tmux && \
       mkdir -p ~/.stt-mcp-server-linux/whisper && chmod 0700 ~/.stt-mcp-server-linux/whisper && \
       bash $STT_MCP_SERVER_LINUX_PATH/scripts/restart_mcp_server.sh"
