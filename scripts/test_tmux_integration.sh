#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export CONTAINER_NAME="stt-mcp-server-linux-integration"
export TMUX_SESSION="stt-mcp-server-linux-integration"
export TMUX_TMPDIR="${HOME}/.tmux-integration"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

cleanup() {
    log "Cleaning up"
    TMUX_TMPDIR="$TMUX_TMPDIR" tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    kill "$CONTAINER_PID" 2>/dev/null || true
    rm -rf "$TMUX_TMPDIR" 2>/dev/null || true
}

trap cleanup EXIT

test_tmux_integration() {
    log "Starting Tmux integration test"
    
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    
    mkdir -p "$TMUX_TMPDIR" && chmod 0700 "$TMUX_TMPDIR"
    TMUX_TMPDIR="$TMUX_TMPDIR" tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_DIR" 'bash'
    
    # Start the MCP server container in interactive mode with stdin open
    log "Starting MCP server container"
    {
        sleep 3
        echo '{"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}}'
        sleep 2
        echo '{"jsonrpc": "2.0", "id": "transcribe", "method": "tools/call", "params": {"name": "transcribe"}}'
        # Keep stdin open with periodic pings
        while true; do
            sleep 30
            echo '{"jsonrpc": "2.0", "id": "ping", "method": "notifications/initialized"}'
        done
    } | bash scripts/restart_mcp_server.sh &
    
    CONTAINER_PID=$!
    
    log "Waiting for transcribe service to initialize"
    sleep 10
    
    if ! docker ps | grep -q "$CONTAINER_NAME"; then
        log "Container stopped during initialization. Final logs:"
        docker logs "$CONTAINER_NAME" 2>&1
        return 1
    fi
    
    log "Testing with audio file"
    docker exec "$CONTAINER_NAME" /home/nonroot/venv/bin/python -c "
import os
import sys

sys.path.insert(0, '/app')
from stt_mcp_server_linux import WhisperEngine, TmuxOutputHandler

wav_path = '/app/tests/213282__aderumoro__hello-female-friendly-professional-16kHz.wav'

audio_data = None
if os.path.exists(wav_path):
    print(f'Found audio file at: {wav_path}')
    try:
        import wave
        with wave.open(wav_path, 'rb') as wav:
            audio_data = wav.readframes(wav.getnframes())
        print(f'Successfully loaded wav file: {wav_path}')
    except Exception as e:
        print(f'Error loading wav file: {e}')
        sys.exit(1)
else:
    print(f'Audio file not found at: {wav_path}')
    sys.exit(1)

# Create engines and test
print('Starting transcription...')
whisper = WhisperEngine('en')
text = whisper.transcribe(audio_data)
print(f'Transcribed text: \"{text}\"')

# Use the container's TMUX_TMPDIR which should already be set to /.tmux
print(f'TMUX_TMPDIR: $TMUX_TMPDIR')
# Send to Tmux with correct socket path
print(f'Sending to Tmux session: $TMUX_SESSION')
import subprocess
tmux_handler = TmuxOutputHandler('$TMUX_SESSION')
tmux_handler.send_text(text)
print('Transcribed text sent to Tmux')
"
 
    log "Checking Tmux session for transcribed text"
    tmux_content=$(TMUX_TMPDIR="$TMUX_TMPDIR" tmux capture-pane -t "$TMUX_SESSION" -p)
    log "Captured Tmux session content $tmux_content"
    
    if echo "$tmux_content" | grep -qi "hello"; then
        log "SUCCESS: Text injection of 'hello' working correctly"
        return 0
    else
        log "FAILED: Expected text 'hello' not found in Tmux session content"
        return 1
    fi
}

log "Tmux Integration Test"
if test_tmux_integration; then
    log "Test PASSED"
    exit 0
else
    log "Test FAILED"
    exit 1
fi
