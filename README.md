[![test](https://github.com/marcindulak/stt-mcp-server-linux/actions/workflows/test.yml/badge.svg)](https://github.com/marcindulak/stt-mcp-server-linux/actions/workflows/test.yml)

[![Mentioned in Awesome Claude Code](https://awesome.re/mentioned-badge.svg)](https://github.com/hesreallyhim/awesome-claude-code)

> Co-Authored-By: Claude

# Functionality overview

Local speech-to-text MCP server for Linux.
The speech-to-text functionality can also be used in a standalone mode in Tmux, without relying on MCP.

Claude Code is required to run inside a Tmux session to enable the transcribed text injection into Claude's input stream.

The MCP server runs in a Docker container with access to host input and audio devices.
The server provides a `transcribe` tool accessible through MCP protocol.
When the tool is activated, the server monitors the `Right Ctrl` key for push-to-talk functionality.
Key press detection uses `/dev/input` keyboard devices.
Audio recording uses `/dev/snd` microphone device.

On `Right Ctrl` key release, speech-to-text transcription occurs (using Whisper tiny model by default).
The transcribed text is injected into Claude's input stream via Tmux send-keys.

The MCP server is Linux-only due to `/dev` device dependencies.

> [!WARNING]
> This project will create `~/.stt-mcp-server-linux` directory.

# Usage examples

The instructions follow below.

1. Install [Docker Engine](https://docs.docker.com/engine/install/) or [Docker Desktop](https://docs.docker.com/desktop/)

2. Install [Tmux](https://github.com/tmux/tmux).
   If you are unfamiliar with, Tmux watch this [YouTube tutorial](https://www.youtube.com/watch?v=UxbiDtEXuxg&list=PLT98CRl2KxKGiyV1u6wHDV8VwcQdzfuKe) and checkout out this [cheat sheet](https://tmuxcheatsheet.com) for a shortcuts reference.

3. Clone this repository, and `cd` into it:

   ```
   git clone https://github.com/marcindulak/stt-mcp-server-linux
   cd stt-mcp-server-linux
   export STT_MCP_SERVER_LINUX_PATH=$(pwd)
   ```

4. Build the Docker image of the MCP server:

   ```
   bash scripts/build_docker_image.sh
   ```

5. Download the Whisper tiny model under `~/.stt-mcp-server-linux/whisper`:

   ```
   bash scripts/download_whisper_model.sh
   ```

6. Configure Tmux, so `~/.tmux.conf` contains at least:

   ```
   # Enable mouse support for scrolling
   set -g mouse on

   # Set large scrollback lines buffer
   set -g history-limit 1000000

   # Hide status bar to reduce flicker
   set -g status off

   # Reduce escape key delay to reduce flicker
   set -g escape-time 0
   ```

## MCP mode

In this mode, the local MCP server running in the Docker container exposes the `transcribe` tool.
The tool is activated from Claude Code, running in a Tmux session.
The tool sends the transcribed text into the Tmux session's input buffer.

7. Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code/setup)

8. Add the MCP server to Claude (MCP client).

   Navigate to any of your Claude directories.

   ```
   bash "${STT_MCP_SERVER_LINUX_PATH}/scripts/add_mcp_server_to_claude.sh"
   ```

   Verify the Claude connection to the MCP server with:

   ```
   claude mcp list --debug
   ```

   Expected output:

   ```
   stt-mcp-server-linux: ... ✓ Connected
   ```

> [!NOTE]
> The addition the MCP server needs to be performed only once, because the server is added with the `--scope user`.
> The first time setup is now complete!

9. Navigate to any of your Claude directories, start Claude in a new Tmux session stored under `~/.stt-mcp-server-linux/tmux`.
   The reason for using a custom `TMUX_TMPDIR` location instead of the default `/tmp/tmux-$(id -u)` is to make it shareable between the Docker host and the container with correct file ownership.

   ```
   TMUX_TMPDIR=~/.stt-mcp-server-linux/tmux tmux new-session -s claude 'claude'
   ```

   and ask to `Run the transcribe tool provided by the stt-mcp-server-linux MCP server`.

   Press the `Right Ctrl` key to activate `Push-to-Talk` functionality.
   Release the key to perform the transcription and inject the resulting text into Claude.

> [!NOTE]
> Give the MCP server some time to initialize.
> You may need to explicitly verify its status with the `/mcp` command.
>
> Use `docker logs stt-mcp-server-linux` to check the progress.
>
> Once a `{... "message": "Waiting for Right Ctrl key press on ... keyboard ..."}` log line appears the transcription feature should be available.
>
> The `transcribe` tool returns immediately and runs in the background.
> Claude Code's terminal remains responsive for your input and commands while keyboard monitoring continues.
>
> To stop the transcription service, use `/quit` or close Claude Code.

## Standalone mode (without MCP)

The speech-to-text transcription can be performed without the MCP protocol.

7. Start a new Tmux session. The example here starts a session for bash:

   ```
   TMUX_TMPDIR=~/.stt-mcp-server-linux/tmux tmux new-session -s bash 'bash'
   ```

8. Start the transcription service in standalone mode:

   ```
   DEBUG=human MODE=standalone OUTPUT=tmux TMUX_SESSION=bash bash scripts/restart_mcp_server.sh
   ```

   Press the `Right Ctrl` key to activate `Push-to-Talk` functionality.

   The transcribed text will be inserted into the Tmux session's input buffer.

> [!WARNING]
> The current limitation is that only one transcription container can be running at a given time.
> If you start a new container, it will stop and replace the previous one.

# Running tests

Tests run inside Docker containers to have access to required dependencies.

## Unit tests

```
bash scripts/test_unit.sh
```

## Integration test

End-to-end integration test verifies the functionality of injecting text into the Tmux input:

```
bash scripts/test_tmux_integration.sh
```

## Type checking

Run mypy static type checking:

```
bash scripts/test_mypy.sh
```

# Implementation overview

The system uses object composition with separated responsibilities across multiple classes:

1. **MCPServer**: Handles JSON-RPC protocol communication with Claude using async/await. Manages an event loop that keeps the server responsive while background tasks execute. Routes MCP requests and schedules the speech-to-text service as background asyncio tasks.

2. **AudioRecorder**: Manages audio stream capture and buffering. Provides start/stop interface for recording sessions.

3. **TranscriptionEngine**: Abstract base with concrete implementations (WhisperEngine, VoskEngine) for different transcription models.

4. **OutputHandler**: Abstract base with concrete implementations (TmuxOutputHandler, StdoutOutputHandler) for different output destinations.

5. **KeyboardMonitor**: Handles keyboard device detection and Right Ctrl key event monitoring using evdev. Runs as an async coroutine allowing non-blocking keyboard monitoring.

6. **SpeechToTextService**: Main coordinator that orchestrates all components. Provides `start_async()` for MCP mode (background execution) and `start()` for standalone mode (blocking execution).

# Abandoned ideas

## ydotool for Wayland text injection
Considered using ydotool for keyboard access on Wayland systems. Abandoned because:
- Requires root privileges for /dev/uinput access
- Python wrappers are unmaintained (pydotool, pyydotool, TotoBotKey)
- Not packaged for Debian

## xdotool for X11 text injection
Attempted using xdotool for keyboard simulation on X11. Abandoned because:
- Most modern Linux systems run on Wayland, not X11 (check you system with `echo $XDG_SESSION_TYPE`)

## Direct MCP text injection
Investigated injecting text directly through MCP protocol. Abandoned because:
- It seems that MCP tools can only return content to Claude (as output), not inject into input stream
