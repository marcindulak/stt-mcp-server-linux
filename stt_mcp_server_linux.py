import argparse
import asyncio
import datetime
import inspect
import json
import logging
import queue
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import evdev  # type: ignore[import-untyped]
import sounddevice  # type: ignore[import-untyped]

KeyEventCallback = Callable[[], None]


class JsonFormatter(logging.Formatter):
    """Custom JSON log formatter for structured logging.

    Args:
        debug_mode: Controls traceback output.
            - "off": no traceback included
            - "json": traceback in JSON exception field (machine-parseable)
            - "human": JSON line + traceback on separate lines (human-readable)
    """

    def __init__(self, debug_mode: str = "off") -> None:
        super().__init__()
        self.debug_mode = debug_mode

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }

        if not hasattr(record, 'method'):
            frame = inspect.currentframe()
            while frame:
                frame = frame.f_back
                if frame and frame.f_code.co_filename == __file__ and 'self' in frame.f_locals:
                    class_name = frame.f_locals['self'].__class__.__name__
                    log_obj["method"] = f"{class_name}.{frame.f_code.co_name}"
                    break

        # Include exception traceback based on debug_mode
        # Only include exception if record explicitly has exc_info set (from logger.exception() call)
        formatted_exception = None
        if self.debug_mode != "off" and record.exc_info:
            try:
                formatted_exception = self.formatException(record.exc_info)
            except Exception as fmt_err:
                formatted_exception = f"Failed to format exception: {fmt_err}"

            if self.debug_mode == "json":
                log_obj["exception"] = formatted_exception

        result = json.dumps(log_obj)

        # For human mode, append traceback on separate lines after JSON
        if self.debug_mode == "human" and formatted_exception:
            result += "\n" + formatted_exception

        return result


def create_logger(name: str = __name__, log_level: int = logging.INFO, use_stderr: bool = True, debug_mode: str = "off") -> logging.Logger:
    """Create and configure a logger with JSON formatting.

    Args:
        name: Logger name.
        log_level: Logging level.
        use_stderr: If True, log to stderr; otherwise log to stdout.
        debug_mode: Controls traceback output ("off", "json", "human").
    """
    logger = logging.getLogger(name)

    logger.handlers.clear()
    logger.setLevel(log_level)
    logger.propagate = False

    log_format = JsonFormatter(debug_mode=debug_mode)
    # MCP stdio server must not log to stdout
    # See https://modelcontextprotocol.io/docs/develop/build-server#logging-in-mcp-servers
    stream = sys.stderr if use_stderr else sys.stdout
    console = logging.StreamHandler(stream)
    console.setFormatter(log_format)
    logger.addHandler(console)

    return logger


class MCPServer:
    """MCP server implementation for speech-to-text functionality."""

    def __init__(self, speech_to_text_service: 'SpeechToTextService') -> None:
        self.speech_to_text_service = speech_to_text_service
        self.logger = logging.getLogger(__name__)
        self.background_tasks: List[asyncio.Task[Any]] = []
    
    def send_response(self, response: Dict[str, Any]) -> None:
        """Send JSON-RPC response to stdout."""
        json.dump(response, sys.stdout)
        sys.stdout.write('\n')
        sys.stdout.flush()
    
    def handle_initialize(self, request_id: Optional[str]) -> None:
        """Handle MCP initialize request."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "stt-mcp-server-linux",
                    "version": "1.0.0"
                }
            }
        }
        self.send_response(response)

    def handle_tools_list(self, request_id: Optional[str]) -> None:
        """Handle MCP tools/list request."""
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "transcribe",
                        "description": "Activate speech-to-text transcription",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                ]
            }
        }
        self.send_response(response)

    def handle_tools_call(self, request_id: Optional[str], params: Dict[str, Any]) -> None:
        """Handle MCP tools/call request."""
        tool_name = params.get("name")
        self.logger.info(f"Handling tool call: {tool_name}")

        if tool_name == "transcribe":
            self.logger.info("Scheduling speech-to-text service to run in background")
            # Remove completed tasks to prevent unbounded list growth
            self.background_tasks = [t for t in self.background_tasks if not t.done()]
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.speech_to_text_service.start_async())
            self.background_tasks.append(task)

            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": "Speech-to-text transcription activated. Press Right Ctrl to start recording."}],
                    "isError": False
                }
            }
            self.send_response(response)
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }
            self.send_response(response)
    
    def handle_request(self, request: Dict[str, Any]) -> None:
        """Handle incoming JSON-RPC request."""
        method = request.get("method")
        request_id = request.get("id")
        self.logger.info(f"Received MCP request: {method}")

        # Handle known methods
        if method == "initialize":
            self.handle_initialize(request_id)
        elif method == "notifications/initialized":
            self.logger.info("MCP client initialized")
        elif method == "notifications/cancelled":
            self.logger.info("MCP request cancelled by client")
        elif method == "tools/list":
            self.handle_tools_list(request_id)
        elif method == "tools/call":
            params = request.get("params", {})
            self.handle_tools_call(request_id, params)
        else:
            # Per JSON-RPC 2.0 spec (https://www.jsonrpc.org/specification):
            # "The Server MUST NOT reply to a Notification, including those that are within a batch request."
            # Notifications are requests without an 'id' field (id: null).
            # Only send error response for actual requests (with id), not notifications.
            if request_id is not None:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}"
                    }
                }
                self.send_response(response)
    
    async def run(self) -> None:
        """Main MCP server loop."""
        self.logger.info("Service Initialized in MCP mode")
        loop = asyncio.get_event_loop()

        try:
            while True:
                # Read stdin in executor to avoid blocking the event loop
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                try:
                    request = json.loads(line.strip())
                    self.handle_request(request)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.exception(f"Error handling request: {e}")
        finally:
            # Clean up background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass


class AudioRecorder:
    """Manages audio recording and buffering."""
    
    def __init__(self) -> None:
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.recording_active = False
        self.audio_stream: Optional[sounddevice.RawInputStream] = None
        self.logger = logging.getLogger(__name__)
    
    def audio_callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        """Callback for audio stream data."""
        self.audio_queue.put(bytes(indata))
    
    def start_recording(self) -> None:
        """Start audio recording."""
        if self.recording_active:
            self.logger.debug("Recording already active, ignoring start request")
            return
            
        self.logger.info("Starting audio recording")
        try:
            self.audio_stream = sounddevice.RawInputStream(
                samplerate=16000, blocksize=2048, dtype='int16', channels=1,
                callback=self.audio_callback
            )
            self.audio_stream.start()
            self.recording_active = True
            self.logger.info("Audio recording started successfully")
        except (OSError, sounddevice.PortAudioError) as e:
            self.logger.exception(f"Failed to start audio recording: {e}")
            self._cleanup_failed_stream()
            self.recording_active = False
            raise
    
    def stop_recording(self) -> bytes:
        """Stop audio recording and return collected audio data."""
        if not self.recording_active:
            self.logger.debug("Recording not active, nothing to stop")
            return b""
        
        if not self.audio_stream:
            self.logger.exception("Audio stream is None but recording is active - inconsistent state")
            self.recording_active = False
            return b""
        
        try:
            self.logger.info("Stopping audio recording")
            self.recording_active = False
            self.audio_stream.stop()
            self.audio_stream.close()
            
            audio_bytes = b""
            
            while not self.audio_queue.empty():
                try:
                    audio_bytes += self.audio_queue.get_nowait()
                except queue.Empty:
                    break
                
            self.logger.info(f"Audio recording stopped, collected {len(audio_bytes)} bytes")
            return audio_bytes
        
        except Exception as e:
            self.logger.exception(f"Error stopping audio recording: {e}")
            self.recording_active = False
            return b""
    
    def __enter__(self) -> 'AudioRecorder':
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
    
    def _cleanup_failed_stream(self) -> None:
        """Clean up audio stream after initialization failure."""
        if self.audio_stream:
            try:
                self.audio_stream.close()
            except Exception as e:
                self.logger.exception(f"Failed to close audio stream during error cleanup: {e}")
            finally:
                self.audio_stream = None

    def cleanup(self) -> None:
        """Clean up audio resources."""
        try:
            if self.audio_stream:
                if self.recording_active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            self.recording_active = False
            self.logger.info("Audio resources cleaned up successfully")
        except Exception as e:
            self.logger.exception(f"Error during audio cleanup: {e}")


class TranscriptionEngine:
    """Abstract base for transcription engines."""
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        raise NotImplementedError


class WhisperEngine(TranscriptionEngine):
    """Whisper-based transcription engine."""

    def __init__(self, language: str = "en", pad_up_to_seconds: float = 0.0) -> None:
        self.language = language
        self.pad_up_to_seconds = pad_up_to_seconds
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading Whisper model with language: {language}")
        import whisper  # type: ignore[import-untyped]
        self.model = whisper.load_model("tiny")
        self.logger.info("Whisper model loaded successfully")
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using Whisper model."""
        if not audio_data:
            self.logger.debug("No audio data provided for transcription")
            return ""

        self.logger.info(f"Starting Whisper transcription of {len(audio_data)} bytes")
        import numpy as np
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if self.pad_up_to_seconds > 0.0:
            sample_rate = 16000
            current_duration = len(audio_np) / sample_rate
            target_samples = int(self.pad_up_to_seconds * sample_rate)

            if len(audio_np) < target_samples:
                padding_samples = target_samples - len(audio_np)
                self.logger.info(f"Padding audio from {current_duration:.2f}s to {self.pad_up_to_seconds}s")
                audio_np = np.pad(audio_np, (0, padding_samples), mode='constant', constant_values=0.0)

        result = self.model.transcribe(audio_np, fp16=False, language=self.language)
        text = result.get("text", "").strip()
        self.logger.info(f"Whisper transcription completed: '{text}'")
        return text


class VoskEngine(TranscriptionEngine):
    """Vosk-based transcription engine."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading Vosk model")
        import vosk  # type: ignore[import-untyped]
        self.model = vosk.Model("/vosk")
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.logger.info("Vosk model loaded successfully")
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using Vosk model."""
        if not audio_data:
            self.logger.debug("No audio data provided for transcription")
            return ""
            
        self.logger.info(f"Starting Vosk transcription of {len(audio_data)} bytes")
        self.recognizer.Reset()
        
        chunk_size = 2048
        result_text = ""
        chunks_processed = 0
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if self.recognizer.AcceptWaveform(chunk):
                res = json.loads(self.recognizer.Result())
                result_text += res.get("text", "") + " "
            chunks_processed += 1
        
        self.logger.debug(f"Processed {chunks_processed} audio chunks")
        final_result = json.loads(self.recognizer.FinalResult())
        result_text += final_result.get("text", "")
        
        text = result_text.strip()
        self.logger.info(f"Vosk transcription completed: '{text}'")
        return text


class OutputHandler:
    """Abstract base for output handling."""
    
    def send_text(self, text: str) -> None:
        """Send transcribed text to output destination."""
        raise NotImplementedError


class TmuxOutputHandler(OutputHandler):
    """Tmux-based output handler."""
    
    def __init__(self, session_name: str) -> None:
        self.session_name = self._validate_session_name(session_name)
        self.logger = logging.getLogger(__name__)
    
    def _validate_session_name(self, session_name: str) -> str:
        """Validate tmux session name against injection attacks."""
        if not session_name or not session_name.strip():
            raise ValueError("Session name cannot be empty")
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_name):
            raise ValueError("Session name contains invalid characters")
        
        if len(session_name) > 64:
            raise ValueError("Session name too long")
            
        return session_name.strip()
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text to prevent command injection."""
        if not text:
            return ""
        
        # Normalize unicode to prevent normalization attacks
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters and format characters
        text = ''.join(char for char in text 
                       if unicodedata.category(char) not in ['Cc', 'Cf'])
        
        # Remove shell metacharacters (ASCII)
        dangerous_chars = ['|', '&', ';', '`', '$', '(', ')', '<', '>', '\\']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        # Restrict to printable characters only
        text = ''.join(char for char in text if char.isprintable())
        
        # Limit text length to prevent excessive input (5000 characters ≈ 5 minutes of speech)
        if len(text) > 5000:
            text = text[:5000]
        
        return text.strip()
    
    def send_text(self, text: str) -> None:
        """Send text to tmux session."""
        try:
            sanitized_text = self._sanitize_text(text)
            sanitized_session = self._sanitize_text(self.session_name)
            
            if not sanitized_text:
                self.logger.info("Empty or entirely filtered transcription text, skipping")
                return

            if sanitized_text != text:
                self.logger.info(f"Sanitized transcription text from '{text}' to '{sanitized_text}'")
            
            result = subprocess.run(
                ["tmux", "send-keys", "-t", sanitized_session, sanitized_text],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.logger.exception(f"tmux send-keys failed with code {result.returncode}: {result.stderr}")
                return
            self.logger.info(f"Sent transcription to tmux session '{self.session_name}': {sanitized_text}")
        except Exception as e:
            self.logger.exception(f"Error sending transcription to tmux session '{self.session_name}': {e}")


class StdoutOutputHandler(OutputHandler):
    """Stdout-based output handler."""
    
    def send_text(self, text: str) -> None:
        """Send text to stdout."""
        print(f"[Transcript] {text}")


class KeyboardMonitor:
    """Monitors keyboard devices for Right Ctrl key events."""
    
    def __init__(self, keyboard_name: Optional[str] = None) -> None:
        self.keyboard_name = keyboard_name
        self.logger = logging.getLogger(__name__)
    
    async def find_keyboards(self) -> List[evdev.InputDevice]:
        """Find keyboard input devices matching the configured name."""
        self.logger.info("Scanning for keyboard devices")
        keyboards = []
        available_keyboards = []
        
        for dev_path in evdev.list_devices():
            try:
                device = evdev.InputDevice(dev_path)
                if "keyboard" in device.name.lower():
                    available_keyboards.append(device.name)
                    if self.keyboard_name is None or self.keyboard_name.lower() == device.name.lower():
                        self.logger.info(f"Found matching keyboard: {device.name} ({dev_path})")
                        keyboards.append(device)
            except Exception as e:
                self.logger.debug(f"Could not access device {dev_path}: {e}")
        
        if not keyboards and self.keyboard_name:
            available_list = ", ".join(available_keyboards) if available_keyboards else "none"
            raise RuntimeError(
                f"No keyboard named '{self.keyboard_name}' found. "
                f"Available keyboards: {available_list}. "
                f"Use --keyboard option or leave empty to use all keyboards."
            )
        
        if not keyboards:
            raise RuntimeError(
                "No keyboard input devices found. "
                "Ensure you have appropriate permissions to access /dev/input devices."
            )
        
        self.logger.info(f"Found {len(keyboards)} matching keyboard devices")
        return keyboards
    
    async def monitor_device(self, dev_path: str, on_key_press: KeyEventCallback, on_key_release: KeyEventCallback) -> None:
        """Monitor a single keyboard device for Right Ctrl events."""
        dev = evdev.InputDevice(dev_path)
        self.logger.info(f"Waiting for Right Ctrl key press on {dev.name} ({dev_path})")
        
        try:
            async for event in dev.async_read_loop():
                if event.type == evdev.ecodes.EV_KEY:
                    key_event = evdev.categorize(event)
                    if key_event.keycode == 'KEY_RIGHTCTRL':  # type: ignore[attr-defined]
                        if key_event.keystate == key_event.key_down:  # type: ignore[attr-defined]
                            self.logger.info("Right Ctrl key pressed")
                            on_key_press()
                        elif key_event.keystate == key_event.key_up:  # type: ignore[attr-defined]
                            self.logger.info("Right Ctrl key released")
                            on_key_release()
        except Exception as e:
            self.logger.exception(f"Error monitoring device {dev_path}: {e}")
    
    async def start_monitoring(self, on_key_press: KeyEventCallback, on_key_release: KeyEventCallback) -> None:
        """Start monitoring all matching keyboards."""
        keyboards = await self.find_keyboards()
        if not keyboards:
            raise RuntimeError("No keyboard input devices found.")
        
        await asyncio.gather(*(
            self.monitor_device(str(dev.path), on_key_press, on_key_release)
            for dev in keyboards
        ))


class SpeechToTextService:
    """Main speech-to-text service coordinating all components."""
    
    def __init__(self, config: 'Config', 
                 audio_recorder: AudioRecorder,
                 keyboard_monitor: KeyboardMonitor,
                 transcription_engine: TranscriptionEngine,
                 output_handler: OutputHandler) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.audio_recorder = audio_recorder
        self.keyboard_monitor = keyboard_monitor
        self.transcription_engine = transcription_engine
        self.output_handler = output_handler
    
    def on_key_press(self) -> None:
        """Handle Right Ctrl key press."""
        self.logger.info("Key press detected, starting audio recording")
        self.audio_recorder.start_recording()
    
    def on_key_release(self) -> None:
        """Handle Right Ctrl key release."""
        self.logger.info("Key release detected, processing audio")
        audio_data = self.audio_recorder.stop_recording()
        if audio_data:
            text = self.transcription_engine.transcribe(audio_data)
            if text:
                self.logger.info(f"Sending transcribed text to output handler")
                self.output_handler.send_text(text)
            else:
                self.logger.info("Transcription returned empty text")
        else:
            self.logger.info("No audio data captured")
    
    async def start_async(self) -> None:
        """Start the speech-to-text service as an async coroutine (for MCP mode)."""
        self.logger.info("Starting speech-to-text functionality in async mode")

        try:
            await self.keyboard_monitor.start_monitoring(
                self.on_key_press,
                self.on_key_release
            )
        finally:
            self.audio_recorder.cleanup()

    def start(self) -> None:
        """Start the speech-to-text service (for standalone mode)."""
        self.logger.info("Starting speech-to-text functionality")

        try:
            asyncio.run(self.keyboard_monitor.start_monitoring(
                self.on_key_press,
                self.on_key_release
            ))
        finally:
            self.audio_recorder.cleanup()


@dataclass(frozen=True)
class Config:
    """Configuration container for the application."""
    debug: str
    keyboard: Optional[str]
    language: str
    mode: str
    model: str
    output_type: str
    pad_up_to_seconds: float
    session: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        return cls(
            debug=args.debug,
            keyboard=args.keyboard,
            language=args.language,
            mode=args.mode,
            model=args.model,
            output_type=args.output,
            pad_up_to_seconds=args.pad_up_to_seconds,
            session=args.session
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Speech-to-text service")

    parser.add_argument(
        "--keyboard", 
        type=str, 
        help="Name of the keyboard to listen to (optional)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for Whisper transcription (default: en). Examples: en, pl, da, pt, zh, ja"
    )
    parser.add_argument(
        "--mode",
        choices=["mcp", "standalone"],
        default="mcp",
        help="Choose operating mode (default: mcp). Use 'standalone' to bypass MCP protocol and run directly"
    )
    parser.add_argument(
        "--model",
        choices=["whisper", "vosk"],
        default="whisper",
        help="Choose speech-to-text transcription model (default: whisper)"
    )
    parser.add_argument(
        "--output",
        choices=["stdout", "tmux"],
        default="tmux",
        help="Choose output destination (default: tmux)"
    )
    parser.add_argument(
        "--pad-up-to-seconds",
        type=float,
        default=0.0,
        help="Pad short audio with silence up to specified duration in seconds (Whisper only, default: 0.0 = no padding), following https://arxiv.org/abs/2412.11272. Example: 30.0 for 30 seconds"
    )
    parser.add_argument(
        "--session",
        type=str,
        default="claude",
        help="Tmux session name (default: claude)"
    )
    parser.add_argument(
        "--debug",
        choices=["off", "json", "human"],
        default="off",
        help="Debug mode for exception tracebacks (default: off). Use 'json' for machine-parseable or 'human' for human-readable"
    )

    return parser


def main() -> None:
    """Application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    config = Config.from_args(args)

    # Configure logging for all modules before creating any components
    create_logger(debug_mode=config.debug)
    logger = logging.getLogger(__name__)

    try:
        audio_recorder = AudioRecorder()
        keyboard_monitor = KeyboardMonitor(config.keyboard)

        transcription_engine: TranscriptionEngine
        if config.model == "whisper":
            transcription_engine = WhisperEngine(config.language, config.pad_up_to_seconds)
        elif config.model == "vosk":
            if config.pad_up_to_seconds > 0.0:
                logger.info("--pad-up-to-seconds option is ignored for Vosk model")
            transcription_engine = VoskEngine()
        else:
            raise ValueError(f"Unknown transcription model: {config.model}")

        output_handler: OutputHandler
        if config.output_type == "tmux":
            output_handler = TmuxOutputHandler(config.session)
        else:
            output_handler = StdoutOutputHandler()

        speech_to_text_service = SpeechToTextService(
            config, audio_recorder, keyboard_monitor, transcription_engine, output_handler
        )

        if config.mode == "standalone":
            logger.info("Service Initialized in standalone mode")
            speech_to_text_service.start()
        elif config.mode == "mcp":
            mcp_server = MCPServer(speech_to_text_service)
            asyncio.run(mcp_server.run())
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
    except Exception as e:
        logger.exception(f"Application error: {e}")


if __name__ == "__main__":
    main()
