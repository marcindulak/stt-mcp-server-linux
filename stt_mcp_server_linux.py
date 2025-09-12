import argparse
import json
import logging
import sys
import asyncio
import subprocess
from typing import Any, Dict, List, Optional, Callable
import datetime
import evdev
import queue
import sounddevice  # type: ignore[import-untyped]
import inspect
import unicodedata


class JsonFormatter(logging.Formatter):
    """Custom JSON log formatter for structured logging."""
    
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
            
        return json.dumps(log_obj)


def create_logger(name: str, log_level: int = logging.INFO, use_stderr: bool = False) -> logging.Logger:
    """Create and configure a logger with JSON formatting."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(log_level)
    logger.propagate = False

    log_format = JsonFormatter()
    stream = sys.stderr if use_stderr else sys.stdout
    console = logging.StreamHandler(stream)
    console.setFormatter(log_format)
    logger.addHandler(console)

    return logger


class MCPServer:
    """MCP server implementation for speech-to-text functionality."""
    
    def __init__(self, speech_to_text_service: 'SpeechToTextService') -> None:
        self.speech_to_text_service = speech_to_text_service
        self.logger = create_logger(__name__)
    
    def send_response(self, response: Dict[str, Any], method_name: str = "") -> None:
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
        self.send_response(response, "handle_initialize")
    
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
        self.send_response(response, "handle_tools_list")
    
    def handle_tools_call(self, request_id: Optional[str], params: Dict[str, Any]) -> None:
        """Handle MCP tools/call request."""
        tool_name = params.get("name")
        self.logger.info(f"Handling tool call: {tool_name}")
        
        if tool_name == "transcribe":
            self.logger.info("Starting speech-to-text service")
            self.speech_to_text_service.start()
            
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": "Speech-to-text transcription activated. Press Right Ctrl to start recording."}],
                    "isError": False
                }
            }
            self.send_response(response, "handle_tools_call")
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
                }
            }
            self.send_response(response, "handle_tools_call")
    
    def handle_request(self, request: Dict[str, Any]) -> None:
        """Handle incoming JSON-RPC request."""
        method = request.get("method")
        request_id = request.get("id")
        self.logger.info(f"Received MCP request: {method}")
        
        if method == "initialize":
            self.handle_initialize(request_id)
        elif method == "notifications/initialized":
            self.logger.info("MCP client initialized")
        elif method == "tools/list":
            self.handle_tools_list(request_id)
        elif method == "tools/call":
            params = request.get("params", {})
            self.handle_tools_call(request_id, params)
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            }
            self.send_response(response, "handle_request")
    
    def run(self) -> None:
        """Main MCP server loop."""
        self.logger.info("Service Initialized in MCP mode")
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                self.handle_request(request)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error: {e}")


class AudioRecorder:
    """Manages audio recording and buffering."""
    
    def __init__(self) -> None:
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.recording_active = False
        self.audio_stream: Optional[sounddevice.RawInputStream] = None
        self.logger = create_logger(__name__)
    
    def audio_callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        """Callback for audio stream data."""
        self.audio_queue.put(bytes(indata))
    
    def start_recording(self) -> None:
        """Start audio recording."""
        if self.recording_active:
            self.logger.debug("Recording already active, ignoring start request")
            return
            
        self.logger.info("Starting audio recording")
        self.recording_active = True
        self.audio_stream = sounddevice.RawInputStream(
            samplerate=16000, blocksize=2048, dtype='int16', channels=1,
            callback=self.audio_callback
        )
        self.audio_stream.start()
        self.logger.info("Audio recording started successfully")
    
    def stop_recording(self) -> bytes:
        """Stop audio recording and return collected audio data."""
        if not self.recording_active or not self.audio_stream:
            self.logger.debug("Recording not active, nothing to stop")
            return b""
            
        self.logger.info("Stopping audio recording")
        self.recording_active = False
        self.audio_stream.stop()
        self.audio_stream.close()
        
        audio_bytes = b""
        while not self.audio_queue.empty():
            audio_bytes += self.audio_queue.get()
        
        self.logger.info(f"Audio recording stopped, collected {len(audio_bytes)} bytes")
        return audio_bytes
    
    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()


class TranscriptionEngine:
    """Abstract base for transcription engines."""
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        raise NotImplementedError


class WhisperEngine(TranscriptionEngine):
    """Whisper-based transcription engine."""
    
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.logger = create_logger(__name__)
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
        result = self.model.transcribe(audio_np, fp16=False, language=self.language)
        text = result.get("text", "").strip()
        self.logger.info(f"Whisper transcription completed: '{text}'")
        return text


class VoskEngine(TranscriptionEngine):
    """Vosk-based transcription engine."""
    
    def __init__(self) -> None:
        self.logger = create_logger(__name__)
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
        self.session_name = session_name
        self.logger = create_logger(__name__)
    
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
                self.logger.warning("Empty or entirely filtered transcription text, skipping")
                return
                
            if sanitized_text != text:
                self.logger.warning(f"Sanitized transcription text from '{text}' to '{sanitized_text}'")
            
            subprocess.run(["tmux", "send-keys", "-t", sanitized_session, sanitized_text, "Enter"])
            self.logger.info(f"Sent transcription to tmux session '{self.session_name}': {sanitized_text}")
        except Exception as e:
            self.logger.error(f"Error sending transcription to tmux session '{self.session_name}': {e}")


class StdoutOutputHandler(OutputHandler):
    """Stdout-based output handler."""
    
    def send_text(self, text: str) -> None:
        """Send text to stdout."""
        print(f"[Transcript] {text}")


class KeyboardMonitor:
    """Monitors keyboard devices for Right Ctrl key events."""
    
    def __init__(self, keyboard_name: Optional[str] = None) -> None:
        self.keyboard_name = keyboard_name
        self.logger = create_logger(__name__)
    
    async def find_keyboards(self) -> List[evdev.InputDevice]:
        """Find keyboard input devices matching the configured name."""
        self.logger.info("Scanning for keyboard devices")
        keyboards = []
        for dev_path in evdev.list_devices():
            try:
                device = evdev.InputDevice(dev_path)
                if "keyboard" in device.name.lower():
                    if self.keyboard_name is None or self.keyboard_name.lower() == device.name.lower():
                        self.logger.info(f"Found matching keyboard: {device.name} ({dev_path})")
                        keyboards.append(device)
            except Exception:
                pass
        self.logger.info(f"Found {len(keyboards)} matching keyboard devices")
        return keyboards
    
    async def monitor_device(self, dev_path: str, on_key_press: Callable[[], None], on_key_release: Callable[[], None]) -> None:
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
            self.logger.error(f"Error monitoring device {dev_path}: {e}")
    
    async def start_monitoring(self, on_key_press: Callable[[], None], on_key_release: Callable[[], None]) -> None:
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
    
    def __init__(self, config: 'Config') -> None:
        self.config = config
        self.logger = create_logger(__name__)
        
        self.audio_recorder = AudioRecorder()
        self.keyboard_monitor = KeyboardMonitor(config.keyboard_name)
        
        if config.use_whisper:
            self.transcription_engine: TranscriptionEngine = WhisperEngine(config.language)
        else:
            self.transcription_engine = VoskEngine()
        
        if config.output_type == "tmux":
            self.output_handler: OutputHandler = TmuxOutputHandler(config.session_name)
        else:
            self.output_handler = StdoutOutputHandler()
    
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
                self.logger.warning("Transcription returned empty text")
        else:
            self.logger.warning("No audio data captured")
    
    def start(self) -> None:
        """Start the speech-to-text service."""
        self.logger.info("Starting speech-to-text functionality")
        
        try:
            asyncio.run(self.keyboard_monitor.start_monitoring(
                self.on_key_press, 
                self.on_key_release
            ))
        finally:
            self.audio_recorder.cleanup()


class Config:
    """Configuration container for the application."""
    
    def __init__(self, args: argparse.Namespace) -> None:
        self.keyboard_name: Optional[str] = args.keyboard
        self.language: str = args.language
        self.use_whisper: bool = args.model == "whisper"
        self.output_type: str = args.output
        self.session_name: str = args.session


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
        "--session", 
        type=str, 
        default="claude",
        help="Tmux session name (default: claude)"
    )

    return parser


def main() -> None:
    """Application entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    config = Config(args)
    logger = create_logger(__name__)
    
    speech_to_text_service = SpeechToTextService(config)
    
    if sys.stdin.isatty():
        logger.info("Service Initialized in standalone mode")
        speech_to_text_service.start()
    else:
        logger.info("Service Initialized in MCP mode")
        mcp_server = MCPServer(speech_to_text_service)
        mcp_server.run()


if __name__ == "__main__":
    main()
