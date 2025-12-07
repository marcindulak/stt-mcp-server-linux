import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_mcp_server_linux import AudioRecorder, Config, SpeechToTextService


class TestAudioRecording:
    """Test audio recording functionality."""
    
    def setup_method(self):
        """Setup real audio recorder instance."""
        self.recorder = AudioRecorder()
    
    def test_audio_callback(self):
        """Test audio callback with real queue."""
        test_data = b"audio_test_data"
        
        self.recorder.audio_callback(test_data, 1024, None, None)
        
        assert not self.recorder.audio_queue.empty()
        retrieved_data = self.recorder.audio_queue.get()
        assert retrieved_data == test_data
    
    def test_stop_recording_no_stream(self):
        """Test stopping recording when no stream exists."""
        result = self.recorder.stop_recording()
        assert result == b""


class TestTranscriptionWorkflow:
    """Test overall transcription workflow."""
    
    def test_key_press_starts_recording(self):
        """Test that key press starts recording."""
        config = Config(
            debug="off",
            keyboard=None,
            language="en",
            mode="standalone",
            model="whisper",
            output_type="stdout",
            pad_up_to_seconds=0.0,
            session="test"
        )

        with patch('stt_mcp_server_linux.AudioRecorder') as mock_audio:
            with patch('stt_mcp_server_linux.KeyboardMonitor') as mock_keyboard:
                with patch('stt_mcp_server_linux.WhisperEngine') as mock_whisper:
                    with patch('stt_mcp_server_linux.StdoutOutputHandler') as mock_output:
                        audio_recorder = mock_audio.return_value
                        keyboard_monitor = mock_keyboard.return_value
                        transcription_engine = mock_whisper.return_value
                        output_handler = mock_output.return_value

                        service = SpeechToTextService(
                            config, audio_recorder, keyboard_monitor,
                            transcription_engine, output_handler
                        )

                        service.on_key_press()
                        audio_recorder.start_recording.assert_called_once()

    def test_key_release_processes_audio(self):
        """Test that key release processes audio data."""
        config = Config(
            debug="off",
            keyboard=None,
            language="en",
            mode="standalone",
            model="whisper",
            output_type="stdout",
            pad_up_to_seconds=0.0,
            session="test"
        )

        with patch('stt_mcp_server_linux.AudioRecorder') as mock_audio:
            with patch('stt_mcp_server_linux.KeyboardMonitor') as mock_keyboard:
                with patch('stt_mcp_server_linux.WhisperEngine') as mock_whisper:
                    with patch('stt_mcp_server_linux.StdoutOutputHandler') as mock_output:
                        audio_recorder = mock_audio.return_value
                        keyboard_monitor = mock_keyboard.return_value
                        transcription_engine = mock_whisper.return_value
                        output_handler = mock_output.return_value
                        
                        service = SpeechToTextService(
                            config, audio_recorder, keyboard_monitor,
                            transcription_engine, output_handler
                        )
                        
                        audio_recorder.stop_recording.return_value = b"audio_data"
                        transcription_engine.transcribe.return_value = "transcribed text"
                        
                        service.on_key_release()
                        
                        audio_recorder.stop_recording.assert_called_once()
                        transcription_engine.transcribe.assert_called_once_with(b"audio_data")
                        output_handler.send_text.assert_called_once_with("transcribed text")
