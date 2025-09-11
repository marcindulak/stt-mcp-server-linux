import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_mcp_server_linux import AudioRecorder, SpeechToTextService


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
        class MockConfig:
            keyboard_name = None
            use_whisper = True
            output_type = "stdout"
            session_name = "test"
            language = "en"
        
        with patch('stt_mcp_server_linux.AudioRecorder') as mock_audio:
            with patch('stt_mcp_server_linux.KeyboardMonitor'):
                with patch('stt_mcp_server_linux.WhisperEngine'):
                    with patch('stt_mcp_server_linux.StdoutOutputHandler'):
                        service = SpeechToTextService(MockConfig())
                        mock_audio_instance = mock_audio.return_value
                        
                        service.on_key_press()
                        mock_audio_instance.start_recording.assert_called_once()
    
    def test_key_release_processes_audio(self):
        """Test that key release processes audio data."""
        class MockConfig:
            keyboard_name = None
            use_whisper = True
            output_type = "stdout"
            session_name = "test"
            language = "en"
        
        with patch('stt_mcp_server_linux.AudioRecorder') as mock_audio:
            with patch('stt_mcp_server_linux.KeyboardMonitor'):
                with patch('stt_mcp_server_linux.WhisperEngine') as mock_whisper:
                    with patch('stt_mcp_server_linux.StdoutOutputHandler') as mock_output:
                        service = SpeechToTextService(MockConfig())
                        
                        mock_audio.return_value.stop_recording.return_value = b"audio_data"
                        mock_whisper.return_value.transcribe.return_value = "transcribed text"
                        
                        service.on_key_release()
                        
                        mock_audio.return_value.stop_recording.assert_called_once()
                        mock_whisper.return_value.transcribe.assert_called_once_with(b"audio_data")
                        mock_output.return_value.send_text.assert_called_once_with("transcribed text")
