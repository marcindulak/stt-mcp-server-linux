import sys
import os
import wave
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_mcp_server_linux import WhisperEngine


class TestWhisperIntegration:
    """Test Whisper transcription with actual audio data."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.whisper_engine = WhisperEngine("en")
    
    def load_real_audio(self) -> bytes:
        """Load the real hello speech WAV file.
        
        Returns raw audio bytes from the test WAV file.
        """
        # https://freesound.org/people/AderuMoro/sounds/213282/
        # The original file is 48kHz mono 16-bit, but Whisper expects 16kHz
        test_file = os.path.join(os.path.dirname(__file__), 
                                "213282__aderumoro__hello-female-friendly-professional-16kHz.wav")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test audio file not found: {test_file}")
        
        with wave.open(test_file, 'rb') as wav_file:
            # Read all audio frames
            audio_data = wav_file.readframes(wav_file.getnframes())
            
            # Get audio parameters for validation
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            
            # Convert to 16kHz mono 16-bit if needed
            if channels != 1 or sample_width != 2 or framerate != 16000:
                try:
                    import numpy as np
                    # Convert bytes to numpy array
                    if sample_width == 1:
                        audio_np = np.frombuffer(audio_data, dtype=np.uint8)
                        audio_np = (audio_np.astype(np.float32) - 128) / 128.0
                    elif sample_width == 2:
                        audio_np = np.frombuffer(audio_data, dtype=np.int16)
                        audio_np = audio_np.astype(np.float32) / 32768.0
                    elif sample_width == 4:
                        audio_np = np.frombuffer(audio_data, dtype=np.int32)
                        audio_np = audio_np.astype(np.float32) / 2147483648.0
                    else:
                        return audio_data  # Use as-is if unknown format
                    
                    # Handle stereo to mono conversion
                    if channels == 2:
                        audio_np = audio_np.reshape(-1, 2).mean(axis=1)
                    
                    # Resample if needed (simple approach)
                    if framerate != 16000:
                        # Simple resampling by taking every nth sample
                        step = framerate / 16000
                        indices = np.arange(0, len(audio_np), step).astype(int)
                        audio_np = audio_np[indices]
                    
                    # Convert back to 16-bit
                    audio_data = (audio_np * 32767).astype(np.int16).tobytes()
                except ImportError:
                    # If numpy not available, use audio as-is
                    pass
            
            return audio_data
    
    def test_whisper_transcription_with_real_speech(self):
        """Test Whisper transcription with real 'hello' speech audio.
        
        Uses actual human speech from the test WAV file to verify
        that Whisper correctly transcribes the word 'hello'.
        """
        # Load real audio data
        audio_bytes = self.load_real_audio()
        
        # Verify we have audio data
        assert len(audio_bytes) > 0
        
        # Test transcription with real speech
        result = self.whisper_engine.transcribe(audio_bytes)
        
        # Verify the method completes without error
        assert isinstance(result, str)
        
        # Verify transcription contains "hello" (case-insensitive)
        # The file is specifically a "hello" greeting
        result_lower = result.lower().strip()
        assert "hello" in result_lower, f"Expected 'hello' in transcription, got: '{result}'"
    
    def test_empty_audio_handling(self):
        """Test that empty audio is handled gracefully."""
        result = self.whisper_engine.transcribe(b"")
        assert result == ""
    
    def test_file_exists(self):
        """Test that the required audio file exists."""
        test_file = os.path.join(os.path.dirname(__file__), 
                                "213282__aderumoro__hello-female-friendly-professional-16kHz.wav")
        assert os.path.exists(test_file), f"Test audio file missing: {test_file}"
    
    def test_short_audio_handling(self):
        """Test that very short audio is handled gracefully."""
        # Create very short audio (0.1 seconds)
        short_audio = b"\x00\x00" * 1600  # 1600 samples = 0.1s at 16kHz
        result = self.whisper_engine.transcribe(short_audio)
        assert isinstance(result, str)
