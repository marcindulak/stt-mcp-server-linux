import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_mcp_server_linux import TmuxOutputHandler


class TestInputSanitization:
    """Test input sanitization functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.handler = TmuxOutputHandler("test_session")
    
    def test_command_injection_prevention(self):
        """Test that dangerous shell metacharacters are removed."""
        dangerous_inputs = [
            ("$(rm -rf /)", "rm -rf /"),
            ("`cat /etc/passwd`", "cat /etc/passwd"),
            ("cmd1;cmd2", "cmd1cmd2"),
            ("pipe|command", "pipecommand"),
            ("test\\escape", "testescape"),
        ]
        
        for malicious, expected in dangerous_inputs:
            result = self.handler._sanitize_text(malicious)
            assert result == expected
    
    def test_unicode_and_control_chars(self):
        """Test Unicode preservation and control character removal."""
        test_cases = [
            ("Hello 世界", "Hello 世界"),
            ("hello\x00world", "helloworld"),
            ("test\u200bstring", "teststring"),
        ]
        
        for input_text, expected in test_cases:
            result = self.handler._sanitize_text(input_text)
            assert result == expected
    
    def test_empty_input_handling(self):
        """Test edge cases with empty inputs."""
        assert self.handler._sanitize_text("") == ""
        assert self.handler._sanitize_text(None) == ""
        assert self.handler._sanitize_text("$()") == ""
    
    def test_character_limit_enforcement(self):
        """Test that text is truncated to 5000 characters."""
        # Test exactly 5000 characters
        text_5000 = "a" * 5000
        result = self.handler._sanitize_text(text_5000)
        assert len(result) == 5000
        assert result == text_5000
        
        # Test 5001 characters (should be truncated)
        text_5001 = "a" * 5001
        result = self.handler._sanitize_text(text_5001)
        assert len(result) == 5000
        assert result == "a" * 5000
