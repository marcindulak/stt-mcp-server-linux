import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_mcp_server_linux import create_logger


class TestLogging:
    """Test logging configuration behavior."""
    
    def test_create_logger_defaults_to_stderr(self):
        """Test that create_logger uses stderr by default."""
        with patch('stt_mcp_server_linux.logging.StreamHandler') as mock_handler_class:
            mock_handler = mock_handler_class.return_value
            
            create_logger("test_logger")
            
            mock_handler_class.assert_called_once_with(sys.stderr)
    
    def test_create_logger_uses_stdout_when_explicitly_set(self):
        """Test that create_logger can use stdout when use_stderr=False."""
        with patch('stt_mcp_server_linux.logging.StreamHandler') as mock_handler_class:
            mock_handler = mock_handler_class.return_value
            
            create_logger("test_logger", use_stderr=False)
            
            mock_handler_class.assert_called_once_with(sys.stdout)
    
    def test_create_logger_uses_stderr_when_explicitly_set(self):
        """Test that create_logger uses stderr when use_stderr=True."""
        with patch('stt_mcp_server_linux.logging.StreamHandler') as mock_handler_class:
            mock_handler = mock_handler_class.return_value
            
            create_logger("test_logger", use_stderr=True)
            
            mock_handler_class.assert_called_once_with(sys.stderr)