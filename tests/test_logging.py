import sys
import os
import logging
import json
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_mcp_server_linux import create_logger, JsonFormatter


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

    def test_debug_mode_off_excludes_exception(self) -> None:
        """Test that debug_mode='off' excludes exception even when present."""
        formatter = JsonFormatter(debug_mode="off")

        try:
            raise RuntimeError("Test error")
        except RuntimeError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error message",
                args=(),
                exc_info=exc_info
            )

        output = formatter.format(record)
        log_obj = json.loads(output)

        assert "exception" not in log_obj
        assert log_obj["message"] == "Error message"
        assert log_obj["level"] == "ERROR"

    def test_debug_mode_json_includes_exception_in_json(self) -> None:
        """Test that debug_mode='json' includes exception in JSON field."""
        formatter = JsonFormatter(debug_mode="json")

        try:
            raise RuntimeError("Test error")
        except RuntimeError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info
            )

        output = formatter.format(record)
        log_obj = json.loads(output)

        assert "exception" in log_obj
        assert "RuntimeError: Test error" in log_obj["exception"]
        assert "Traceback" in log_obj["exception"]

    def test_debug_mode_human_appends_traceback_to_output(self) -> None:
        """Test that debug_mode='human' appends traceback on separate lines."""
        formatter = JsonFormatter(debug_mode="human")

        try:
            raise ValueError("Human readable error")
        except ValueError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Human error",
                args=(),
                exc_info=exc_info
            )

        output = formatter.format(record)
        lines = output.split("\n", 1)

        # First line should be valid JSON
        log_obj = json.loads(lines[0])
        assert log_obj["message"] == "Human error"
        assert "exception" not in log_obj

        # Remaining lines should be the traceback
        assert len(lines) > 1
        assert "Traceback" in lines[1]
        assert "ValueError: Human readable error" in lines[1]
