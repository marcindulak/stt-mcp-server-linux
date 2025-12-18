import asyncio
import json
import sys
import os
from io import StringIO
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_mcp_server_linux import MCPServer


class TestMCPProtocol:
    """Test MCP protocol communication."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_service = Mock()
        self.mock_service.start_async = AsyncMock()
        self.server = MCPServer(self.mock_service)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_initialize_request(self, mock_stdout):
        """Test MCP initialize request handling."""
        request = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "initialize"
        }
        
        self.server.handle_request(request)
        
        output = mock_stdout.getvalue().strip()
        response = json.loads(output)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id"
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "stt-mcp-server-linux"
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_transcribe_tool_call(self, mock_stdout):
        """Test transcribe tool call handling.

        Runs within an event loop because handle_tools_call uses
        asyncio.get_running_loop() to schedule background tasks.
        """
        request = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "tools/call",
            "params": {
                "name": "transcribe"
            }
        }

        async def run_in_loop():
            self.server.handle_request(request)

        asyncio.run(run_in_loop())

        self.mock_service.start_async.assert_called_once()

        output = mock_stdout.getvalue().strip()
        response = json.loads(output)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id"
        assert "result" in response
        assert response["result"]["isError"] is False
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_unknown_tool_call(self, mock_stdout):
        """Test handling of unknown tool calls."""
        request = {
            "jsonrpc": "2.0",
            "id": "test-id",
            "method": "tools/call", 
            "params": {
                "name": "unknown_tool"
            }
        }
        
        self.server.handle_request(request)
        
        output = mock_stdout.getvalue().strip()
        response = json.loads(output)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id"
        assert "error" in response
        assert response["error"]["code"] == -32601
