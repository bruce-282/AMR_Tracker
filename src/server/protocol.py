"""Protocol handler for TCP/IP communication."""

import json
from typing import Dict, Optional, Any, Union, List
from enum import IntEnum


class Command(IntEnum):
    """Command codes for TCP/IP protocol."""
    START_VISION = 1
    END_VISION = 2
    START_CAM_1 = 3
    START_CAM_2 = 4
    START_CAM_3 = 5
    CALC_RESULT = 6
    NOTIFY_CONNECTION = 7


class ProtocolHandler:
    """Handles protocol parsing and response generation."""
    
    @staticmethod
    def parse_request(data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse incoming request.
        
        Args:
            data: Raw bytes from client
            
        Returns:
            Parsed request dictionary or None if invalid
        """
        try:
            text = data.decode('utf-8').strip()
            if not text:
                return None
            return json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"⚠ Failed to parse request: {e}")
            return None
    
    @staticmethod
    def create_response(
        cmd: int,
        success: bool = True,
        error_code: Optional[str] = None,
        error_desc: Optional[str] = None,
        data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> bytes:
        """
        Create response message.
        
        Args:
            cmd: Command code
            success: Success status
            error_code: Error code if failed
            error_desc: Error description if failed
            data: Response data
            
        Returns:
            JSON-encoded bytes
        """
        response = {
            "cmd": cmd,
            "success": success
        }
        
        if error_code:
            response["error_code"] = error_code
        if error_desc:
            response["error_desc"] = error_desc
        if data:
            response["data"] = data
        
        return json.dumps(response).encode('utf-8')
    
    @staticmethod
    def create_notification(
        camera_id: int,
        is_connected: bool,
        error_code: Optional[str] = None,
        error_desc: Optional[str] = None
    ) -> bytes:
        """
        Create connection notification message.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            is_connected: Connection status
            error_code: Error code if disconnected
            error_desc: Error description if disconnected
            
        Returns:
            JSON-encoded bytes
        """
        response = {
            "cmd": Command.NOTIFY_CONNECTION,
            "camera_id": camera_id,
            "is_connected": is_connected
        }
        
        if error_code:
            response["error_code"] = error_code
        if error_desc:
            response["error_desc"] = error_desc
        
        return json.dumps(response).encode('utf-8')

