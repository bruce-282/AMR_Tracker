"""Test client for Vision Tracking TCP/IP Server."""

import socket
import json
import time
import logging
from pathlib import Path


class VisionClient:
    """Test client for vision server."""
    
    def __init__(self, host="127.0.0.1", port=10000, model_config_path="model_config.json"):
        # Setup logger with timestamp
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.host = host
        self.port = port
        self.socket = None
        self.model_config_path = Path(model_config_path)
        self.selected_model = None
        self._load_model_config()
    
    def _load_model_config(self):
        """Load model configuration from model_config.json."""
        if self.model_config_path.exists():
            try:
                with open(self.model_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.selected_model = config.get("selected_model")
                    if self.selected_model:
                        self.logger.info(f"Loaded selected_model from {self.model_config_path}: {self.selected_model}")
                    else:
                        self.logger.warning(f"No selected_model in {self.model_config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load model config from {self.model_config_path}: {e}")
        else:
            self.logger.warning(f"Model config file not found: {self.model_config_path}")
    
    def connect(self):
        """Connect to server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.logger.info(f"Connected to {self.host}:{self.port}")
    
    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            self.socket.close()
            self.logger.info("Disconnected")
    
    def send_request(self, request: dict) -> dict:
        """Send request and receive response."""
        # Log request
        self.logger.info(f"Request: {json.dumps(request, indent=2)}")
        
        # Send request
        request_json = json.dumps(request)
        self.socket.sendall(request_json.encode('utf-8'))
        
        # Get expected command code from request
        expected_cmd = request.get("cmd")
        
        # Receive response - read until we get a response with matching cmd
        buffer = b""
        max_attempts = 10  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            chunk = self.socket.recv(4096)
            if not chunk:
                break
            buffer += chunk
            
            # Try to parse JSON
            try:
                text = buffer.decode('utf-8')
                # Find all complete JSON objects
                brace_count = 0
                json_start = -1
                json_end = -1
                
                for i, char in enumerate(text):
                    if char == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and json_start >= 0:
                            json_end = i + 1
                            # Parse this JSON object
                            json_str = text[json_start:json_end]
                            response = json.loads(json_str)
                            
                            # Handle NOTIFY_CONNECTION (cmd: 7) separately
                            if response.get("cmd") == 7:
                                # Log NOTIFY_CONNECTION message as-is
                                self.logger.info(f"[NOTIFY_CONNECTION] {json.dumps(response, indent=2)}")
                                
                                # Remove this JSON from buffer and continue looking for expected response
                                buffer = text[json_end:].encode('utf-8')
                                json_start = -1
                                json_end = -1
                                continue
                            
                            # Check if this is the response we're waiting for
                            if expected_cmd is None or response.get("cmd") == expected_cmd:
                                # Remove this JSON from buffer
                                remaining = text[json_end:].encode('utf-8')
                                # Log response
                                self.logger.info(f"Response: {json.dumps(response, indent=2)}")
                                return response
                            
                            # Not the response we want, continue looking
                            json_start = -1
                            json_end = -1
                
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Not enough data yet or incomplete JSON, continue reading
                attempts += 1
                continue
            
            attempts += 1
        
        # If we get here, try to parse whatever we have
        if buffer:
            try:
                response = json.loads(buffer.decode('utf-8'))
                return response
            except json.JSONDecodeError as e:
                raise ConnectionError(f"Failed to parse response: {e}, buffer: {buffer[:200]}")
        else:
            raise ConnectionError("No response received from server")
    
    def test_start_vision(self, model=None, use_area_scan=False):
        """Test START VISION command.
        
        Args:
            model: Model name (optional, uses selected_model from model_config.json if not provided)
            use_area_scan: Whether to use area scan mode
        """
        self.logger.info("\n[TEST] START VISION")
        
        # Use selected_model from config if model not provided
        if model is None:
            if self.selected_model:
                model = self.selected_model
                self.logger.info(f"Using selected_model from config: {model}")
            else:
                raise ValueError("No model provided and no selected_model in model_config.json")
        
        request = {
            "cmd": 1,
            "model": model,
            "use_area_scan": use_area_scan
        }
        response = self.send_request(request)
        return response
    
    def test_end_vision(self):
        """Test END VISION command."""
        self.logger.info("\n[TEST] END VISION")
        request = {"cmd": 2}
        response = self.send_request(request)
        return response
    
    def test_start_cam(self, camera_id: int):
        """Test START CAM command.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
        """
        self.logger.info(f"\n[TEST] START CAM {camera_id}")
        cmd_map = {1: 3, 2: 4, 3: 5}  # cmd 3, 4, 5 for cam 1, 2, 3
        request = {"cmd": cmd_map[camera_id]}
        
        response = self.send_request(request)
        return response
    
    def test_calc_result(self, path_csv=""):
        """Test CALC RESULT command."""
        self.logger.info("\n[TEST] CALC RESULT")
        request = {
            "cmd": 6,
            "path_csv": path_csv
        }
        response = self.send_request(request)
        return response


def main():
    """Run test client."""
    # Setup logger
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info("=" * 60)
    logger.info("Vision Tracking TCP/IP Server - Test Client")
    logger.info("=" * 60)
    
    client = VisionClient()
    
    try:
        # Connect to server
        client.connect()
        
        # Test 1: START VISION (uses selected_model from model_config.json)
        use_area_scan = False  # Set use_area_scan mode
        for i in range(2):
            response = client.test_start_vision(use_area_scan=use_area_scan)
            if not response.get("success"):
                logger.error(f"START VISION failed: {response.get('error_desc')}")
                return
            
            time.sleep(1)  # Wait for initialization
            
            if use_area_scan:
                # use_area_scan=true: client sends requests
                # Test 2: START CAM 1
                response = client.test_start_cam(1)
                if response.get("success"):
                    data = response.get("data", {})
                    logger.info(f"  [OK] Position: x={data.get('x', 0):.2f}mm, y={data.get('y', 0):.2f}mm, rz={data.get('rz', 0):.4f}rad")
                    if "result_image" in data:
                        logger.info(f"  [OK] Result image: {data['result_image']}")
                else:
                    logger.error(f"  [FAIL] Failed: {response.get('error_desc', 'Unknown error')}")
                
                time.sleep(3)  # Wait for tracking to process frames
                
                # Test 3: Get updated position
                logger.info("\nGetting updated position...")
                response = client.test_start_cam(1)
                if response.get("success"):
                    data = response.get("data", {})
                    logger.info(f"  [OK] Updated Position: x={data.get('x', 0):.2f}mm, y={data.get('y', 0):.2f}mm, rz={data.get('rz', 0):.4f}rad")
                else:
                    logger.error(f"  [FAIL] Failed: {response.get('error_desc', 'Unknown error')}")
            else:
                # use_area_scan=false: client does NOT send requests, only waits for periodic responses
                logger.info("\n[INFO] use_area_scan=false: Waiting for responses from server...")
                logger.info("  [INFO] Server will send responses automatically (1→2→3→1→2→3...)")
                logger.info("  [INFO] Waiting for 2 cycles, then sending END_VISION")
                
                # Wait for 2 cycles: 1→2→3→1→2→3
                cycles_to_complete = 2
                cycle_count = 0
                response_count = {3: 0, 4: 0, 5: 0}  # Track responses for cmd 3, 4, 5
                
                client.socket.settimeout(None)  # No timeout
                buffer = b""
                
                try:
                    while cycle_count < cycles_to_complete:
                        chunk = client.socket.recv(4096)
                        if not chunk:
                            break
                        buffer += chunk
                        
                        # Try to parse JSON
                        try:
                            text = buffer.decode('utf-8')
                            # Find complete JSON object
                            brace_count = 0
                            json_start = -1
                            json_end = -1
                            
                            for i, char in enumerate(text):
                                if char == '{':
                                    if brace_count == 0:
                                        json_start = i
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0 and json_start >= 0:
                                        json_end = i + 1
                                        json_str = text[json_start:json_end]
                                        response = json.loads(json_str)
                                        
                                        cmd = response.get("cmd")
                                        
                                        # Handle NOTIFY_CONNECTION (cmd: 7)
                                        if cmd == 7:
                                            # Log NOTIFY_CONNECTION message as-is
                                            logger.info(f"[NOTIFY_CONNECTION] {json.dumps(response, indent=2)}")
                                            
                                            # Remove processed JSON from buffer
                                            buffer = text[json_end:].encode('utf-8')
                                            json_start = -1
                                            json_end = -1
                                            continue
                                        
                                        # Handle camera responses (cmd: 3, 4, 5)
                                        if cmd in [3, 4, 5] and response.get("success"):
                                            # Log response in same format as send_request
                                            logger.info(f"Response: {json.dumps(response, indent=2)}")
                                            
                                            response_count[cmd] += 1
                                            
                                            # if cmd == 3:  # Camera 1
                                            #     data = response.get("data", {})
                                            #     logger.info(f"  [OK] Camera 1 Response #{response_count[cmd]}: x={data.get('x', 0):.2f}mm, y={data.get('y', 0):.2f}mm, rz={data.get('rz', 0):.4f}rad")
                                            #     if "result_image" in data:
                                            #         logger.info(f"  [OK] Result image: {data['result_image']}")
                                            
                                            # elif cmd == 4:  # Camera 2
                                            #     # Log entire response as-is (same as NOTIFY_CONNECTION)
                                            #     logger.info(f"  [OK] Camera 2 Response #{response_count[cmd]}: {json.dumps(response, indent=2)}")
                                            
                                            # elif cmd == 5:  # Camera 3
                                            #     data = response.get("data", {})
                                            #     logger.info(f"  [OK] Camera 3 Response #{response_count[cmd]}: x={data.get('x', 0):.2f}mm, y={data.get('y', 0):.2f}mm, rz={data.get('rz', 0):.4f}rad")
                                            #     if "result_image" in data:
                                            #         logger.info(f"  [OK] Result image: {data['result_image']}")
                                            
                                            # Check if we completed a cycle (received 1, 2, 3 in sequence)
                                            # A cycle is complete when we have equal counts for all cameras
                                            min_count = min(response_count[3], response_count[4], response_count[5])
                                            if min_count > cycle_count:
                                                cycle_count = min_count
                                                logger.info(f"\n[INFO] === Cycle {cycle_count}/{cycles_to_complete} completed ===")
                                            
                                            # Remove processed JSON from buffer
                                            buffer = text[json_end:].encode('utf-8')
                                            
                                            if cycle_count >= cycles_to_complete:
                                                break
                                        
                                        json_start = -1
                                        json_end = -1
                            
                            if cycle_count >= cycles_to_complete:
                                break
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                    
                    logger.info(f"\n[INFO] All {cycles_to_complete} cycles completed. Sending END_VISION...")
                except KeyboardInterrupt:
                    logger.info("  [INFO] Interrupted by user")
                except Exception as e:
                    logger.error(f"  [ERROR] Error during response wait: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Test 4: CALC RESULT
            
            # Test 5: END VISION
            time.sleep(3)
            client.test_end_vision()

            response = client.test_calc_result(path_csv="data/20251118-154122_zoom1_raw_data.csv")
            if not response.get("success"):
                logger.error(f"CALC RESULT failed: {response}")
                return
            else:
                logger.info(f"CALC RESULT success: {response}")
            
            logger.info("\n" + "=" * 60)
            logger.info("[OK] All tests completed")
            logger.info("=" * 60)
        
    except ConnectionRefusedError:
        logger.error("Connection refused. Make sure server is running.")
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

