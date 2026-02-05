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
            self.logger.setLevel(logging.DEBUG)
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
                                
                                # Calculate memory size
                                import sys
                                json_size_bytes = len(json_str.encode('utf-8'))
                                json_size_kb = json_size_bytes / 1024
                                json_size_mb = json_size_kb / 1024
                                
                                # Estimate Python object size
                                object_size_bytes = sys.getsizeof(response)
                                if isinstance(response.get("data"), list):
                                    # For trajectory data (list of dicts)
                                    for item in response.get("data", []):
                                        object_size_bytes += sys.getsizeof(item)
                                        if isinstance(item, dict):
                                            for key, value in item.items():
                                                object_size_bytes += sys.getsizeof(key) + sys.getsizeof(value)
                                elif isinstance(response.get("data"), dict):
                                    # For single detection data
                                    for key, value in response.get("data", {}).items():
                                        object_size_bytes += sys.getsizeof(key) + sys.getsizeof(value)
                                
                                object_size_kb = object_size_bytes / 1024
                                object_size_mb = object_size_kb / 1024
                                
                                # Log response with memory info
                                self.logger.info(f"Response: {json.dumps(response, indent=2)}")
                                self.logger.info(
                                    f"Response memory size: "
                                    f"JSON={json_size_bytes} bytes ({json_size_kb:.2f} KB, {json_size_mb:.3f} MB), "
                                    f"Python object={object_size_bytes} bytes ({object_size_kb:.2f} KB, {object_size_mb:.3f} MB)"
                                )
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

    def test_start_cam_manual(self):
        """Test START CAM 1 Manual command (cmd: 8).

        Performs a single-shot detection on camera 1 and returns position data.
        Can be called at any time during the camera cycle.
        """
        self.logger.info("\n[TEST] START CAM 1 Manual")
        request = {"cmd": 8}
        response = self.send_request(request)
        return response

    def test_manual_calc_result(self, path_csv: str):
        """Test MANUAL CALC RESULT command (cmd: 9).

        Calculates performance metrics from manual measurement CSV data.

        Args:
            path_csv: Path to CSV file with x, y, rz columns
        """
        self.logger.info("\n[TEST] MANUAL CALC RESULT")
        request = {
            "cmd": 9,
            "path_csv": path_csv
        }
        response = self.send_request(request)
        return response


def run_interactive_mode(client, logger):
    """Run interactive mode with keyboard input for manual commands.

    Keyboard commands:
      m - START CAM 1 Manual (single-shot detection)
      c - MANUAL CALC RESULT (calculate from CSV)
      q - Quit interactive mode
    """
    import select
    import sys
    import threading

    logger.info("\n" + "=" * 60)
    logger.info("Interactive Mode - Keyboard Commands:")
    logger.info("  m : START CAM 1 Manual (single-shot detection)")
    logger.info("  c : MANUAL CALC RESULT (calculate from CSV)")
    logger.info("  q : Quit interactive mode")
    logger.info("=" * 60)

    # Storage for manual measurements
    manual_measurements = []
    manual_csv_path = Path("data/manual_measurements.csv")

    # Flag to stop the response listener thread
    stop_listener = threading.Event()

    def response_listener():
        """Background thread to receive and log server responses."""
        buffer = b""
        client.socket.settimeout(0.5)  # Short timeout for checking stop flag

        while not stop_listener.is_set():
            try:
                chunk = client.socket.recv(4096)
                if not chunk:
                    continue
                buffer += chunk

                # Parse JSON responses
                text = buffer.decode('utf-8')
                brace_count = 0
                json_start = -1

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
                            try:
                                response = json.loads(json_str)
                                cmd = response.get("cmd")

                                if cmd == 7:  # NOTIFY_CONNECTION
                                    logger.info(f"[NOTIFY_CONNECTION] {json.dumps(response, indent=2)}")
                                elif cmd in [3, 4, 5]:  # Camera responses
                                    logger.info(f"[CAM {cmd-2}] {json.dumps(response, indent=2)}")
                                elif cmd == 8:  # Manual camera response
                                    logger.info(f"[MANUAL CAM 1] {json.dumps(response, indent=2)}")
                                elif cmd == 9:  # Manual calc result
                                    logger.info(f"[MANUAL CALC] {json.dumps(response, indent=2)}")
                                else:
                                    logger.info(f"[CMD {cmd}] {json.dumps(response, indent=2)}")

                                buffer = text[json_end:].encode('utf-8')
                            except json.JSONDecodeError:
                                pass
                            json_start = -1

            except socket.timeout:
                continue
            except Exception as e:
                if not stop_listener.is_set():
                    logger.debug(f"Listener error: {e}")
                break

    # Start response listener thread
    listener_thread = threading.Thread(target=response_listener, daemon=True)
    listener_thread.start()

    try:
        while True:
            try:
                cmd_input = input("\nEnter command (m/c/q): ").strip().lower()
            except EOFError:
                break

            if cmd_input == 'q':
                logger.info("Exiting interactive mode...")
                break

            elif cmd_input == 'm':
                # START CAM 1 Manual
                logger.info("\n[INPUT] Sending START CAM 1 Manual...")
                try:
                    # Send request directly (bypass send_request to avoid blocking)
                    request = {"cmd": 8}
                    request_json = json.dumps(request)
                    client.socket.sendall(request_json.encode('utf-8'))
                    logger.info(f"Request sent: {request_json}")

                    # Wait briefly for response
                    time.sleep(0.5)

                    # Try to read response
                    client.socket.settimeout(5.0)
                    try:
                        response_data = b""
                        while True:
                            chunk = client.socket.recv(4096)
                            if not chunk:
                                break
                            response_data += chunk
                            # Try to parse
                            try:
                                text = response_data.decode('utf-8')
                                # Find cmd:8 response
                                if '"cmd": 8' in text or '"cmd":8' in text:
                                    response = json.loads(text)
                                    if response.get("cmd") == 8:
                                        logger.info(f"Response: {json.dumps(response, indent=2)}")
                                        if response.get("success"):
                                            data = response.get("data", {})
                                            x = data.get("x", 0)
                                            y = data.get("y", 0)
                                            rz = data.get("rz", 0)
                                            logger.info(f"  [OK] Manual Detection: x={x:.3f}mm, y={y:.3f}mm, rz={rz:.3f}deg")

                                            # Store measurement
                                            manual_measurements.append({"x": x, "y": y, "rz": rz})
                                            logger.info(f"  [INFO] Stored measurement #{len(manual_measurements)}")
                                        else:
                                            logger.error(f"  [FAIL] {response.get('error_code')}: {response.get('error_desc')}")
                                        break
                            except json.JSONDecodeError:
                                continue
                    except socket.timeout:
                        logger.warning("Timeout waiting for manual camera response")
                    finally:
                        client.socket.settimeout(0.5)

                except Exception as e:
                    logger.error(f"Error sending manual camera command: {e}")

            elif cmd_input == 'c':
                # MANUAL CALC RESULT
                if not manual_measurements:
                    logger.warning("No manual measurements stored. Use 'm' to collect measurements first.")
                    csv_path = input("Or enter CSV path (empty to cancel): ").strip()
                    if not csv_path:
                        continue
                else:
                    # Save measurements to CSV
                    import pandas as pd
                    df = pd.DataFrame(manual_measurements)
                    manual_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(str(manual_csv_path), index=False)
                    logger.info(f"  [INFO] Saved {len(manual_measurements)} measurements to {manual_csv_path}")
                    csv_path = str(manual_csv_path)

                logger.info(f"\n[INPUT] Sending MANUAL CALC RESULT for: {csv_path}")
                try:
                    request = {"cmd": 9, "path_csv": csv_path}
                    request_json = json.dumps(request)
                    client.socket.sendall(request_json.encode('utf-8'))
                    logger.info(f"Request sent: {request_json}")

                    # Wait for response
                    time.sleep(0.5)
                    client.socket.settimeout(5.0)
                    try:
                        response_data = b""
                        while True:
                            chunk = client.socket.recv(4096)
                            if not chunk:
                                break
                            response_data += chunk
                            try:
                                text = response_data.decode('utf-8')
                                if '"cmd": 9' in text or '"cmd":9' in text:
                                    response = json.loads(text)
                                    if response.get("cmd") == 9:
                                        logger.info(f"Response: {json.dumps(response, indent=2)}")
                                        if response.get("success"):
                                            data = response.get("data", {})
                                            stats = data.get("statistics", {})
                                            logger.info(f"  [OK] Analysis completed for {data.get('n_measurements')} measurements")
                                            if stats:
                                                for axis in ['x', 'y', 'rz']:
                                                    s = stats.get(axis, {})
                                                    logger.info(f"    {axis}: mean={s.get('mean', 0):.3f}, std={s.get('std', 0):.3f}, "
                                                              f"range={s.get('range', 0):.3f}, 3σ={s.get('repeatability_3sigma', 0):.3f}")
                                        else:
                                            logger.error(f"  [FAIL] {response.get('error_code')}: {response.get('error_desc')}")
                                        break
                            except json.JSONDecodeError:
                                continue
                    except socket.timeout:
                        logger.warning("Timeout waiting for calc result response")
                    finally:
                        client.socket.settimeout(0.5)

                except Exception as e:
                    logger.error(f"Error sending calc result command: {e}")

            else:
                logger.warning(f"Unknown command: '{cmd_input}'. Use m/c/q")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        stop_listener.set()
        listener_thread.join(timeout=1.0)


def main():
    """Run test client."""
    import argparse

    parser = argparse.ArgumentParser(description="Vision Tracking TCP/IP Server - Test Client")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode for manual testing")
    parser.add_argument("--cycles", type=int, default=2,
                        help="Number of cycles to wait in automatic mode (default: 2)")
    args = parser.parse_args()

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

        if args.interactive:
            # Interactive mode: START VISION then allow keyboard commands
            response = client.test_start_vision(use_area_scan=False)
            if not response.get("success"):
                logger.error(f"START VISION failed: {response.get('error_desc')}")
                return

            time.sleep(1)  # Wait for initialization
            run_interactive_mode(client, logger)

            # End vision before disconnecting
            client.test_end_vision()
        else:
            # Original automatic test mode
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
                    logger.info(f"  [INFO] Waiting for {args.cycles} cycles, then sending END_VISION")

                    # Wait for cycles: 1→2→3→1→2→3
                    cycles_to_complete = args.cycles
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
                                                # Calculate memory size
                                                import sys
                                                json_size_bytes = len(json_str.encode('utf-8'))
                                                json_size_kb = json_size_bytes / 1024
                                                json_size_mb = json_size_kb / 1024

                                                # Estimate Python object size
                                                object_size_bytes = sys.getsizeof(response)
                                                if isinstance(response.get("data"), list):
                                                    # For trajectory data (list of dicts)
                                                    for item in response.get("data", []):
                                                        object_size_bytes += sys.getsizeof(item)
                                                        if isinstance(item, dict):
                                                            for key, value in item.items():
                                                                object_size_bytes += sys.getsizeof(key) + sys.getsizeof(value)
                                                elif isinstance(response.get("data"), dict):
                                                    # For single detection data
                                                    for key, value in response.get("data", {}).items():
                                                        object_size_bytes += sys.getsizeof(key) + sys.getsizeof(value)

                                                object_size_kb = object_size_bytes / 1024
                                                object_size_mb = object_size_kb / 1024

                                                # Log response in same format as send_request
                                                logger.info(f"Response: {json.dumps(response, indent=2)}")
                                                logger.info(
                                                    f"Response memory size: "
                                                    f"JSON={json_size_bytes} bytes ({json_size_kb:.2f} KB, {json_size_mb:.3f} MB), "
                                                    f"Python object={object_size_bytes} bytes ({object_size_kb:.2f} KB, {object_size_mb:.3f} MB)"
                                                )

                                                response_count[cmd] += 1

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

