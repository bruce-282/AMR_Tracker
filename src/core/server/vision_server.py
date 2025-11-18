"""TCP/IP Vision Server for AMR Tracking System."""

import socket
import threading
import json
import time
import logging
from typing import Dict, Optional, Any, Callable, Tuple, List, Union
from pathlib import Path
import cv2
import numpy as np

from .protocol import ProtocolHandler, Command
from .model_config import ModelConfig
from src.core.detection import YOLODetector, Detection
from src.core.tracking import KalmanTracker, associate_detections_to_trackers
from src.utils.sequence_loader import create_sequence_loader, BaseLoader
from config import SystemConfig

# Import visualizer
try:
    from src.visualization.display import Visualizer
    from src.core.measurement.size_measurement import SizeMeasurement
    ENHANCED_VISUALIZATION_AVAILABLE = True
except ImportError:
    ENHANCED_VISUALIZATION_AVAILABLE = False


# Constants
LOADER_MODE_MAP = {
    "video": "video_file",
    "sequence": "image_sequence",
    "camera": "camera_device"
}

SPEED_THRESHOLD_PIX_PER_FRAME = 5.0  # pixels/frame (속도가 이 값보다 크면 tracking loop 종료)
DETECTION_LOSS_THRESHOLD_FRAMES = 30
CAMERA2_TRAJECTORY_MAX_FRAMES = 50
SPEED_NEAR_ZERO_THRESHOLD = 3.0  # pixels/frame (속도가 이 값 이하면 0에 가까운 것으로 간주)
SPEED_ZERO_FRAMES_THRESHOLD = 10  # 프레임 수 (이 프레임 수 동안 속도가 0에 가까우면 response 전송)


class VisionServer:
    """TCP/IP server for vision tracking system."""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10000,
        config_path: Optional[str] = None,
        preset_name: Optional[str] = None
    ):
        """
        Initialize vision server.
        
        Args:
            host: Server host address
            port: Server port
            config_path: Path to configuration file
            preset_name: Preset name to use (overrides config's use_preset)
        """
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
        self.running = False
        self.client_socket = None
        self.client_address = None
        self.preset_name = preset_name  # Command-line preset override
        
        # Load configuration
        self.config = None
        if config_path and Path(config_path).exists():
            try:
                self.config = SystemConfig.load(config_path)
                self.logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}")
        
        # Model configuration
        self.model_config = ModelConfig()
        
        # Result file paths
        self.result_base_path = Path("C:/CMES_AI/Result")
        self.result_base_path.mkdir(parents=True, exist_ok=True)
        
        # Debug data path (separate from studio results)
        self.debug_base_path = Path("tracking_results")
        self.debug_base_path.mkdir(parents=True, exist_ok=True)
        
        # System state
        self.vision_active = False
        # Get model file path from selected product model
        selected_product_model = self.model_config.get_selected_model()
        if selected_product_model:
            self.model_path = self.model_config.get_model_file_path(selected_product_model)
        else:
            self.model_path = Path("weights/last.pt")  # Fallback
        self.use_area_scan = False
        self.detector = None
        self.trackers = {}  # camera_id -> tracker dict
        self.camera_loaders = {}  # camera_id -> loader
        self.next_track_ids = {}  # camera_id -> next_track_id
        self.tracking_threads = {}  # camera_id -> thread
        self.frame_numbers = {}  # camera_id -> frame_number
        self.latest_detections = {}  # camera_id -> latest Detection object
        
        # Visualization components (same as main.py)
        self.visualizer = None
        self.size_measurement = None
        if ENHANCED_VISUALIZATION_AVAILABLE and self.config:
            try:
                # Load calibration data if available (same as main.py)
                calibration_path = None
                try:
                    if hasattr(self.config, 'calibration') and hasattr(self.config.calibration, 'calibration_data_path'):
                        calibration_path = self.config.calibration.calibration_data_path
                except (KeyError, AttributeError) as e:
                    self.logger.debug(f"Error accessing calibration_data_path: {e}")
                    calibration_path = None
                
                if calibration_path and Path(calibration_path).exists():
                    with open(calibration_path, "r") as f:
                        import json
                        calibration_data = json.load(f)
                    
                    # Initialize size measurement and visualizer if calibration data exists
                    self.size_measurement = SizeMeasurement(
                        homography=np.array(calibration_data["homography"]),
                        camera_height=self.config.calibration.camera_height,
                        pixel_size=calibration_data.get("pixel_size", 1.0),
                        calibration_image_size=self.config.calibration.calibration_image_size,
                    )
                    self.logger.info("Size measurement initialized")
                    
                    self.visualizer = Visualizer(
                        homography=np.array(calibration_data["homography"])
                    )
                    self.logger.info("Visualizer initialized")
                else:
                    self.logger.debug("No calibration data found. Size measurement disabled.")
            except Exception as e:
                self.logger.warning(f"Error initializing visualization components: {e}")
        
        # For camera 2: store all frame positions
        self.camera2_trajectory = []  # List of {"track_idx": int, "x": float, "y": float, "rz": float}
        self.camera2_trajectory_sent = False  # Flag to prevent duplicate sends in the same cycle
        
        # For cameras 1, 3: track speed and detection loss
        self.camera_speed_history = {}  # camera_id -> list of speeds (mm/s)
        self.camera_detection_loss_frames = {}  # camera_id -> frames since last detection
        self.camera_has_reached_speed_threshold = {}  # camera_id -> bool (speed > 1 mm/s)
        self.camera_speed_near_zero_frames = {}  # camera_id -> frames with speed near zero (for response condition)
        
        # Track if response has been sent for cameras 1, 3 (to avoid sending twice)
        self.camera_response_sent = {}  # {camera_id: bool}
        
        # Camera connection status
        self.camera_status = {1: False, 2: False, 3: False}
        
        # For periodic response when use_area_scan is false
        self.periodic_response_threads = {}  # camera_id -> thread
        self.periodic_response_interval = 0.1  # 100ms (10 Hz)
        
        # Protocol handler
        self.protocol = ProtocolHandler()
    
    # ==================== Helper Methods ====================
    
    def _normalize_loader_mode(self, loader_mode: str) -> str:
        """Normalize loader mode string to sequence_loader format."""
        return LOADER_MODE_MAP.get(loader_mode, loader_mode)
    
    def _get_pixel_size(self) -> float:
        """Get pixel size from config."""
        return self.config.measurement.pixel_size if self.config else 1.0
    
    def _get_fps_from_loader(self, loader: BaseLoader) -> float:
        """Get FPS from loader or config."""
        if loader and hasattr(loader, 'fps'):
            return loader.fps
        elif self.config and hasattr(self.config, 'measurement'):
            return self.config.measurement.fps
        return 30.0
    
    def _create_tracker(self, camera_id: int, track_id: int, fps: Optional[float] = None) -> KalmanTracker:
        """Create a KalmanTracker instance for a camera."""
        if fps is None:
            loader = self.camera_loaders.get(camera_id)
            fps = self._get_fps_from_loader(loader)
        
        pixel_size = self._get_pixel_size()
        
        return KalmanTracker(
            fps=fps,
            pixel_size=pixel_size,
            track_id=track_id,
        )
    
    def _reset_camera_state(self, camera_id: int):
        """Reset camera state for restart (cameras 1, 3)."""
        if camera_id in self.latest_detections:
            del self.latest_detections[camera_id]
        if camera_id in self.camera_speed_history:
            self.camera_speed_history[camera_id] = []
        if camera_id in self.camera_detection_loss_frames:
            self.camera_detection_loss_frames[camera_id] = 0
        if camera_id in self.camera_speed_near_zero_frames:
            self.camera_speed_near_zero_frames[camera_id] = 0
        if camera_id in self.camera_response_sent:
            del self.camera_response_sent[camera_id]
    
    def _send_response_to_client(self, cmd: int, success: bool, 
                                  data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                                  error_code: Optional[str] = None,
                                  error_desc: Optional[str] = None) -> bool:
        """Send response to client. Returns True if successful."""
        if not self.client_socket:
            return False
        
        try:
            response = self.protocol.create_response(
                cmd, success=success, data=data,
                error_code=error_code, error_desc=error_desc
            )
            self.client_socket.sendall(response)
            # Add small delay to prevent TCP packet merging (especially after multiple NOTIFY_CONNECTION)
            time.sleep(0.005)  # 5ms delay
            return True
        except (ConnectionError, OSError) as e:
            self.logger.warning(f"Failed to send response (cmd={cmd}): {e}")
            return False
    
    def _initialize_camera_tracker(self, camera_id: int, fps: float, track_id: int = 0):
        """Initialize tracker instance for a camera (without starting tracking thread)."""
        if camera_id not in self.trackers:
            self.trackers[camera_id] = {}
            self.next_track_ids[camera_id] = track_id + 1
            self.frame_numbers[camera_id] = 0
        
        tracker = self._create_tracker(camera_id, track_id, fps)
        self.trackers[camera_id][track_id] = tracker
        
        # Initialize speed tracking for cameras 1, 3
        if camera_id in [1, 3]:
            self.camera_speed_history[camera_id] = []
            self.camera_detection_loss_frames[camera_id] = 0
            self.camera_has_reached_speed_threshold[camera_id] = False
            self.camera_speed_near_zero_frames[camera_id] = 0
        
        self.logger.info(
            f"Camera {camera_id}: Tracker created (track_id={track_id}, "
            f"fps={fps:.2f}, pixel_size={self._get_pixel_size()})"
        )
    
    def _ensure_camera_initialized(self, camera_id: int, product_model_name: Optional[str] = None) -> bool:
        """Ensure camera is initialized. Returns True if successful."""
        if camera_id in self.camera_loaders and camera_id in self.trackers:
            return True
        
        try:
            loader_mode, source, fps = self._get_camera_config(camera_id, product_model_name)
            loader_mode = self._normalize_loader_mode(loader_mode)
            
            if camera_id not in self.camera_loaders:
                self._initialize_camera(camera_id, loader_mode=loader_mode, source=source, fps=fps)
            
            if camera_id not in self.trackers:
                self._initialize_camera_tracker(camera_id, fps, track_id=0)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize camera {camera_id}: {e}")
            return False
    
    def _calculate_speed_pix_per_frame(self, tracker: KalmanTracker) -> float:
        """Calculate object speed in pixels/frame from tracker state."""
        state = tracker.kf.statePost.flatten()
        vx = state[3]  # velocity x (pixels/frame)
        vy = state[4]  # velocity y (pixels/frame)
        # Return speed in pixels/frame (no fps multiplication)
        return np.sqrt(vx**2 + vy**2)
    
    def _send_first_detection_response(self, camera_id: int, detection: Detection, frame: np.ndarray):
        """Send first detection response for cameras 1, 3 (use_area_scan=false)."""
        if camera_id in self.camera_response_sent:
            return  # Already sent
        
        # Get detection data (use detection center, not tracker position, for cameras 1, 3)
        pixel_size = self._get_pixel_size()
        center = detection.get_center()
        x_mm = center[0] * pixel_size
        y_mm = center[1] * pixel_size
        rz = detection.get_orientation() if detection.get_orientation() is not None else 0.0  # Already in degrees
        
        # Save result image on first detection
        result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
        try:
            # For cameras 1, 3: use detection center (not tracker position) to match response data
            # Create tracking_results with detection center position
            tracking_results = []
            tracking_result = {
                "track_id": 0,
                "position": {"x": center[0], "y": center[1]},  # Use detection center, not tracker position
                "orientation": {"theta_deg": rz},  # Already in degrees
                "bbox": detection.bbox
            }
            tracking_results.append(tracking_result)
            
            # For cameras 1, 3: draw only current point, not trajectory
            draw_trajectory = camera_id not in [1, 3]
            frame_with_results = self.visualize_results(camera_id, frame.copy(), [detection], tracking_results, draw_trajectory=draw_trajectory)
            cv2.imwrite(str(result_image_path), frame_with_results)
        except Exception as e:
            self.logger.warning(f"Failed to save result image for camera {camera_id}: {e}")
        
        # Send response (round to 3 decimal places)
        response_data = {
            "x": round(float(x_mm), 3),
            "y": round(float(y_mm), 3),
            "rz": round(float(rz), 3),
            "result_image": str(result_image_path)
        }
        
        cmd = Command.START_CAM_1 + camera_id - 1
        if self._send_response_to_client(cmd, success=True, data=response_data):
            self.logger.info(f"Camera {camera_id}: First detection response sent")
            self.camera_response_sent[camera_id] = True
    
    def start(self):
        """Start the TCP/IP server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.logger.info(f"Vision Server started on {self.host}:{self.port}")
            self.running = True
            
            while self.running:
                self.logger.info("Waiting for client connection...")
                client_socket, client_address = self.socket.accept()
                self.logger.info(f"Client connected from {client_address}")
                
                # Configure client socket for efficient data transmission
                # TCP_NODELAY: Disable Nagle's algorithm (send data immediately, don't wait for ACK)
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Increase send buffer size for large trajectory data
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # 64KB send buffer
                
                self.client_socket = client_socket
                self.client_address = client_address
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
        except Exception as e:
            self.logger.warning(f"Server error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.socket:
            self.socket.close()
        self._stop_all_cameras()
        self.logger.info("Vision Server stopped")
    
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle client connection."""
        try:
            while self.running:
                # Receive data
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                except (ConnectionError, OSError) as e:
                    self.logger.warning(f"Connection error while receiving data: {e}")
                    break
                
                # Parse request
                request = self.protocol.parse_request(data)
                if not request:
                    continue
                
                # Log request
                self.logger.info(f"Request: {json.dumps(request, indent=2)}")
                
                # Handle command
                response = self._handle_command(request)
                if response:
                    try:
                        client_socket.sendall(response)
                        # Add small delay to prevent TCP packet merging
                        time.sleep(0.005)  # 5ms delay
                        # Log response
                        try:
                            response_dict = json.loads(response.decode('utf-8'))
                            self.logger.info(f"Response: {json.dumps(response_dict, indent=2)}")
                        except:
                            pass  # If response is not JSON, skip logging
                    except (ConnectionError, OSError) as e:
                        self.logger.warning(f"Connection error while sending response: {e}")
                        break
                    
        except Exception as e:
            self.logger.warning(f"Client handling error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Client disconnected - cleanup all resources (same as END_VISION)
            self.logger.info(f"Client {client_address} disconnected. Cleaning up all resources...")
            
            # Set vision_active to False to stop all tracking loops (same as END_VISION)
            self.vision_active = False
            
            # Stop all cameras and cleanup (same as END_VISION)
            self._stop_all_cameras()
            
            try:
                client_socket.close()
            except:
                pass
            self.client_socket = None
            self.client_address = None
            self.logger.info("All resources cleaned up after client disconnection")
    
    def _handle_command(self, request: Dict[str, Any]) -> Optional[bytes]:
        """Handle incoming command."""
        cmd = request.get("cmd")
        
        try:
            if cmd == Command.START_VISION:
                return self._handle_start_vision(request)
            elif cmd == Command.END_VISION:
                return self._handle_end_vision(request)
            elif cmd == Command.START_CAM_1:
                return self._handle_start_cam(1, request)
            elif cmd == Command.START_CAM_2:
                return self._handle_start_cam(2, request)
            elif cmd == Command.START_CAM_3:
                return self._handle_start_cam(3, request)
            elif cmd == Command.CALC_RESULT:
                return self._handle_calc_result(request)
            else:
                return self.protocol.create_response(
                    cmd or 0,
                    success=False,
                    error_code="INVALID_CMD",
                    error_desc=f"Unknown command: {cmd}"
                )
        except Exception as e:
            return self.protocol.create_response(
                cmd or 0,
                success=False,
                error_code="INTERNAL_ERROR",
                error_desc=str(e)
            )
    
    def _handle_start_vision(self, request: Dict[str, Any]) -> bytes:
        """Handle START VISION command."""
        start_time = time.time()
        self.logger.info("START VISION command received")
        try:
            # Update product model if provided
            # model can be: product model name (string) or index (int)
            if "model" in request:
                model = request["model"]
                if isinstance(model, str):
                    # Product model name (e.g., "zoom1", "zoom2")
                    self.model_config.set_selected_model(model)
                    product_model_name = model
                elif isinstance(model, int):
                    # Use model index from model_list
                    model_list = self.model_config.get_model_list()
                    if 0 <= model < len(model_list):
                        product_model_name = model_list[model]
                        self.model_config.set_selected_model(product_model_name)
                    else:
                        raise ValueError(f"Invalid model index: {model}. Available models: {model_list}")
                else:
                    # Use selected model from config
                    product_model_name = self.model_config.get_selected_model()
                    if not product_model_name:
                        raise ValueError("No model selected and no model provided")
            else:
                # Use selected model from config
                product_model_name = self.model_config.get_selected_model()
                if not product_model_name:
                    raise ValueError("No model selected. Please provide model in request or set in config.")
            
            # Get detector configuration from product model config
            detector_config = self.model_config.get_detector_config(product_model_name)
            model_path_str = detector_config.get("model_path")
            self.model_path = Path(model_path_str)
            
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path} "
                    f"(product model: {product_model_name})"
                )
            
            # Update use_area_scan
            if "use_area_scan" in request:
                self.use_area_scan = bool(request["use_area_scan"])
            
            # Initialize detector with config parameters
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.detector = YOLODetector(
                str(self.model_path),
                confidence_threshold=detector_config.get("confidence_threshold", 0.2),
                nms_threshold=detector_config.get("nms_threshold", 0.4),
                device=device,
                imgsz=detector_config.get("imgsz", 640),
                target_classes=detector_config.get("target_classes", [0]),
            )
            
            self.vision_active = True
            
            self.logger.info(f"Vision started with product model: {product_model_name}")
            self.logger.info(f"  Model file: {self.model_path}")
            
            # Initialize all 3 cameras and initialize trackers (without starting tracking threads)
            for cam_id in [1, 2, 3]:
                try:
                    self.logger.info(f"Initializing camera {cam_id}...")
                    
                    loader_mode, source, fps = self._get_camera_config(cam_id, product_model_name)
                    loader_mode = self._normalize_loader_mode(loader_mode)
                    
                    # Initialize camera loader
                    self._initialize_camera(cam_id, loader_mode=loader_mode, source=source, fps=fps)
                    
                    # Check connection and send NOTIFY_CONNECTION
                    loader = self.camera_loaders.get(cam_id)
                    if loader and hasattr(loader, 'check_connection'):
                        is_connected = loader.check_connection()
                        if is_connected:
                            self._send_notification(cam_id, True)
                            self.logger.info(f"Camera {cam_id} connection confirmed - NOTIFY_CONNECTION sent")
                        else:
                            self._send_notification(cam_id, False, error_code="CONNECTION_FAILED", error_desc="Camera connection check failed")
                            self.logger.warning(f"Camera {cam_id} connection check failed - NOTIFY_CONNECTION sent")
                    else:
                        self._send_notification(cam_id, False, error_code="LOADER_ERROR", error_desc="Loader does not support connection check")
                        self.logger.warning(f"Camera {cam_id} loader does not support connection check")
                    
                    # For cameras 1 and 3: create tracker instance (track_id=0) without initialization
                    # For camera 2: do not create tracker yet (will be created when cameras 1/3 stop)
                    if cam_id in [1, 3]:
                        self._initialize_camera_tracker(cam_id, fps, track_id=0)
                    
                    self.logger.info(f"Camera {cam_id} initialized successfully")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize camera {cam_id}: {e}")
                    # Send NOTIFY_CONNECTION with error
                    self._send_notification(cam_id, False, error_code="INIT_ERROR", error_desc=str(e))
                    # Continue initializing other cameras even if one fails
            
            # All cameras initialized
            elapsed_time = time.time() - start_time
            self.logger.info(f"All cameras initialized. Total time: {elapsed_time:.3f}s")
            
            # If use_area_scan is false, automatically start camera 1 tracking
            if not self.use_area_scan:
                self.logger.info("use_area_scan=false: Automatically starting camera 1 tracking")
                
                # Start camera 1 tracking thread
                if 1 in self.trackers and 1 in self.camera_loaders:
                    if 1 not in self.tracking_threads or not self.tracking_threads[1].is_alive():
                        self._start_camera_tracking(1)
                        time.sleep(0.1)
            
            return self.protocol.create_response(
                Command.START_VISION,
                success=True
            )
        except Exception as e:
            return self.protocol.create_response(
                Command.START_VISION,
                success=False,
                error_code="INIT_ERROR",
                error_desc=str(e)
            )
    
    def _get_camera_config(self, camera_id: int, product_model_name: Optional[str] = None) -> tuple:
        """
        Get camera configuration (loader_mode, source, fps) from config file.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            product_model_name: Product model name (optional, uses selected_model if None)
        
        Returns:
            Tuple of (loader_mode, source, fps)
        """
        if product_model_name is None:
            product_model_name = self.model_config.get_selected_model()
        
        loader_mode = None
        source = None
        fps = 30.0
        
        if self.config and hasattr(self.config, 'execution') and self.config.execution:
            exec_config = self.config.execution
            
            # Check if preset is specified (command-line arg takes priority)
            preset_name = self.preset_name or exec_config.get("use_preset")
            if preset_name:
                # Use specified preset
                presets = exec_config.get("presets", {})
                preset = presets.get(preset_name, {})
                if preset:
                    loader_mode = preset.get("loader_mode")
                    source = preset.get("source")
                    fps = preset.get("fps", 30.0)
                    
                    # If camera mode, use camera device ID from product model config
                    if loader_mode == "camera" and product_model_name:
                        source = self.model_config.get_camera_device_id(product_model_name, camera_id)
                        self.logger.info(f"Camera {camera_id}: Using device_id={source} from product config '{product_model_name}'")
                    
                    self.logger.info(f"Using preset: {preset_name} (loader_mode={loader_mode}, source={source})")
                else:
                    self.logger.warning(f"Preset '{preset_name}' not found in config")
            
            # If preset not found or not specified, use default settings
            if loader_mode is None or source is None:
                loader_mode = exec_config.get("default_loader_mode", "camera")
                source = exec_config.get("default_source", camera_id - 1)
                fps = exec_config.get("default_fps", 30.0)
                
                # If camera mode, try to use camera device ID from product model config
                if loader_mode == "camera" and product_model_name:
                    source = self.model_config.get_camera_device_id(product_model_name, camera_id)
                    self.logger.info(f"Camera {camera_id}: Using device_id={source} from product config '{product_model_name}'")
                
                self.logger.info(f"Using default settings (loader_mode={loader_mode}, source={source})")
        else:
            # Fallback defaults if no config
            loader_mode = "camera"
            if product_model_name:
                source = self.model_config.get_camera_device_id(product_model_name, camera_id)
            else:
                source = camera_id - 1
        
        return loader_mode, source, fps
    
    def _handle_end_vision(self, request: Dict[str, Any]) -> bytes:
        """Handle END VISION command.
        
        Response format:
        {
            "cmd": 2,
            "success": bool,
            "error_code": string (optional),
            "error_desc": string (optional)
        }
        """
        try:
            self.vision_active = False
            self.logger.info("END VISION command received")
            self._stop_all_cameras()
            
            return self.protocol.create_response(
                Command.END_VISION,  # cmd: 2
                success=True
            )
        except Exception as e:
            return self.protocol.create_response(
                Command.END_VISION,  # cmd: 2
                success=False,
                error_code="STOP_ERROR",
                error_desc=str(e)
            )
    
    def _handle_start_cam(self, camera_id: int, request: Dict[str, Any]) -> bytes:
        """Handle START CAM command.
        
        Loader mode and source are read from config file, not from request.
        For camera mode, uses camera_id from product model config.
        """
        try:
            if not self.vision_active:
                return self.protocol.create_response(
                    Command.START_CAM_1 + camera_id - 1,
                    success=False,
                    error_code="VISION_NOT_ACTIVE",
                    error_desc="Vision system not started. Call START VISION first."
                )
            
            # Camera should already be initialized from START VISION
            # Just verify it's initialized
            product_model_name = self.model_config.get_selected_model()
            if not self._ensure_camera_initialized(camera_id, product_model_name):
                return self.protocol.create_response(
                    Command.START_CAM_1 + camera_id - 1,
                    success=False,
                    error_code="CAM_INIT_ERROR",
                    error_desc=f"Failed to initialize camera {camera_id}"
                )
            
            # Handle tracking thread start based on use_area_scan
            if self.use_area_scan:
                # use_area_scan is true: start tracking thread only when client requests
                # All cameras (1, 2, 3) can be started via START_CAM command
                if camera_id not in self.tracking_threads or not self.tracking_threads[camera_id].is_alive():
                    self._start_camera_tracking(camera_id)
                    time.sleep(0.1)
            else:
                # use_area_scan is false: tracking threads are started automatically
                # Camera 1: already started in START_VISION
                # Camera 2: will be started automatically when camera 1 stops
                # Camera 3: will be started automatically when camera 2 stops
                # Only start tracking thread for cameras 1 and 3 if not already running
                if camera_id in [1, 3]:
                    if camera_id not in self.tracking_threads or not self.tracking_threads[camera_id].is_alive():
                        self._start_camera_tracking(camera_id)
                        time.sleep(0.1)
                # Camera 2: do not start tracking thread here (will be started when cameras 1/3 stop)
            
            # Handle response based on use_area_scan
            if self.use_area_scan:
                # use_area_scan is true: client will send requests periodically
                # Just respond to this request
                data = self._get_tracking_data(camera_id)
                result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
                self._save_result_image(camera_id, result_image_path)
                
                response_data = {
                    "x": data["x"],
                    "y": data["y"],
                    "rz": data["rz"]
                }
                
                return self.protocol.create_response(
                    Command.START_CAM_1 + camera_id - 1,
                    success=True,
                    data=response_data
                )
            else:
                # use_area_scan is false: client won't send requests
                # Cameras 1, 3: response will be sent on first detection in tracking loop
                # Camera 2: sends trajectory data directly from tracking loop when detection is lost
                # Return success response (detection result will be sent from tracking loop)
                return self.protocol.create_response(
                    Command.START_CAM_1 + camera_id - 1,
                    success=True
                )
        except Exception as e:
            return self.protocol.create_response(
                Command.START_CAM_1 + camera_id - 1,
                success=False,
                error_code="CAM_START_ERROR",
                error_desc=str(e)
            )
    
    def _initialize_camera(self, camera_id: int, loader_mode: str = "camera", source = None, fps: float = 30.0):
        """Initialize camera and tracker.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            loader_mode: Loader mode ("camera", "video", "image_sequence")
            source: Source path or device ID (depends on loader_mode)
            fps: Frame rate for video/image sequence
        """
        # Default source if not provided
        if source is None:
            if loader_mode == "camera":
                source = camera_id - 1  # Camera device index
            else:
                raise ValueError(f"Source must be provided for loader_mode: {loader_mode}")
        
        # Create loader based on mode
        # Each camera gets its own independent loader instance
        loader = create_sequence_loader(source, fps=fps, loader_mode=loader_mode)
        
        if loader is None:
            raise RuntimeError(f"Failed to create loader for camera {camera_id} (mode: {loader_mode}, source: {source})")
        
        # Reset loader to start from first frame (important for video files)
        # Each camera should start from frame 0 independently
        if hasattr(loader, 'cap') and loader.cap is not None:
            # For VideoFileLoader, reset to first frame
            loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            loader.frame_number = 0
        elif hasattr(loader, 'frame_number'):
            # For ImageSequenceLoader, reset frame counter
            loader.frame_number = 0
        
        self.camera_loaders[camera_id] = loader
        self.trackers[camera_id] = {}
        self.next_track_ids[camera_id] = 0
        self.frame_numbers[camera_id] = 0
        self.camera_status[camera_id] = True
        
        self.logger.info(f"Camera {camera_id} initialized")
        
        # Note: Connection notification is not sent automatically
        # Client can check camera status via START CAM response
    
    def _tracking_loop(self, camera_id: int):
        """Main tracking loop for camera."""
        loader = self.camera_loaders.get(camera_id)
        if not loader:
            return
        
        trackers = self.trackers[camera_id]
        frame_number = 0
        
        self.logger.info(f"Camera {camera_id} tracking started")
        
        while self.vision_active and camera_id in self.camera_loaders:
            try:
                ret, frame = loader.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                frame_number += 1
                self.frame_numbers[camera_id] = frame_number
                
                # Get frame timestamp (use current time, or from loader if available)
                frame_timestamp = time.time()
                if hasattr(loader, 'get_timestamp'):
                    loader_timestamp = loader.get_timestamp()
                    if loader_timestamp is not None:
                        frame_timestamp = loader_timestamp
                
                # Detect objects
                detections = self.detector.detect(
                    frame,
                    frame_number=frame_number,
                    timestamp=frame_timestamp
                )
                
                # Store latest detection (for cameras 1 and 3, or when tracker not initialized)
                # Note: For cameras 1, 3 with use_area_scan=false, this is handled in camera-specific logic below
                has_detection = len(detections) > 0
                if has_detection:
                    # Only update if not already handled in camera-specific logic
                    if not (not self.use_area_scan and camera_id in [1, 3]):
                        self.latest_detections[camera_id] = detections[0]  # Use first detection
                    # Reset detection loss counter
                    if camera_id in [1, 3] and not self.use_area_scan:
                        self.camera_detection_loss_frames[camera_id] = 0
                else:
                    # Increment detection loss counter for cameras 1, 3
                    if camera_id in [1, 3] and not self.use_area_scan:
                        if camera_id not in self.camera_detection_loss_frames:
                            self.camera_detection_loss_frames[camera_id] = 0
                        self.camera_detection_loss_frames[camera_id] += 1
                
                # Track objects
                tracking_results = []
                used_detections = set()
                
                # Update existing trackers
                for track_id, tracker in list(trackers.items()):
                    # Find best matching detection
                    best_detection_idx = None
                    best_distance = float('inf')
                    
                    for i, detection in enumerate(detections):
                        if i in used_detections:
                            continue
                        
                        # Calculate distance
                        tracker_state = tracker.kf.statePost.flatten()
                        tracker_pos = (tracker_state[0], tracker_state[1])
                        detection_center = detection.get_center()
                        distance = np.sqrt(
                            (tracker_pos[0] - detection_center[0]) ** 2 +
                            (tracker_pos[1] - detection_center[1]) ** 2
                        )
                        
                        if distance < best_distance and distance < 500:  # Max association distance
                            best_distance = distance
                            best_detection_idx = i
                    
                    if best_detection_idx is not None:
                        detection = detections[best_detection_idx]
                        tracking_result = tracker.update(
                            frame,
                            detection.bbox,
                            frame_number=frame_number,
                            orientation=detection.get_orientation(),
                            timestamp=frame_timestamp,  # Pass timestamp
                        )
                        tracking_result["track_id"] = track_id
                        tracking_result["class_name"] = getattr(detection, "class_name", "Unknown")
                        tracking_results.append(tracking_result)
                        used_detections.add(best_detection_idx)
                    else:
                        # No detection found, predict only
                        tracking_result = tracker.update(
                            frame, 
                            None, 
                            frame_number=frame_number,
                            timestamp=frame_timestamp  # Pass timestamp
                        )
                        tracking_result["track_id"] = track_id
                        tracking_results.append(tracking_result)
                
                # Create new trackers for unassigned detections
                for i, detection in enumerate(detections):
                    if i not in used_detections:
                        # Check if there's an uninitialized tracker (track_id=0 with state at origin)
                        tracker = None
                        track_id = None
                        
                        # Look for uninitialized tracker (state is all zeros)
                        for existing_track_id, existing_tracker in trackers.items():
                            state = existing_tracker.kf.statePost.flatten()
                            # Check if tracker is uninitialized (position is at origin)
                            if state[0] == 0.0 and state[1] == 0.0 and state[2] == 0.0:
                                tracker = existing_tracker
                                track_id = existing_track_id
                                break
                        
                        # If no uninitialized tracker found, create new one
                        if tracker is None:
                            track_id = self.next_track_ids[camera_id]
                            self.next_track_ids[camera_id] += 1
                            
                            fps = self._get_fps_from_loader(loader)
                            tracker = self._create_tracker(camera_id, track_id, fps)
                            trackers[track_id] = tracker
                        
                        # Initialize tracker with detection
                        tracker.initialize_with_detection(detection.bbox)
                        
                        # Update with orientation if available
                        if detection.get_orientation() is not None:
                            state = tracker.kf.statePost.flatten()
                            state[2] = detection.get_orientation()
                            tracker.kf.statePost = state.reshape(-1, 1)
                        
                        tracking_result = tracker.update(
                            frame,
                            detection.bbox,
                            frame_number=frame_number,
                            orientation=detection.get_orientation(),
                            timestamp=frame_timestamp,  # Pass timestamp
                        )
                        tracking_result["track_id"] = track_id
                        tracking_result["class_name"] = getattr(detection, "class_name", "Unknown")
                        tracking_results.append(tracking_result)
                
                # Visualize and display tracking window
                vis_frame = self.visualize_results(camera_id, frame, detections, tracking_results)
                
                # Store original vis_frame before resize (for saving image)
                vis_frame_original = vis_frame.copy()
                
                # Resize window if too large (same as main.py)
                height, width = vis_frame.shape[:2]
                if width > 1280 or height > 720:
                    scale = min(1280 / width, 720 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    vis_frame = cv2.resize(vis_frame, (new_width, new_height))
                
                # Show tracking window
                window_name = f"Camera {camera_id} - AMR Tracking"
                cv2.imshow(window_name, vis_frame)
                
                # Handle key press (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.logger.info(f"Camera {camera_id}: 'q' pressed, stopping tracking")
                    break
                elif key == ord("s"):
                    snapshot_path = f"snapshot_cam{camera_id}_{frame_number:06d}.jpg"
                    cv2.imwrite(snapshot_path, vis_frame)
                    self.logger.info(f"Camera {camera_id}: Snapshot saved: {snapshot_path}")
                
                # Camera-specific logic
                if not self.use_area_scan:
                    # Cameras 1, 3: track speed and handle detection loss
                    if camera_id in [1, 3]:
                        # Store detection
                        if detections:
                            self.latest_detections[camera_id] = detections[0]
                        
                        # Check speed and send response when speed is near zero for threshold frames
                        if tracking_results:
                            tracker = next(iter(trackers.values())) if trackers else None
                            if tracker:
                                speed_pix_per_frame = self._calculate_speed_pix_per_frame(tracker)
                                
                                # Track speed history
                                if camera_id not in self.camera_speed_history:
                                    self.camera_speed_history[camera_id] = []
                                self.camera_speed_history[camera_id].append(speed_pix_per_frame)
                                if len(self.camera_speed_history[camera_id]) > 10:
                                    self.camera_speed_history[camera_id].pop(0)
                                
                                self.logger.info(
                                        f"Camera {camera_id}: Speed near zero check - "
                                        f"speed={abs(speed_pix_per_frame):.3f} pix/frame, "
                                        f"threshold={SPEED_NEAR_ZERO_THRESHOLD}, "
                                        f"count={self.camera_speed_near_zero_frames[camera_id]}/{SPEED_ZERO_FRAMES_THRESHOLD}, "
                                        f"response_sent={camera_id in self.camera_response_sent}, "
                                        f"has_detection={camera_id in self.latest_detections}"
                                )
                                    
                                # Check if speed is near zero (in pixels/frame)
                                if abs(speed_pix_per_frame) <= SPEED_NEAR_ZERO_THRESHOLD:
                                    # Initialize counter if not exists
                                    if camera_id not in self.camera_speed_near_zero_frames:
                                        self.camera_speed_near_zero_frames[camera_id] = 0
                                    self.camera_speed_near_zero_frames[camera_id] += 1
                                    
                         
                                    # Send response if speed has been near zero for threshold frames
                                    if (self.camera_speed_near_zero_frames[camera_id] >= SPEED_ZERO_FRAMES_THRESHOLD and
                                        camera_id not in self.camera_response_sent and
                                        camera_id in self.latest_detections):
                                        self.logger.info(
                                            f"Camera {camera_id}: Sending first detection response - "
                                            f"speed={speed_pix_per_frame:.3f} pix/frame, "
                                            f"count={self.camera_speed_near_zero_frames[camera_id]}"
                                        )
                                        self._send_first_detection_response(camera_id, self.latest_detections[camera_id], frame)
                                        self.camera_speed_near_zero_frames[camera_id] = 0
                                else:
                                    # Reset counter if speed is not near zero
                                    if camera_id in self.camera_speed_near_zero_frames:
                                        if self.camera_speed_near_zero_frames[camera_id] > 0:
                                            self.logger.debug(
                                                f"Camera {camera_id}: Speed not near zero - "
                                                f"speed={speed_pix_per_frame:.3f} pix/frame > "
                                                f"threshold={SPEED_NEAR_ZERO_THRESHOLD}, "
                                                f"resetting count from {self.camera_speed_near_zero_frames[camera_id]}"
                                            )
                                        self.camera_speed_near_zero_frames[camera_id] = 0
                                
                                # Check if speed threshold reached (for stopping tracking)
                                # Only check if response has already been sent
                                if (abs(speed_pix_per_frame) > SPEED_THRESHOLD_PIX_PER_FRAME and
                                    camera_id in self.camera_response_sent):
                                    self.logger.info(f"Camera {camera_id}: Speed threshold reached ({speed_pix_per_frame:.3f} pix/frame > {SPEED_THRESHOLD_PIX_PER_FRAME} pix/frame). Stopping camera {camera_id} tracking loop.")
                                    
                                    # Cameras 1, 3: Response already sent on first detection, no need to send again
                                    
                                    # Stop this camera's tracking thread (but keep loader for reuse)
                                    
                                    # Camera 1: Start camera 2 tracking immediately
                                    if camera_id == 1:
                                        self.logger.info("Camera 1: Starting camera 2 tracking loop.")
                                        if 2 not in self.tracking_threads or not self.tracking_threads[2].is_alive():
                                            product_model_name = self.model_config.get_selected_model()
                                            if self._ensure_camera_initialized(2, product_model_name):
                                                # Clear camera 2 trajectory before starting
                                                self.camera2_trajectory = []
                                                self.camera_detection_loss_frames[2] = 0
                                                self.camera2_trajectory_sent = False  # Reset flag for new cycle
                                                # Clear visualizer trajectory for camera 2 (remove previous camera's trajectory)
                                                if self.visualizer is not None:
                                                    track_id = 0
                                                    if track_id in self.visualizer.track_trajs:
                                                        self.visualizer.track_trajs[track_id] = []
                                                self._start_camera_tracking(2)
                                                self.logger.info("Camera 2 tracking thread started")
                                    
                                    # Camera 3: Start camera 1 tracking again (infinite loop: 1→2→3→1→...)
                                    elif camera_id == 3:
                                        self.logger.info("Camera 3: Starting camera 1 tracking loop again (infinite cycle).")
                                        # Camera 1 is already initialized from START_VISION
                                        # Just clear state and start tracking thread
                                        self._reset_camera_state(1)
                                        # Clear visualizer trajectory for camera 1 (remove previous camera's trajectory)
                                        if self.visualizer is not None:
                                            track_id = 0
                                            if track_id in self.visualizer.track_trajs:
                                                self.visualizer.track_trajs[track_id] = []
                                        if 1 not in self.tracking_threads or not self.tracking_threads[1].is_alive():
                                            if 1 in self.camera_loaders and 1 in self.trackers:
                                                self._start_camera_tracking(1)
                                                self.logger.info("Camera 1 tracking thread started (cycle restarted)")
                                            else:
                                                self.logger.warning("Camera 1 not initialized, cannot start tracking")
                                    
                                    # Exit tracking loop for this camera
                                    break
                    
                    # Camera 2: store all frame positions
                    elif camera_id == 2:
                        if tracking_results:
                            tracker = next(iter(trackers.values())) if trackers else None
                            if tracker:
                                state = tracker.kf.statePost.flatten()
                                pixel_size = self.config.measurement.pixel_size if self.config else 1.0
                                x_mm = state[0] * pixel_size
                                y_mm = state[1] * pixel_size
                                rz_deg = state[2]  # Already in degrees
                                
                                # Store trajectory with index (round to 3 decimal places)
                                trajectory_index = len(self.camera2_trajectory)
                                self.camera2_trajectory.append({
                                    "track_idx": trajectory_index,
                                    "x": round(float(x_mm), 3),
                                    "y": round(float(y_mm), 3),
                                    "rz": round(float(rz_deg), 3)  # in degrees
                                })
                        
                        # Check if detection lost for camera 2
                        if not has_detection:
                            if camera_id not in self.camera_detection_loss_frames:
                                self.camera_detection_loss_frames[camera_id] = 0
                            self.camera_detection_loss_frames[camera_id] += 1
                        
                        # Check if should send trajectory data and exit:
                        # Condition 1: Detection lost for threshold frames
                        # Condition 2: OR trajectory has max frames
                        should_send_trajectory = False
                        if not has_detection and self.camera_detection_loss_frames.get(camera_id, 0) >= DETECTION_LOSS_THRESHOLD_FRAMES:
                            should_send_trajectory = True
                            reason = f"detection lost for {DETECTION_LOSS_THRESHOLD_FRAMES}+ frames"
                        elif len(self.camera2_trajectory) >= CAMERA2_TRAJECTORY_MAX_FRAMES:
                            should_send_trajectory = True
                            reason = f"trajectory reached {len(self.camera2_trajectory)} frames (>= {CAMERA2_TRAJECTORY_MAX_FRAMES})"
                        
                        if should_send_trajectory and len(self.camera2_trajectory) > 0 and not self.camera2_trajectory_sent:
                            # Set flag IMMEDIATELY to prevent duplicate sends (even within same frame)
                            self.camera2_trajectory_sent = True
                            
                            self.logger.info(f"Camera 2: {reason}. Sending trajectory data to client ({len(self.camera2_trajectory)} frames).")
                            
                            # Save result image at the last frame before sending trajectory data
                            # Use original vis_frame (before resize) to match camera 1, 3 image size
                            result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
                            self._save_result_image(camera_id, result_image_path, frame=vis_frame_original, detections=detections, tracking_results=tracking_results)
                            
                            # Send trajectory data directly from tracking loop
                            # Protocol: {cmd: 4, success: bool, data: array<object>}
                            trajectory_data = self.camera2_trajectory.copy()
                            cmd = Command.START_CAM_2  # cmd: 4 for camera 2
                            
                            # Send trajectory array directly (without trajectory key wrapper)
                            # result_image is saved but not included in response (as per protocol)
                            if self._send_response_to_client(cmd, success=True, data=trajectory_data):
                                self.logger.info(f"Camera 2 trajectory data sent ({len(trajectory_data)} frames) with result_image")
                            
                            # Clear trajectory after sending
                            self.camera2_trajectory = []
                            self.camera_detection_loss_frames[camera_id] = 0
                            
                            # Start camera 3 tracking after camera 2 exits (only when use_area_scan=false)
                            if not self.use_area_scan:
                                self.logger.info("Camera 2: Starting camera 3 tracking loop.")
                                if 3 not in self.tracking_threads or not self.tracking_threads[3].is_alive():
                                    self._reset_camera_state(3)
                                    # Clear visualizer trajectory for camera 3 (remove previous camera's trajectory)
                                    if self.visualizer is not None:
                                        track_id = 0
                                        if track_id in self.visualizer.track_trajs:
                                            self.visualizer.track_trajs[track_id] = []
                                    if 3 in self.camera_loaders and 3 in self.trackers:
                                        self._start_camera_tracking(3)
                                    else:
                                        self.logger.warning("Camera 3 not initialized, cannot start tracking")
                            
                            # Exit tracking loop immediately after sending trajectory
                            # This prevents duplicate sends in the same cycle
                            self.logger.info(f"Camera 2: Tracking loop exiting after sending trajectory.")
                            break
                
                # Clean up lost trackers
                trackers_to_remove = []
                for track_id, tracker in trackers.items():
                    if tracker.is_lost(max_frames_lost=30):
                        trackers_to_remove.append(track_id)
                
                for track_id in trackers_to_remove:
                    del trackers[track_id]
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.warning(f"Camera {camera_id} tracking error: {e}")
                time.sleep(0.1)
        
        self.logger.info(f"Camera {camera_id} tracking stopped")
        
        # Close tracking window for this camera
        window_name = f"Camera {camera_id} - AMR Tracking"
        try:
            cv2.destroyWindow(window_name)
        except:
            pass  # Window may not exist
    
    def _start_camera_tracking(self, camera_id: int):
        """Start tracking thread for a camera."""
        if camera_id not in self.tracking_threads or not self.tracking_threads[camera_id].is_alive():
            thread = threading.Thread(
                target=self._tracking_loop,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.tracking_threads[camera_id] = thread
            self.logger.info(f"Camera {camera_id} tracking thread started")
    
    def _get_tracking_data(self, camera_id: int) -> Dict[str, float]:
        """Get current tracking data for camera.
        
        For cameras 1, 3 (use_area_scan=false): return detection data
        For camera 2: return tracker data
        
        Returns:
            Dictionary with x, y (in mm) and rz (yaw angle in degrees)
            rz = rotation around Z-axis = yaw angle (in degrees)
        """
        # Cameras 1, 3: use detection data (not tracker) when use_area_scan is false
        self.logger.debug(f"Camera {camera_id}: use_area_scan={self.use_area_scan}, camera_id in [1,3]={camera_id in [1, 3]}")
        if camera_id in [1, 3] and not self.use_area_scan:
            detection = self.latest_detections.get(camera_id)
            self.logger.debug(f"Camera {camera_id}: latest_detections has data: {detection is not None}")
            if detection is not None:
                center = detection.get_center()
                orientation = detection.get_orientation()
                
                # Convert to mm
                pixel_size = self.config.measurement.pixel_size if self.config else 1.0
                x_mm = center[0] * pixel_size
                y_mm = center[1] * pixel_size
                rz_deg = orientation if orientation is not None else 0.0  # Already in degrees
                
                self.logger.debug(f"Camera {camera_id}: Returning DETECTION data: x={x_mm:.2f}, y={y_mm:.2f}, rz={rz_deg:.4f}deg")
                return {
                    "x": round(float(x_mm), 3),
                    "y": round(float(y_mm), 3),
                    "rz": round(float(rz_deg), 3)  # in degrees
                }
            else:
                # No detection yet
                self.logger.debug(f"Camera {camera_id}: No detection available, returning zeros")
                return {"x": 0.0, "y": 0.0, "rz": 0.0}
        
        # Camera 2 or use_area_scan=true: use tracker data
        trackers = self.trackers.get(camera_id, {})
        
        if not trackers:
            # Return default values if no tracker
            self.logger.debug(f"Camera {camera_id}: No tracker available, returning zeros")
            return {"x": 0.0, "y": 0.0, "rz": 0.0}
        
        # Get first tracker (primary object)
        tracker = next(iter(trackers.values()))
        state = tracker.kf.statePost.flatten()
        
        # Convert to mm if config available
        pixel_size = self.config.measurement.pixel_size if self.config else 1.0
        x_mm = state[0] * pixel_size
        y_mm = state[1] * pixel_size
        rz_deg = state[2]  # rz = yaw = rotation around Z-axis (in degrees)
        
        self.logger.debug(f"Camera {camera_id}: Returning TRACKER data: x={x_mm:.2f}, y={y_mm:.2f}, rz={rz_deg:.4f}deg")
        return {
            "x": round(float(x_mm), 3),
            "y": round(float(y_mm), 3),
            "rz": round(float(rz_deg), 3)  # yaw angle in degrees
        }
    
    def _handle_calc_result(self, request: Dict[str, Any]) -> bytes:
        """Handle CALC RESULT command."""
        try:
            # Calculate performance metrics for all active cameras
            results = {}
            
            for camera_id in self.trackers.keys():
                # Get tracking data
                tracking_data = self._get_tracking_data(camera_id)
                
                # Save result image (overwrite existing file)
                result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
                self._save_result_image(camera_id, result_image_path)
                
                results[f"cam_{camera_id}"] = {
                    "position": tracking_data,
                    "result_image": str(result_image_path)
                }
            
            return self.protocol.create_response(
                Command.CALC_RESULT,
                success=True,
                data=results
            )
        except Exception as e:
            return self.protocol.create_response(
                Command.CALC_RESULT,
                success=False,
                error_code="CALC_ERROR",
                error_desc=str(e)
            )
    
    def _save_result_image(self, camera_id: int, image_path: Path, frame: Optional[np.ndarray] = None, detections: Optional[List[Detection]] = None, tracking_results: Optional[List[Dict]] = None):
        """Save result image (overwrite existing file)."""
        try:
            # Get frame if not provided
            if frame is None:
                loader = self.camera_loaders.get(camera_id)
                if loader:
                    ret, frame = loader.read()
                    if not ret or frame is None:
                        return
                else:
                    return
            
            # Get detections and tracking results if not provided
            if detections is None:
                detections = []
                if camera_id in self.latest_detections:
                    detections = [self.latest_detections[camera_id]]
            
            if tracking_results is None:
                trackers = self.trackers.get(camera_id, {})
                tracking_results = []
                for track_id, tracker in trackers.items():
                    state = tracker.kf.statePost.flatten()
                    bbox = detections[0].bbox if detections else None
                    tracking_result = {
                        "track_id": track_id,
                        "position": {"x": state[0], "y": state[1]},
                        "orientation": {"theta_deg": state[2]},  # Already in degrees
                        "bbox": bbox
                    }
                    tracking_results.append(tracking_result)
            
            # Draw tracking results on frame using visualize_results
            # For cameras 1, 3: draw only current point, not trajectory
            draw_trajectory = camera_id not in [1, 3]
            frame_with_results = self.visualize_results(camera_id, frame.copy(), detections, tracking_results, draw_trajectory=draw_trajectory)
            cv2.imwrite(str(image_path), frame_with_results)
            self.logger.info(f"Result image saved: {image_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save result image: {e}")
    
    def visualize_results(
        self, camera_id: int, frame: np.ndarray, detections: List[Detection], tracking_results: List[Dict], draw_trajectory: bool = True
    ) -> np.ndarray:
        """Visualize results using appropriate visualizer (same as main.py).
        
        Args:
            camera_id: Camera ID
            frame: Input frame
            detections: List of detections
            tracking_results: List of tracking results
            draw_trajectory: Whether to draw trajectory (default: True). Set to False for cameras 1, 3 to show only current point.
        """
        # Filter out uninitialized trackers (state at origin: 0, 0, 0)
        filtered_tracking_results = []
        for result in tracking_results:
            position = result.get("position", {})
            if position:
                x = position.get("x", 0)
                y = position.get("y", 0)
                # Skip if position is at origin (uninitialized tracker)
                if x == 0.0 and y == 0.0:
                    continue
            filtered_tracking_results.append(result)
        
        if self.visualizer is not None:
            # For cameras 1, 3: clear trajectory before drawing to show only current point
            if not draw_trajectory and camera_id in [1, 3]:
                track_id = 0
                if track_id in self.visualizer.track_trajs:
                    # Clear trajectory and keep only current point
                    if filtered_tracking_results:
                        position = filtered_tracking_results[0].get("position", {})
                        if position:
                            x = int(position.get("x", 0))
                            y = int(position.get("y", 0))
                            self.visualizer.track_trajs[track_id] = [(x, y)]
                        else:
                            self.visualizer.track_trajs[track_id] = []
                    else:
                        self.visualizer.track_trajs[track_id] = []
            
            vis_frame = self.visualizer.draw_detections(
                frame, detections, filtered_tracking_results
            )
        else:
            # Fallback: return frame as-is if visualizer is not available
            vis_frame = frame.copy()
        
        return vis_frame
    
    def _stop_all_cameras(self):
        """Stop all camera tracking."""
        self.logger.info("Stopping all cameras and tracking threads...")
        
        # vision_active is already set to False in _handle_end_vision
        # This will cause tracking loops to exit (while self.vision_active and ...)
        
        # Wait for tracking threads to finish first (before closing windows)
        # This ensures cv2.waitKey is no longer blocking
        for camera_id in list(self.tracking_threads.keys()):
            thread = self.tracking_threads.get(camera_id)
            if thread and thread.is_alive():
                self.logger.info(f"Waiting for camera {camera_id} tracking thread to finish...")
                thread.join(timeout=2.0)  # Wait up to 2 seconds
                if thread.is_alive():
                    self.logger.warning(f"Camera {camera_id} tracking thread did not finish within timeout")
        
        # Stop all periodic response threads
        for camera_id in list(self.periodic_response_threads.keys()):
            thread = self.periodic_response_threads.get(camera_id)
            if thread and thread.is_alive():
                self.logger.info(f"Waiting for camera {camera_id} periodic response thread to finish...")
                thread.join(timeout=1.0)  # Wait up to 1 second
                if thread.is_alive():
                    self.logger.warning(f"Camera {camera_id} periodic response thread did not finish within timeout")
        
        # Close all tracking windows (after threads are stopped)
        # Use destroyAllWindows to avoid blocking issues
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.debug(f"Failed to destroy windows: {e}")
        
        # Stop all cameras (release loaders, delete trackers, etc.)
        # Send NOTIFY_CONNECTION for each camera before stopping
        for camera_id in list(self.camera_loaders.keys()):
            # Send NOTIFY_CONNECTION before stopping
            self._send_notification(camera_id, False, error_code="VISION_ENDED", error_desc="Vision ended")
            self.logger.info(f"Camera {camera_id} disconnection notified - NOTIFY_CONNECTION sent")
            # Stop camera
            self._stop_camera(camera_id)
        
        # Clear tracking threads dictionary
        self.tracking_threads.clear()
        self.periodic_response_threads.clear()
        
        self.logger.info("All cameras stopped")
    
    def _stop_camera(self, camera_id: int):
        """Stop camera tracking."""
        if camera_id in self.camera_loaders:
            loader = self.camera_loaders[camera_id]
            if hasattr(loader, 'release'):
                loader.release()
            del self.camera_loaders[camera_id]
        
        if camera_id in self.trackers:
            del self.trackers[camera_id]
        
        self.camera_status[camera_id] = False
        # Note: NOTIFY_CONNECTION is sent in _stop_all_cameras before calling _stop_camera
    
    # def _periodic_response_loop(self, camera_id: int):
    #     """Periodically send tracking data when use_area_scan is false.
        
    #     For cameras 1, 3: send detection result once and exit.
    #     For camera 2: send trajectory data when detection is lost for 30+ frames.
    #     """
    #     # Cameras 1, 3: send detection result once and exit
    #     if camera_id in [1, 3]:
    #         # Wait for detection to be available
    #         max_wait_time = 5.0  # Wait up to 5 seconds for detection
    #         wait_start = time.time()
    #         while time.time() - wait_start < max_wait_time:
    #             data = self._get_tracking_data(camera_id)
    #             if not (data["x"] == 0.0 and data["y"] == 0.0 and data["rz"] == 0.0):
    #                 # Detection available, send once and exit
    #                 # Image should already be saved in _tracking_loop on first detection
    #                 result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
                    
    #                 response_data = {
    #                     "x": data["x"],
    #                     "y": data["y"],
    #                     "rz": data["rz"],
    #                     "result_image": str(result_image_path)
    #                 }
                    
    #                 cmd = Command.START_CAM_1 + camera_id - 1
    #                 if self._send_response_to_client(cmd, success=True, data=response_data):
    #                     self.logger.info(f"Camera {camera_id}: Detection result sent to client (one-time)")
                    
    #                 # Exit after sending once
    #                 return
                
    #             time.sleep(0.1)  # Check every 100ms
            
    #         # No detection available after waiting
    #         self.logger.warning(f"Camera {camera_id}: No detection available after {max_wait_time}s, exiting periodic response loop")
    #         return
        
    #     # Camera 2: does not use periodic response loop
    #     # Trajectory data is sent directly from _tracking_loop when detection is lost for 30+ frames
    #     elif camera_id == 2:
    #         # Camera 2 should not have a periodic response loop
    #         # It sends data directly from _tracking_loop
    #         self.logger.warning(f"Camera {camera_id}: Periodic response loop should not be called for camera 2")
    #         return
    
    def _send_notification(self, camera_id: int, is_connected: bool, 
                          error_code: Optional[str] = None,
                          error_desc: Optional[str] = None):
        """Send connection notification to client."""
        if self.client_socket:
            try:
                notification = self.protocol.create_notification(
                    camera_id,
                    is_connected,
                    error_code,
                    error_desc
                )
                self.client_socket.sendall(notification)
                # Add small delay to prevent TCP packet merging
                # 5ms delay ensures OS has time to send the packet before next sendall()
                time.sleep(0.005)  # 5ms delay
            except Exception as e:
                self.logger.warning(f"Failed to send notification: {e}")


def main():
    """Run vision server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Tracking TCP/IP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=10000, help="Server port")
    parser.add_argument("--config", default="tracker_config.json", help="Config file path")
    parser.add_argument("--result-path", default="C:/CMES_AI/Result", help="Result file base path")
    parser.add_argument("--debug-path", default="tracking_results", help="Debug data base path")
    
    args = parser.parse_args()
    
    server = VisionServer(host=args.host, port=args.port, config_path=args.config)
    
    # Override paths if provided
    if args.result_path:
        server.result_base_path = Path(args.result_path)
        server.result_base_path.mkdir(parents=True, exist_ok=True)
    if args.debug_path:
        server.debug_base_path = Path(args.debug_path)
        server.debug_base_path.mkdir(parents=True, exist_ok=True)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[OK] Shutting down server...")
        server.stop()


if __name__ == "__main__":
    main()

