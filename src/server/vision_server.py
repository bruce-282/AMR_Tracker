"""TCP/IP Vision Server for AMR Tracking System."""

import os

# Fix OpenMP library conflict - must be set before importing other libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import socket
import threading
import json
import time
import logging
from collections import deque
from typing import Dict, Optional, Any, Callable, Tuple, List, Union
from pathlib import Path
import cv2
import numpy as np

from .protocol import ProtocolHandler, Command
from .model_config import ModelConfig
from .camera_state import CameraState, CameraStateManager
from .camera_manager import CameraManager
from .tracking_manager import TrackingManager
from .response_builder import ResponseBuilder
from src.core.detection import YOLODetector, Detection
from src.core.tracking import KalmanTracker
from src.core.amr_tracker import EnhancedAMRTracker
from src.utils.sequence_loader import create_sequence_loader, BaseLoader
from src.utils.trajectory_repeatability import TrajectoryRepeatability
from src.utils.config_loader import (
    load_product_model_config,
    get_camera_config,
    get_camera_pixel_sizes,
)
from config import SystemConfig, TrackingConfig



# Constants
LOADER_MODE_MAP = {
    "video": "video_file",
    "sequence": "image_sequence",
    "camera": "camera_device"
}

# Default tracking config (used when config file not loaded)
DEFAULT_TRACKING_CONFIG = TrackingConfig()


class VisionServer:
    """TCP/IP server for vision tracking system."""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10000,
        preset_name: Optional[str] = None
    ):
        """
        Initialize vision server.
        
        Args:
            host: Server host address
            port: Server port
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
            self.logger.setLevel(logging.DEBUG)
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.client_socket = None
        self.client_address = None
        self.preset_name = preset_name  # Command-line preset override
        
        # Load configuration
        self.config = None
        
        # Model configuration
        self.model_config = ModelConfig()
        
        # Result file paths
        self.result_base_path = Path("C:/CMES_AI/Result")
        self.result_base_path.mkdir(parents=True, exist_ok=True)

        self.summary_base_path = Path("C:/CMES_AI/Summary")
        self.summary_base_path.mkdir(parents=True, exist_ok=True)

        # Debug data path (separate from studio results)
        self.debug_base_path = Path("C:/CMES_AI/Debug")
        self.debug_base_path.mkdir(parents=True, exist_ok=True)
        
        # System state
        self.vision_active = False
        self.use_area_scan = False
        self.visualize_stream = True  # Enable/disable visualization stream (cv2.imshow)
        
        # Camera state manager - centralized state management for all cameras
        self.camera_state_manager = CameraStateManager()

        # Tracking configuration
        self.tracking_config = self.config.tracking if self.config and self.config.tracking else DEFAULT_TRACKING_CONFIG

        # Initialize managers (delegates for camera, tracking, and response handling)
        self.camera_manager = CameraManager(
            model_config=self.model_config,
            system_config=self.config,
            preset_name=self.preset_name
        )
        
        self.tracking_manager = TrackingManager(
            camera_manager=self.camera_manager,
            camera_state_manager=self.camera_state_manager,
            tracking_config=self.tracking_config,
            use_area_scan=self.use_area_scan,
            visualize_stream=self.visualize_stream
        )
        
        # Set callbacks for TrackingManager
        self.tracking_manager.on_camera_1_3_stop = self._start_next_camera_after_1_3
        self.tracking_manager.on_camera_2_stop = self._start_camera_3_after_2
        self.tracking_manager.on_camera_1_3_first_detection = self._send_first_detection_response
        self.tracking_manager.on_camera_2_trajectory = self._send_camera2_trajectory
        
        # Share camera2_trajectory with TrackingManager
        # This will be initialized in START_VISION
        self.camera2_trajectory = None
        self.camera2_trajectory_sent = False
        
        self.response_builder = ResponseBuilder(
            camera_manager=self.camera_manager,
            tracking_manager=self.tracking_manager,
            result_base_path=self.result_base_path,
            debug_base_path=self.debug_base_path
        )
        
        # Protocol handler
        self.protocol = ProtocolHandler()
        
        # Periodic response threads (for area scan mode)
        self.periodic_response_threads: Dict[int, threading.Thread] = {}
    
    # ==================== Property Delegation ====================
    # Properties delegate to manager attributes for cleaner access
    
    @property
    def camera_loaders(self):
        """Access camera_loaders from CameraManager."""
        return self.camera_manager.camera_loaders
    
    @property
    def amr_trackers(self):
        """Access amr_trackers from CameraManager."""
        return self.camera_manager.amr_trackers
    
    @property
    def trackers(self):
        """Access trackers from CameraManager."""
        return self.camera_manager.trackers
    
    @property
    def tracking_threads(self):
        """Access tracking_threads from TrackingManager."""
        return self.tracking_manager.tracking_threads
    
    @property
    def latest_detections(self):
        """Access latest_detections from TrackingManager."""
        return self.tracking_manager.latest_detections
    
    @property
    def camera_pixel_sizes(self):
        """Access camera_pixel_sizes from CameraManager."""
        return self.camera_manager.camera_pixel_sizes
    
    @property
    def frame_numbers(self):
        """Access frame_numbers from CameraManager."""
        return self.camera_manager.frame_numbers
    
    @property
    def camera_status(self):
        """Access camera_status from CameraManager."""
        return self.camera_manager.camera_status
    
    # ==================== Helper Methods ====================
    
    
    
    def _create_tracker(self, camera_id: int, track_id: int, fps: Optional[float] = None) -> KalmanTracker:
        """Create a KalmanTracker instance for a camera."""
        if fps is None:
            loader = self.camera_loaders.get(camera_id)
            fps = self.camera_manager.get_fps_from_loader(loader)
        
        # Get pixel_size for this specific camera
        pixel_size = self.camera_manager.get_pixel_size(camera_id)
        
        return KalmanTracker(
            fps=fps,
            pixel_size=pixel_size,
            track_id=track_id,
        )
    
    def _reset_camera_state(self, camera_id: int):
        """Reset camera state for restart (cameras 1, 3)."""
        if camera_id in self.latest_detections:
            del self.latest_detections[camera_id]
        # Reset state via CameraStateManager
        state = self.camera_state_manager.get(camera_id)
        if state:
            state.reset()
    
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

        # Initialize camera state via CameraStateManager
        state = self.camera_state_manager.get_or_create(camera_id)
        state.tracker = tracker
        state.next_track_id = track_id + 1
        state.reset()  # Clear any previous state

        self.logger.info(
            f"Camera {camera_id}: Tracker created (track_id={track_id}, "
            f"fps={fps:.2f}, pixel_size={self.camera_manager.get_pixel_size()})"
        )
    
    def _ensure_camera_initialized(self, camera_id: int, product_model_name: Optional[str] = None) -> bool:
        """Ensure camera is initialized. Returns True if successful."""
        if camera_id in self.camera_loaders and camera_id in self.trackers:
            return True
        
        try:
            loader_mode, source, fps, config_path = self.camera_manager.get_camera_config(camera_id, product_model_name)
           # loader_mode = self._normalize_loader_mode(loader_mode)
            
            if camera_id not in self.camera_loaders:
                self._initialize_camera(camera_id, loader_mode=loader_mode, source=source, fps=fps, camera_config_path=config_path)
            
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


    def _handle_camera_1_3_tracking(
        self,
        camera_id: int,
        trackers: Dict[int, KalmanTracker],
        detections: List[Detection],
        tracking_results: List[Dict],
        frame: np.ndarray,
        cam_state: CameraState
    ) -> bool:
        """Handle camera 1, 3 specific tracking logic.

        Returns:
            True if tracking should continue, False if should break.
        """
        # Store detection
        if detections:
            self.latest_detections[camera_id] = detections[0]

        # Get tracking config thresholds
        speed_near_zero_thresh = self.tracking_config.speed_near_zero_threshold
        speed_zero_frames_thresh = self.tracking_config.speed_zero_frames_threshold
        speed_thresh = self.tracking_config.speed_threshold_pix_per_frame

        if not tracking_results:
            return True

        # Try to get tracker from EnhancedAMRTracker first, then fallback to trackers dict
        amr_tracker = self.amr_trackers.get(camera_id)
        if amr_tracker and amr_tracker.tracker and amr_tracker.track_id is not None:
            tracker = amr_tracker.tracker
        else:
            tracker = next(iter(trackers.values())) if trackers else None
        
        if not tracker:
            return True

        speed_pix_per_frame = self._calculate_speed_pix_per_frame(tracker)
        cam_state.update_speed(speed_pix_per_frame)

        self.logger.info(
            f"Camera {camera_id}: Speed near zero check - "
            f"speed={abs(speed_pix_per_frame):.3f} pix/frame, "
            f"threshold={speed_near_zero_thresh}, "
            f"count={cam_state.speed_near_zero_frames}/{speed_zero_frames_thresh}, "
            f"response_sent={cam_state.response_sent}, "
            f"has_detection={camera_id in self.latest_detections}"
        )

        # Check if speed is near zero
        if abs(speed_pix_per_frame) <= speed_near_zero_thresh:
            cam_state.speed_near_zero_frames += 1

            # Send response if speed has been near zero for threshold frames
            if (cam_state.speed_near_zero_frames >= speed_zero_frames_thresh and
                not cam_state.response_sent and
                camera_id in self.latest_detections):
                self.logger.info(
                    f"Camera {camera_id}: Sending first detection response - "
                    f"speed={speed_pix_per_frame:.3f} pix/frame, "
                    f"count={cam_state.speed_near_zero_frames}"
                )
                self._send_first_detection_response(camera_id, self.latest_detections[camera_id], frame)
                cam_state.speed_near_zero_frames = 0
        else:
            if cam_state.speed_near_zero_frames > 0:
                self.logger.debug(
                    f"Camera {camera_id}: Speed not near zero - "
                    f"speed={speed_pix_per_frame:.3f} pix/frame > threshold={speed_near_zero_thresh}, "
                    f"resetting count from {cam_state.speed_near_zero_frames}"
                )
            cam_state.speed_near_zero_frames = 0

        # Check if speed threshold reached (for stopping tracking)
        if abs(speed_pix_per_frame) > speed_thresh and cam_state.response_sent:
          
            self.logger.info(
                    f"Camera {camera_id}: Speed threshold reached "
                    f"({speed_pix_per_frame:.3f} pix/frame > {speed_thresh} pix/frame). "
            )
            self.logger.info(f"Camera {camera_id}: AGV가 다시 움직이기 시작하였으므로 다음 카메라의 Tracking Loop를 시작합니다.") 
            self._start_next_camera_after_1_3(camera_id)
            return False  # Break tracking loop

        return True  # Continue tracking

    def _start_next_camera_after_1_3(self, camera_id: int):
        """Start next camera after camera 1 or 3 finishes tracking."""
        if camera_id == 1:
            # Camera 1 -> Start camera 2
            self.logger.info("Camera 1: Starting camera 2 tracking loop.")
            if 2 not in self.tracking_threads or not self.tracking_threads[2].is_alive():
                product_model_name = self.model_config.get_selected_model()
                if self._ensure_camera_initialized(2, product_model_name):
                    self.camera2_trajectory.clear()
                    cam2_state = self.camera_state_manager.get_or_create(2)
                    cam2_state.reset_detection_loss()
                    self.camera2_trajectory_sent = False
                    self.tracking_manager.start_tracking(2)

        elif camera_id == 3:
            # Camera 3 -> Start camera 1 (cycle)
            self.logger.info("Camera 3: Starting camera 1 tracking loop again (infinite cycle).")
            self._reset_camera_state(1)
            if 1 not in self.tracking_threads or not self.tracking_threads[1].is_alive():
                if 1 in self.camera_loaders and 1 in self.trackers:
                    self.tracking_manager.start_tracking(1)
                    self.logger.info("Camera 1 tracking thread started (cycle restarted)")
                else:
                    self.logger.warning("Camera 1 not initialized, cannot start tracking")

    def _handle_camera_2_tracking(
        self,
        camera_id: int,
        trackers: Dict[int, KalmanTracker],
        tracking_results: List[Dict],
        has_detection: bool,
        vis_frame_original: np.ndarray,
        detections: List[Detection],
        cam_state: CameraState
    ) -> bool:
        """Handle camera 2 specific tracking logic.

        Returns:
            True if tracking should continue, False if should break.
        """
        detection_loss_thresh = self.tracking_config.detection_loss_threshold_frames
        camera2_trajectory_max_frames = self.tracking_config.camera2_trajectory_max_frames

        if tracking_results:
            # Try to get tracker from EnhancedAMRTracker first, then fallback to trackers dict
            amr_tracker = self.amr_trackers.get(camera_id)
            if amr_tracker and amr_tracker.tracker and amr_tracker.track_id is not None:
                tracker = amr_tracker.tracker
            else:
                tracker = next(iter(trackers.values())) if trackers else None
            
            if tracker:
                kf_state = tracker.kf.statePost.flatten()
                pixel_size = self.camera_manager.get_pixel_size(camera_id)
                x_mm = kf_state[0] * pixel_size
                y_mm = kf_state[1] * pixel_size
                rz_deg = kf_state[2]

                trajectory_index = len(self.camera2_trajectory)
                self.camera2_trajectory.append({
                    "track_idx": trajectory_index,
                    "x": round(float(x_mm), 3),
                    "y": round(float(y_mm), 3),
                    "rz": round(float(rz_deg), 3)
                })

        # Check if detection lost
        if not has_detection:
            cam_state.increment_detection_loss()
        else:
            # Reset detection loss counter when detection is found
            cam_state.reset_detection_loss()
        
        # Log Camera 2 tracking status periodically
        self.logger.info(
            f"Camera 2: Trajectory tracking - "
            f"trajectory_frames={len(self.camera2_trajectory)}/{camera2_trajectory_max_frames}, "
            f"has_detection={has_detection}, "
            f"detection_loss_frames={cam_state.detection_loss_frames}/{detection_loss_thresh}"
        )

        # Check if should send trajectory data
        end_tracking = False
        reason = ""
        if not has_detection and cam_state.detection_loss_frames >= detection_loss_thresh and len(self.camera2_trajectory) > 0:
            end_tracking = True
            reason = f"detection lost for {cam_state.detection_loss_frames} frames (>= {detection_loss_thresh})"
        elif len(self.camera2_trajectory) >= camera2_trajectory_max_frames:
            end_tracking = True
            reason = f"trajectory reached {len(self.camera2_trajectory)} frames (>= {camera2_trajectory_max_frames})"

        if end_tracking:
            if self._send_camera2_trajectory(
                camera_id, 
                frame=vis_frame_original, 
                detections=detections, 
                tracking_results=tracking_results,
                reason=reason
            ):
                cam_state.reset_detection_loss()
                self.logger.info("Camera 2: Tracking loop exiting after sending trajectory.")
                return False  # Break tracking loop

        return True  # Continue tracking

    def _send_camera2_trajectory(
        self,
        camera_id: int,
        frame: Optional[np.ndarray],
        detections: List[Detection],
        tracking_results: List[Dict],
        reason: str
    ) -> bool:
        """Send Camera 2 trajectory data to client (callback for TrackingManager).
        
        Args:
            camera_id: Camera ID (should be 2)
            frame: Frame for result image
            detections: Detections for result image
            tracking_results: Tracking results for result image
            reason: Reason for sending trajectory
            
        Returns:
            True if trajectory was sent, False otherwise
        """
        if camera_id != 2:
            return False
        
        if len(self.camera2_trajectory) == 0 or self.camera2_trajectory_sent:
            return False
        
        self.camera2_trajectory_sent = True
        self.logger.info(f"Camera 2: {reason}. Sending trajectory data to client ({len(self.camera2_trajectory)} frames).")
        
        # Save result image (frame can be None, save_result_image will try to read from loader)
        result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
        self.logger.info(f"Camera {camera_id}: Saving result image to {result_image_path} (frame is {'not None' if frame is not None else 'None'})")
        self.response_builder.save_result_image(
            camera_id, 
            result_image_path, 
            frame=frame, 
            detections=detections or [], 
            tracking_results=tracking_results or []
        )
        
        # Send trajectory data
        trajectory_data = list(self.camera2_trajectory)
        cmd = Command.START_CAM_2
        if self._send_response_to_client(cmd, success=True, data=trajectory_data):
            self.logger.info(f"Camera 2 trajectory data sent ({len(trajectory_data)} frames)")
        
        self.camera2_trajectory.clear()
        
        # Start camera 3 if not use_area_scan
        if not self.use_area_scan:
            self.logger.info(f"Camera {camera_id}: AGV가 카메라 상에서 벗어났으므로 다음 카메라의 Tracking Loop를 시작합니다.")
            self._start_camera_3_after_2()
        
        return True

    def _start_camera_3_after_2(self):
        """Start camera 3 after camera 2 finishes tracking."""
        self.logger.info("Camera 2: Starting camera 3 tracking loop.")
        if 3 not in self.tracking_threads or not self.tracking_threads[3].is_alive():
            self._reset_camera_state(3)
            if 3 in self.camera_loaders and 3 in self.trackers:
                self.tracking_manager.start_tracking(3)
            else:
                self.logger.warning("Camera 3 not initialized, cannot start tracking")

    def _send_first_detection_response(self, camera_id: int, detection: Detection, frame: np.ndarray):
        """Send first detection response for cameras 1, 3 (use_area_scan=false)."""
        cam_state = self.camera_state_manager.get(camera_id)
        if cam_state and cam_state.response_sent:
            return  # Already sent
        
        # Build response using ResponseBuilder
        response_data = self.response_builder.build_first_detection_response(camera_id, detection, frame)
        
        # Send response
        cmd = Command.START_CAM_1 + camera_id - 1
        if self._send_response_to_client(cmd, success=True, data=response_data):
            self.logger.info(f"Camera {camera_id}: First detection response sent")
            if cam_state:
                cam_state.response_sent = True
    
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
                        except (json.JSONDecodeError, UnicodeDecodeError):
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
            except OSError:
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
            
            # Get all configurations from model_config (consistent interface)
            detector_config = self.model_config.get_detector_config(product_model_name)
            tracking_config = self.model_config.get_tracking_config(
                product_model_name=product_model_name,
                main_config_tracking=self.config.tracking if self.config and self.config.tracking else None
            )
            calibration_data = self.model_config.get_calibration_config(
                product_model_name=product_model_name,
                main_config_calibration=self.config.calibration if self.config and self.config.calibration else None
            )
            
            # Get execution config settings
            image_undistortion = False
            product_model_config = load_product_model_config(product_model_name)
            if product_model_config and "execution" in product_model_config:
                exec_config = product_model_config["execution"]
                image_undistortion = exec_config.get("image_undistortion", False)
                
                # Update result paths from execution config
                if "result_base_path" in exec_config:
                    self.result_base_path = Path(exec_config["result_base_path"])
                    self.result_base_path.mkdir(parents=True, exist_ok=True)
                    self.response_builder.result_base_path = self.result_base_path
                    self.logger.info(f"Result base path set to: {self.result_base_path}")
                
                if "summary_base_path" in exec_config:
                    self.summary_base_path = Path(exec_config["summary_base_path"])
                    self.summary_base_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Summary base path set to: {self.summary_base_path}")
                
                if "debug_base_path" in exec_config:
                    self.debug_base_path = Path(exec_config["debug_base_path"])
                    self.debug_base_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Debug base path set to: {self.debug_base_path}")
                    # Update ResponseBuilder's debug_base_path
                    self.response_builder.debug_base_path = self.debug_base_path
            
            # Add image_undistortion to calibration_data so it can be passed to loaders
            if calibration_data:
                calibration_data["enable_undistortion"] = image_undistortion
            
            # Get detector type and model path from detector config
            detector_type = detector_config.get("detector_type", "yolo")
            self.logger.info(f"Detector config: {detector_config}")
            
            # Model path is only required for YOLO detector
            if detector_type == "yolo":
                model_path_str = detector_config.get("model_path")
                if not model_path_str:
                    raise ValueError("model_path is required for YOLO detector")
                self.model_path = Path(model_path_str)
                
                if not self.model_path.exists():
                    raise FileNotFoundError(
                        f"Model file not found: {self.model_path} "
                        f"(product model: {product_model_name})"
                    )
            else:
                # Binary detector doesn't need model_path
                self.model_path = None
            
            # Update use_area_scan
            if "use_area_scan" in request:
                self.use_area_scan = bool(request["use_area_scan"])
                # Update TrackingManager
                self.tracking_manager.use_area_scan = self.use_area_scan
            
            # Store configs for EnhancedAMRTracker initialization
            # All components (detector, tracker, calibration) will be initialized by EnhancedAMRTracker._initialize_components()
            self.detector_config = detector_config
            self.calibration_config = calibration_data  # Store calibration config for EnhancedAMRTracker
            
            # Store tracking config
            self.tracking_config = tracking_config
            # Update camera2_trajectory maxlen and share with TrackingManager
            self.camera2_trajectory = deque(maxlen=self.tracking_config.camera2_trajectory_max_frames * 2)
            self.tracking_manager.camera2_trajectory = self.camera2_trajectory
            self.tracking_manager.camera2_trajectory_sent = False
            
            self.vision_active = True
            # Update TrackingManager
            self.tracking_manager.set_vision_active(True)
            
            self.logger.info(f"Vision started with product model: {product_model_name}")
            self.logger.info(f"  Model file: {self.model_path}")
            self.logger.info(f"  Tracking config loaded: {tracking_config}")
            if calibration_data:
                self.logger.info(f"  Calibration config loaded: {calibration_data.get('calibration_data_path', 'N/A')}")
                if calibration_data.get('enable_undistortion', False):
                    self.logger.info(f"  Image undistortion: ENABLED")
                else:
                    self.logger.info(f"  Image undistortion: DISABLED")
            
            # Pre-load pixel_sizes and distance_map_paths for all cameras from preset (efficient - done once at initialization)
            exec_config = self.config.execution if self.config and hasattr(self.config, 'execution') and self.config.execution else {}
            preset_name = self.preset_name or exec_config.get("use_preset")
            self.camera_manager.load_camera_pixel_sizes(preset_name, product_model_name)
            self.camera_manager.load_camera_distance_map_paths(preset_name, product_model_name)
            
            # Log loaded distance map paths
            for camera_id in [1, 2, 3]:
                distance_map_path = self.camera_manager.get_distance_map_path(camera_id)
                if distance_map_path:
                    # Try to load distance map to get info
                    try:
                        from scripts.pixel_distance_mapper import PixelDistanceMapper
                        distance_map_data = PixelDistanceMapper.load_distance_map(distance_map_path)
                        if distance_map_data:
                            image_shape = distance_map_data.get('image_shape', 'unknown')
                            reference_world = distance_map_data.get('reference_world', [1, 1])
                            self.logger.info(f"Camera {camera_id}: Using distance_map_path={distance_map_path}")
                            self.logger.info(f"  Image shape: {image_shape}, Reference point: ({reference_world[0]:.2f}, {reference_world[1]:.2f}) mm")
                        else:
                            self.logger.warning(f"Camera {camera_id}: Failed to load distance_map_path={distance_map_path}")
                    except Exception as e:
                        self.logger.warning(f"Camera {camera_id}: Error loading distance_map_path={distance_map_path}: {e}")
                else:
                    pixel_size = self.camera_manager.get_pixel_size(camera_id)
                    self.logger.info(f"Camera {camera_id}: Using pixel_size={pixel_size} (no distance_map_path)")
             
            # Load visualize_stream from product model config (execution.visualize_stream)
            product_model_config = load_product_model_config(product_model_name)
            if product_model_config and "execution" in product_model_config:
                exec_config_from_product = product_model_config["execution"]
                if "visualize_stream" in exec_config_from_product:
                    self.visualize_stream = exec_config_from_product["visualize_stream"]
                    # Update TrackingManager
                    self.tracking_manager.visualize_stream = self.visualize_stream
                    self.logger.info(f"visualize_stream set to {self.visualize_stream} from {product_model_name}.json")
                else:
                    self.logger.debug(f"visualize_stream not found in {product_model_name}.json, using default: {self.visualize_stream}")
            else:
                self.logger.debug(f"execution config not found in {product_model_name}.json, using default: {self.visualize_stream}")
            
            # Initialize all 3 cameras and initialize trackers (without starting tracking threads)
            for cam_id in [1, 2, 3]:
                try:
                    self.logger.info(f"Initializing camera {cam_id}...")
                    
                    loader_mode, source, fps, config_path = self.camera_manager.get_camera_config(cam_id, product_model_name)
                    
                    # Initialize camera loader
                    self._initialize_camera(cam_id, loader_mode=loader_mode, source=source, fps=fps, camera_config_path=config_path)
                    
                    # Check connection and send NOTIFY_CONNECTION
                    is_connected = self.camera_manager.check_camera_connection(cam_id)
                    if is_connected:
                        self._send_notification(cam_id, True)
                        self.logger.info(f"Camera {cam_id} connection confirmed - NOTIFY_CONNECTION sent")
                    else:
                        self._send_notification(cam_id, False, error_code="CONNECTION_FAILED", error_desc="Camera connection check failed")
                        self.logger.warning(f"Camera {cam_id} connection check failed - NOTIFY_CONNECTION sent")
                    
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
                        self.tracking_manager.start_tracking(1)
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
            # Update TrackingManager
            self.tracking_manager.set_vision_active(False)
            self.logger.info("END VISION command received")
            self._stop_all_cameras()
            # Reset all camera states
            self.camera_state_manager.reset_all()

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
                    self.tracking_manager.start_tracking(camera_id)
                    time.sleep(0.1)
            else:
                # use_area_scan is false: tracking threads are started automatically
                # Camera 1: already started in START_VISION
                # Camera 2: will be started automatically when camera 1 stops
                # Camera 3: will be started automatically when camera 2 stops
                # Only start tracking thread for cameras 1 and 3 if not already running
                if camera_id in [1, 3]:
                    if camera_id not in self.tracking_threads or not self.tracking_threads[camera_id].is_alive():
                        self.tracking_manager.start_tracking(camera_id)
                        time.sleep(0.1)
                # Camera 2: do not start tracking thread here (will be started when cameras 1/3 stop)
            
            # Handle response based on use_area_scan
            if self.use_area_scan:
                # use_area_scan is true: client will send requests periodically
                # Just respond to this request
                data = self.response_builder.get_tracking_data(camera_id, self.use_area_scan)
                result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
                self.response_builder.save_result_image(camera_id, result_image_path)
                
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
    
    def _initialize_camera(self, camera_id: int, loader_mode: str = "camera", source = None, fps: float = 30.0, camera_config_path: Optional[str] = None):
        """Initialize camera and tracker."""
        # Get detector and calibration configs (stored during START_VISION)
        detector_config = getattr(self, 'detector_config', {})
        calibration_config = getattr(self, 'calibration_config', None)
        model_path = getattr(self, 'model_path', None)
        
        # For binary detector, model_path can be None
        detector_type = detector_config.get("detector_type", "yolo")
        if detector_type == "binary" and model_path is None:
            # Binary detector doesn't need model_path, this is OK
            pass
        elif detector_type == "yolo" and model_path is None:
            raise ValueError("model_path is required for YOLO detector")
        
        # Delegate to CameraManager
        self.camera_manager.initialize_camera(
            camera_id=camera_id,
            loader_mode=loader_mode,
            source=source,
            fps=fps,
            model_path=model_path,
            detector_config=detector_config,
            camera_config_path=camera_config_path
        )
        
        self.logger.info(f"Camera {camera_id} initialized with EnhancedAMRTracker")
    
    
    
    def _handle_calc_result(self, request: Dict[str, Any]) -> bytes:
        """Handle CALC RESULT command.
        
        Request format:
        {
            "cmd": 6,
            "path_csv": "data/20251118-154122_zoom1_raw_data.csv"
        }
        
        Response format:
        {
            "cmd": 6,
            "success": bool,
            "error_code": string (optional),
            "error_desc": string (optional),
            "data": {
                "summary_csv": string,
                "cam2_detailed_csv": string,
                "cam1_measurements_csv": string,
                "cam3_measurements_csv": string,
                "analysis_image": string
            }
        }
        """
        try:
            # Get CSV path from request
            path_csv = request.get("path_csv")
            if not path_csv:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="MISSING_PARAM",
                    error_desc="path_csv parameter is required"
                )
            
            # Validate CSV file exists
            csv_path = Path(path_csv)
            if not csv_path.exists():
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="FILE_NOT_FOUND",
                    error_desc=f"CSV file not found: {path_csv}"
                )
            
            # Get sampling interval from request (optional, default 20.0)
            #sampling_interval_mm = request.get("sampling_interval_mm", 20.0)
            
            # Use result_base_path as output directory
            output_dir = str(self.summary_base_path)
            
            self.logger.info(f"Starting trajectory repeatability analysis: {path_csv}")
            
            # Initialize analyzer
            analyzer = TrajectoryRepeatability(str(csv_path))
            
            # Run analysis
            analyzer.run_analysis(sampling_interval_mm=20.0)
            
            # Save results to CSV
            csv_paths = analyzer.save_results_to_csv(output_dir=output_dir)
            
            # Generate plot
            analyzer.plot_results(output_dir=output_dir)
            # analysis_image_path = str(Path(output_dir) / "repeatability_analysis.png")
            
            # self.logger.info(f"Trajectory repeatability analysis completed. Results saved to {output_dir}")
            
            # Prepare response data
            response_data = {
                "summary": None,
            }
            #response_data = {}
            
            return self.protocol.create_response(
                Command.CALC_RESULT,
                success=True,
                data=response_data
            )
        except Exception as e:
            self.logger.error(f"Error in trajectory repeatability analysis: {e}")
            import traceback
            traceback.print_exc()
            return self.protocol.create_response(
                Command.CALC_RESULT,
                success=False,
                error_code="CALC_ERROR",
                error_desc=str(e)
            )
    
    
    
    def _stop_all_cameras(self):
        """Stop all camera tracking."""
        self.logger.info("Stopping all cameras and tracking threads...")
        
        # vision_active is already set to False in _handle_end_vision
        # This will cause tracking loops to exit (while self.vision_active and ...)
        
        # Stop all tracking threads via TrackingManager
        self.tracking_manager.stop_all_tracking(timeout=2.0)
        
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
            self.camera_manager.release_camera(camera_id)
        
        # Clear tracking threads dictionary
        self.tracking_threads.clear()
        self.periodic_response_threads.clear()
        
        # Release all cameras via CameraManager
        self.camera_manager.release_all_cameras()
        
        self.logger.info("All cameras stopped")
    
    
    
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

