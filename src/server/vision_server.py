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
from src.utils.trajectory_repeatability import TrajectoryRepeatability, ManualRepeatability
from src.utils.config_loader import (
    load_product_model_config,
    get_camera_config,
    get_camera_pixel_sizes,
    get_camera_homographies,
    get_camera_tracker_config,
    load_tracking_config,
    load_camera_tracking_config,
    load_camera_detector_config,
    load_calibration_config,
    get_execution_config,
)
from src.utils.image_utils import (
    draw_trajectory_on_frame,
    transform_detection_with_homography,
    transform_tracking_result_with_homography,
    transform_trajectory_data_with_homography,
    warp_frame_with_homography,
    save_image,
)
# Config classes removed - all configs loaded directly json and tracker_config files



# Constants
LOADER_MODE_MAP = {
    "video": "video_file",
    "sequence": "image_sequence",
    "camera": "camera_device"
}

# Novitec 수동: 데몬/워커 서브프로세스 경로 (camera_manager 구현은 유지).
# False — 미사용: CAM1 스트림 상시 유지 전제. True — START VISION 시 데몬 기동·스트림 OFF 시 워커 캡처.
USE_NOVITEC_MANUAL_SUBPROCESS = False

# No default tracking config - must be loaded from tracker_config file


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
        # 중복 출력 방지: root로 전파되지 않도록 (main.py basicConfig와 이 로거가 둘 다 출력하던 문제)
        self.logger.propagate = False
        
        # Log configuration will be set after config is loaded
        self.log_base_path = None
        self.log_level = "DEBUG"
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
        self.model_config = ModelConfig(config_path="config/model_config.json", weights_path="weights")
        
        # Load selected model's configuration and apply to SystemConfig
        selected_model = self.model_config.get_selected_model()
        if selected_model:
            self.logger.info(f"Loading configuration from selected model: {selected_model}")
            # Load product model config file
            product_model_config = load_product_model_config(selected_model)
            if product_model_config:
                # Load configs - tracking config is loaded per camera from tracker_config files
                # No global tracking_config needed
                self.tracking_config = None
                calibration_data = load_calibration_config(selected_model, None)
                execution_config = get_execution_config(selected_model, None)
                
                # Apply execution config settings
                if execution_config:
                    # Set visualize_stream
                    if "visualize_stream" in execution_config:
                        self.visualize_stream = bool(execution_config["visualize_stream"])
                        self.logger.info(f"Visualize stream: {self.visualize_stream}")
                    else:
                        self.visualize_stream = True  # Default
                    
                    # Set draw_masks
                    if "draw_masks" in execution_config:
                        self.draw_masks = bool(execution_config["draw_masks"])
                        self.logger.info(f"Draw masks: {self.draw_masks}")
                    else:
                        self.draw_masks = False  # Default
                    
                    # Set result paths
                    if "result_base_path" in execution_config:
                        self.result_base_path = Path(execution_config["result_base_path"])
                        self.result_base_path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Result base path: {self.result_base_path}")
                    else:
                        self.result_base_path = Path("C:/CMES_AI/Result")
                        self.result_base_path.mkdir(parents=True, exist_ok=True)
                    
                    if "summary_base_path" in execution_config:
                        self.summary_base_path = Path(execution_config["summary_base_path"])
                        self.summary_base_path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Summary base path: {self.summary_base_path}")
                    else:
                        self.summary_base_path = Path("C:/CMES_AI/Summary")
                        self.summary_base_path.mkdir(parents=True, exist_ok=True)
                    
                    if "debug_base_path" in execution_config:
                        self.debug_base_path = Path(execution_config["debug_base_path"])
                        self.debug_base_path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Debug base path: {self.debug_base_path}")
                    else:
                        self.debug_base_path = Path("C:/CMES_AI/Debug")
                        self.debug_base_path.mkdir(parents=True, exist_ok=True)
                    
                    # Set log configuration
                    if "log_base_path" in execution_config:
                        self.log_base_path = Path(execution_config["log_base_path"])
                        self.log_base_path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Log base path: {self.log_base_path}")
                    else:
                        self.log_base_path = Path("C:/CMES_AI/Log")
                        self.log_base_path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Log base path: {self.log_base_path}")
                    
                    if "log_level" in execution_config:
                        self.log_level = execution_config["log_level"].upper()
                    else:
                        self.log_level = "DEBUG"
                    
                    # Setup file logging
                    self._setup_file_logging()
                else:
                    # Default paths if execution config not found
                    self.visualize_stream = True  # Default
                    self.draw_masks = False  # Default
                    self.result_base_path = Path("C:/CMES_AI/Result")
                    self.result_base_path.mkdir(parents=True, exist_ok=True)
                    self.summary_base_path = Path("C:/CMES_AI/Summary")
                    self.summary_base_path.mkdir(parents=True, exist_ok=True)
                    self.debug_base_path = Path("C:/CMES_AI/Debug")
                    self.debug_base_path.mkdir(parents=True, exist_ok=True)
                    self.log_base_path = Path("C:/CMES_AI/Log")
                    self.log_base_path.mkdir(parents=True, exist_ok=True)
                    self.log_level = "DEBUG"
                    # Setup file logging
                    self._setup_file_logging()
            else:
                # Default paths if config file not found
                self.visualize_stream = True  # Default
                self.draw_masks = False  # Default
                self.result_base_path = Path("C:/CMES_AI/Result")
                self.result_base_path.mkdir(parents=True, exist_ok=True)
                self.summary_base_path = Path("C:/CMES_AI/Summary")
                self.summary_base_path.mkdir(parents=True, exist_ok=True)
                self.debug_base_path = Path("C:/CMES_AI/Debug")
                self.debug_base_path.mkdir(parents=True, exist_ok=True)
                self.log_base_path = Path("C:/CMES_AI/Log")
                self.log_base_path.mkdir(parents=True, exist_ok=True)
                self.log_level = "DEBUG"
                self.tracking_config = None
                self.logger.warning(f"Failed to load config file for model: {selected_model}")
                # Setup file logging
                self._setup_file_logging()
        else:
            # Default paths if no model selected
            self.visualize_stream = True  # Default
            self.draw_masks = False  # Default
            self.result_base_path = Path("C:/CMES_AI/Result")
            self.result_base_path.mkdir(parents=True, exist_ok=True)
            self.summary_base_path = Path("C:/CMES_AI/Summary")
            self.summary_base_path.mkdir(parents=True, exist_ok=True)
            self.debug_base_path = Path("C:/CMES_AI/Debug")
            self.debug_base_path.mkdir(parents=True, exist_ok=True)
            self.log_base_path = Path("C:/CMES_AI/Log")
            self.log_base_path.mkdir(parents=True, exist_ok=True)
            self.log_level = "DEBUG"
            self.tracking_config = None
            self.logger.warning("No model selected in model_config.json")
            # Setup file logging
            self._setup_file_logging()
        
        # System state
        self.vision_active = False
        self.use_area_scan = False
        # visualize_stream is set above from config file (or default True)
        
        # Camera state manager - centralized state management for all cameras
        self.camera_state_manager = CameraStateManager()

        # Tracking configuration is loaded from tracker_config files per camera
        # No global tracking_config needed - each camera loads its own from tracker_config file

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
        self.tracking_manager.on_camera_3_trajectory = self._send_camera3_trajectory

        # Share camera2/camera3 trajectory with TrackingManager
        # This will be initialized in START_VISION
        self.camera2_trajectory = None
        self.camera2_trajectory_sent = False
        self.camera3_trajectory = None
        self.camera3_trajectory_sent = False
        
        # Protocol handler
        self.protocol = ProtocolHandler()
        
        self.response_builder = ResponseBuilder(
            camera_manager=self.camera_manager,
            tracking_manager=self.tracking_manager,
            result_base_path=self.result_base_path,
            debug_base_path=self.debug_base_path,
            protocol_handler=self.protocol
        )
    
    def _setup_file_logging(self):
        """Setup file logging based on config."""
        # Set log level for console handler (if exists)
        log_level = getattr(logging, self.log_level, logging.DEBUG)
        
        # Suppress verbose logs from third-party libraries
        logging.getLogger("numba").setLevel(logging.WARNING)
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        
        # Update console handler level if it exists
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)
        
        # Update logger level
        self.logger.setLevel(log_level)
        
        # Setup file logging if log_base_path is set
        if self.log_base_path is None:
            return
        
        try:
            from datetime import datetime
            # Create log filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_base_path / f"vision_server_{timestamp}.log"
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            
            # Add file handler to root logger to capture all logs
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(log_level)
            
            # Also add to this logger
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"File logging enabled: {log_file} (level: {self.log_level})")
        except Exception as e:
            self.logger.warning(f"Failed to setup file logging: {e}")
        
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
        
        # Get pixel_size for this specific camera (dict with x, y, average)
        pixel_size = self.camera_manager.get_pixel_size_dict(camera_id)
        
        # Get max_frames_lost from tracking_config (dict or None)
        max_frames_lost = self.tracking_config.get('max_frames_lost', 500) if isinstance(self.tracking_config, dict) else 500
        
        return KalmanTracker(
            fps=fps,
            pixel_size=pixel_size,
            track_id=track_id,
            max_frames_lost=max_frames_lost,
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

        ps_dict = self.camera_manager.get_pixel_size_dict(camera_id)
        ps_str = f"x={ps_dict['x']:.4f}, y={ps_dict['y']:.4f}"
        self.logger.info(
            f"Camera {camera_id}: Tracker created (track_id={track_id}, "
            f"fps={fps:.2f}, pixel_size={ps_str})"
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

        # Get tracking config thresholds (dict or None)
        if isinstance(self.tracking_config, dict):
            speed_near_zero_thresh = self.tracking_config.get('speed_near_zero_threshold', 3.0)
            speed_zero_frames_thresh = self.tracking_config.get('speed_zero_frames_threshold', 20)
            speed_thresh = self.tracking_config.get('speed_threshold_pix_per_frame', 5.0)
        else:
            # Default values if tracking_config not loaded
            speed_near_zero_thresh = 3.0
            speed_zero_frames_thresh = 20
            speed_thresh = 5.0

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
        """Start next camera after camera 1 finishes tracking.

        Note: Camera 3 no longer uses this method (it uses trajectory callback instead).
        """
        if camera_id == 1:
            # Camera 1 -> Start camera 2
            # CAM1 Novitec 스트림은 상시 유지(향후 CAM1 전용 별도 프로세스와 맞추기 위함). 여기서 끊지 않음.
            # self.camera_manager.stop_camera_stream(1)

            if self.use_area_scan:
                self.logger.info(f"Camera {camera_id}: Tracking finished. Waiting for client request (use_area_scan=true).")
                return

            if 2 not in self.tracking_threads or not self.tracking_threads[2].is_alive():
                product_model_name = self.model_config.get_selected_model()
                if self._ensure_camera_initialized(2, product_model_name):
                    self.camera2_trajectory.clear()
                    cam2_state = self.camera_state_manager.get_or_create(2)
                    cam2_state.reset_detection_loss()
                    self.camera2_trajectory_sent = False
                    # Start camera 2 stream before starting tracking via CameraManager
                    self.camera_manager.start_camera_stream(2)
                    self.tracking_manager.start_tracking(2)

        elif camera_id == 3:
            # Camera 3 should use trajectory callback (_send_camera3_trajectory -> _start_camera_1_after_3)
            self.logger.warning("Camera 3: _start_next_camera_after_1_3 called unexpectedly (should use trajectory callback)")
            self.camera_manager.stop_camera_stream(3)

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
        # Get tracking config thresholds (dict or None)
        if isinstance(self.tracking_config, dict):
            detection_loss_thresh = self.tracking_config.get('detection_loss_threshold_frames', 30)
            camera2_trajectory_max_frames = self.tracking_config.get('camera2_trajectory_max_frames', 300)
        else:
            # Default values if tracking_config not loaded
            detection_loss_thresh = 30
            camera2_trajectory_max_frames = 300

        if tracking_results:
            # Try to get tracker from EnhancedAMRTracker first, then fallback to trackers dict
            amr_tracker = self.amr_trackers.get(camera_id)
            if amr_tracker and amr_tracker.tracker and amr_tracker.track_id is not None:
                tracker = amr_tracker.tracker
            else:
                tracker = next(iter(trackers.values())) if trackers else None
            
            if tracker:
                kf_state = tracker.kf.statePost.flatten()
                # Use pixel_size_x and pixel_size_y separately (same as KalmanTracker)
                pixel_size_dict = self.camera_manager.get_pixel_size_dict(camera_id)
                x_pix = kf_state[0]
                y_pix = kf_state[1]
                x_mm = x_pix * pixel_size_dict['x']
                y_mm = y_pix * pixel_size_dict['y']
                rz_deg = kf_state[2]

                self.logger.debug(
                    f"Camera {camera_id}: Tracking - "
                    f"x={x_mm:.3f}mm, y={y_mm:.3f}mm, yaw={rz_deg:.3f}deg, "
                    f"x_pix={x_pix:.1f}, y_pix={y_pix:.1f}"
                )

                # 미초기화/리셋 직후 (0,0) 위치는 trajectory에 넣지 않음 (track_idx 0~3, 99 등 0으로 채워지는 현상 방지)
                if x_pix != 0 or y_pix != 0:
                    trajectory_index = len(self.camera2_trajectory)
                    # Store pixel coordinates only - mm values will be calculated after homography transformation
                    self.camera2_trajectory.append({
                        "track_idx": trajectory_index,
                        "x_pix": round(float(x_pix), 1),  # Store pixel coords for homography transformation
                        "y_pix": round(float(y_pix), 1),  # Store pixel coords for homography transformation
                        "rz": round(float(rz_deg), 3)
                        # x, y (mm) will be calculated after homography transformation in _send_camera2_trajectory
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
        
        # Apply homography transformation at save time only
        homography = self.camera_manager.get_homography(camera_id)
        trajectory_data = list(self.camera2_trajectory)
        
        # Transform trajectory points if homography is available
        if homography is not None and len(trajectory_data) > 0:
            # Use pixel_size_x and pixel_size_y separately
            pixel_size_dict = self.camera_manager.get_pixel_size_dict(camera_id)
            trajectory_data = transform_trajectory_data_with_homography(
                trajectory_data, homography, pixel_size_dict
            )
            self.logger.debug(f"Camera {camera_id}: Applied homography transformation to {len(trajectory_data)} trajectory points")
        
        # Save result image with trajectory drawn from transformed trajectory data
        result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
        try:
            # Get frame to draw on
            if frame is None:
                loader = self.camera_manager.camera_loaders.get(camera_id)
                if loader:
                    ret, frame = loader.read()
            
            if frame is not None:
                # Apply homography to frame
                if homography is not None:
                    frame = warp_frame_with_homography(frame, homography)
                
                # Draw trajectory from transformed trajectory data
                vis_frame = draw_trajectory_on_frame(frame, trajectory_data)
                
                result_image_path.parent.mkdir(parents=True, exist_ok=True)
                success = cv2.imwrite(str(result_image_path), vis_frame)
                if success:
                    self.logger.info(f"Camera {camera_id}: Saved result image with trajectory ({len(trajectory_data)} points) to {result_image_path}")
                else:
                    self.logger.error(f"Camera {camera_id}: Failed to save result image")
            else:
                self.logger.warning(f"Camera {camera_id}: No frame available for result image")
        except Exception as e:
            self.logger.error(f"Camera {camera_id}: Failed to save result image: {e}")
        
        # Remove x_pix, y_pix from response (only x, y in mm and rz are sent)
        response_data = []
        for point in trajectory_data:
            response_point = {
                "track_idx": point.get("track_idx", 0),
                "x": point.get("x", 0),
                "y": point.get("y", 0),
                "rz": point.get("rz", 0)
            }
            response_data.append(response_point)
        
        cmd = Command.START_CAM_2
        if self._send_response_to_client(cmd, success=True, data=response_data):
            self.logger.info(f"Camera 2 trajectory data sent ({len(response_data)} frames)")
        
        self.camera2_trajectory.clear()
        
        # Start camera 3 (or wait for client request if use_area_scan=true)
        self._start_camera_3_after_2()
        
        return True

    def _start_camera_3_after_2(self):
        """Start camera 3 after camera 2 finishes tracking."""
        self.logger.info("Camera 2: Stopping stream.")
        # Stop camera 2 stream via CameraManager
        self.camera_manager.stop_camera_stream(2)

        if self.use_area_scan:
            self.logger.info(f"Camera 2: Tracking finished. Waiting for client request (use_area_scan=true).")
            return
        
        self.logger.info("Camera 3: Starting camera 3 stream.")
        if 3 not in self.tracking_threads or not self.tracking_threads[3].is_alive():
            self._reset_camera_state(3)
            # Clear camera 3 trajectory for new tracking session
            if self.camera3_trajectory is not None:
                self.camera3_trajectory.clear()
            self.camera3_trajectory_sent = False
            if 3 in self.camera_loaders and 3 in self.trackers:
                # Start camera 3 stream before starting tracking via CameraManager
                self.camera_manager.start_camera_stream(3)
                self.tracking_manager.start_tracking(3)
            else:
                self.logger.warning("Camera 3 not initialized, cannot start tracking")

    def _send_camera3_trajectory(
        self,
        camera_id: int,
        frame: Optional[np.ndarray],
        detections: List[Detection],
        tracking_results: List[Dict],
        reason: str
    ) -> bool:
        """Send Camera 3 trajectory data to client (callback for TrackingManager).

        Args:
            camera_id: Camera ID (should be 3)
            frame: Frame for result image
            detections: Detections for result image
            tracking_results: Tracking results for result image
            reason: Reason for sending trajectory

        Returns:
            True if trajectory was sent, False otherwise
        """
        if camera_id != 3:
            return False

        if len(self.camera3_trajectory) == 0 or self.camera3_trajectory_sent:
            return False

        self.camera3_trajectory_sent = True
        self.logger.info(f"Camera 3: {reason}. Sending trajectory data to client ({len(self.camera3_trajectory)} frames).")

        # Apply homography transformation at save time only
        homography = self.camera_manager.get_homography(camera_id)
        trajectory_data = list(self.camera3_trajectory)

        # Transform trajectory points if homography is available
        if homography is not None and len(trajectory_data) > 0:
            # Use pixel_size_x and pixel_size_y separately
            pixel_size_dict = self.camera_manager.get_pixel_size_dict(camera_id)
            trajectory_data = transform_trajectory_data_with_homography(
                trajectory_data, homography, pixel_size_dict
            )
            self.logger.debug(f"Camera {camera_id}: Applied homography transformation to {len(trajectory_data)} trajectory points")

        # Save result image with trajectory drawn from transformed trajectory data
        result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
        try:
            # Get frame to draw on
            if frame is None:
                loader = self.camera_manager.camera_loaders.get(camera_id)
                if loader:
                    ret, frame = loader.read()

            if frame is not None:
                # Apply homography to frame
                if homography is not None:
                    frame = warp_frame_with_homography(frame, homography)

                # Draw trajectory from transformed trajectory data
                vis_frame = draw_trajectory_on_frame(frame, trajectory_data)

                result_image_path.parent.mkdir(parents=True, exist_ok=True)
                success = cv2.imwrite(str(result_image_path), vis_frame)
                if success:
                    self.logger.info(f"Camera {camera_id}: Saved result image with trajectory ({len(trajectory_data)} points) to {result_image_path}")
                else:
                    self.logger.error(f"Camera {camera_id}: Failed to save result image")
            else:
                self.logger.warning(f"Camera {camera_id}: No frame available for result image")
        except Exception as e:
            self.logger.error(f"Camera {camera_id}: Failed to save result image: {e}")

        # Remove x_pix, y_pix from response (only x, y in mm and rz are sent)
        response_data = []
        for point in trajectory_data:
            response_point = {
                "track_idx": point.get("track_idx", 0),
                "x": point.get("x", 0),
                "y": point.get("y", 0),
                "rz": point.get("rz", 0)
            }
            response_data.append(response_point)

        cmd = Command.START_CAM_3
        if self._send_response_to_client(cmd, success=True, data=response_data):
            self.logger.info(f"Camera 3 trajectory data sent ({len(response_data)} frames)")

        self.camera3_trajectory.clear()

        # Start camera 1 (cycle restart)
        self._start_camera_1_after_3()

        return True

    def _start_camera_1_after_3(self):
        """Start camera 1 after camera 3 finishes trajectory tracking."""
        self.logger.info("Camera 3: Stopping stream.")
        self.camera_manager.stop_camera_stream(3)

        if self.use_area_scan:
            self.logger.info("Camera 3: Tracking finished. Waiting for client request (use_area_scan=true).")
            return

        # Prepare all cameras for next cycle (advance list sources, reset video loaders)
        cycle_results = self.camera_manager.prepare_all_cameras_for_next_cycle()
        if cycle_results:
            for cam_id, success in cycle_results.items():
                info = self.camera_manager.get_video_source_info(cam_id)
                if info:
                    self.logger.info(f"Camera {cam_id}: Cycle advanced to source [{info['current_index']}/{info['total']}]: {info['current_source']}")
                elif success:
                    self.logger.info(f"Camera {cam_id}: Video loader reset for next cycle")

        self.logger.info("Camera 3: Starting camera 1 stream.")
        self._reset_camera_state(1)
        if 1 not in self.tracking_threads or not self.tracking_threads[1].is_alive():
            if 1 in self.camera_loaders and 1 in self.trackers:
                self.camera_manager.start_camera_stream(1)
                self.tracking_manager.start_tracking(1)
                self.logger.info("Camera 1 tracking thread started (cycle restarted)")
            else:
                self.logger.warning("Camera 1 not initialized, cannot start tracking")

    def _send_first_detection_response(self, camera_id: int, detection: Detection, tracking_result: Dict, frame: np.ndarray):
        """Send first detection response for camera 1 (use_area_scan=false).
        
        Args:
            camera_id: Camera ID
            detection: Detection object
            tracking_result: Kalman filtered tracking result with position and orientation
            frame: Frame image
        """
        cam_state = self.camera_state_manager.get(camera_id)
        if cam_state and cam_state.response_sent:
            return  # Already sent
        
        # Apply homography transformation at save time only
        # Transform data for both response calculation and image saving
        homography = self.camera_manager.get_homography(camera_id)
        if homography is not None:
            # Transform frame
            frame = warp_frame_with_homography(frame, homography)
            
            # Get transformed frame size for mask extraction
            transformed_h, transformed_w = frame.shape[:2]
            
            # Transform detection (bbox, masks, oriented_box_info)
            # Pass transformed image size and frame so oriented_box_info can be re-extracted and refined
            # Also pass debug_base_path for refinement debug images
            debug_base_path = getattr(self.response_builder, 'debug_base_path', None)
            
            # Get edge refinement config from camera_manager
            edge_config = self.camera_manager.get_edge_refinement_config(camera_id)
            detection = transform_detection_with_homography(
                detection, homography, 
                transformed_image_size=(transformed_w, transformed_h),
                frame=frame,
                debug_base_path=debug_base_path,
                camera_id=camera_id,
                enable_edge_refinement=edge_config["enable"],
                edge_search_range_px=edge_config["search_range_px"]
            )
            
            # Transform tracking result (position, trajectory, bbox)
            if tracking_result:
                tracking_result = transform_tracking_result_with_homography(tracking_result, homography)
            
            self.logger.debug(f"Camera {camera_id}: Applied homography transformation for response")
        
        # Build response using ResponseBuilder with tracking result (Kalman filtered position)
        # Note: tracking_result is already transformed, so response will use transformed coordinates
        response_data = self.response_builder.build_first_detection_response(
            camera_id, detection, tracking_result, frame
        )
        
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
                self.logger.info(f"Request: {json.dumps(request, indent=2, ensure_ascii=False)}")
                
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
                            self.logger.info(f"Response: {json.dumps(response_dict, indent=2, ensure_ascii=False)}")
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
            elif cmd == Command.START_CAM_1_MANUAL:
                return self._handle_start_cam_manual(request)
            elif cmd == Command.MANUAL_CALC_RESULT:
                return self._handle_manual_calc_result(request)
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
            
            # Get execution config settings first (needed for camera-specific config loading)
            image_undistortion = False
            product_model_config = load_product_model_config(product_model_name)
            exec_config = {}
            if product_model_config and "execution" in product_model_config:
                exec_config = product_model_config["execution"]
                image_undistortion = exec_config.get("image_undistortion", False)
                
                # Update ResponseBuilder paths (paths are already set in __init__)
                self.response_builder.result_base_path = self.result_base_path
                self.response_builder.debug_base_path = self.debug_base_path
            
            # Get all configurations from model_config (consistent interface)
            # Note: This is a fallback - actual camera initialization uses camera-specific configs
            detector_config = self.model_config.get_detector_config(product_model_name)
            # Tracking config is loaded per camera from tracker_config files
            # No global tracking_config needed
            tracking_config = None
            calibration_data = self.model_config.get_calibration_config(
                product_model_name=product_model_name,
                main_config_calibration=self.config.calibration if self.config and self.config.calibration else None
            )
            
            # Add image_undistortion to calibration_data so it can be passed to loaders
            if calibration_data:
                calibration_data["enable_undistortion"] = image_undistortion
            
            # Check if camera-specific detector configs exist (for logging)
            main_config_execution = exec_config if isinstance(exec_config, dict) else {}
            preset_name = self.preset_name or main_config_execution.get("use_preset")
            camera_specific_configs_found = False
            for camera_id in [1, 2, 3]:
                camera_detector_config = load_camera_detector_config(
                    camera_id=camera_id,
                    product_model_name=product_model_name,
                    main_config_execution=main_config_execution,
                    preset_name=preset_name
                )
                if camera_detector_config:
                    camera_specific_configs_found = True
                    break
            
            # Get detector type and model path from detector config
            detector_type = detector_config.get("detector_type", "aruco")
            if camera_specific_configs_found:
                self.logger.info(f"Detector config (fallback, camera-specific configs will be used): {detector_config}")
            else:
                self.logger.warning(f"Detector config (using fallback/default values - no camera-specific configs found): {detector_config}")
            
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
            
            # Tracking config is loaded per camera from tracker_config files
            # No global tracking_config needed
            self.tracking_config = None
            # Update camera2/camera3 trajectory maxlen and share with TrackingManager
            # Use default value - actual max_frames comes from camera-specific config
            camera2_trajectory_max_frames = 300  # Default, will be overridden by camera-specific config
            self.camera2_trajectory = deque(maxlen=camera2_trajectory_max_frames * 2)
            self.tracking_manager.camera2_trajectory = self.camera2_trajectory
            self.tracking_manager.camera2_trajectory_sent = False

            camera3_trajectory_max_frames = 300  # Default, will be overridden by camera-specific config
            self.camera3_trajectory = deque(maxlen=camera3_trajectory_max_frames * 2)
            self.tracking_manager.camera3_trajectory = self.camera3_trajectory
            self.tracking_manager.camera3_trajectory_sent = False
            
            self.vision_active = True
            # Update TrackingManager
            self.tracking_manager.set_vision_active(True)
            
            # Store product_model_name for camera-specific config loading
            self.product_model_name = product_model_name
            
            self.logger.info(f"Vision started with product model: {product_model_name}")
            self.logger.info(f"  Model file: {self.model_path}")
            self.logger.info(f"  Tracking config loaded: {tracking_config}")
            if calibration_data:
                self.logger.info(f"Calibration config loaded: {calibration_data.get('calibration_data_path', 'N/A')}")
                if calibration_data.get('enable_undistortion', False):
                    self.logger.info(f"Image undistortion: ENABLED")
                else:
                    self.logger.info(f"Image undistortion: DISABLED")
            
            # Pre-load pixel_sizes and distance_map_paths for all cameras from preset (efficient - done once at initialization)
            exec_config = self.config.execution if self.config and hasattr(self.config, 'execution') and self.config.execution else {}
            preset_name = self.preset_name or exec_config.get("use_preset")
            self.camera_manager.load_camera_pixel_sizes(preset_name, product_model_name)
            self.camera_manager.load_camera_distance_map_paths(preset_name, product_model_name)
            self.camera_manager.load_camera_homographies(preset_name, product_model_name)
            
            # Log loaded homographies
            for camera_id in [1, 2, 3]:
                homography = self.camera_manager.get_homography(camera_id)
                if homography is not None:
                    self.logger.info(f"Camera {camera_id}: Homography loaded (will warp frames)")
            
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
                    pixel_size_dict = self.camera_manager.get_pixel_size_dict(camera_id)
                    self.logger.info(f"Camera {camera_id}: Using pixel_size (x={pixel_size_dict['x']:.6f}, y={pixel_size_dict['y']:.6f}) (no distance_map_path)")
             
            # Load visualize_stream and draw_masks from product model config (execution section)
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
                
                if "draw_masks" in exec_config_from_product:
                    self.draw_masks = exec_config_from_product["draw_masks"]
                    self.logger.info(f"draw_masks set to {self.draw_masks} from {product_model_name}.json")
                else:
                    self.logger.debug(f"draw_masks not found in {product_model_name}.json, using default: {self.draw_masks}")
            else:
                self.logger.debug(f"execution config not found in {product_model_name}.json, using default: {self.visualize_stream}, draw_masks={self.draw_masks}")
            
            # Initialize all 3 cameras and initialize trackers (without starting tracking threads)
            failed_cameras = []
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
                        failed_cameras.append(cam_id)
                    
                    # Create tracker instance (track_id=0) for all cameras
                    # Note: Camera 2 tracker will be used when cameras 1/3 stop
                    self._initialize_camera_tracker(cam_id, fps, track_id=0)
                    
                    self.logger.info(f"Camera {cam_id} initialized successfully")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize camera {cam_id}: {e}")
                    # Send NOTIFY_CONNECTION with error
                    self._send_notification(cam_id, False, error_code="INIT_ERROR", error_desc=str(e))
                    failed_cameras.append(cam_id)
                    # Continue initializing other cameras even if one fails
            
            # Check if any cameras failed
            if failed_cameras:
                if not self.use_area_scan:
                    # For non-area-scan mode, any camera failure should be reported
                    error_msg = f"Failed to initialize cameras: {failed_cameras}. Cannot start tracking."
                    self.logger.error(error_msg)
                    return self.protocol.create_response(
                        Command.START_VISION,
                        success=False,
                        error_code="CAM_INIT_ERROR",
                        error_desc=error_msg
                    )
                else:
                    # For area-scan mode, all cameras can be optional
                    self.logger.warning(f"Some cameras failed to initialize: {failed_cameras}. Continuing with available cameras.")
            
            # All cameras initialized (or at least critical ones)
            elapsed_time = time.time() - start_time
            if failed_cameras:
                self.logger.info(f"Camera initialization completed with {len(failed_cameras)} failures. Total time: {elapsed_time:.3f}s")
            else:
                self.logger.info(f"All cameras initialized. Total time: {elapsed_time:.3f}s")
            
            # Stop all camera streams to ensure clean state (Novitec cameras only)
            # This prevents the last initialized camera from having an active stream
            self.camera_manager.stop_all_camera_streams()
            self.logger.info("All camera streams stopped (clean state)")
            
            # If use_area_scan is false, automatically start camera 1 tracking
            if not self.use_area_scan:
                self.logger.info("use_area_scan=false: Automatically starting camera 1 tracking")
                
                # Check if camera 1 is initialized before starting tracking
                if 1 not in self.trackers or 1 not in self.camera_loaders:
                    error_msg = "Camera 1 not initialized. Cannot start tracking."
                    self.logger.error(error_msg)
                    return self.protocol.create_response(
                        Command.START_VISION,
                        success=False,
                        error_code="CAM_INIT_ERROR",
                        error_desc=error_msg
                    )
                
                # Start camera 1 tracking thread
                if 1 not in self.tracking_threads or not self.tracking_threads[1].is_alive():
                    # Explicitly start camera 1 stream before starting tracking
                    # This ensures camera 1 stream is active and other streams are stopped
                    self.camera_manager.start_camera_stream(1)
                    self.tracking_manager.start_tracking(1)
                    time.sleep(0.1)
            
            if USE_NOVITEC_MANUAL_SUBPROCESS:
                try:
                    if self.camera_manager.ensure_novitec_manual_daemon():
                        self.logger.info(
                            "Novitec manual capture daemon pre-started (persistent worker)"
                        )
                except Exception as e:
                    self.logger.debug(f"Novitec manual daemon pre-start skipped: {e}")

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
            if USE_NOVITEC_MANUAL_SUBPROCESS:
                self.camera_manager.stop_novitec_manual_daemon()
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
                # Only start tracking thread for camera 1 if not already running
                if camera_id == 1:
                    if camera_id not in self.tracking_threads or not self.tracking_threads[camera_id].is_alive():
                        self.tracking_manager.start_tracking(camera_id)
                        time.sleep(0.1)
                elif camera_id == 3:
                    if camera_id not in self.tracking_threads or not self.tracking_threads[camera_id].is_alive():
                        # Clear camera 3 trajectory for new tracking session
                        if self.camera3_trajectory is not None:
                            self.camera3_trajectory.clear()
                        self.camera3_trajectory_sent = False
                        self.tracking_manager.start_tracking(camera_id)
                        time.sleep(0.1)
                # Camera 2: do not start tracking thread here (will be started when camera 1 stops)
            
            # Handle response based on use_area_scan
            if self.use_area_scan:
                # use_area_scan is true: client will send requests periodically
                # Get detection and frame, apply transform + refine before saving
                detection = self.latest_detections.get(camera_id)
                
                # Read frame from camera loader
                frame = None
                loader = self.camera_loaders.get(camera_id)
                if loader is not None:
                    ret, frame = loader.read()
                    if not ret:
                        frame = None
                
                # Apply homography transformation + edge refinement
                homography = self.camera_manager.get_homography(camera_id)
                if detection is not None and homography is not None and frame is not None:
                    frame = warp_frame_with_homography(frame, homography)
                    transformed_h, transformed_w = frame.shape[:2]
                    
                    # Get edge refinement config from camera_manager
                    edge_config = self.camera_manager.get_edge_refinement_config(camera_id)
                    detection = transform_detection_with_homography(
                        detection, homography,
                        transformed_image_size=(transformed_w, transformed_h),
                        frame=frame,
                        debug_base_path=self.response_builder.debug_base_path,
                        camera_id=camera_id,
                        enable_edge_refinement=edge_config["enable"],
                        edge_search_range_px=edge_config["search_range_px"]
                    )
                
                # Get tracking data for response
                data = self.response_builder.get_tracking_data(camera_id, self.use_area_scan)
                
                # Save result image with already refined detection
                result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
                self.response_builder.save_result_image(
                    camera_id, result_image_path,
                    frame=frame,
                    detections=[detection] if detection else None,
                    apply_homography=False  # Already transformed + refined
                )
                
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
        # Get calibration config (stored during START_VISION)
        calibration_config = getattr(self, 'calibration_config', None)
        
        # Get camera-specific detector config from tracker_config file
        product_model_name = getattr(self, 'product_model_name', None)
        exec_config = self.config.execution if self.config and hasattr(self.config, 'execution') and self.config.execution else {}
        main_config_execution = exec_config if isinstance(exec_config, dict) else (exec_config.__dict__ if hasattr(exec_config, '__dict__') else {})
        preset_name = self.preset_name or main_config_execution.get("use_preset")
        
        # Try to load camera-specific detector config
        camera_detector_config = load_camera_detector_config(
            camera_id=camera_id,
            product_model_name=product_model_name,
            main_config_execution=main_config_execution,
            preset_name=preset_name
        )
        
        if camera_detector_config:
            self.logger.info(f"Camera {camera_id}: Using camera-specific detector config from tracker_config file")
            detector_config = camera_detector_config
        else:
            # Fallback to global detector config (stored during START_VISION)
            detector_config = getattr(self, 'detector_config', {})
            self.logger.debug(f"Camera {camera_id}: Using global detector config (no camera-specific config found)")
        
        # Get model_path from camera-specific or global config
        model_path = None
        detector_type = detector_config.get("detector_type", "aruco")
        if detector_type == "yolo":
            model_path_str = detector_config.get("model_path")
            if model_path_str:
                model_path = Path(model_path_str)
                if not model_path.exists():
                    # Fallback to global model_path
                    model_path = getattr(self, 'model_path', None)
            else:
                model_path = getattr(self, 'model_path', None)
        
        # For binary detector, model_path can be None
        if detector_type == "binary" and model_path is None:
            # Binary detector doesn't need model_path, this is OK
            pass
        elif detector_type == "yolo" and model_path is None:
            raise ValueError("model_path is required for YOLO detector")
        
        # Get enable_undistortion json -> execution.image_undistortion
        enable_undistortion = False
        try:
            product_model_config = load_product_model_config(product_model_name)
            if product_model_config and "execution" in product_model_config:
                exec_config = product_model_config["execution"]
                enable_undistortion = exec_config.get("image_undistortion", False)
                self.logger.info(f"Camera {camera_id}: Image undistortion from {product_model_name}.json: {enable_undistortion}")
            else:
                self.logger.debug(f"Camera {camera_id}: No execution section in {product_model_name}.json, undistortion disabled")
        except Exception as e:
            self.logger.warning(f"Camera {camera_id}: Failed to load image_undistortion from {product_model_name}.json: {e}")
        
        # Load camera-specific tracking config from tracker_config file and set to TrackingManager
        camera_tracking_config = load_camera_tracking_config(
            camera_id=camera_id,
            product_model_name=product_model_name,
            main_config_execution=main_config_execution,
            main_config_tracking=None,  # Not used - all configs from tracker_config files
            preset_name=preset_name
        )
        self.tracking_manager.set_camera_tracking_config(camera_id, camera_tracking_config)
        self.logger.info(f"Camera {camera_id}: Loaded tracking config - "
                         f"speed_near_zero={camera_tracking_config.get('speed_near_zero_threshold', 3.0)}, "
                         f"speed_zero_frames={camera_tracking_config.get('speed_zero_frames_threshold', 20)}, "
                         f"speed_threshold={camera_tracking_config.get('speed_threshold_pix_per_frame', 5.0)}")
        
        # Get raw tracker config dict for KalmanTracker (boundary_margin_ratio, etc.)
        raw_tracker_config = get_camera_tracker_config(
            camera_id=camera_id,
            product_model_name=product_model_name,
            main_config_execution=main_config_execution,
            preset_name=preset_name
        )
        tracker_config_dict = raw_tracker_config.get("tracker", {})
        
        # Delegate to CameraManager
        self.camera_manager.initialize_camera(
            camera_id=camera_id,
            loader_mode=loader_mode,
            source=source,
            fps=fps,
            model_path=model_path,
            detector_config=detector_config,
            tracker_config=tracker_config_dict,
            enable_undistortion=enable_undistortion,
            camera_config_path=camera_config_path,
            draw_masks=getattr(self, 'draw_masks', False)
        )
        
        self.logger.info(f"Camera {camera_id} initialized with EnhancedAMRTracker")
    
    
    
    def _handle_calc_result(self, request: Dict[str, Any]) -> bytes:
        """Handle CALC RESULT command.
        
        Request format:
        {
            "cmd": 6,
            "path_csv": "data/20251118-154122_zoom1_raw_data.csv",
            "sampling_interval_mm": 20.0  (optional, default: 20.0)
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
        
        Error codes:
        - MISSING_PARAM: path_csv 파라미터 누락
        - INVALID_PATH: 파일 경로 형식 오류
        - FILE_NOT_FOUND: CSV 파일을 찾을 수 없음
        - INVALID_FORMAT: CSV 파일 형식 오류 (확장자, 인코딩 등)
        - FILE_READ_ERROR: 파일 읽기 오류
        - INVALID_CSV_STRUCTURE: CSV 구조 오류 (필수 컬럼 누락)
        - INSUFFICIENT_DATA: 분석에 필요한 데이터 부족 (최소 2개 trial 필요)
        - OUTPUT_DIR_ERROR: 출력 디렉토리 생성/쓰기 오류
        - CALC_ERROR: 분석 계산 중 오류
        """
        import pandas as pd
        
        try:
            # 1. 필수 파라미터 검증
            path_csv = request.get("path_csv")
            if not path_csv:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="MISSING_PARAM",
                    error_desc="path_csv 파라미터가 필요합니다. 분석할 CSV 파일 경로를 지정해주세요."
                )
            
            # 2. 경로 형식 검증
            if not isinstance(path_csv, str) or len(path_csv.strip()) == 0:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INVALID_PATH",
                    error_desc="path_csv는 유효한 문자열 경로여야 합니다."
                )
            
            csv_path = Path(path_csv)
            
            # 3. 파일 존재 여부 검증
            if not csv_path.exists():
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="FILE_NOT_FOUND",
                    error_desc=f"CSV 파일을 찾을 수 없습니다: {path_csv}"
                )
            
            # 4. 파일 형식 검증 (확장자)
            if csv_path.suffix.lower() != '.csv':
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INVALID_FORMAT",
                    error_desc=f"CSV 파일만 지원됩니다. 제공된 파일: {csv_path.suffix} (확장자: {csv_path.name})"
                )
            
            # 5. 파일 읽기 가능 여부 검증
            try:
                df = pd.read_csv(str(csv_path))
            except pd.errors.EmptyDataError:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INVALID_FORMAT",
                    error_desc=f"CSV 파일이 비어있습니다: {path_csv}"
                )
            except pd.errors.ParserError as e:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INVALID_FORMAT",
                    error_desc=f"CSV 파일 파싱 오류: {str(e)}"
                )
            except UnicodeDecodeError as e:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INVALID_FORMAT",
                    error_desc=f"CSV 파일 인코딩 오류 (UTF-8 권장): {str(e)}"
                )
            except PermissionError:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="FILE_READ_ERROR",
                    error_desc=f"CSV 파일 읽기 권한이 없습니다: {path_csv}"
                )
            except Exception as e:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="FILE_READ_ERROR",
                    error_desc=f"CSV 파일 읽기 오류: {str(e)}"
                )
            
            # 6. CSV 구조 검증 (필수 컬럼 확인). cam_1_x 또는 cam_1_x(mm) 형식 모두 허용
            aliases_cam1 = [['cam_1_x', 'cam_1_y', 'cam_1_rz'], ['cam_1_x(mm)', 'cam_1_y(mm)', 'cam_1_rz(deg)']]
            aliases_cam3 = [['cam_3_x', 'cam_3_y', 'cam_3_rz'], ['cam_3_x(mm)', 'cam_3_y(mm)', 'cam_3_rz(deg)'], ['cam_3_x_0', 'cam_3_y_0', 'cam_3_rz_0']]
            aliases_cam2 = [['cam_2_x_0', 'cam_2_y_0', 'cam_2_rz_0']]  # cam_2는 인덱스만 붙는 형식만 사용

            def _has_columns(aliases_list):
                return any(all(c in df.columns for c in group) for group in aliases_list)

            missing_columns = []
            if not _has_columns(aliases_cam1):
                missing_columns.extend(aliases_cam1[0])
            if not _has_columns(aliases_cam3):
                missing_columns.extend(aliases_cam3[0])
            if not _has_columns(aliases_cam2):
                missing_columns.extend(aliases_cam2[0])

            if missing_columns:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INVALID_CSV_STRUCTURE",
                    error_desc=f"CSV 파일에 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}. "
                              f"필요한 컬럼: cam_1_x/y/rz (또는 cam_1_x(mm), cam_1_y(mm), cam_1_rz(deg)), "
                              f"cam_3 동일, cam_2_x/y/rz_0~N"
                )
            
            # 7. 데이터 양 검증 (최소 2개 trial 필요)
            n_trials = len(df)
            if n_trials < 2:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INSUFFICIENT_DATA",
                    error_desc=f"반복정밀도 분석에는 최소 2개 이상의 trial 데이터가 필요합니다. 현재: {n_trials}개"
                )
            
            # 8. 출력 디렉토리 검증
            output_dir = str(self.summary_base_path)
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                # 쓰기 권한 테스트
                test_file = output_path / ".write_test"
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="OUTPUT_DIR_ERROR",
                    error_desc=f"출력 디렉토리에 쓰기 권한이 없습니다: {output_dir}"
                )
            except Exception as e:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="OUTPUT_DIR_ERROR",
                    error_desc=f"출력 디렉토리 접근 오류: {output_dir}, {str(e)}"
                )
            
            # 9. 샘플링 간격 파라미터 검증 (optional)
            sampling_interval_mm = request.get("sampling_interval_mm", 20.0)
            try:
                sampling_interval_mm = float(sampling_interval_mm)
                if sampling_interval_mm <= 0:
                    return self.protocol.create_response(
                        Command.CALC_RESULT,
                        success=False,
                        error_code="INVALID_PARAM",
                        error_desc=f"sampling_interval_mm은 양수여야 합니다. 현재 값: {sampling_interval_mm}"
                    )
            except (ValueError, TypeError):
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="INVALID_PARAM",
                    error_desc=f"sampling_interval_mm은 숫자여야 합니다. 현재 값: {request.get('sampling_interval_mm')}"
                )
            
            self.logger.info(f"Starting trajectory repeatability analysis: {path_csv}")
            self.logger.info(f"  Trials: {n_trials}, Sampling interval: {sampling_interval_mm}mm")
            
            # 10. 분석 실행
            try:
                analyzer = TrajectoryRepeatability(str(csv_path))
                analyzer.run_analysis(sampling_interval_mm=sampling_interval_mm)
            except ValueError as e:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="CALC_ERROR",
                    error_desc=f"분석 데이터 오류: {str(e)}"
                )
            except Exception as e:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="CALC_ERROR",
                    error_desc=f"분석 실행 중 오류: {str(e)}"
                )
            
            # 11. 결과 저장
            try:
                csv_paths = analyzer.save_results_to_csv(output_dir=output_dir)
                analyzer.plot_results(output_dir=output_dir)
            except Exception as e:
                return self.protocol.create_response(
                    Command.CALC_RESULT,
                    success=False,
                    error_code="OUTPUT_DIR_ERROR",
                    error_desc=f"결과 저장 중 오류: {str(e)}"
                )
            
            self.logger.info(f"Trajectory repeatability analysis completed. Results saved to {output_dir}")
            
            # 12. 응답 데이터 준비
            response_data = {
                "summary": None,
                "output_dir": output_dir,
                "n_trials": n_trials
            }
            
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
                error_code="INTERNAL_ERROR",
                error_desc=f"예상치 못한 내부 오류: {str(e)}"
            )

    def _handle_start_cam_manual(self, request: Dict[str, Any]) -> bytes:
        """Handle START CAM 1 Manual command (cmd: 8).

        Performs a single-shot detection on camera 1.

        Novitec CAM1: If **stream is on**, use the latest tracking frame when available,
        else one in-process capture. If **stream is off** and ``USE_NOVITEC_MANUAL_SUBPROCESS``:
        subprocess worker/daemon (fresh frame). If that flag is False (default), stream off →
        error (CAM1 상시 스트림 유지 전제). Non-Novitec: last_frames if present, else VideoCapture(source).

        Request format:
        {
            "cmd": 8
        }

        Response format:
        {
            "cmd": 8,
            "success": bool,
            "error_code": string (optional),
            "error_desc": string (optional),
            "data": {
                "x": float,    # X position in mm
                "y": float,    # Y position in mm
                "rz": float    # Rotation angle in degrees
            }
        }

        Error codes:
        - VISION_NOT_ACTIVE: Vision system not started
        - CAM_NOT_INITIALIZED: Camera 1 not initialized
        - FRAME_READ_ERROR: Failed to read frame from camera
        - DETECTION_FAILED: No object detected in frame
        - MANUAL_CAM_ERROR: General error during manual detection
        """
        camera_id = 1
        manual_capture = None

        try:
            if not self.vision_active:
                return self.protocol.create_response(
                    Command.START_CAM_1_MANUAL,
                    success=False,
                    error_code="VISION_NOT_ACTIVE",
                    error_desc="Vision system not started. Call START VISION first.",
                    data={"x": 0.0, "y": 0.0, "rz": 0.0}
                )

            product_model_name = self.model_config.get_selected_model()
            try:
                loader_mode, source, fps, config_path = self.camera_manager.get_camera_config(
                    camera_id, product_model_name
                )
            except Exception as e:
                return self.protocol.create_response(
                    Command.START_CAM_1_MANUAL,
                    success=False,
                    error_code="CAM_NOT_INITIALIZED",
                    error_desc=f"Failed to get camera 1 config: {e}",
                    data={"x": 0.0, "y": 0.0, "rz": 0.0}
                )

            frame = None
            is_novitec = (
                loader_mode == "camera" and self.camera_manager.is_novitec_camera(camera_id)
            )

            if is_novitec:
                if self.camera_manager.is_camera_stream_active(camera_id):
                    # 스트리밍 중: 최신 트래킹 프레임 또는 인프로세스 1장
                    frame = self.tracking_manager.last_frames.get(camera_id)
                    if frame is not None:
                        frame = frame.copy()
                        self.logger.info(
                            "Camera 1 Manual: Latest tracking frame (Novitec stream active)"
                        )
                    else:
                        frame = self.camera_manager.grab_novitec_single_frame_from_stream(camera_id)
                        if frame is not None:
                            self.logger.info(
                                "Camera 1 Manual: Novitec in-process single capture "
                                "(stream active, no last_frames yet)"
                            )
                else:
                    # 스트림 꺼짐: last_frames 미사용. 서브프로세스 경로는 플래그로만 (기본 비활성)
                    if USE_NOVITEC_MANUAL_SUBPROCESS:
                        frame = self.camera_manager.grab_novitec_manual_frame_subprocess(
                            camera_id
                        )
                        if frame is not None:
                            self.logger.info(
                                "Camera 1 Manual: Novitec subprocess trigger (stream stopped; "
                                "fresh frame, not last_frames)"
                            )
                    else:
                        self.logger.warning(
                            "Camera 1 Manual: Novitec stream is OFF and subprocess manual "
                            "is disabled (USE_NOVITEC_MANUAL_SUBPROCESS=False). Keep CAM1 streaming."
                        )
                        frame = None
            else:
                frame = self.tracking_manager.last_frames.get(camera_id)
                if frame is not None:
                    frame = frame.copy()
                    self.logger.info(
                        "Camera 1 Manual: Using latest frame from tracking loop"
                    )
                else:
                    self.logger.info(
                        f"Camera 1 Manual: No tracking frame yet, opening capture from {source}"
                    )
                    manual_capture = cv2.VideoCapture(source)
                    if not manual_capture.isOpened():
                        return self.protocol.create_response(
                            Command.START_CAM_1_MANUAL,
                            success=False,
                            error_code="FRAME_READ_ERROR",
                            error_desc=f"Failed to open camera 1 source: {source}",
                            data={"x": 0.0, "y": 0.0, "rz": 0.0}
                        )
                    ret, frame = manual_capture.read()
                    manual_capture.release()
                    manual_capture = None
                    if not ret or frame is None:
                        return self.protocol.create_response(
                            Command.START_CAM_1_MANUAL,
                            success=False,
                            error_code="FRAME_READ_ERROR",
                            error_desc="Failed to read frame from camera 1",
                            data={"x": 0.0, "y": 0.0, "rz": 0.0}
                        )

            if frame is None:
                return self.protocol.create_response(
                    Command.START_CAM_1_MANUAL,
                    success=False,
                    error_code="FRAME_READ_ERROR",
                    error_desc=(
                        "Failed to acquire camera 1 frame (Novitec: keep CAM1 stream on, or enable "
                        "USE_NOVITEC_MANUAL_SUBPROCESS)"
                        if is_novitec
                        else "Failed to acquire camera 1 frame"
                    ),
                    data={"x": 0.0, "y": 0.0, "rz": 0.0}
                )

            # Get detector from AMR tracker (detector is stateless, thread-safe)
            amr_tracker = self.amr_trackers.get(camera_id)
            if amr_tracker is None or amr_tracker.detector is None:
                # Fallback: try to create detector from config
                return self.protocol.create_response(
                    Command.START_CAM_1_MANUAL,
                    success=False,
                    error_code="CAM_NOT_INITIALIZED",
                    error_desc="Camera 1 detector not available",
                    data={"x": 0.0, "y": 0.0, "rz": 0.0}
                )

            # Detect object in frame (detector.detect is stateless)
            detections = amr_tracker.detector.detect(frame)

            if not detections:
                # Save debug image even on failure
                debug_image_path = self.result_base_path / "cam_1_result_manual_no_detection.png"
                cv2.imwrite(str(debug_image_path), frame)
                self.logger.warning(f"No detection - debug image saved to {debug_image_path}")

                return self.protocol.create_response(
                    Command.START_CAM_1_MANUAL,
                    success=False,
                    error_code="DETECTION_FAILED",
                    error_desc="No object detected in camera 1 frame",
                    data={"x": 0.0, "y": 0.0, "rz": 0.0}
                )

            detection = detections[0]

            # NOTE: We do NOT update the Kalman tracker here to keep manual measurement
            # completely independent from the tracking process
            tracking_result = None

            # Apply homography transformation
            homography = self.camera_manager.get_homography(camera_id)
            if homography is not None:
                frame = warp_frame_with_homography(frame, homography)
                transformed_h, transformed_w = frame.shape[:2]

                # Get edge refinement config
                edge_config = self.camera_manager.get_edge_refinement_config(camera_id)
                debug_base_path = getattr(self.response_builder, 'debug_base_path', None)

                detection = transform_detection_with_homography(
                    detection, homography,
                    transformed_image_size=(transformed_w, transformed_h),
                    frame=frame,
                    debug_base_path=debug_base_path,
                    camera_id=camera_id,
                    enable_edge_refinement=edge_config["enable"],
                    edge_search_range_px=edge_config["search_range_px"]
                )

            # Calculate response data
            pixel_size_dict = self.camera_manager.get_pixel_size_dict(camera_id)

            # Get position from detection's oriented_box_info
            if (hasattr(detection, 'oriented_box_info') and
                detection.oriented_box_info is not None and
                "center" in detection.oriented_box_info):
                center = detection.oriented_box_info["center"]
                x_pix, y_pix = center[0], center[1]
            else:
                center = detection.get_center()
                x_pix, y_pix = center[0], center[1]

            x_mm = x_pix * pixel_size_dict['x']
            y_mm = y_pix * pixel_size_dict['y']

            # Get orientation
            if (hasattr(detection, 'oriented_box_info') and
                detection.oriented_box_info is not None and
                "angle" in detection.oriented_box_info):
                rz = detection.oriented_box_info["angle"]
            else:
                rz = detection.get_orientation() or 0.0

            # Create tracking_result for visualization (with mm coordinates)
            tracking_result = {
                "track_id": 0,
                "position": {
                    "x": x_pix,
                    "y": y_pix,
                    "x_mm": x_mm,
                    "y_mm": y_mm
                },
                "orientation": {"theta_deg": rz},
                "bbox": detection.bbox
            }

            # Save result image to cam_1_result_manual.png
            result_image_path = self.result_base_path / "cam_1_result_manual.png"
            self.response_builder.save_result_image(
                camera_id, result_image_path,
                frame=frame,
                detections=[detection],
                tracking_results=[tracking_result],
                apply_homography=False  # Already transformed
            )

            self.logger.info(
                f"Camera 1 Manual: Detection result - "
                f"position: ({x_mm:.2f}, {y_mm:.2f}) mm, yaw: {rz:.2f} deg"
            )

            response_data = {
                "x": round(float(x_mm), 3),
                "y": round(float(y_mm), 3),
                "rz": round(float(rz), 3)
            }

            return self.protocol.create_response(
                Command.START_CAM_1_MANUAL,
                success=True,
                data=response_data
            )

        except Exception as e:
            self.logger.error(f"Error in manual camera 1 detection: {e}")
            import traceback
            traceback.print_exc()
            return self.protocol.create_response(
                Command.START_CAM_1_MANUAL,
                success=False,
                error_code="MANUAL_CAM_ERROR",
                error_desc=str(e),
                data={"x": 0.0, "y": 0.0, "rz": 0.0}
            )
        finally:
            # Ensure manual capture is released even on error
            if manual_capture is not None:
                try:
                    manual_capture.release()
                except Exception:
                    pass

    def _handle_manual_calc_result(self, request: Dict[str, Any]) -> bytes:
        """Handle MANUAL CALC RESULT command (cmd: 9).

        Calculates performance metrics for manual measurements from CSV data.

        Request format:
        {
            "cmd": 9,
            "path_csv": "path/to/manual_measurements.csv"
        }

        Response format:
        {
            "cmd": 9,
            "success": bool,
            "error_code": string (optional),
            "error_desc": string (optional),
            "data": { ... }
        }

        Error codes:
        - MISSING_PARAM: path_csv parameter missing
        - INVALID_PATH: Invalid file path format
        - FILE_NOT_FOUND: CSV file not found
        - INVALID_FORMAT: CSV file format error
        - FILE_READ_ERROR: File read error
        - INVALID_CSV_STRUCTURE: CSV structure error (missing required columns)
        - INSUFFICIENT_DATA: Insufficient data for analysis
        - OUTPUT_DIR_ERROR: Output directory creation/write error
        - CALC_ERROR: Calculation error
        """
        import pandas as pd

        try:
            # 1. Validate required parameter
            path_csv = request.get("path_csv")
            if not path_csv:
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="MISSING_PARAM",
                    error_desc="path_csv 파라미터가 필요합니다. 분석할 CSV 파일 경로를 지정해주세요."
                )

            # 2. Validate path format
            if not isinstance(path_csv, str) or len(path_csv.strip()) == 0:
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="INVALID_PATH",
                    error_desc="path_csv는 유효한 문자열 경로여야 합니다."
                )

            # Normalize path (handle Windows forward slashes)
            path_csv = path_csv.strip()
            csv_path = Path(path_csv)
            
            # Convert to absolute path and resolve
            try:
                if csv_path.is_absolute():
                    # For absolute paths, ensure it's properly normalized
                    csv_path = csv_path.resolve()
                else:
                    # For relative paths, resolve from current working directory
                    csv_path = csv_path.resolve()
            except (OSError, ValueError) as e:
                # If resolve fails, try with the original path
                self.logger.warning(f"Path resolution failed for {path_csv}: {e}, using original path")
            
            self.logger.debug(f"Checking CSV file: original={path_csv}, resolved={csv_path}, exists={csv_path.exists()}")

            # 3. Validate file existence
            if not csv_path.exists():
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="FILE_NOT_FOUND",
                    error_desc=f"CSV 파일을 찾을 수 없습니다: {path_csv} (확인한 경로: {csv_path})"
                )

            # 4. Validate file format (extension)
            if csv_path.suffix.lower() != '.csv':
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="INVALID_FORMAT",
                    error_desc=f"CSV 파일만 지원됩니다. 제공된 파일: {csv_path.suffix}"
                )

            # 5. Validate output directory
            output_dir = str(self.summary_base_path)
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                # Write permission test
                test_file = output_path / ".write_test"
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="OUTPUT_DIR_ERROR",
                    error_desc=f"출력 디렉토리에 쓰기 권한이 없습니다: {output_dir}"
                )
            except Exception as e:
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="OUTPUT_DIR_ERROR",
                    error_desc=f"출력 디렉토리 접근 오류: {output_dir}, {str(e)}"
                )

            # 6. 분석 실행 (TrajectoryRepeatability와 동일 패턴)
            try:
                analyzer = ManualRepeatability(str(csv_path))
                analyzer.run_analysis()
            except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, PermissionError) as e:
                code = "INVALID_FORMAT" if isinstance(e, (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError)) else "FILE_READ_ERROR"
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code=code,
                    error_desc=f"CSV 파일 읽기/파싱 오류: {str(e)}"
                )
            except ValueError as e:
                msg = str(e)
                if "필수 컬럼" in msg or "컬럼" in msg:
                    return self.protocol.create_response(
                        Command.MANUAL_CALC_RESULT,
                        success=False,
                        error_code="INVALID_CSV_STRUCTURE",
                        error_desc=msg
                    )
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="INSUFFICIENT_DATA",
                    error_desc=msg
                )
            except Exception as e:
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="CALC_ERROR",
                    error_desc=f"분석 실행 중 오류: {str(e)}"
                )

            self.logger.info(f"Starting manual measurement analysis: {path_csv}")
            self.logger.info(f"  Measurements: {analyzer.results['n_measurements']}")

            # 7. 결과 저장 (CSV + PNG, TrajectoryRepeatability와 동일)
            try:
                result_csv_path = analyzer.save_results_to_csv(output_dir=output_dir)
                result_png_path = analyzer.plot_results(output_dir=output_dir)
            except Exception as e:
                return self.protocol.create_response(
                    Command.MANUAL_CALC_RESULT,
                    success=False,
                    error_code="OUTPUT_DIR_ERROR",
                    error_desc=f"결과 저장 중 오류: {str(e)}"
                )

            self.logger.info(f"Manual (Cam1) measurement analysis completed. Results saved to {output_dir}")

            # 8. 응답 데이터 준비 (모두 summary_base_path = C:\CMES_AI\Summary 에 저장됨)
            r = analyzer.results
            response_data = {
                "n_measurements": r["n_measurements"],
                "output_dir": output_dir,
                "output_csv": result_csv_path,
                "output_png": result_png_path,
                "statistics": {
                    "x": {
                        "mean": round(r["x_mean"], 3),
                        "std": round(r["x_std"], 3),
                        "min": round(r["x_min"], 3),
                        "max": round(r["x_max"], 3),
                        "range": round(r["x_range"], 3),
                        "repeatability": round(r["x_std"], 3)
                    },
                    "y": {
                        "mean": round(r["y_mean"], 3),
                        "std": round(r["y_std"], 3),
                        "min": round(r["y_min"], 3),
                        "max": round(r["y_max"], 3),
                        "range": round(r["y_range"], 3),
                        "repeatability": round(r["y_std"], 3)
                    },
                    "rz": {
                        "mean": round(r["rz_mean"], 3),
                        "std": round(r["rz_std"], 3),
                        "min": round(r["rz_min"], 3),
                        "max": round(r["rz_max"], 3),
                        "range": round(r["rz_range"], 3),
                        "repeatability": round(r["rz_std"], 3)
                    }
                }
            }

            return self.protocol.create_response(
                Command.MANUAL_CALC_RESULT,
                success=True,
                data=response_data
            )

        except Exception as e:
            self.logger.error(f"Error in manual calculation: {e}")
            import traceback
            traceback.print_exc()
            return self.protocol.create_response(
                Command.MANUAL_CALC_RESULT,
                success=False,
                error_code="INTERNAL_ERROR",
                error_desc=f"예상치 못한 내부 오류: {str(e)}"
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

