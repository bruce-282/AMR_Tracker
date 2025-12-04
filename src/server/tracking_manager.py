"""Tracking management for vision server."""

import logging
import time
import threading
from typing import Dict, Optional, List, Callable
import cv2
import numpy as np

from src.core.detection import Detection
from src.core.amr_tracker import EnhancedAMRTracker
from src.utils.sequence_loader import BaseLoader
from .camera_manager import CameraManager
from .camera_state import CameraStateManager
from config import TrackingConfig

logger = logging.getLogger(__name__)


class TrackingManager:
    """Manages tracking loops and camera-specific tracking logic."""
    
    def __init__(
        self,
        camera_manager: CameraManager,
        camera_state_manager: CameraStateManager,
        tracking_config: TrackingConfig,
        use_area_scan: bool = False,
        visualize_stream: bool = True
    ):
        """
        Initialize tracking manager.
        
        Args:
            camera_manager: Camera manager instance
            camera_state_manager: Camera state manager instance
            tracking_config: Tracking configuration
            use_area_scan: Whether to use area scan mode
            visualize_stream: Whether to visualize tracking stream
        """
        self.camera_manager = camera_manager
        self.camera_state_manager = camera_state_manager
        self.tracking_config = tracking_config
        self.use_area_scan = use_area_scan
        self.visualize_stream = visualize_stream
        
        self.tracking_threads: Dict[int, threading.Thread] = {}
        self.vision_active = False
        
        # Camera-specific data
        self.latest_detections: Dict[int, Detection] = {}
        self.camera2_trajectory = None  # Will be initialized with deque
        self.camera2_trajectory_sent = False
        self.last_frames: Dict[int, np.ndarray] = {}  # Store last frame for each camera
        
        # Callbacks for camera-specific logic
        self.on_camera_1_3_stop: Optional[Callable[[int], None]] = None
        self.on_camera_2_stop: Optional[Callable[[], None]] = None
        self.on_camera_1_3_first_detection: Optional[Callable[[int, Detection, np.ndarray], None]] = None
        self.on_camera_2_trajectory: Optional[Callable[[int, Optional[np.ndarray], List[Detection], List[Dict], str], bool]] = None
    
    def set_vision_active(self, active: bool):
        """Set vision active state."""
        self.vision_active = active
    
    def start_tracking(self, camera_id: int):
        """Start tracking thread for a camera."""
        if camera_id not in self.tracking_threads or not self.tracking_threads[camera_id].is_alive():
            thread = threading.Thread(
                target=self._tracking_loop,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.tracking_threads[camera_id] = thread
            logger.info(f"Camera {camera_id} tracking thread started")
    
    def stop_all_tracking(self, timeout: float = 2.0):
        """Stop all tracking threads."""
        self.vision_active = False
        
        for camera_id in list(self.tracking_threads.keys()):
            thread = self.tracking_threads.get(camera_id)
            if thread and thread.is_alive():
                logger.info(f"Waiting for camera {camera_id} tracking thread to finish...")
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Camera {camera_id} tracking thread did not finish within timeout")
        
        self.tracking_threads.clear()
    
    def _tracking_loop(self, camera_id: int):
        """Main tracking loop for camera."""
        loader = self.camera_manager.camera_loaders.get(camera_id)
        if not loader:
            return

        amr_tracker = self.camera_manager.amr_trackers.get(camera_id)
        if not amr_tracker:
            logger.error(f"Camera {camera_id}: EnhancedAMRTracker not found")
            return
        
        trackers = self.camera_manager.trackers[camera_id]
        frame_number = 0
        
        logger.info(f"Camera {camera_id} tracking started")
        
        while self.vision_active and camera_id in self.camera_manager.camera_loaders:
            try:
                ret, frame = loader.read()
                if not ret:
                    logger.info(f"Camera {camera_id}: No more frames available. Exiting tracking loop.")
                    self._handle_no_frames(camera_id)
                    break

                

                # 모든 카메라에서 프레임얻어 와보기. 아래 예시시
                # for cam_id, cam_loader in self.camera_manager.camera_loaders.items():
                #     if cam_id != camera_id:
                #         ret, frame = cam_loader.read()
                #         if not ret:
                #             logger.info(f"Camera {cam_id}: No more frames available. Exiting tracking loop.")
                #             self._handle_no_frames(cam_id)
                #             break
                #        imwrite(f"output/Debug/frame_{cam_id}.png", frame)
                #        print(f"Camera {cam_id}: Frame saved successfully")


                
                frame_number += 1
                self.camera_manager.frame_numbers[camera_id] = frame_number
                
                # Get frame timestamp
                frame_timestamp = time.time()
                if hasattr(loader, 'get_timestamp'):
                    loader_timestamp = loader.get_timestamp()
                    if loader_timestamp is not None:
                        frame_timestamp = loader_timestamp
                
                # Detect and track using EnhancedAMRTracker
                detections = amr_tracker.detect_objects(
                    frame=frame,
                    frame_number=frame_number,
                    timestamp=frame_timestamp
                )
                
                tracking_results = amr_tracker.track_objects(
                    frame=frame,
                    detections=detections,
                    frame_number=frame_number
                )
                
                # Sync EnhancedAMRTracker's tracker to trackers dict for compatibility
                if amr_tracker.tracker and amr_tracker.track_id is not None:
                    track_id = amr_tracker.track_id
                    trackers[track_id] = amr_tracker.tracker
                else:
                    trackers.clear()
                
                # Visualize results
                vis_frame = amr_tracker.visualize_results(
                    frame=frame,
                    detections=detections,
                    tracking_results=tracking_results
                )
                
                # Store last frame for result image saving (especially for camera 2)
                self.last_frames[camera_id] = frame.copy()
                
                has_detection = len(detections) > 0
                cam_state = self.camera_state_manager.get_or_create(camera_id)
                
                # Update detection state
                if has_detection:
                    if not (not self.use_area_scan and camera_id in [1, 3]):
                        self.latest_detections[camera_id] = detections[0]
                    if camera_id in [1, 3] and not self.use_area_scan:
                        # Mark that first detection has occurred
                        if not cam_state.has_first_detection:
                            cam_state.has_first_detection = True
                        cam_state.reset_detection_loss()
                else:
                    if camera_id in [1, 3] and not self.use_area_scan:
                        # Only increment if first detection has already occurred
                        if cam_state.has_first_detection:
                            cam_state.increment_detection_loss()
                
                # Display visualization
                if self.visualize_stream:
                    height, width = vis_frame.shape[:2]
                    if width > 1920 or height > 1080:
                        scale = min(1920 / width, 1080 / height)
                        vis_frame = cv2.resize(vis_frame, (int(width * scale), int(height * scale)))
                    
                    window_name = f"Camera {camera_id} - AMR Tracking"
                    cv2.imshow(window_name, vis_frame)
                    cv2.waitKey(1)
                
                # Camera-specific logic
                if not self.use_area_scan:
                    should_continue = self._handle_camera_specific_logic(
                        camera_id, trackers, detections, tracking_results, 
                        frame, vis_frame, has_detection, cam_state
                    )
                    if not should_continue:
                        break
                
                # Clean up lost trackers
                if amr_tracker.track_id is None:
                    trackers.clear()
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.warning(f"Camera {camera_id} tracking error: {e}")
                time.sleep(0.1)
        
        logger.info(f"Camera {camera_id} tracking stopped")
        
        # Reset camera state before cleanup
        cam_state = self.camera_state_manager.get(camera_id)
        if cam_state:
            cam_state.has_first_detection = False
            logger.debug(f"Camera {camera_id}: has_first_detection reset to False")
        
        # Reset loader to beginning if it's a video/sequence loader
        if loader and hasattr(loader, 'reset'):
            try:
                loader.reset()
                logger.info(f"Camera {camera_id}: Loader reset to beginning")
            except Exception as e:
                logger.warning(f"Camera {camera_id}: Failed to reset loader: {e}")
        
        # Close tracking window
        window_name = f"Camera {camera_id} - AMR Tracking"
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
    
    def _handle_no_frames(self, camera_id: int):
        """Handle case when no more frames are available."""
        if camera_id == 2 and self.on_camera_2_trajectory:
            # Use last frame if available, otherwise None (save_result_image will try to read from loader)
            last_frame = self.last_frames.get(camera_id)
            if last_frame is not None:
                logger.info(f"Camera {camera_id}: Using last frame for result image")
            else:
                logger.warning(f"Camera {camera_id}: No last frame available, save_result_image will try to read from loader")
            
            self.on_camera_2_trajectory(
                camera_id,
                frame=last_frame,
                detections=[],
                tracking_results=[],
                reason="No more frames available"
            )
        
        if not self.use_area_scan:
            if camera_id in [1, 3] and self.on_camera_1_3_stop:
                self.on_camera_1_3_stop(camera_id)
            elif camera_id == 2 and self.on_camera_2_stop:
                self.on_camera_2_stop()
    
    def _handle_camera_specific_logic(
        self,
        camera_id: int,
        trackers: Dict,
        detections: List[Detection],
        tracking_results: List[Dict],
        frame: np.ndarray,
        vis_frame: np.ndarray,
        has_detection: bool,
        cam_state
    ) -> bool:
        """Handle camera-specific tracking logic. Returns False if tracking should stop."""
        if camera_id in [1, 3]:
            return self._handle_camera_1_3_tracking(
                camera_id, trackers, detections, tracking_results, frame, cam_state
            )
        elif camera_id == 2:
            return self._handle_camera_2_tracking(
                camera_id, trackers, tracking_results, has_detection, vis_frame, detections, cam_state, frame
            )
        return True
    
    def _handle_camera_1_3_tracking(
        self,
        camera_id: int,
        trackers: Dict,
        detections: List[Detection],
        tracking_results: List[Dict],
        frame: np.ndarray,
        cam_state
    ) -> bool:
        """Handle camera 1, 3 specific tracking logic."""
        if detections:
            self.latest_detections[camera_id] = detections[0]
        
        speed_near_zero_thresh = self.tracking_config.speed_near_zero_threshold
        speed_zero_frames_thresh = self.tracking_config.speed_zero_frames_threshold
        speed_thresh = self.tracking_config.speed_threshold_pix_per_frame
        
        if not tracking_results:
            return True
        
        # Get tracker from EnhancedAMRTracker
        amr_tracker = self.camera_manager.amr_trackers.get(camera_id)
        if amr_tracker and amr_tracker.tracker and amr_tracker.track_id is not None:
            tracker = amr_tracker.tracker
        else:
            tracker = next(iter(trackers.values())) if trackers else None
        
        if not tracker:
            return True
        
        speed_pix_per_frame = self._calculate_speed_pix_per_frame(tracker)
        cam_state.update_speed(speed_pix_per_frame)
        
        logger.info(
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
                logger.info(
                    f"Camera {camera_id}: Sending first detection response - "
                    f"speed={speed_pix_per_frame:.3f} pix/frame, "
                    f"count={cam_state.speed_near_zero_frames}"
                )
                if self.on_camera_1_3_first_detection:
                    self.on_camera_1_3_first_detection(
                        camera_id, self.latest_detections[camera_id], frame
                    )
                cam_state.speed_near_zero_frames = 0
        else:
            if cam_state.speed_near_zero_frames > 0:
                logger.debug(
                    f"Camera {camera_id}: Speed not near zero - "
                    f"speed={speed_pix_per_frame:.3f} pix/frame > threshold={speed_near_zero_thresh}, "
                    f"resetting count from {cam_state.speed_near_zero_frames}"
                )
            cam_state.speed_near_zero_frames = 0
        
        # Check if speed threshold reached (for stopping tracking)
        if abs(speed_pix_per_frame) > speed_thresh and cam_state.response_sent:
            logger.info(
                f"Camera {camera_id}: Speed threshold reached "
                f"({speed_pix_per_frame:.3f} pix/frame > {speed_thresh} pix/frame). "
            )
            logger.info(f"Camera {camera_id}: AGV가 다시 움직이기 시작하였으므로 다음 카메라의 Tracking Loop를 시작합니다.")
            if self.on_camera_1_3_stop:
                self.on_camera_1_3_stop(camera_id)
            return False  # Break tracking loop
        
        return True
    
    def _handle_camera_2_tracking(
        self,
        camera_id: int,
        trackers: Dict,
        tracking_results: List[Dict],
        has_detection: bool,
        vis_frame: np.ndarray,
        detections: List[Detection],
        cam_state,
        frame: Optional[np.ndarray] = None
    ) -> bool:
        """Handle camera 2 specific tracking logic."""
        detection_loss_thresh = self.tracking_config.detection_loss_threshold_frames
        camera2_trajectory_max_frames = self.tracking_config.camera2_trajectory_max_frames
        
        if tracking_results:
            amr_tracker = self.camera_manager.amr_trackers.get(camera_id)
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
        # Only increment detection loss after first detection has occurred
        if has_detection:
            # Mark that first detection has occurred
            if not cam_state.has_first_detection:
                cam_state.has_first_detection = True
            cam_state.reset_detection_loss()
        else:
            # Only increment if first detection has already occurred
            if cam_state.has_first_detection:
                cam_state.increment_detection_loss()
        
        logger.info(
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
            if self.on_camera_2_trajectory:
                # Use original frame if available, otherwise use vis_frame
                frame_to_save = frame if frame is not None else vis_frame
                logger.info(f"Camera {camera_id}: End tracking triggered ({reason}), frame_to_save is {'not None' if frame_to_save is not None else 'None'}")
                if self.on_camera_2_trajectory(
                    camera_id,
                    frame_to_save,
                    detections,
                    tracking_results,
                    reason
                ):
                    cam_state.reset_detection_loss()
                    logger.info("Camera 2: Tracking loop exiting after sending trajectory.")
                    return False  # Break tracking loop
        
        return True
    
    def _calculate_speed_pix_per_frame(self, tracker) -> float:
        """Calculate object speed in pixels/frame from tracker state."""
        state = tracker.kf.statePost.flatten()
        vx = state[3]  # velocity x (pixels/frame)
        vy = state[4]  # velocity y (pixels/frame)
        return np.sqrt(vx**2 + vy**2)
    
    def initialize_camera2_trajectory(self, maxlen: int):
        """Initialize camera 2 trajectory deque."""
        from collections import deque
        self.camera2_trajectory = deque(maxlen=maxlen)
        self.camera2_trajectory_sent = False

