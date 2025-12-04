"""Response building and image saving for vision server."""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import cv2
import numpy as np

from src.core.detection import Detection
from .camera_manager import CameraManager
from .tracking_manager import TrackingManager
from .protocol import ProtocolHandler, Command

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Builds responses and saves result images."""
    
    def __init__(
        self,
        camera_manager: CameraManager,
        tracking_manager: TrackingManager,
        result_base_path: Path,
        debug_base_path: Optional[Path] = None,
        protocol_handler: Optional[ProtocolHandler] = None
    ):
        """
        Initialize response builder.
        
        Args:
            camera_manager: Camera manager instance
            tracking_manager: Tracking manager instance
            result_base_path: Base path for result images
            debug_base_path: Base path for debug images (optional)
            protocol_handler: Protocol handler instance (optional, creates new if not provided)
        """
        self.camera_manager = camera_manager
        self.tracking_manager = tracking_manager
        self.result_base_path = result_base_path
        self.debug_base_path = debug_base_path
        self.protocol = protocol_handler if protocol_handler else ProtocolHandler()
    
    def get_tracking_data(self, camera_id: int, use_area_scan: bool) -> Dict[str, float]:
        """
        Get current tracking data for camera.
        
        Args:
            camera_id: Camera ID
            use_area_scan: Whether area scan mode is enabled
        
        Returns:
            Dictionary with x, y (in mm) and rz (yaw angle in degrees)
        """
        # Cameras 1, 3: use detection data (not tracker) when use_area_scan is false
        if camera_id in [1, 3] and not use_area_scan:
            detection = self.tracking_manager.latest_detections.get(camera_id)
            if detection is not None:
                center = detection.get_center()
                orientation = detection.get_orientation()
                
                pixel_size = self.camera_manager.get_pixel_size(camera_id)
                x_mm = center[0] * pixel_size
                y_mm = center[1] * pixel_size
                rz_deg = orientation if orientation is not None else 0.0
                
                logger.debug(f"Camera {camera_id}: Returning DETECTION data: x={x_mm:.2f}, y={y_mm:.2f}, rz={rz_deg:.4f}deg")
                return {
                    "x": round(float(x_mm), 3),
                    "y": round(float(y_mm), 3),
                    "rz": round(float(rz_deg), 3)
                }
            else:
                logger.debug(f"Camera {camera_id}: No detection available, returning zeros")
                return {"x": 0.0, "y": 0.0, "rz": 0.0}
        
        # Camera 2 or use_area_scan=true: use tracker data
        amr_tracker = self.camera_manager.amr_trackers.get(camera_id)
        if amr_tracker and amr_tracker.tracker and amr_tracker.track_id is not None:
            tracker = amr_tracker.tracker
            state = tracker.kf.statePost.flatten()
            
            pixel_size = self.camera_manager.get_pixel_size(camera_id)
            x_mm = state[0] * pixel_size
            y_mm = state[1] * pixel_size
            rz_deg = state[2]
            
            logger.debug(f"Camera {camera_id}: Returning EnhancedAMRTracker data: x={x_mm:.2f}, y={y_mm:.2f}, rz={rz_deg:.4f}deg")
            return {
                "x": round(float(x_mm), 3),
                "y": round(float(y_mm), 3),
                "rz": round(float(rz_deg), 3)
            }
        
        # Fallback: get tracking results from trackers dict
        trackers = self.camera_manager.trackers.get(camera_id, {})
        if not trackers:
            logger.debug(f"Camera {camera_id}: No tracker available, returning zeros")
            return {"x": 0.0, "y": 0.0, "rz": 0.0}
        
        tracker = next(iter(trackers.values()))
        state = tracker.kf.statePost.flatten()
        
        pixel_size = self.camera_manager.get_pixel_size(camera_id)
        x_mm = state[0] * pixel_size
        y_mm = state[1] * pixel_size
        rz_deg = state[2]
        
        logger.debug(f"Camera {camera_id}: Returning TRACKER data: x={x_mm:.2f}, y={y_mm:.2f}, rz={rz_deg:.4f}deg")
        return {
            "x": round(float(x_mm), 3),
            "y": round(float(y_mm), 3),
            "rz": round(float(rz_deg), 3)
        }
    
    def build_first_detection_response(
        self,
        camera_id: int,
        detection: Detection,
        frame: np.ndarray
    ) -> Dict[str, any]:
        """
        Build first detection response for cameras 1, 3.
        
        Args:
            camera_id: Camera ID
            detection: Detection object
            frame: Frame image
        
        Returns:
            Response data dictionary
        """
        pixel_size = self.camera_manager.get_pixel_size(camera_id)
        center = detection.get_center()
        x_mm = center[0] * pixel_size
        y_mm = center[1] * pixel_size
        rz = detection.get_orientation() if detection.get_orientation() is not None else 0.0
        
        # Save result image
        result_image_path = self.result_base_path / f"cam_{camera_id}_result.png"
        self.save_result_image(
            camera_id,
            result_image_path,
            frame=frame,
            detections=[detection],
            tracking_results=self._create_tracking_result_from_detection(detection, center, rz)
        )
        
        return {
            "x": round(float(x_mm), 3),
            "y": round(float(y_mm), 3),
            "rz": round(float(rz), 3),
            "result_image": str(result_image_path)
        }
    
    def _create_tracking_result_from_detection(
        self,
        detection: Detection,
        center: tuple,
        rz: float
    ) -> List[Dict]:
        """Create tracking result from detection for visualization."""
        return [{
            "track_id": 0,
            "position": {"x": center[0], "y": center[1]},
            "orientation": {"theta_deg": rz},
            "bbox": detection.bbox
        }]
    
    def save_result_image(
        self,
        camera_id: int,
        image_path: Path,
        frame: Optional[np.ndarray] = None,
        detections: Optional[List[Detection]] = None,
        tracking_results: Optional[List[Dict]] = None
    ):
        """
        Save result image with tracking visualization.
        
        Args:
            camera_id: Camera ID
            image_path: Path to save image
            frame: Optional frame (will be read from loader if not provided)
            detections: Optional detections
            tracking_results: Optional tracking results
        """
        try:
            # Get frame if not provided
            if frame is None:
                loader = self.camera_manager.camera_loaders.get(camera_id)
                if loader:
                    ret, frame = loader.read()
                    frame = self.camera_manager.warp_frame(camera_id, frame)
                    if not ret or frame is None:
                        return
                else:
                    return
            
            # Get detections if not provided
            if detections is None:
                detections = []
                if camera_id in self.tracking_manager.latest_detections:
                    detections = [self.tracking_manager.latest_detections[camera_id]]
            
            # Get tracking results if not provided
            if tracking_results is None:
                trackers = self.camera_manager.trackers.get(camera_id, {})
                tracking_results = []
                for track_id, tracker in trackers.items():
                    state = tracker.kf.statePost.flatten()
                    bbox = detections[0].bbox if detections else None
                    tracking_results.append({
                        "track_id": track_id,
                        "position": {"x": state[0], "y": state[1]},
                        "orientation": {"theta_deg": state[2]},
                        "bbox": bbox
                    })
            
            # Visualize and save
            vis_frame = self.visualize_results(
                camera_id, frame, detections, tracking_results
            )
            # Ensure directory exists
            image_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(image_path), vis_frame)
            if success:
                logger.info(f"Result image saved: {image_path}")
            else:
                logger.error(f"Failed to save result image: {image_path} (cv2.imwrite returned False)")
            
            # Save binary detector debug image if using binary detector
            self._save_binary_debug_image(camera_id, frame)
            
        except Exception as e:
            logger.error(f"Failed to save result image: {e}", exc_info=True)
    
    def visualize_results(
        self,
        camera_id: int,
        frame: np.ndarray,
        detections: List[Detection],
        tracking_results: List[Dict],
        draw_trajectory: bool = True
    ) -> np.ndarray:
        """
        Visualize tracking results on frame.
        
        Args:
            camera_id: Camera ID
            frame: Input frame
            detections: List of detections
            tracking_results: List of tracking results
            draw_trajectory: Whether to draw trajectory
        
        Returns:
            Visualized frame
        """
        # Try to use EnhancedAMRTracker's visualizer first
        amr_tracker = self.camera_manager.amr_trackers.get(camera_id)
        if amr_tracker and amr_tracker.visualizer and amr_tracker.size_measurement:
            return amr_tracker.visualize_results(frame, detections, tracking_results)
        
        # Fallback: filter uninitialized trackers
        filtered_tracking_results = []
        for result in tracking_results:
            position = result.get("position", {})
            if position:
                x = position.get("x", 0)
                y = position.get("y", 0)
                if x == 0.0 and y == 0.0:
                    continue
            filtered_tracking_results.append(result)
        
        # Try to get visualizer from first camera's EnhancedAMRTracker
        visualizer = None
        for cam_id in [1, 2, 3]:
            amr_tracker = self.camera_manager.amr_trackers.get(cam_id)
            if amr_tracker and amr_tracker.visualizer:
                visualizer = amr_tracker.visualizer
                break
        
        if visualizer:
            # For cameras 1, 3: do not draw tracking information, only detections
            # For camera 2: draw tracking information with trajectory
            if camera_id == 2:
                # Camera 2: use tracking results with trajectory, but force track_id to 0
                camera2_trackings = []
                for result in filtered_tracking_results:
                    result_copy = result.copy()
                    result_copy["track_id"] = 0
                    camera2_trackings.append(result_copy)
                return visualizer.draw_single_object(frame, detections, camera2_trackings)
            else:
                # Camera 1, 3: create empty tracking dicts for each detection (to draw detections only)
                empty_trackings = [{"track_id": 0, "trajectory": []} for _ in detections]
                return visualizer.draw_single_object(frame, detections, empty_trackings)
        
        # Final fallback: return frame as-is
        return frame.copy()
    
    def _save_binary_debug_image(self, camera_id: int, frame: np.ndarray):
        """
        Save binary detector debug image if binary detector is being used.
        
        Args:
            camera_id: Camera ID
            frame: Original frame
        """
        if self.debug_base_path is None:
            return
        
        try:
            amr_tracker = self.camera_manager.amr_trackers.get(camera_id)
            if not amr_tracker or not amr_tracker.detector:
                return
            
            # Check if detector is BinaryDetector
            from src.core.detection import BinaryDetector
            if not isinstance(amr_tracker.detector, BinaryDetector):
                return
            
            # Get debug image from binary detector
            debug_image = amr_tracker.detector.get_debug_image(frame)
            if debug_image is None:
                return
            
            # Save debug image
            debug_path = self.debug_base_path / f"cam_{camera_id}_binary_debug.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(debug_path), debug_image)
            if success:
                logger.info(f"Binary debug image saved: {debug_path}")
            else:
                logger.warning(f"Failed to save binary debug image: {debug_path}")
        except Exception as e:
            logger.debug(f"Could not save binary debug image: {e}")

