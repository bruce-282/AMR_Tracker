"""Enhanced AMR Tracker - Core tracking system."""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

import cv2
import numpy as np

from config import SystemConfig
from src.core.detection import Detection, YOLODetector, BinaryDetector
from src.core.measurement.size_measurement import SizeMeasurement
from src.core.tracking import KalmanTracker, MAX_FRAMES_LOST
from src.visualization import Visualizer

logger = logging.getLogger(__name__)


class EnhancedAMRTracker:
    """
    AMR (Autonomous Mobile Robot) tracking system with YOLO detection and Kalman filtering.

    Features:
    - YOLO-based object detection with configurable classes
    - Kalman filter tracking with IoU-based association
    - Real-time speed and orientation estimation
    - CSV data logging for tracking results
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        detector_type: str = "yolo",
        tracker_type: str = "kalman",
        pixel_size: float = 1.0,
        distance_map_path: Optional[str] = None,
        model_path: Optional[str] = None,
        detector_config: Optional[Dict[str, Any]] = None,
        calibration_config: Optional[Dict[str, Any]] = None,
        fps: Optional[float] = None,
    ):
        """
        Initialize enhanced AMR system

        Args:
            config: System configuration
            detector_type: Type of detector ("yolo")
            tracker_type: Type of tracker ("kalman", "speed")
            pixel_size: Pixel size in mm (used if distance_map_path is None)
            distance_map_path: Path to distance map .npz file (optional, overrides pixel_size)
            model_path: Path to YOLO model file (default: "weights/zoom1/best.pt")
            detector_config: Detector configuration dictionary (optional)
            calibration_config: Calibration configuration dictionary (optional)
            fps: Frame rate (optional, defaults to config.measurement.fps or 30)
        """
        self.config = config if config else SystemConfig()
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.pixel_size = pixel_size
        self.distance_map_path = distance_map_path
        self.distance_map_data = None  # Will be loaded if distance_map_path is provided

        # Initialize components (will be set in _initialize_components)
        self.detector = None
        self.tracker = None
        self.size_measurement = None
        self.visualizer = None
        self.next_track_id = 0
        self.track_id = None

        # Store configs for initialization
        self.model_path = model_path or "weights/zoom1/best.pt"
        self.detector_config = detector_config or {}
        self.calibration_config = calibration_config
        
        # Get fps from parameter, config, or default
        if fps is not None:
            self.fps = fps
        elif self.config and hasattr(self.config, "measurement"):
            self.fps = self.config.measurement.fps
        else:
            self.fps = 30

        # Load distance map if path is provided
        if self.distance_map_path:
            self._load_distance_map()

        self._initialize_components()
    
    def _load_distance_map(self):
        """Load distance map from file."""
        try:
            from scripts.pixel_distance_mapper import PixelDistanceMapper
            self.distance_map_data = PixelDistanceMapper.load_distance_map(self.distance_map_path)
            if self.distance_map_data:
                logger.info(f"Distance map loaded from: {self.distance_map_path}")
                logger.info(f"  Image shape: {self.distance_map_data['image_shape']}")
                logger.info(f"  Reference point: ({self.distance_map_data['reference_world'][0]:.2f}, {self.distance_map_data['reference_world'][1]:.2f}) mm")
            else:
                logger.warning(f"Failed to load distance map from: {self.distance_map_path}")
                self.distance_map_path = None
        except Exception as e:
            logger.error(f"Error loading distance map from {self.distance_map_path}: {e}")
            self.distance_map_path = None
            self.distance_map_data = None

    def _initialize_components(self):
        """Initialize all system components"""
        logger.info(
            f"Initializing Enhanced AMR System (detector={self.detector_type}, tracker={self.tracker_type})"
        )

        # Initialize detector
        if self.detector_type == "yolo":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Use detector_config if provided, otherwise use defaults
                self.detector = YOLODetector(
                    self.model_path,
                    confidence_threshold=self.detector_config.get("confidence_threshold", 0.5),
                    device=device,
                    imgsz=self.detector_config.get("imgsz", 1536),
                    target_classes=self.detector_config.get("target_classes", [0]),
                )
                logger.info("YOLO detector initialized")
            except ImportError:
                raise ImportError("ultralytics module is not installed.")
            except FileNotFoundError:
                raise FileNotFoundError(f"weights file not found: {self.model_path}")
        elif self.detector_type == "binary":
            # Initialize binary detector
            # Get binary detector parameters from detector_config
            self.detector = BinaryDetector(
                threshold=self.detector_config.get("threshold", 50),
                min_area=self.detector_config.get("min_area", 1000),
                max_area=self.detector_config.get("max_area", None),
                width_height_ratio_min=self.detector_config.get("width_height_ratio_min", 0.8),
                width_height_ratio_max=self.detector_config.get("width_height_ratio_max", 1.2),
                mask_area_ratio=self.detector_config.get("mask_area_ratio", 0.9),
                inverse=self.detector_config.get("inverse", True),  # True: dark objects, False: bright objects
                use_adaptive=self.detector_config.get("use_adaptive", True),  # Use adaptive threshold
                adaptive_block_size=self.detector_config.get("adaptive_block_size", 11),
                adaptive_c=self.detector_config.get("adaptive_c", 2.0),
            )
            logger.info("Binary detector initialized")
        else:
            raise ValueError(
                f"Unsupported detector type: {self.detector_type}. Supported types: 'yolo', 'binary'"
            )

        # Initialize tracker
        if self.tracker_type == "kalman":
            self.tracker = KalmanTracker(
                fps=self.fps,
                pixel_size=self.pixel_size,
                distance_map_data=self.distance_map_data,
                track_id=0,
            )
            if self.distance_map_data:
                logger.info(f"Kalman filter tracker initialized (fps={self.fps}, using distance map)")
            else:
                logger.info(f"Kalman filter tracker initialized (fps={self.fps}, pixel_size={self.pixel_size})")
        else:
            raise ValueError(
                f"Unsupported tracker type: {self.tracker_type}. Only 'kalman' is supported."
            )

        self.next_track_id = 0
        self.track_id = None

        # Initialize calibration components
        self._initialize_calibration()

    def _initialize_calibration(self):
        """Initialize calibration components (size measurement and visualizer)"""
        calibration_path = None
        
        # Try to get calibration path from calibration_config first
        if self.calibration_config:
            calibration_path = self.calibration_config.get("calibration_data_path")
            camera_height = self.calibration_config.get("camera_height", 0.0)
            calibration_image_size = self.calibration_config.get("calibration_image_size", (3840, 2160))
        # Fallback to config object
        elif self.config:
            try:
                calibration_path = self.config.calibration.calibration_data_path
                camera_height = self.config.calibration.camera_height
                calibration_image_size = self.config.calibration.calibration_image_size
            except AttributeError as e:
                logger.debug(f"Error accessing calibration from config: {e}")
                camera_height = 0.0
                calibration_image_size = (3840, 2160)
        else:
            camera_height = 0.0
            calibration_image_size = (3840, 2160)

        if calibration_path and Path(calibration_path).exists():
            try:
                with open(calibration_path, "r") as f:
                    calibration_data = json.load(f)

                # Convert calibration_image_size to tuple if it's a list
                if isinstance(calibration_image_size, list):
                    calibration_image_size = tuple(calibration_image_size)

                # Use distance map if available, otherwise use pixel_size from calibration
                pixel_size_for_measurement = calibration_data.get("pixel_size", 1.0)
                if self.distance_map_data:
                    # Distance map will be used instead of pixel_size
                    pixel_size_for_measurement = 1.0  # Placeholder, actual conversion uses distance map
                
                self.size_measurement = SizeMeasurement(
                    homography=np.array(calibration_data["homography"]),
                    camera_height=camera_height,
                    pixel_size=pixel_size_for_measurement,
                    distance_map_data=self.distance_map_data,
                    calibration_image_size=calibration_image_size,
                )
                logger.info("Size measurement initialized from calibration config")

                self.visualizer = Visualizer(
                    homography=np.array(calibration_data["homography"])
                )
                logger.info("Visualizer initialized from calibration config")
            except Exception as e:
                logger.warning(f"Error loading calibration data from {calibration_path}: {e}")
        else:
            if calibration_path:
                logger.debug(f"Calibration data file not found: {calibration_path}")
            else:
                logger.debug("No calibration config provided - size measurement disabled")

    def detect_objects(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = None
    ) -> List[Detection]:
        """Detect objects in frame using selected detector"""
        if timestamp is None:
            timestamp = time.time()

        # YOLO detection using YOLODetector
        detections = self.detector.detect(
            image=frame, frame_number=frame_number, timestamp=timestamp
        )
        return detections

    def track_objects(
        self, frame: np.ndarray, detections: List[Detection], frame_number: int = 0
    ) -> List[Dict]:
        """Track objects using selected tracker"""
        results = []

        if self.tracker_type == "kalman":
            # Multi-object detection, but track only the first tracked object
            if len(detections) > 0:
                # If no primary tracker exists, create one with the best detection
                best_detection = detections[0]

                if self.track_id is None:
                    # Create primary tracker
                    self.track_id = 0

                    self.tracker.initialize_with_detection(
                        best_detection.oriented_box_info["center"],
                        best_detection.oriented_box_info["angle"],
                    )
                    self.next_track_id += 1

                    # Update with first detection
                    # Get image size for boundary checking
                    img_height, img_width = frame.shape[:2]
                    tracking_result = self.tracker.update(
                        bbox=best_detection.bbox,
                        center=best_detection.oriented_box_info["center"],
                        frame_number=frame_number,
                        theta=best_detection.oriented_box_info["angle"],
                        image_size=(img_width, img_height),
                    )
                    tracking_result["detection_type"] = getattr(
                        best_detection, "class_name", "unknown"
                    )
                    tracking_result["track_id"] = self.track_id
                    tracking_result["color"] = self.tracker.color

                    # Add size measurement if available (only store initial size)
                    if self.size_measurement:
                        size_measurement = self.size_measurement.measure(best_detection)
                        tracking_result["size_measurement"] = size_measurement
                        self.tracker.last_size_measurement = size_measurement
                        self.tracker.initial_size_measurement = (
                            size_measurement  # Store initial size
                        )

                    results.append(tracking_result)
                else:
                    # Primary tracker exists
                    if self.tracker is None:
                        # Primary tracker was lost, reset
                        self.track_id = None
                        return results

                    # Get image size for boundary checking
                    img_height, img_width = frame.shape[:2]
                    # Get image size for boundary checking
                    img_height, img_width = frame.shape[:2]
                    tracking_result = self.tracker.update(
                        bbox=best_detection.bbox,
                        center=best_detection.oriented_box_info["center"],
                        theta=best_detection.oriented_box_info["angle"],
                        frame_number=frame_number,
                        image_size=(img_width, img_height),
                    )
                    
                    # Check if reset was required (position outside 70% of image)
                    if tracking_result.get("reset_required", False):
                        logger.info(f"Tracker reset due to position outside bounds, reinitializing with next detection")
                        self.tracker.reset()
                        self.track_id = None
                        # Don't add this result, wait for next detection to reinitialize
                        return results
                    
                    tracking_result["detection_type"] = "predicted"
                    tracking_result["track_id"] = self.track_id
                    tracking_result["color"] = self.tracker.color
                    results.append(tracking_result)

                    if self.tracker.is_lost(max_frames_lost=MAX_FRAMES_LOST):
                        logger.info(
                            f"Removing lost primary tracker ID: {self.track_id}"
                        )
                        self.tracker.reset()
                        if self.visualizer:
                            self.visualizer.reset()
                        self.track_id = None

            else:
                # No detections: predict with primary tracker
                if self.track_id is not None:
                    if self.tracker is not None:
                        # Get image size for boundary checking
                        img_height, img_width = frame.shape[:2]
                        tracking_result = self.tracker.update(
                            bbox=None,
                            center=None,
                            theta=None,
                            frame_number=frame_number,
                            image_size=(img_width, img_height),
                        )
                        
                        # Check if reset was required (position outside 70% of image)
                        if tracking_result.get("reset_required", False):
                            logger.info(f"Tracker reset due to position outside bounds, reinitializing with next detection")
                            self.tracker.reset()
                            self.track_id = None
                            # Don't add this result, wait for next detection to reinitialize
                            return results
                        
                        tracking_result["detection_type"] = "predicted"
                        tracking_result["track_id"] = self.track_id
                        tracking_result["color"] = self.tracker.color
                        results.append(tracking_result)

                        # Clean up if tracker is lost
                        if self.tracker.is_lost(max_frames_lost=MAX_FRAMES_LOST):
                            logger.info(
                                f"Removing lost primary tracker ID: {self.track_id}"
                            )
                            self.tracker.reset()
                            if self.visualizer:
                                self.visualizer.reset()
                            self.track_id = None

        else:
            raise ValueError(
                f"Unsupported tracker type: {self.tracker_type}. Only 'kalman' is supported."
            )
        return results

    def visualize_results(
        self, frame: np.ndarray, detections: List[Detection], tracking_results: List[Dict]
    ) -> np.ndarray:
        """Visualize results using appropriate visualizer"""
        # Use multi-object AMR tracker visualization
        if self.visualizer and self.size_measurement:
            # Use enhanced visualizer if available
            detection_objects = detections
            vis_frame = self.visualizer.draw_single_object(
                frame, detection_objects, tracking_results
            )
        else:
            # Use basic AMR tracker visualization
            vis_frame = frame.copy()

            # Draw all tracking results
            for i, result in enumerate(tracking_results):
                if "track_id" in result:
                    if result["track_id"] == 0:
                        vis_frame = self.tracker.draw_visualization(vis_frame, result)
        
        # If using binary detector, overlay binary debug info in corner
        if isinstance(self.detector, BinaryDetector):
            debug_image = self.detector.get_debug_image(frame)
            if debug_image is not None:
                # Resize debug image to fit in corner (larger size - about 1/3 of width)
                h, w = vis_frame.shape[:2]
                debug_h, debug_w = debug_image.shape[:2]
                # Use larger scale for better visibility
                scale = min(w // 3 / debug_w, h // 3 / debug_h)
                if scale < 1.0:
                    new_w = int(debug_w * scale)
                    new_h = int(debug_h * scale)
                    debug_resized = cv2.resize(debug_image, (new_w, new_h))
                else:
                    debug_resized = debug_image
                
                # Place in top-right corner
                dh, dw = debug_resized.shape[:2]
                y_offset = 10
                x_offset = w - dw - 10
                
                # Create overlay with transparency
                overlay = vis_frame.copy()
                overlay[y_offset:y_offset+dh, x_offset:x_offset+dw] = debug_resized
                vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)

        return vis_frame

    def reset(self):
        """Reset tracker state"""
        if self.tracker:
            self.tracker.reset()
        if self.visualizer:
            self.visualizer.reset()
        self.track_id = None
        self.next_track_id = 0

