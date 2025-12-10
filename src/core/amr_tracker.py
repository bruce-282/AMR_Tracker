"""Enhanced AMR Tracker - Core tracking system."""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

import cv2
import numpy as np

# SystemConfig removed - all configs loaded directly json and tracker_config files
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
        config: Optional[Any] = None,  # Not used - kept for compatibility
        detector_type: str = "yolo",
        tracker_type: str = "kalman",
        pixel_size: Union[float, Dict[str, float]] = 1.0,
        distance_map_path: Optional[str] = None,
        model_path: Optional[str] = None,
        detector_config: Optional[Dict[str, Any]] = None,
        tracker_config: Optional[Dict[str, Any]] = None,
        calibration_config: Optional[Dict[str, Any]] = None,
        fps: Optional[float] = None,
        max_frames_lost: Optional[int] = None,
    ):
        """
        Initialize enhanced AMR system

        Args:
            config: Not used (deprecated - all configs json and tracker_config files)
            detector_type: Type of detector ("yolo")
            tracker_type: Type of tracker ("kalman", "speed")
            pixel_size: Pixel size in mm - dict {'x': float, 'y': float, 'average': float} (required)
            distance_map_path: Path to distance map .npz file (optional, overrides pixel_size)
            model_path: Path to YOLO model file (default: "weights/zoom1/best.pt")
            detector_config: Detector configuration dictionary (optional)
            tracker_config: Tracker configuration dictionary (optional) - boundary_margin_ratio, etc.
            calibration_config: Calibration configuration dictionary (optional)
            fps: Frame rate (optional, defaults to 30)
            max_frames_lost: Maximum frames without detection before track is lost
        """
        self.config = None  # Not used - all configs json and tracker_config files
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        
        # pixel_size는 반드시 dict 형태여야 함
        if not isinstance(pixel_size, dict):
            raise ValueError(f"pixel_size must be a dict with 'x' and 'y' keys, got {type(pixel_size)}")
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
        self.tracker_config = tracker_config or {}
        self.calibration_config = calibration_config
        
        # Get fps from parameter, config, or default
        # FPS and max_frames_lost are passed as parameters or use defaults
        # No need to read from SystemConfig
        self.fps = fps if fps is not None else 30.0
        self.max_frames_lost = max_frames_lost if max_frames_lost is not None else MAX_FRAMES_LOST

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
                    min_area=self.detector_config.get("min_area"),
                    max_area=self.detector_config.get("max_area"),
                    width_height_ratio_tolerance=self.detector_config.get("width_height_ratio_tolerance"),
                    mask_area_ratio=self.detector_config.get("mask_area_ratio"),
                    boundary_margin_ratio=self.detector_config.get("boundary_margin_ratio"),
                )
                logger.info(f"YOLO detector initialized (boundary_margin={self.detector_config.get('boundary_margin_ratio')})")
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
            # Get boundary_margin_ratio from tracker_config
            boundary_margin_ratio = self.tracker_config.get("boundary_margin_ratio", 0.1)
            
            self.tracker = KalmanTracker(
                fps=self.fps,
                pixel_size=self.pixel_size,
                distance_map_data=self.distance_map_data,
                track_id=0,
                max_frames_lost=self.max_frames_lost,
                boundary_margin_ratio=boundary_margin_ratio,
            )
            if self.distance_map_data:
                logger.info(f"Kalman filter tracker initialized (fps={self.fps}, using distance map, boundary_margin={boundary_margin_ratio})")
            else:
                # pixel_size는 항상 dict 형태
                ps_str = f"x={self.pixel_size.get('x', 1.0):.4f}, y={self.pixel_size.get('y', 1.0):.4f}"
                logger.info(f"Kalman filter tracker initialized (fps={self.fps}, pixel_size={ps_str}, boundary_margin={boundary_margin_ratio})")
        else:
            raise ValueError(
                f"Unsupported tracker type: {self.tracker_type}. Only 'kalman' is supported."
            )

        self.next_track_id = 0
        self.track_id = None

        # Initialize calibration components
        self._initialize_calibration()

    def _initialize_calibration(self):
        """Initialize calibration components (size measurement and visualizer)
        
        All calibration data (homography, pixel_size) should be provided via calibration_config
        from camera_manager, which loads from camX_tracker_config.json.
        """
        import numpy as np
        
        # Get calibration parameters from calibration_config (loaded from tracker_config)
        if not self.calibration_config:
            logger.debug("No calibration_config provided - size measurement and visualizer disabled")
            return
        
        # Get homography from calibration_config
        homography = None
        if "homography" in self.calibration_config:
            try:
                homography = np.array(self.calibration_config["homography"])
                logger.info("Homography loaded from calibration_config")
            except Exception as e:
                logger.warning(f"Failed to parse homography from calibration_config: {e}")
                return
        
        if homography is None:
            logger.debug("No homography in calibration_config - size measurement and visualizer disabled")
            return
        
        # Initialize SizeMeasurement
        # Note: camera_height, calibration_image_size, and pixel_size are not actually used
        # SizeMeasurement.measure() only uses homography for transformation
        # These parameters are kept for backward compatibility but use default values
        try:
            self.size_measurement = SizeMeasurement(
                homography=homography,
                camera_height=None,  # Not used (only needed for _adjust_homography_for_height which is commented out)
                pixel_size=1.0,  # Not used (only needed for _get_scaled_pixel_size which is never called)
                distance_map_data=self.distance_map_data,
                calibration_image_size=None,  # Not used (only needed for _get_scaled_pixel_size which is never called)
            )
            logger.info("Size measurement initialized with homography from calibration_config")
        except Exception as e:
            logger.warning(f"Error initializing SizeMeasurement: {e}")

        # Initialize Visualizer
        try:
            self.visualizer = Visualizer(homography=homography)
            logger.info("Visualizer initialized with homography from calibration_config")
        except Exception as e:
            logger.warning(f"Error initializing Visualizer: {e}")

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

                    # Safely get center and angle (oriented_box_info may be None for binary detector)
                    center = best_detection.get_center()
                    angle = best_detection.oriented_box_info["angle"] if best_detection.oriented_box_info else 0.0

                    self.tracker.initialize_with_detection(center, angle)
                    self.next_track_id += 1

                    # Update with first detection
                    # Get image size for boundary checking
                    img_height, img_width = frame.shape[:2]
                    tracking_result = self.tracker.update(
                        bbox=best_detection.bbox,
                        center=center,
                        frame_number=frame_number,
                        theta=angle,
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
                    # Safely get center and angle
                    center = best_detection.get_center()
                    angle = best_detection.oriented_box_info["angle"] if best_detection.oriented_box_info else 0.0
                    tracking_result = self.tracker.update(
                        bbox=best_detection.bbox,
                        center=center,
                        theta=angle,
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
        self, frame: np.ndarray, detections: List[Detection], tracking_results: List[Dict],
        draw_oriented_box: bool = False
    ) -> np.ndarray:
        """Visualize results using appropriate visualizer"""
        # Use multi-object AMR tracker visualization
        if self.visualizer and self.size_measurement:
            # Use enhanced visualizer if available
            detection_objects = detections
            vis_frame = self.visualizer.draw_single_object(
                frame, detection_objects, tracking_results, draw_oriented_box=draw_oriented_box
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