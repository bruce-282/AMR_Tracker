"""Enhanced AMR Tracker - Core tracking system."""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

from config import SystemConfig
from src.core.detection import Detection, YOLODetector
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
        model_path: Optional[str] = None,
    ):
        """
        Initialize enhanced AMR system

        Args:
            config: System configuration
            detector_type: Type of detector ("yolo")
            tracker_type: Type of tracker ("kalman", "speed")
            pixel_size: Pixel size in mm
            model_path: Path to YOLO model file (default: "weights/zoom1/best.pt")
        """
        self.config = config if config else SystemConfig()
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.pixel_size = pixel_size

        # Initialize components
        self.detector = None
        self.size_measurement = None
        self.visualizer = None
        
        # Get fps from config if available
        fps = 30
        if self.config and hasattr(self.config, "measurement"):
            fps = self.config.measurement.fps

        self.tracker = KalmanTracker(
            fps=fps,
            pixel_size=self.pixel_size,
            track_id=0,
        )

        # Store model path for initialization
        self.model_path = model_path or "weights/zoom1/best.pt"

        self._initialize_components()

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
                
                self.detector = YOLODetector(
                    self.model_path,
                    confidence_threshold=0.5,
                    device=device,
                    imgsz=1536,
                    target_classes=[0],
                )
                logger.info("YOLO detector initialized")
            except ImportError:
                raise ImportError("ultralytics module is not installed.")
            except FileNotFoundError:
                raise FileNotFoundError(f"weights file not found: {self.model_path}")
        else:
            raise ValueError(
                f"Unsupported detector type: {self.detector_type}. Only 'yolo' is supported."
            )

        self.next_track_id = 0
        self.track_id = None
        logger.info("Multi-object Kalman filter tracker initialized")

        # Initialize additional components if using new modules
        if self.config:
            try:
                # Load calibration data if available
                try:
                    # SystemConfig object
                    calibration_path = self.config.calibration.calibration_data_path
                except AttributeError as e:
                    logger.debug(f"Error accessing calibration_data_path: {e}")
                    calibration_path = None
                if calibration_path and Path(calibration_path).exists():
                    with open(calibration_path, "r") as f:
                        calibration_data = json.load(f)

                    self.size_measurement = SizeMeasurement(
                        homography=np.array(calibration_data["homography"]),
                        camera_height=self.config.calibration.camera_height,
                        pixel_size=calibration_data.get("pixel_size", 1.0),
                        calibration_image_size=self.config.calibration.calibration_image_size,
                    )
                    logger.info("Size measurement initialized")

                    self.visualizer = Visualizer(
                        homography=np.array(calibration_data["homography"])
                    )
                    logger.info("Visualizer initialized")
                else:
                    logger.info("No calibration data found - size measurement disabled")
            except Exception as e:
                logger.warning(f"Error initializing modules: {e}")

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
                    tracking_result = self.tracker.update(
                        bbox=best_detection.bbox,
                        center=best_detection.oriented_box_info["center"],
                        frame_number=frame_number,
                        theta=best_detection.oriented_box_info["angle"],
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

                    tracking_result = self.tracker.update(
                        bbox=best_detection.bbox,
                        center=best_detection.oriented_box_info["center"],
                        theta=best_detection.oriented_box_info["angle"],
                        frame_number=frame_number,
                    )
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
                        tracking_result = self.tracker.update(
                            bbox=None,
                            center=None,
                            theta=None,
                            frame_number=frame_number,
                        )
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

        return vis_frame

    def reset(self):
        """Reset tracker state"""
        if self.tracker:
            self.tracker.reset()
        if self.visualizer:
            self.visualizer.reset()
        self.track_id = None
        self.next_track_id = 0

