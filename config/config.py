"""Configuration settings for the AGV measurement system."""

import json
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CalibrationConfig:
    """Calibration configuration parameters."""

    checkerboard_size: Tuple[int, int] = (8, 7)  # (columns-1, rows-1)
    square_size: float = 20.0  # mm
    num_calibration_images: int = 15
    camera_height: float = 2000.0  # mm from ground
    calibration_image_size: Tuple[int, int] = (3840, 2160)  # 4K 이미지 크기
    # File paths
    calibration_data_path: str = "calibration_data.json"
    calibration_images_dir: str = "calibration_images/"


@dataclass
class MeasurementConfig:
    """Measurement configuration parameters."""

    min_agv_area: float = 1000  # minimum pixel area for detection
    max_tracking_distance: float = 500  # mm, for tracking between frames
    fps: int = 30  # camera frame rate
    pixel_size: float = 1.0  # pixel size in mm

    # AGV types and their known heights
    agv_heights = {"small": 200, "medium": 300, "large": 400}  # mm


@dataclass
class TrackingConfig:
    """Tracking configuration parameters."""

    # Speed thresholds
    speed_threshold_pix_per_frame: float = 5.0  # tracking stops when speed exceeds this
    speed_near_zero_threshold: float = 3.0  # speed considered near zero below this
    speed_zero_frames_threshold: int = 10  # frames before sending response when speed is near zero

    # Detection thresholds
    detection_loss_threshold_frames: int = 30  # frames before tracker is considered lost
    max_frames_lost: int = 500  # frames before removing tracker

    # Trajectory settings
    camera2_trajectory_max_frames: int = 300  # trajectory frames for camera 2

    # Kalman filter parameters
    velocity_damping_threshold: float = 0.5  # pixels - threshold for velocity damping
    velocity_damping_factor: float = 0.9  # damping factor when stationary
    velocity_damping_max: float = 5.0  # only damp velocities below this

    # Visualization
    arrow_length: int = 50  # pixels for orientation arrow



@dataclass
class SystemConfig:
    """Overall system configuration."""

    calibration: CalibrationConfig = None
    measurement: MeasurementConfig = None
    tracking: TrackingConfig = None

    # Display settings
    display_scale: float = 0.5  # scale factor for display window
    record_video: bool = False
    output_video_path: str = "output.mp4"

    # Execution settings (optional)
    execution: dict = None

    def __post_init__(self):
        if self.calibration is None:
            self.calibration = CalibrationConfig()
        if self.measurement is None:
            self.measurement = MeasurementConfig()
        if self.tracking is None:
            self.tracking = TrackingConfig()

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Create nested objects
        calibration_data = data.get("calibration", {})
        measurement_data = data.get("measurement", {})
        tracking_data = data.get("tracking", {})

        calibration = CalibrationConfig(**calibration_data)
        measurement = MeasurementConfig(**measurement_data)
        tracking = TrackingConfig(**tracking_data)

        return cls(
            calibration=calibration,
            measurement=measurement,
            tracking=tracking,
            display_scale=data.get("display_scale", 0.5),
            record_video=data.get("record_video", False),
            output_video_path=data.get("output_video_path", "output.mp4"),
            execution=data.get("execution", None),
        )


# python main.py --source "data/Test.mkv" --mode basic
