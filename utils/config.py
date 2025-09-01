"""Configuration settings for the AGV measurement system."""

import json
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CalibrationConfig:
    """Calibration configuration parameters."""

    checkerboard_size: Tuple[int, int] = (9, 6)  # (columns-1, rows-1)
    square_size: float = 25.0  # mm
    num_calibration_images: int = 15
    camera_height: float = 2000.0  # mm from ground

    # File paths
    calibration_data_path: str = "calibration_data.json"
    calibration_images_dir: str = "calibration_images/"


@dataclass
class MeasurementConfig:
    """Measurement configuration parameters."""

    min_agv_area: float = 1000  # minimum pixel area for detection
    max_tracking_distance: float = 500  # mm, for tracking between frames
    fps: int = 30  # camera frame rate

    # AGV types and their known heights
    agv_heights = {"small": 200, "medium": 300, "large": 400}  # mm  # mm  # mm


@dataclass
class SystemConfig:
    """Overall system configuration."""

    calibration: CalibrationConfig = CalibrationConfig()
    measurement: MeasurementConfig = MeasurementConfig()

    # Display settings
    display_scale: float = 0.5  # scale factor for display window
    record_video: bool = False
    output_video_path: str = "output.mp4"

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
