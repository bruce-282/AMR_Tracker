"""Module for measuring physical dimensions of detected objects."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..detection.detection import Detection


class SizeMeasurement:
    """
    Measures physical dimensions of detected AGVs.

    Converts pixel measurements to real-world dimensions using
    calibrated homography transformation.
    """

    def __init__(
        self,
        homography: np.ndarray,
        camera_height: float = None,
        pixel_size: float = 1.0,
        distance_map_data: Optional[Dict] = None,
        calibration_image_size: tuple = None,
    ):
        """
        Initialize size measurement.

        Args:
            homography: Homography matrix for ground plane
            camera_height: Height of camera above ground in mm
            pixel_size: Pixel size in mm (from calibration, used if distance_map_data is None)
            distance_map_data: Distance map data dict from PixelDistanceMapper.load_distance_map() (optional)
            calibration_image_size: Original image size used for calibration (width, height)
        """
        self.H = homography
        self.camera_height = camera_height
        self.pixel_size = pixel_size
        self.distance_map_data = distance_map_data
        self.calibration_image_size = calibration_image_size

    def measure(
        self,
        detection: Detection,
        object_height: Optional[float] = None,
        current_image_size: tuple = None,
    ) -> Dict:
        """
        Measure physical dimensions of detected object.

        Args:
            detection: Detection object with bounding box
            object_height: Height of object above ground in mm
            current_image_size: Current image size (width, height) for pixel_size scaling

        Returns:
            Dictionary with measurement results
        """
        # Adjust homography for object height
        # H_adjusted = self._adjust_homography_for_height(object_height)

        # Get bounding box corners
        x, y, w, h = detection.bbox
        corners = np.array(
            [
                [x, y],  # top-left
                [x + w, y],  # top-right
                [x + w, y + h],  # bottom-right
                [x, y + h],  # bottom-left
            ]
        )

        # Transform corners to world coordinates
        world_corners = self._transform_to_world(corners, self.H)

        # Calculate dimensions
        measurements = self._calculate_dimensions(world_corners)

        # Add additional information
        measurements.update(
            {
                "center_world": self._calculate_center(world_corners),
                "orientation": self._calculate_orientation(world_corners),
                "area": self._calculate_area(world_corners),
                "timestamp": detection.timestamp,
                "confidence": detection.confidence,
            }
        )

        return measurements

    def _get_scaled_pixel_size(self, current_image_size: tuple = None) -> float:
        """
        Get pixel size scaled for current image size.

        Args:
            current_image_size: Current image size (width, height)

        Returns:
            Scaled pixel size in mm
        """
        if current_image_size is None or self.calibration_image_size is None:
            return self.pixel_size

        # Calculate scaling factor
        calib_width, calib_height = self.calibration_image_size
        curr_width, curr_height = current_image_size

        # Use width scaling (assuming same aspect ratio)
        scale_factor = curr_width / calib_width

        return self.pixel_size * scale_factor

    def _adjust_homography_for_height(
        self, height: Optional[float] = None
    ) -> np.ndarray:
        """Adjust homography matrix for object height."""
        if height is None:
            return self.H
        if self.camera_height is None:
            raise ValueError("Camera height is not set")
        # Scale factor based on height
        scale = (self.camera_height - height) / self.camera_height

        # Adjust homography
        H_adjusted = self.H.copy()
        H_adjusted[:2, :] *= scale

        return H_adjusted

    def _transform_to_world(
        self, image_points: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Transform image points to world coordinates."""
        # Convert to homogeneous coordinates
        pts_homo = np.column_stack([image_points, np.ones(len(image_points))])

        # Apply homography
        world_pts_homo = (H @ pts_homo.T).T

        # Normalize
        world_pts = world_pts_homo[:, :2] / world_pts_homo[:, 2:3]

        return world_pts

    def _calculate_dimensions(self, corners: np.ndarray) -> Dict:
        """
        Calculate object dimensions from corners.

        Assumes corners are ordered as:
        0 --- 1
        |     |
        3 --- 2
        """
        # Calculate edge lengths
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])

        # Calculate diagonals for rectangularity check
        diagonal1 = np.linalg.norm(corners[2] - corners[0])
        diagonal2 = np.linalg.norm(corners[3] - corners[1])

        # Average dimensions
        width = (width_top + width_bottom) / 2
        height = (height_left + height_right) / 2

        # Quality metrics
        width_variance = abs(width_top - width_bottom)
        height_variance = abs(height_left - height_right)
        rectangularity = 1 - abs(diagonal1 - diagonal2) / max(diagonal1, diagonal2)

        return {
            "width": width,
            "height": height,
            "width_variance": width_variance,
            "height_variance": height_variance,
            "rectangularity": rectangularity,
            "diagonal1": diagonal1,
            "diagonal2": diagonal2,
        }

    def _calculate_center(self, corners: np.ndarray) -> Tuple[float, float]:
        """Calculate center point of corners."""
        center = np.mean(corners, axis=0)
        return tuple(center)

    def _calculate_orientation(self, corners: np.ndarray) -> float:
        """
        Calculate orientation angle in degrees.
        Assumes front edge is corners[0] to corners[1].
        """
        front_vector = corners[1] - corners[0]
        angle_rad = np.arctan2(front_vector[1], front_vector[0])
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _calculate_area(self, corners: np.ndarray) -> float:
        """Calculate area using Shoelace formula."""
        x = corners[:, 0]
        y = corners[:, 1]

        area = 0.5 * abs(
            sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1))
        )

        return area
