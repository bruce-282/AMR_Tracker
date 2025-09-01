"""Module for measuring physical dimensions of detected objects."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from detection.agv_detector import Detection


class SizeMeasurement:
    """
    Measures physical dimensions of detected AGVs.

    Converts pixel measurements to real-world dimensions using
    calibrated homography transformation.
    """

    def __init__(self, homography: np.ndarray, camera_height: float):
        """
        Initialize size measurement.

        Args:
            homography: Homography matrix for ground plane
            camera_height: Height of camera above ground in mm
        """
        self.H = homography
        self.camera_height = camera_height

    def measure(self, detection: Detection, object_height: float = 0) -> Dict:
        """
        Measure physical dimensions of detected object.

        Args:
            detection: Detection object with bounding box
            object_height: Height of object above ground in mm

        Returns:
            Dictionary with measurement results
        """
        # Adjust homography for object height
        H_adjusted = self._adjust_homography_for_height(object_height)

        # Transform corners to world coordinates
        world_corners = self._transform_to_world(detection.bbox, H_adjusted)

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

    def _adjust_homography_for_height(self, height: float) -> np.ndarray:
        """Adjust homography matrix for object height."""
        if height == 0:
            return self.H

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
