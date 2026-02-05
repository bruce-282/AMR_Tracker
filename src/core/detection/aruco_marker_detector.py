"""
ArUco marker detector for AMR tracking system.

Detects a single ArUco marker with a given ID in each frame.
Returns detection in the same format as BinaryDetector/YOLODetector for optional use.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .detection import Detection

logger = logging.getLogger(__name__)

# Map string dictionary names to OpenCV constants
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


class ArUcoMarkerDetector:
    """
    ArUco marker detector for a single marker ID per frame.

    Configurable marker ID and (optional) physical marker size.
    Returns at most one Detection per frame (the marker with the given ID).
    """

    def __init__(
        self,
        marker_id: int = 0,
        marker_size: float = 0.1,
        dictionary: str = "DICT_4X4_50",
        min_marker_perimeter: float = 0.0,
        max_marker_perimeter: float = 0.0,
        class_name: str = "aruco",
    ):
        """
        Initialize ArUco marker detector.

        Args:
            marker_id: ArUco marker ID to detect (only this ID is returned).
            marker_size: Physical marker size in meters (for logging/future pose; not used for 2D bbox).
            dictionary: ArUco dictionary name (e.g. "DICT_4X4_50", "DICT_6X6_250").
            min_marker_perimeter: Optional min contour perimeter (0 = disabled).
            max_marker_perimeter: Optional max contour perimeter (0 = disabled).
            class_name: class_name for Detection (e.g. "aruco" or "agv").
        """
        self.marker_id = int(marker_id)
        self.marker_size = float(marker_size)
        self.dictionary_name = dictionary
        self.min_marker_perimeter = float(min_marker_perimeter)
        self.max_marker_perimeter = float(max_marker_perimeter)
        self.class_name = class_name

        dict_enum = ARUCO_DICT_MAP.get(
            dictionary.upper() if isinstance(dictionary, str) else "DICT_4X4_50",
            cv2.aruco.DICT_4X4_50,
        )
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_enum)

        try:
            self._detector = cv2.aruco.ArucoDetector(
                self._aruco_dict,
                cv2.aruco.DetectorParameters(),
            )
            self._use_aruco_detector = True
        except AttributeError:
            self._detector = None
            self._use_aruco_detector = False

    def detect(
        self,
        image: np.ndarray,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect the configured ArUco marker (single ID) in the image.

        Args:
            image: Input image (BGR or grayscale).
            frame_number: Frame number.
            timestamp: Optional timestamp.

        Returns:
            List of 0 or 1 Detection (only the marker with marker_id).
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if self._use_aruco_detector and self._detector is not None:
                corners, ids, _ = self._detector.detectMarkers(gray)
            else:
                params = cv2.aruco.DetectorParameters()
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self._aruco_dict, parameters=params
                )

            if ids is None or len(ids) == 0:
                return []

            ids_flat = ids.flatten()
            idx = np.where(ids_flat == self.marker_id)[0]
            if len(idx) == 0:
                return []

            i = int(idx[0])
            corner_pts = corners[i][0]  # (4, 2)

            # Optional perimeter filter
            if self.min_marker_perimeter > 0 or self.max_marker_perimeter > 0:
                perim = cv2.arcLength(corner_pts.astype(np.float32), True)
                if self.min_marker_perimeter > 0 and perim < self.min_marker_perimeter:
                    return []
                if self.max_marker_perimeter > 0 and perim > self.max_marker_perimeter:
                    return []

            x, y, w, h = cv2.boundingRect(corner_pts.astype(np.int32))
            image_size = (image.shape[1], image.shape[0])
            img_w, img_h = image.shape[1], image.shape[0]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img_w or y + h > img_h:
                return []

            poly_xy = corner_pts.astype(np.float32).tolist()
            # ArUco corners are already 4 points (TL, TR, BR, BL); use them directly (no minAreaRect)
            pts = corner_pts.astype(np.float32)
            center = np.mean(pts, axis=0)
            # Edge lengths: [0-1]=top, [1-2]=right, [2-3]=bottom, [3-0]=left
            e01 = float(np.linalg.norm(pts[1] - pts[0]))
            e12 = float(np.linalg.norm(pts[2] - pts[1]))
            e23 = float(np.linalg.norm(pts[3] - pts[2]))
            e30 = float(np.linalg.norm(pts[0] - pts[3]))
            width_avg = (e01 + e23) * 0.5
            height_avg = (e12 + e30) * 0.5
            # Angle from top edge (pts[0] -> pts[1])
            top_vec = pts[1] - pts[0]
            angle_deg = float(np.degrees(np.arctan2(top_vec[1], top_vec[0])))
            if height_avg > width_avg:
                long_axis, short_axis = height_avg, width_avg
                normalized_angle = angle_deg - 90
            else:
                long_axis, short_axis = width_avg, height_avg
                normalized_angle = angle_deg
            while normalized_angle > 90:
                normalized_angle -= 180
            while normalized_angle < -90:
                normalized_angle += 180

            oriented_box_info = {
                "center": tuple(center),
                "width": long_axis,
                "height": short_axis,
                "angle": normalized_angle,
                "angle_rad": np.deg2rad(normalized_angle),
                "box_points": pts,
            }

            detection = Detection(
                bbox=[float(x), float(y), float(w), float(h)],
                confidence=1.0,
                class_id=self.marker_id,
                class_name=self.class_name,
                timestamp=timestamp,
                masks=poly_xy,
                frame_number=frame_number,
                image_size=image_size,
                xywhr=None,
            )
            detection.oriented_box_info = oriented_box_info
            return [detection]

        except Exception as e:
            logger.debug("ArUco detection failed: %s", e)
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Return detector configuration for logging/debug."""
        return {
            "detector_type": "aruco",
            "marker_id": self.marker_id,
            "marker_size": self.marker_size,
            "dictionary": self.dictionary_name,
            "class_name": self.class_name,
        }
