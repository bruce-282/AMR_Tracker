"""Detection data structures for object tracking."""

import time
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class Detection:
    """Represents a detected object with bounding box and metadata."""

    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        class_id: int = 0,
        class_name: str = "",
        timestamp: Optional[float] = None,
        masks: Optional[List[List[float]]] = None,
        frame_number: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        xywhr: Optional[np.ndarray] = None,
    ):
        """
        Initialize detection.

        Args:
            bbox: Bounding box [x, y, width, height] (fallback if no mask/xywhr)
            confidence: Detection confidence score
            class_id: Object class ID
            class_name: Object class name
            timestamp: Detection timestamp
            masks: Masks (polygon points)
            frame_number: Frame number where detection occurred
            image_size: Image size (width, height) for mask processing
            xywhr: OBB in [x_center, y_center, width, height, rotation] format (from model)
        """
        self.original_bbox = bbox  # Keep original YOLO bbox as fallback
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.timestamp = timestamp or time.time()
        self.masks = masks
        self.frame_number = frame_number
        
        # Priority: 1) xywhr (from model), 2) mask extraction, 3) original bbox
        self.oriented_box_info = None
        
        # First, try to use xywhr if available (from OBB model)
        if xywhr is not None:
            self.oriented_box_info = self._extract_box_from_xywhr(xywhr)
        
        if self.oriented_box_info is None and masks is not None and image_size is not None:
            self.oriented_box_info = Detection.extract_box_from_mask(masks, image_size)
            #self.logger.info(f"Extracted oriented box from mask: {self.oriented_box_info}")
        
        # Use oriented box if available, otherwise use original bbox
        if self.oriented_box_info is not None:
            # Convert oriented box to [x, y, w, h] format for compatibility
            center = self.oriented_box_info["center"]
            width = self.oriented_box_info["width"]
            height = self.oriented_box_info["height"]
            x = center[0] - width / 2
            y = center[1] - height / 2
            self.bbox = [x, y, width, height]
        else:
            self.bbox = bbox

    def _extract_box_from_xywhr(self, xywhr: np.ndarray) -> Optional[Dict]:
        """
        Extract oriented bounding box from xywhr format.
        
        Args:
            xywhr: Array in [x_center, y_center, width, height, rotation] format
                   rotation is in radians
        
        Returns:
            Dictionary with center, width, height, angle, and box_points
            - width: Long axis (major axis)
            - height: Short axis (minor axis)
            - angle: 0 when long axis is horizontal (parallel to x-axis)
        """
        try:
            if xywhr is None or len(xywhr) < 5:
                return None
            
            x_center, y_center, w, h, rotation_rad = xywhr[:5]
            center = (float(x_center), float(y_center))
            w = float(w)
            h = float(h)
            angle_rad = float(rotation_rad)
            angle_deg = np.rad2deg(angle_rad)
            
            # Normalize angle so that the LONG axis (major axis) is the reference
            # When long axis is parallel to image horizontal (x-axis), angle should be 0
            if h > w:
                # Height is the long axis, rotate reference by 90 degrees
                normalized_angle = angle_deg - 90
                long_axis = h
                short_axis = w
            else:
                # Width is the long axis
                normalized_angle = angle_deg
                long_axis = w
                short_axis = h
            
            # Normalize angle to -90 ~ 90 range
            while normalized_angle > 90:
                normalized_angle -= 180
            while normalized_angle < -90:
                normalized_angle += 180
            
            # Create box points from center, size, and original angle
            rect = ((x_center, y_center), (w, h), angle_deg)
            box_points = cv2.boxPoints(rect).astype(np.float32)
            
            return {
                "center": center,  # (x, y)
                "width": long_axis,  # Long axis (major axis)
                "height": short_axis,  # Short axis (minor axis)
                "angle": normalized_angle,  # degrees (0 when long axis is horizontal)
                "angle_rad": np.deg2rad(normalized_angle),  # radians
                "box_points": box_points,  # 4 corner points
            }
        except Exception as e:
            print(f"[WARN] Failed to extract box from xywhr: {e}")
            return None
    
    @staticmethod
    def extract_box_from_mask(
        masks: List[List[float]], 
        image_size: Tuple[int, int], 
        min_area: int = 200
    ) -> Optional[Dict]:
        """
        Extract oriented bounding box from mask using minAreaRect.
        
        Args:
            masks: Mask polygon points
            image_size: Image size (width, height)
            min_area: Minimum area threshold for safety check (YOLO detector already filters by min_area)
        
        Returns:
            Dictionary with center, width, height, angle, and box_points, or None if failed
        """
        if masks is None:
            return None
        
        try:
            poly = np.asarray(masks, dtype=np.float32)
            if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
                return None
            
            # # Optional: filter by min_area as safety check (YOLO detector already filters)
            # poly_area = cv2.contourArea(poly)
            # if poly_area < min_area:
            #     return None
            
            # Approximate polygon to 4 points using approxPolyDP (can represent trapezoid)
            # epsilon: approximation accuracy (percentage of perimeter)
            epsilon = 0.01 * cv2.arcLength(poly, True)
            approx = cv2.approxPolyDP(poly, epsilon, True)
            
            # Ensure we have 4 points (if not, use minAreaRect as fallback)
            if len(approx) != 4:
                # Fallback to minAreaRect if approximation doesn't give 4 points
                rect = cv2.minAreaRect(poly)
                center, (w, h), angle = rect
                
                # Normalize angle
                if h > w:
                    normalized_angle = angle - 90
                    long_axis = h
                    short_axis = w
                else:
                    normalized_angle = angle
                    long_axis = w
                    short_axis = h
                
                # Normalize angle to -90 ~ 90 range
                while normalized_angle > 90:
                    normalized_angle -= 180
                while normalized_angle < -90:
                    normalized_angle += 180
                
                box_points = cv2.boxPoints(rect).astype(np.float32)
            else:
                # Use approximated 4 points (can be trapezoid)
                box_points = approx.reshape(4, 2).astype(np.float32)
                
                # Calculate center as centroid of 4 points
                center = np.mean(box_points, axis=0)
                
                # Calculate dimensions: use minAreaRect for width/height/angle
                # (approxPolyDP gives points but not dimensions)
                rect = cv2.minAreaRect(box_points)
                _, (w, h), angle = rect
                
                # Normalize angle
                if h > w:
                    normalized_angle = angle - 90
                    long_axis = h
                    short_axis = w
                else:
                    normalized_angle = angle
                    long_axis = w
                    short_axis = h
                
                # Normalize angle to -90 ~ 90 range
                while normalized_angle > 90:
                    normalized_angle -= 180
                while normalized_angle < -90:
                    normalized_angle += 180
            
            return {
                "center": tuple(center),  # (x, y)
                "width": long_axis,  # Long axis (major axis)
                "height": short_axis,  # Short axis (minor axis)
                "angle": normalized_angle,  # degrees (0 when long axis is horizontal)
                "angle_rad": np.deg2rad(normalized_angle),  # radians
                "box_points": box_points,  # 4 corner points (can be trapezoid)
                "rect": rect,  # minAreaRect result for refinement (from approximated points)
            }
        except Exception as e:
            print(f"⚠ Failed to extract box from mask: {e}")
            return None

    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box (from mask if available, otherwise from bbox)."""
        if self.oriented_box_info is not None:
            return self.oriented_box_info["center"]
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)
    def get_width(self) -> float:
        """Get width of bounding box (from mask if available, otherwise from bbox)."""
        if self.oriented_box_info is not None:
            return self.oriented_box_info["width"]
        _, _, w, _ = self.bbox
        return w
    def get_height(self) -> float:
        """Get height of bounding box (from mask if available, otherwise from bbox)."""
        if self.oriented_box_info is not None:
            return self.oriented_box_info["height"]
        _, _, _, h = self.bbox
        return h

    def get_area(self) -> float:
        """Get area of bounding box (from mask if available, otherwise from bbox)."""
        if self.oriented_box_info is not None:
            return self.oriented_box_info["width"] * self.oriented_box_info["height"]
        _, _, w, h = self.bbox
        return w * h
    
    def get_orientation(self) -> Optional[float]:
        """Get orientation angle in degrees (from mask if available)."""
        if self.oriented_box_info is not None:
            return self.oriented_box_info["angle"]
        return None

    def to_dict(self) -> dict:
        """Convert detection to dictionary format."""
        result = {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "timestamp": self.timestamp,
            "type": self.class_name,  # For backward compatibility
            "masks": self.masks,
            "frame_number": self.frame_number,
        }
        
        # Add oriented box info if available
        if self.oriented_box_info is not None:
            result["oriented_box"] = {
                "center": self.oriented_box_info["center"],
                "width": self.oriented_box_info["width"],
                "height": self.oriented_box_info["height"],
                "angle": self.oriented_box_info["angle"],
                "angle_rad": self.oriented_box_info["angle_rad"],
            }
            result["orientation"] = self.oriented_box_info["angle_rad"]
        
        return result

    def __str__(self) -> str:
        """String representation of detection."""
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"

    def __repr__(self) -> str:
        """Detailed string representation of detection."""
        return f"Detection(bbox={self.bbox}, confidence={self.confidence}, class_id={self.class_id}, class_name='{self.class_name}', timestamp={self.timestamp}, frame_number={self.frame_number})"
