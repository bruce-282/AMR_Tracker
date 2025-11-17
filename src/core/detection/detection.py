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
        
        # If no xywhr, try to extract from mask
        if self.oriented_box_info is None and masks is not None and image_size is not None:
            self.oriented_box_info = self._extract_box_from_mask(image_size)
        
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
        """
        try:
            if xywhr is None or len(xywhr) < 5:
                return None
            
            x_center, y_center, width, height, rotation_rad = xywhr[:5]
            center = (float(x_center), float(y_center))
            width = float(width)
            height = float(height)
            angle_rad = float(rotation_rad)
            angle_deg = np.rad2deg(angle_rad)
            
            # Create box points from center, size, and angle
            # OpenCV's minAreaRect uses angle in degrees, but we'll use radians internally
            # Convert to OpenCV format: angle in degrees, measured from horizontal
            rect = ((x_center, y_center), (width, height), np.rad2deg(angle_rad))
            box_points = cv2.boxPoints(rect).astype(np.float32)
            
            return {
                "center": center,  # (x, y)
                "width": width,
                "height": height,
                "angle": angle_deg,  # degrees
                "angle_rad": angle_rad,  # radians
                "box_points": box_points,  # 4 corner points
            }
        except Exception as e:
            print(f"[WARN] Failed to extract box from xywhr: {e}")
            return None
    
    def _clean_mask(self, mask: np.ndarray, min_area: int = 200) -> np.ndarray:
        """
        Clean mask by removing small components.
        
        Args:
            mask: Binary mask
            min_area: Minimum area threshold
            
        Returns:
            Cleaned mask
        """
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        for cnt in cnts:
            if cv2.contourArea(cnt) >= min_area:
                cv2.fillPoly(out, [cnt], 255)
        return out

    def _extract_box_from_mask(
        self, image_size: Tuple[int, int], min_area: int = 200
    ) -> Optional[Dict]:
        """
        Extract oriented bounding box from mask using minAreaRect.
        
        Args:
            image_size: Image size (width, height)
            min_area: Minimum area threshold for mask cleaning
            
        Returns:
            Dictionary with center, width, height, angle, and box_points, or None if failed
        """
        if self.masks is None:
            return None
        
        try:
            poly = np.asarray(self.masks, dtype=np.float32)
            if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
                return None
            
            pts = poly.reshape((-1, 1, 2)).astype(np.int32)
            
            # Build binary mask from polygon
            h, w_img = image_size[1], image_size[0]
            bin_mask = np.zeros((h, w_img), dtype=np.uint8)
            cv2.fillPoly(bin_mask, [pts], 255)
            
            # Clean mask
            m = self._clean_mask(bin_mask, min_area=min_area)
            if m.max() == 0:
                return None
            
            # Erode to get core
            erode_px = 3
            core = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8), 1)
            if int(core.sum()) < 50:
                core = m
            
            # Find contours
            cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            
            # Get largest contour
            cnt = max(cnts, key=cv2.contourArea)
            
            # Get minAreaRect
            rect = cv2.minAreaRect(cnt)
            center, (width, height), angle = rect
            
            # Get box points
            box_points = cv2.boxPoints(rect).astype(np.float32)
            
            return {
                "center": center,  # (x, y)
                "width": width,
                "height": height,
                "angle": angle,  # degrees
                "angle_rad": np.deg2rad(angle),  # radians
                "box_points": box_points,  # 4 corner points
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
