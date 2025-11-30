"""
Binary threshold-based detector for AMR tracking system.

This module provides binary threshold-based object detection functionality
for detecting dark objects (e.g., black AGVs) in video frames.
"""

import cv2
import numpy as np
from typing import List, Optional
from .detection import Detection


class BinaryDetector:
    """
    Binary threshold-based detector for dark objects.
    
    Uses binary thresholding to detect dark objects and returns
    detection results in a standardized format compatible with YOLODetector.
    """
    
    def __init__(
        self,
        threshold: int = 50,
        min_area: int = 1000,
        max_area: Optional[int] = None,
        width_height_ratio_min: float = 0.8,
        width_height_ratio_max: float = 1.2,
        mask_area_ratio: float = 0.9,
        inverse: bool = True,
        use_adaptive: bool = True,
        adaptive_block_size: int = 11,
        adaptive_c: float = 2.0,
    ):
        """
        Initialize binary detector.
        
        Args:
            threshold: Binary threshold value (0-255) - used only if use_adaptive=False
            min_area: Minimum area (in pixels) for detected objects
            max_area: Maximum area (in pixels) for detected objects (None for no limit)
            width_height_ratio_min: Minimum width/height ratio
            width_height_ratio_max: Maximum width/height ratio
            mask_area_ratio: Minimum ratio of mask area to bbox area
            inverse: If True, detects dark objects (THRESH_BINARY_INV). 
                    If False, detects bright objects (THRESH_BINARY).
            use_adaptive: If True, use adaptive threshold. If False, use simple threshold.
            adaptive_block_size: Size of a pixel neighborhood for adaptive threshold (must be odd, e.g., 3, 5, 7, 11)
            adaptive_c: Constant subtracted from the mean (typically 2-10)
        """
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.width_height_ratio_min = width_height_ratio_min
        self.width_height_ratio_max = width_height_ratio_max
        self.mask_area_ratio = mask_area_ratio
        self.inverse = inverse
        self.use_adaptive = use_adaptive
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        
        # Ensure block_size is odd
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1
        
        # Store last binary image for debugging
        self.last_binary_image: Optional[np.ndarray] = None
        self.last_gray_image: Optional[np.ndarray] = None
    
    def detect(
        self, image: np.ndarray, frame_number: int = 0, timestamp: float = None
    ) -> List[Detection]:
        """
        Detect objects in image using binary thresholding.
        
        Args:
            image: Input image (BGR format)
            frame_number: Frame number
            timestamp: Timestamp
            
        Returns:
            List of Detection objects (at most one - the largest object)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Store for debugging
            self.last_gray_image = gray.copy()
            
            # Apply binary threshold
            if self.use_adaptive:
                # Use adaptive threshold for uneven lighting conditions
                # If inverse=True: detect dark objects (THRESH_BINARY_INV)
                # If inverse=False: detect bright objects (THRESH_BINARY)
                threshold_type = cv2.THRESH_BINARY_INV if self.inverse else cv2.THRESH_BINARY
                binary = cv2.adaptiveThreshold(
                    gray,
                    255,  # maxValue
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # adaptiveMethod
                    threshold_type,
                    self.adaptive_block_size,  # blockSize
                    self.adaptive_c  # C
                )
            else:
                # Use simple threshold
                threshold_type = cv2.THRESH_BINARY_INV if self.inverse else cv2.THRESH_BINARY
                _, binary = cv2.threshold(gray, self.threshold, 255, threshold_type)
            
            # Store for debugging
            self.last_binary_image = binary.copy()
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return []
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Filter by area
            if contour_area < self.min_area:
                return []
            
            if self.max_area is not None and contour_area > self.max_area:
                return []
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Filter by width/height ratio
            if h > 0:
                width_height_ratio = w / h
                if (width_height_ratio < self.width_height_ratio_min or 
                    width_height_ratio > self.width_height_ratio_max):
                    return []
            
            # Convert contour to polygon (mask)
            # Simplify contour to reduce points
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Convert to list of [x, y] points
            poly_xy = approx.reshape(-1, 2).astype(np.float32)
            
            # Calculate mask area from polygon
            mask_area = cv2.contourArea(largest_contour)
            bbox_area = w * h
            
            # Filter by mask area ratio
            if mask_area < bbox_area * self.mask_area_ratio:
                return []
            
            # Get image size
            image_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Validate bbox bounds
            img_w, img_h = image.shape[1], image.shape[0]
            if x < 0 or y < 0 or w < 0 or h < 0 or x + w > img_w or y + h > img_h:
                return []
            
            # Create Detection object (compatible with YOLODetector output)
            detection = Detection(
                bbox=[float(x), float(y), float(w), float(h)],
                confidence=1.0,  # Binary detector always has full confidence
                class_id=0,  # Default class
                class_name="agv",
                timestamp=timestamp,
                masks=poly_xy.tolist(),  # Convert to list of lists
                frame_number=frame_number,
                image_size=image_size,
                xywhr=None,  # No oriented bounding box for binary detection
            )
            
            return [detection]
            
        except Exception as e:
            print(f"⚠ Binary detection failed: {e}")
            return []
    
    def get_binary_image(self) -> Optional[np.ndarray]:
        """
        Get the last binary threshold result image for debugging.
        
        Returns:
            Binary image (grayscale, 0-255) or None if not available
        """
        return self.last_binary_image
    
    def get_debug_image(self, original_image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get a debug visualization showing binary and binary with contour.
        
        Args:
            original_image: Original BGR image (optional, uses last processed if not provided)
        
        Returns:
            Combined debug image showing binary and binary+contour side by side, or None if not available
        """
        if self.last_binary_image is None:
            return None
        
        # Binary display (convert to BGR)
        binary_display = cv2.cvtColor(self.last_binary_image, cv2.COLOR_GRAY2BGR)
        
        # Binary with contour overlay
        binary_contour = binary_display.copy()
        contours, _ = cv2.findContours(self.last_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Draw all contours in green
            cv2.drawContours(binary_contour, contours, -1, (0, 255, 0), 2)
            # Draw largest contour in red
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(binary_contour, [largest_contour], -1, (0, 0, 255), 3)
        
        # Resize to same size (use original image size as reference, or binary image size)
        h, w = binary_display.shape[:2]
        # Use larger size for better visibility (about 1/3 of original width)
        target_size = (w // 3, h // 3)
        
        binary_resized = cv2.resize(binary_display, target_size)
        binary_contour_resized = cv2.resize(binary_contour, target_size)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        
        mode_text = "Dark" if self.inverse else "Bright"
        thresh_text = f"Adaptive (BS={self.adaptive_block_size}, C={self.adaptive_c})" if self.use_adaptive else f"TH={self.threshold}"
        cv2.putText(binary_resized, f"Binary {mode_text} ({thresh_text})", (10, 30), font, font_scale, color, thickness)
        cv2.putText(binary_contour_resized, "Binary + Contours", (10, 30), font, font_scale, color, thickness)
        
        # Combine into 1x2 grid (side by side)
        combined = np.hstack([binary_resized, binary_contour_resized])
        
        return combined
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with detector information
        """
        return {
            "detector_type": "binary",
            "threshold": self.threshold,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "inverse": self.inverse,
            "use_adaptive": self.use_adaptive,
            "adaptive_block_size": self.adaptive_block_size,
            "adaptive_c": self.adaptive_c,
        }

