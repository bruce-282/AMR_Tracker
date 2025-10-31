"""Detection data structures for object tracking."""

import time
from typing import List, Tuple, Optional


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
    ):
        """
        Initialize detection.

        Args:
            bbox: Bounding box [x, y, width, height]
            confidence: Detection confidence score
            class_id: Object class ID
            class_name: Object class name
            timestamp: Detection timestamp
            masks: Masks
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.timestamp = timestamp or time.time()
        self.masks = masks

    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    def get_area(self) -> float:
        """Get area of bounding box."""
        _, _, w, h = self.bbox
        return w * h

    def to_dict(self) -> dict:
        """Convert detection to dictionary format."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "timestamp": self.timestamp,
            "type": self.class_name,  # For backward compatibility
            "masks": self.masks,
        }

    def __str__(self) -> str:
        """String representation of detection."""
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"

    def __repr__(self) -> str:
        """Detailed string representation of detection."""
        return f"Detection(bbox={self.bbox}, confidence={self.confidence}, class_id={self.class_id}, class_name='{self.class_name}', timestamp={self.timestamp})"
