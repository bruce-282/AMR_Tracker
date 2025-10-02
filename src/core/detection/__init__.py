"""Detection package for AGV measurement system."""

from .detection import Detection
from .yolo_detector import YOLODetector

__all__ = ["Detection", "YOLODetector"]
