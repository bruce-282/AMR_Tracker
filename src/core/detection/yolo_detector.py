"""
YOLO-based detector for AMR tracking system.

This module provides YOLO-based object detection functionality
for detecting AGVs and other objects in video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from .detection import Detection


class YOLODetector:
    """
    YOLO-based detector for AMR objects.

    Uses YOLO model to detect objects in video frames and
    returns detection results in a standardized format.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cuda",
        target_classes: Optional[List[int]] = None,
        imgsz: int = 768,
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu' or 'cuda')
            target_classes: List of class IDs to detect (None for all classes)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.target_classes = target_classes
        self.imgsz = imgsz
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            import ultralytics

            self.model = ultralytics.YOLO(self.model_path)
            # Set device for inference
            self.model.to(self.device)
            print(f"✓ YOLO model loaded from {self.model_path} on {self.device}")
        except ImportError:
            raise ImportError("ultralytics package is required for YOLO detection")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(
        self, image: np.ndarray, frame_number: int = 0, timestamp: float = None
    ) -> List[Detection]:
        """
        Detect objects in image.

        Args:
            image: Input image (BGR format)
            frame_number: Frame number
            timestamp: Timestamp
        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []

        try:
            # Run YOLO inference
            results = self.model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                retina_masks=True,
                verbose=False,
            )

            detections = []
            for result in results:
                if result.obb is not None:
                    obbs = result.obb.xyxy.cpu().numpy()
                    print(f"obbs: {obbs}")
                if result.boxes is not None:
                    masks = result.masks

                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    print(f"boxes: {boxes}")
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for i, (box, conf, class_id) in enumerate(
                        zip(boxes, confidences, class_ids)
                    ):
                        # Filter by target classes if specified
                        if (
                            self.target_classes is not None
                            and class_id not in self.target_classes
                        ):
                            continue

                        # Convert to [x, y, w, h] format
                        x1, y1, x2, y2 = box
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1

                        # Prepare polygon mask per detection if available
                        poly_xy = None
                        try:
                            if (
                                masks is not None
                                and hasattr(masks, "xy")
                                and masks.xy is not None
                            ):
                                poly_xy = masks.xy[i]  # Nx2 numpy array (float)
                        except Exception:
                            poly_xy = None

                        # Create Detection object
                        detection = Detection(
                            bbox=[x, y, w, h],
                            confidence=float(conf),
                            class_id=int(class_id),
                            class_name=self._get_class_name(class_id),
                            timestamp=timestamp,
                            masks=poly_xy,
                            frame_number=frame_number,
                        )
                        detections.append(detection)

            return detections

        except Exception as e:
            print(f"⚠ YOLO detection failed: {e}")
            return []

    def _get_class_name(self, class_id: int) -> str:
        """
        Get class name from class ID.

        Args:
            class_id: Class ID

        Returns:
            Class name
        """
        # Default class names (can be customized based on your model)
        class_names = {0: "amr", 1: "agv", 2: "robot", 3: "vehicle", 4: "person"}
        return class_names.get(class_id, f"class_{class_id}")

    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "device": self.device,
            "model_loaded": self.model is not None,
        }
