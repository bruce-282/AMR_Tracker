"""
YOLO-based detector for AMR tracking system.

This module provides YOLO-based object detection functionality
for detecting AGVs and other objects in video frames.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from .detection import Detection

logger = logging.getLogger(__name__)


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
        device: str = "cuda",
        target_classes: Optional[List[int]] = None,
        imgsz: int = 768,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
        width_height_ratio_tolerance: Optional[float] = None,
        mask_area_ratio: Optional[float] = None,
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu' or 'cuda')
            target_classes: List of class IDs to detect (None for all classes)
            imgsz: Image size for inference
            min_area: Minimum detection area (pixels)
            max_area: Maximum detection area (pixels)
            width_height_ratio_tolerance: Tolerance for width/height ratio (e.g., 0.2 means 1.0 ± 0.2)
            mask_area_ratio: Minimum mask area ratio (mask_area / detection_area)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = target_classes
        self.imgsz = imgsz
        self.min_area = min_area
        self.max_area = max_area
        self.width_height_ratio_tolerance = width_height_ratio_tolerance
        self.mask_area_ratio = mask_area_ratio
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
                # Check if OBB (Oriented Bounding Box) is available
                #has_obb = result.obb is not None
                has_obb = False
                if has_obb:
                    # OBB model: use xywhr format
                    try:
                        xywhr_data = result.obb.xywhr.cpu().numpy()  # [N, 5] array
                        confidences = result.obb.conf.cpu().numpy()
                        class_ids = result.obb.cls.cpu().numpy().astype(int)
                        
                        for i, (xywhr, conf, class_id) in enumerate(
                            zip(xywhr_data, confidences, class_ids)
                        ):
                            # Filter by target classes if specified
                            if (
                                self.target_classes is not None
                                and class_id not in self.target_classes
                            ):
                                continue
                            
                            # Convert xywhr to [x, y, w, h] for fallback bbox
                            x_center, y_center, width, height, _ = xywhr
                            x = x_center - width / 2
                            y = y_center - height / 2
                            
                            # Get image size
                            image_size = (image.shape[1], image.shape[0])  # (width, height)
                            
                            # Create Detection object with xywhr
                            detection = Detection(
                                bbox=[x, y, width, height],
                                confidence=float(conf),
                                class_id=int(class_id),
                                class_name=self._get_class_name(class_id),
                                timestamp=timestamp,
                                masks=None,  # OBB model doesn't use masks
                                frame_number=frame_number,
                                image_size=image_size,
                                xywhr=xywhr,  # Pass xywhr directly
                            )
                            detections.append(detection)
                    except Exception as e:
                        print(f"[WARN] Failed to process OBB: {e}")
                        has_obb = False  # Fallback to regular boxes
                
                # Regular detection (boxes + masks)
                
                else :
                    if result.boxes is None or result.masks is None:
                        raise ValueError("boxes or masks is None")
                        
                    masks = result.masks
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
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
                        mask_area = None
                        try:
                            if (
                                masks is not None
                                and hasattr(masks, "xy")
                                and masks.xy is not None
                            ):
                                poly_xy = masks.xy[i]  # Nx2 numpy array (float)
                                
                                # Calculate area from polygon using cv2.contourArea
                                # 또는 ultralytics가 제공하는 area 속성 사용
                                if poly_xy is not None and len(poly_xy) >= 3:
                                    # Method 1: Try ultralytics masks.area if available
                                    if hasattr(masks, "area") and masks.area is not None:
                                        try:
                                            mask_area = float(masks.area[i])
                                        except (IndexError, TypeError):
                                            # Method 2: Fallback to cv2.contourArea
                                            contour = poly_xy.reshape((-1, 1, 2)).astype(np.int32)
                                            mask_area = cv2.contourArea(contour)
                                    else:
                                        # Method 2: Use cv2.contourArea
                                        contour = poly_xy.reshape((-1, 1, 2)).astype(np.int32)
                                        mask_area = cv2.contourArea(contour)
                        except Exception:
                            poly_xy = None
                            mask_area = None

                        # Get image size for mask processing
                        image_size = (image.shape[1], image.shape[0])  # (width, height)
                        img_w, img_h = image.shape[1], image.shape[0]
                        
                        # Validate bbox bounds
                        if x < 0 or y < 0 or w < 0 or h < 0 or x + w > img_w or y + h > img_h:
                            continue

                        # Create Detection object (no xywhr, will use mask if available)
                        detection = Detection(
                            bbox=[x, y, w, h],
                            confidence=float(conf),
                            class_id=int(class_id),
                            class_name=self._get_class_name(class_id),
                            timestamp=timestamp,
                            masks=poly_xy,
                            frame_number=frame_number,
                            image_size=image_size,
                            xywhr=None,  # No xywhr for regular detection
                        )
                        
                        # Apply filtering based on config
                        detection_area = detection.get_area()
                        
                        # Filter by area
                        if self.min_area is not None and detection_area < self.min_area:
                            logger.debug(
                                f"Detection filtered by min_area: area={detection_area:.0f} < min_area={self.min_area} "
                                f"(bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}], conf={conf:.3f})"
                            )
                            continue
                        if self.max_area is not None and detection_area > self.max_area:
                            logger.debug(
                                f"Detection filtered by max_area: area={detection_area:.0f} > max_area={self.max_area} "
                                f"(bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}], conf={conf:.3f})"
                            )
                            continue
                        
                        # Filter by width/height ratio (1.0 ± tolerance)
                        if self.width_height_ratio_tolerance is not None:
                            width_height_ratio = detection.get_width() / detection.get_height()
                            ratio_min = 1.0 - self.width_height_ratio_tolerance
                            ratio_max = 1.0 + self.width_height_ratio_tolerance
                            if width_height_ratio < ratio_min or width_height_ratio > ratio_max:
                                logger.debug(
                                    f"Detection filtered by width_height_ratio: ratio={width_height_ratio:.3f} "
                                    f"not in [{ratio_min:.3f}, {ratio_max:.3f}] "
                                    f"(bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}], conf={conf:.3f})"
                                )
                                continue
                        
                        # Filter by mask area ratio
                        if self.mask_area_ratio is not None and mask_area is not None:
                            mask_ratio = mask_area / detection_area if detection_area > 0 else 0.0
                            if mask_area < detection_area * self.mask_area_ratio:
                                logger.debug(
                                    f"Detection filtered by mask_area_ratio: mask_ratio={mask_ratio:.3f} "
                                    f"< required={self.mask_area_ratio} "
                                    f"(mask_area={mask_area:.0f}, detection_area={detection_area:.0f}, "
                                    f"bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}], conf={conf:.3f})"
                                )
                                continue
                        
                        detections.append(detection)

            return detections

        except Exception as e:
            #print(f"⚠ YOLO detection failed: {e}")
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
            "device": self.device,
            "model_loaded": self.model is not None,
        }
