import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque
import time
import argparse
from typing import Optional, Dict, List, Tuple
import json
from pathlib import Path
import glob
import os
import csv
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from src.utils.sequence_loader import (
    BaseLoader,
    CameraDeviceLoader,
    VideoFileLoader,
    ImageSequenceLoader,
    create_sequence_loader,
    create_camera_device_loader,
    create_video_file_loader,
    create_image_sequence_loader,
    LoaderMode,
)

# Import config first (always available)
from config import SystemConfig

# Import new modules
try:
    from src.core.detection import Detection, YOLODetector
    from src.core.measurement.size_measurement import SizeMeasurement
    from src.core.tracking import (
        SpeedTracker,
        KalmanTracker,
        TrackingDataLogger,
        associate_detections_to_trackers,
    )
    from src.visualization.display import Visualizer

    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False
    print(
        "Warning: New AGV Measurement System modules not available. Using basic mode."
    )


class EnhancedAMRSystem:
    """
    AMR (Autonomous Mobile Robot) tracking system with YOLO detection and Kalman filtering.

    Features:
    - YOLO-based object detection with configurable classes
    - Kalman filter tracking with IoU-based association
    - Real-time speed and orientation estimation
    - CSV data logging for tracking results
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        detector_type: str = "yolo",
        tracker_type: str = "kalman",
        pixel_size: float = 1.0,
    ):
        """
        Initialize enhanced AMR system

        Args:
            config: System configuration
            detector_type: Type of detector ("yolo")
            tracker_type: Type of tracker ("kalman", "speed")
        """
        self.config = config
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.pixel_size = pixel_size
        self.stationary_threshold = (
            5.0  # Speed threshold in mm/s to consider object as stationary
        )

        # Initialize data logger
        self.data_logger = TrackingDataLogger()

        # Initialize components
        self.detector = None
        self.tracker = None
        self.size_measurement = None
        self.speed_tracker = None
        self.visualizer = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all system components"""
        print(f"Initializing Enhanced AMR System...")
        print(f"Detector: {self.detector_type}")
        print(f"Tracker: {self.tracker_type}")
        print(f"New modules: {ENHANCED_MODULES_AVAILABLE}")

        # Initialize detector
        if self.detector_type == "yolo":
            try:
                # Check CUDA availability
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")

                self.detector = YOLODetector(
                    "weights/last.pt",
                    confidence_threshold=0.88,
                    device=device,  # Auto-detect CUDA availability
                    imgsz=768,
                    target_classes=[0],  # Only detect class 0 (AGV)
                )
                print("✓ YOLO detector initialized (weights/last.pt)")
            except ImportError:
                raise ImportError("ultralytics 모듈이 설치되지 않았습니다.")
            except FileNotFoundError:
                raise FileNotFoundError("weights 파일을 찾을 수 없습니다. ")
        else:
            raise ValueError(
                f"지원하지 않는 감지기 타입: {self.detector_type}. 'yolo'만 지원됩니다."
            )

        # Initialize tracker
        if self.tracker_type == "kalman":
            self.trackers = {}  # Dictionary to store multiple trackers
            self.next_track_id = 0
            print("✓ Multi-object Kalman filter tracker initialized")
        elif self.tracker_type == "speed" and ENHANCED_MODULES_AVAILABLE:
            self.speed_tracker = SpeedTracker(max_history=30, max_tracking_distance=500)
            print("✓ Speed tracker initialized")
            print(f"Debug - SpeedTracker created: {self.speed_tracker is not None}")
        else:
            self.trackers = {}  # Dictionary to store multiple trackers
            self.next_track_id = 0
            print("✓ Fallback multi-object Kalman filter tracker initialized")

        # Initialize additional components if using new modules
        if ENHANCED_MODULES_AVAILABLE and self.config:
            try:
                # Load calibration data if available
                try:
                    if isinstance(self.config, dict):
                        # dict object
                        calibration_path = self.config["calibration"][
                            "calibration_data_path"
                        ]
                    else:
                        # SystemConfig object
                        calibration_path = self.config.calibration.calibration_data_path
                except (KeyError, AttributeError) as e:
                    print(f"⚠ Error accessing calibration_data_path: {e}")
                    calibration_path = "calibration_data.json"  # 기본값
                if Path(calibration_path).exists():
                    with open(calibration_path, "r") as f:
                        calibration_data = json.load(f)

                    # Always initialize size measurement and visualizer if calibration data exists
                    self.size_measurement = SizeMeasurement(
                        homography=np.array(calibration_data["homography"]),
                        camera_height=0.0,  # AGV 위에 캘판을 놓았으므로 높이 0
                        pixel_size=calibration_data.get("pixel_size", 1.0),
                        calibration_image_size=(3840, 2160),  # 4K 이미지 크기
                    )
                    print("✓ Size measurement initialized")

                    self.visualizer = Visualizer(
                        homography=np.array(calibration_data["homography"])
                    )
                    print("✓ Enhanced visualizer initialized")
                else:
                    print("⚠ No calibration data found. Size measurement disabled.")
            except Exception as e:
                print(f"⚠ Error initializing new modules: {e}")

    def detect_objects(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = None
    ) -> List[Dict]:
        """Detect objects in frame using selected detector"""
        if timestamp is None:
            timestamp = time.time()

        detections = []
        classes = [1]
        if self.detector_type == "yolo":
            # YOLO detection using YOLODetector
            yolo_detections = self.detector.detect(frame)

            for detection in yolo_detections:
                # Set timestamp
                detection.timestamp = timestamp

                detections.append(detection)

        return detections

    def track_objects(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> List[Dict]:
        """Track objects using selected tracker"""
        results = []

        if self.tracker_type == "kalman":
            # Use multi-object Kalman filter tracking with IoU-based association
            if detections:
                # Use IoU + Hungarian algorithm for optimal association
                matches, unmatched_detections, unmatched_trackers = (
                    associate_detections_to_trackers(
                        detections, self.trackers, iou_threshold=0.1
                    )
                )

                # Update matched trackers
                for detection_idx, tracker_id in matches:
                    detection = detections[detection_idx]
                    tracker = self.trackers[tracker_id]

                    tracking_result = tracker.update(
                        frame, detection.bbox, frame_number=len(results)
                    )
                    tracking_result["detection_type"] = getattr(
                        detection, "class_name", "unknown"
                    )
                    tracking_result["track_id"] = tracker_id
                    tracking_result["color"] = tracker.color

                    # Add size measurement if available
                    if self.size_measurement:
                        size_measurement = self.size_measurement.measure(detection)
                        tracking_result["size_measurement"] = size_measurement

                    results.append(tracking_result)

                # Handle unmatched trackers (predict only)
                for tracker_id in unmatched_trackers:
                    tracker = self.trackers[tracker_id]
                    tracking_result = tracker.update(
                        frame, None, frame_number=len(results)
                    )
                    tracking_result["detection_type"] = "predicted"
                    tracking_result["track_id"] = tracker_id
                    tracking_result["color"] = tracker.color
                    results.append(tracking_result)

                # Create new trackers for unmatched detections
                for detection_idx in unmatched_detections:
                    detection = detections[detection_idx]
                    new_tracker = KalmanTracker(
                        fps=30,
                        pixel_size=self.pixel_size,
                        track_id=self.next_track_id,
                    )
                    # Initialize tracker with detection position
                    new_tracker.initialize_with_detection(detection.bbox)
                    self.trackers[self.next_track_id] = new_tracker

                    tracking_result = new_tracker.update(
                        frame, detection.bbox, frame_number=len(results)
                    )
                    tracking_result["detection_type"] = getattr(
                        detection, "class_name", "unknown"
                    )
                    tracking_result["track_id"] = self.next_track_id
                    tracking_result["color"] = new_tracker.color

                    # Add size measurement if available
                    if self.size_measurement:
                        size_measurement = self.size_measurement.measure(detection)
                        tracking_result["size_measurement"] = size_measurement

                    results.append(tracking_result)
                    self.next_track_id += 1

                # Clean up lost trackers
                trackers_to_remove = []
                for track_id, tracker in self.trackers.items():
                    if tracker.is_lost(max_frames_lost=10):
                        trackers_to_remove.append(track_id)

                for track_id in trackers_to_remove:
                    print(f"Removing lost tracker ID: {track_id}")
                    del self.trackers[track_id]

        elif self.tracker_type == "speed" and ENHANCED_MODULES_AVAILABLE:
            # Use speed tracker (multiple objects)
            measurements = []
            for detection in detections:
                if self.size_measurement:
                    # Measure size if calibration is available
                    measurement = self.size_measurement.measure(detection)
                    measurement["bbox"] = detection.bbox
                    measurement["confidence"] = detection.confidence
                    measurements.append(measurement)
                else:
                    # Basic measurement - convert to mm coordinates
                    bbox = detection.bbox
                    cx = bbox[0] + bbox[2] / 2
                    cy = bbox[1] + bbox[3] / 2
                    # Convert pixel coordinates to mm using pixel_size
                    cx_mm = cx * self.pixel_size
                    cy_mm = cy * self.pixel_size
                    measurements.append(
                        {
                            "center_world": (cx_mm, cy_mm),
                            "bbox": detection.bbox,
                            "confidence": detection.confidence,
                            "timestamp": time.time(),
                        }
                    )

            # Update speed tracker
            print(f"Debug - measurements before speed tracker: {len(measurements)}")
            for i, m in enumerate(measurements):
                print(f"Debug - measurement {i}: {list(m.keys())}")

            results = self.speed_tracker.update(measurements)
            print(f"Debug - speed tracker results: {len(results)}")
            for i, r in enumerate(results):
                print(f"Debug - result {i}: {list(r.keys())}")

        return results

    def visualize_results(
        self, frame: np.ndarray, detections: List[Dict], tracking_results: List[Dict]
    ) -> np.ndarray:
        """Visualize results using appropriate visualizer"""
        if self.tracker_type == "kalman" and tracking_results:
            # Use multi-object AMR tracker visualization
            if self.visualizer and self.size_measurement:
                # Use enhanced visualizer if available
                detection_objects = detections
                vis_frame = self.visualizer.draw_detections(
                    frame, detection_objects, tracking_results
                )

                # Add summary information
                # cv2.putText(
                #     vis_frame,
                #     f"Objects: {len(tracking_results)} (Enhanced)",
                #     (10, vis_frame.shape[0] - 30),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2,
                # )
            else:
                # Use basic AMR tracker visualization
                vis_frame = frame.copy()

                # Draw all tracking results
                for i, result in enumerate(tracking_results):
                    if "track_id" in result:
                        track_id = result["track_id"]
                        tracker = self.trackers.get(track_id)
                        if tracker:
                            vis_frame = tracker.draw_visualization(vis_frame, result)

                # Add summary information
                # cv2.putText(
                #     vis_frame,
                #     f"Objects: {len(tracking_results)}",
                #     (10, vis_frame.shape[0] - 30),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2,
                # )

            return vis_frame

        elif self.tracker_type == "speed" and ENHANCED_MODULES_AVAILABLE:
            # Use enhanced visualizer with speed information
            if self.visualizer:
                detection_objects = detections
                vis_frame = self.visualizer.draw_detections(
                    frame, detection_objects, tracking_results
                )
            else:
                # Fallback to basic visualization with speed info
                vis_frame = frame.copy()
                print("Debug - Using fallback visualization for SpeedTracker")

            # Add speed information overlay
            for result in tracking_results:
                # 디버깅: 결과 정보 출력
                print(f"Debug - tracking_result keys: {list(result.keys())}")
                if "speed" in result:
                    print(f"Debug - speed: {result['speed']}")

                if "bbox" in result and "speed" in result:
                    bbox = result["bbox"]
                    x, y, w, h = map(int, bbox)
                    speed = result["speed"]

                    # 이미지 크기에 따라 텍스트 크기 동적 조정
                    height, width = vis_frame.shape[:2]
                    font_scale = max(1.0, min(3.0, width / 400))  # 더 큰 텍스트
                    thickness = max(2, int(font_scale * 3))

                    # 속도 정보 표시
                    speed_text = f"Speed: {speed:.1f}mm/s"
                    # cv2.putText(
                    #     vis_frame,
                    #     speed_text,
                    #     (x, y - int(30 * font_scale)),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     font_scale,
                    #     (0, 255, 255),  # 노란색
                    #     thickness,
                    # )
                else:
                    # bbox만 있는 경우 기본 정보 표시
                    if "bbox" in result:
                        bbox = result["bbox"]
                        x, y, w, h = map(int, bbox)

                        # 이미지 크기에 따라 텍스트 크기 동적 조정
                        height, width = vis_frame.shape[:2]
                        font_scale = max(1.0, min(3.0, width / 400))
                        thickness = max(2, int(font_scale * 3))

                        # 기본 정보 표시
                        info_text = f"Track: {result.get('track_id', 'N/A')}"
                        # cv2.putText(
                        #     vis_frame,
                        #     info_text,
                        #     (x, y - int(30 * font_scale)),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     font_scale,
                        #     (255, 0, 0),  # 빨간색
                        #     thickness,
                        # )

            return vis_frame

        else:
            # Basic visualization
            vis_frame = frame.copy()
            for detection in detections:
                x, y, w, h = map(int, detection.bbox)
                # 이미지 크기에 따라 텍스트 크기 동적 조정
                height, width = vis_frame.shape[:2]
                font_scale = max(1.0, min(3.0, width / 400))  # 더 큰 텍스트
                thickness = max(2, int(font_scale * 3))

                # cv2.putText(
                #     vis_frame,
                #     f"Conf: {detection.confidence:.2f}",
                #     (x, y - int(10 * font_scale)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     font_scale,
                #     (0, 255, 0),
                #     thickness,
                # )
            return vis_frame


def load_execution_preset(config_path: str, preset_name: str) -> dict:
    """Load execution preset from config file"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "execution" in config and "presets" in config["execution"]:
            presets = config["execution"]["presets"]
            if preset_name in presets:
                print(f"✓ Loaded preset: {preset_name}")
                return presets[preset_name]
            else:
                available_presets = list(presets.keys())
                print(
                    f"⚠ Preset '{preset_name}' not found. Available presets: {available_presets}"
                )
                return {}
        else:
            print("⚠ No execution presets found in config file")
            return {}
    except Exception as e:
        print(f"⚠ Error loading preset: {e}")
        return {}


def apply_preset_to_args(args: argparse.Namespace, preset: dict) -> argparse.Namespace:
    """Apply preset values to args if not explicitly set"""
    if not preset:
        return args

    # Apply preset values only if args are not explicitly set
    if args.mode == "basic" and "mode" in preset:
        args.mode = preset["mode"]
    if args.loader_mode == "auto" and "loader_mode" in preset:
        args.loader_mode = preset["loader_mode"]
    if args.source == "0" and "source" in preset:
        args.source = preset["source"]
    if args.fps == 30 and "fps" in preset:
        args.fps = preset["fps"]
    if args.detector == "yolo" and "detector" in preset:
        args.detector = preset["detector"]
    if args.tracker == "kalman" and "tracker" in preset:
        args.tracker = preset["tracker"]

    return args


def main():
    """Enhanced main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced AMR Tracking System")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source (0 for camera, file path, video file, or image sequence folder)",
    )
    parser.add_argument(
        "--detector",
        choices=["yolo"],
        default="yolo",
        help="Type of detector to use (enhanced mode only)",
    )
    parser.add_argument(
        "--tracker",
        choices=["kalman", "speed"],
        default="kalman",
        help="Type of tracker to use (enhanced mode only)",
    )
    parser.add_argument(
        "--config",
        default="tracker_config.json",
        help="Configuration file path (enhanced mode only)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        help="Load execution preset from config file (e.g., 'enhanced_stationary', 'enhanced_tracking', 'basic_tracking', 'camera_basic')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "enhanced"],
        default="basic",
        help="System mode: basic (AMR tracker only) or enhanced (AGV measurement system)",
    )
    parser.add_argument(
        "--stationary-mode",
        action="store_true",
        help="Stationary mode: only measure position and orientation, skip velocity calculations",
    )
    parser.add_argument(
        "--loader-mode",
        type=str,
        choices=["auto", "camera", "video", "sequence"],
        default="auto",
        help="Loader mode: auto (default), camera, video, or sequence",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for image sequences (default: 30)",
    )

    args = parser.parse_args()

    # Load preset if provided
    if args.preset:
        preset_config = load_execution_preset(args.config, args.preset)
        args = apply_preset_to_args(args, preset_config)

    # Load configuration for pixel_size
    # pixel_size = 1.0  # default value (1 pixel = 1mm)
    if Path(args.config).exists():
        try:
            config = SystemConfig.load(args.config)
        except Exception as e:
            print(f"⚠ Error loading config, using default: {e}")
            import traceback

            traceback.print_exc()
            config = None
    else:
        print(f"⚠ Config file not found: {args.config}")
        config = None

    # Initialize system
    if args.mode == "basic":
        print("Running in basic mode...")
        if config:
            if isinstance(config, dict):
                # dict object
                pixel_size = config["measurement"]["pixel_size"]
            else:
                # SystemConfig object
                pixel_size = config.measurement.pixel_size
        else:
            pixel_size = 1.0
        run_basic_mode(
            args.source,
            args.fps,
            args.loader_mode,
            pixel_size,
        )
    elif args.mode == "enhanced":
        if not ENHANCED_MODULES_AVAILABLE:
            print("Error: Enhanced mode requires AGV Measurement System modules.")
            print("Please install the required modules or use basic mode.")
            return
        print("Running in enhanced mode...")
        if config:
            print(f"Debug - config type: {type(config)}")
            print(f"Debug - config is dict: {isinstance(config, dict)}")
            try:
                if isinstance(config, dict):
                    # dict object
                    pixel_size = config["measurement"]["pixel_size"]
                elif hasattr(config, "measurement") and hasattr(
                    config.measurement, "pixel_size"
                ):
                    # SystemConfig object
                    pixel_size = config.measurement.pixel_size
                else:
                    print(
                        "⚠ Config object doesn't have expected structure, using default pixel_size"
                    )
                    pixel_size = 1.0
            except (KeyError, AttributeError) as e:
                print(f"⚠ Error accessing pixel_size: {e}")
                pixel_size = 1.0
        else:
            print("⚠ No config loaded, using default pixel_size")
            pixel_size = 1.0
        run_enhanced_mode(
            args,
            args.source,
            args.fps,
            args.loader_mode,
            pixel_size,
        )
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        return


def run_basic_mode(
    video_source,
    fps=30,
    mode="auto",
    pixel_size=1.0,
):
    """Run basic multi-object AMR tracker"""
    import ultralytics

    detector = ultralytics.YOLO("weights/last.pt")
    trackers = {}  # Dictionary to store multiple trackers
    next_track_id = 0
    data_logger = TrackingDataLogger()
    print("Basic multi-object AMR tracker initialized")

    # Create sequence loader based on mode
    if mode == "camera":
        cap = create_camera_device_loader(int(video_source))
    elif mode == "video":
        cap = create_video_file_loader(video_source)
    elif mode == "sequence":
        cap = create_image_sequence_loader(video_source, fps)
    else:  # auto mode
        cap = create_sequence_loader(video_source, fps)

    if cap is None:
        print("Error: Cannot create sequence loader")
        return

    frame_number = 0
    print(f"Processing frames... Press 'q' to quit, 's' to save snapshot")

    classes = range(0, 90, 1)

    # print(classes)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection - filter only cars and trucks
        # COCO class IDs: car=2, truck=7, bus=5

        results = detector(
            frame, classes=[1], imgsz=768, conf=0.88, retina_masks=F, class_id=1
        )  # Only detect cars and trucks

        # Get all detections
        detections = []
        if len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                bbox = [x1, y1, x2 - x1, y2 - y1]
                confidence = results[0].boxes.conf[i].cpu().numpy()

                # 클래스 이름 가져오기
                class_id = int(results[0].boxes.cls[i].cpu().numpy())

                class_name = detector.names[class_id]
                print(f"Class ID: {class_id} Class Name: {class_name}")
                from src.core.detection import Detection

                detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=float(confidence),
                        class_id=class_id,
                        class_name=class_name,
                        timestamp=time.time(),
                    )
                )

        # Multi-object tracking
        tracking_results = []
        used_detections = set()

        # Update existing trackers
        for track_id, tracker in list(trackers.items()):
            if not detections:
                break

            # Find closest detection
            best_detection_idx = None
            best_distance = float("inf")

            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue

                bbox = detection.bbox
                cx, cy = detection.get_center()

                # Get tracker's last position
                tracker_pos = tracker.kf.statePost[:2].flatten()
                distance = np.sqrt(
                    (cx - tracker_pos[0]) ** 2 + (cy - tracker_pos[1]) ** 2
                )

                if (
                    distance < best_distance and distance < 100
                ):  # Max association distance
                    best_distance = distance
                    best_detection_idx = i

            if best_detection_idx is not None:
                detection = detections[best_detection_idx]
                tracking_result = tracker.update(
                    frame, detection.bbox, frame_number=frame_number
                )
                tracking_result["detection_type"] = getattr(
                    detection, "class_name", "unknown"
                )
                tracking_result["track_id"] = track_id
                tracking_result["color"] = tracker.color
                # 클래스 정보 추가
                tracking_result["class_name"] = getattr(
                    detection, "class_name", "Unknown"
                )
                tracking_result["class_id"] = getattr(detection, "class_id", -1)
                tracking_results.append(tracking_result)
                used_detections.add(best_detection_idx)
            else:
                # No detection found, predict only
                tracking_result = tracker.update(frame, None, frame_number=frame_number)
                tracking_result["detection_type"] = "predicted"
                tracking_result["track_id"] = track_id
                tracking_result["color"] = tracker.color
                tracking_results.append(tracking_result)

        # Create new trackers for unassigned detections
        for i, detection in enumerate(detections):
            if i not in used_detections:

                new_tracker = KalmanTracker(
                    fps=fps,
                    pixel_size=pixel_size,  # mm 단위로 직접 전달
                    track_id=next_track_id,
                )
                trackers[next_track_id] = new_tracker

                tracking_result = new_tracker.update(
                    frame, detection.bbox, frame_number=frame_number
                )
                tracking_result["detection_type"] = getattr(
                    detection, "class_name", "unknown"
                )
                tracking_result["track_id"] = next_track_id
                tracking_result["color"] = new_tracker.color
                # 클래스 정보 추가
                tracking_result["class_name"] = getattr(
                    detection, "class_name", "Unknown"
                )
                tracking_result["class_id"] = getattr(detection, "class_id", -1)
                tracking_results.append(tracking_result)

                next_track_id += 1

                # Clean up lost trackers (더 관대한 설정)
        trackers_to_remove = []
        max_frames_lost = 30  # 30프레임까지 기다림 (약 1초)
        for track_id, tracker in trackers.items():
            if tracker.is_lost(max_frames_lost=max_frames_lost):
                trackers_to_remove.append(track_id)

        for track_id in trackers_to_remove:
            print(f"Removing lost tracker ID: {track_id}")
            del trackers[track_id]

        # Log tracking data
        for result in tracking_results:
            data_logger.log_tracking_result(frame_number, result)

        # Visualize all tracking results
        vis_frame = frame.copy()
        for result in tracking_results:
            if "track_id" in result:
                track_id = result["track_id"]
                tracker = trackers.get(track_id)
                if tracker:
                    vis_frame = tracker.draw_visualization(vis_frame, result)

        # Add summary information
        # cv2.putText(
        #     vis_frame,
        #     f"Objects: {len(tracking_results)}",
        #     (10, vis_frame.shape[0] - 30),  # 위쪽으로 이동
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2,
        # )

        # Add frame info
        if hasattr(cap.loader, "frame_number"):  # Image sequence
            # 이미지 크기에 따라 텍스트 크기 동적 조정
            height, width = vis_frame.shape[:2]
            font_scale = max(0.5, min(2.0, width / 800))  # 800px 기준으로 스케일링
            thickness = max(1, int(font_scale * 2))

            # cv2.putText(
            #     vis_frame,
            #     f"Frame: {frame_number}",
            #     (
            #         int(10 * font_scale),
            #         vis_frame.shape[0] - int(10 * font_scale),
            #     ),  # 맨 아래
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     font_scale,
            #     (255, 255, 255),
            #     thickness,
            # )

        # Print results
        if tracking_results:
            print(f"Frame {frame_number}: {len(tracking_results)} objects tracked")
            for result in tracking_results:
                track_id = result.get("track_id", "Unknown")
            print(
                f"  ID {track_id}: Position ({result['position']['x']:.1f}, {result['position']['y']:.1f}), "
                f"Speed: {result['velocity']['linear_speed_mm_per_sec']:.2f} mm/s, "
                f"Orientation: {result['orientation']['theta_normalized_deg']:.1f}°"
            )

        # 창 크기 조절 (화면이 너무 클 때)
        height, width = vis_frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280 / width, 720 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            vis_frame = cv2.resize(vis_frame, (new_width, new_height))

        cv2.imshow("AMR Tracking (Basic)", vis_frame)

        # Handle key press
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"snapshot_{frame_number:06d}.jpg", vis_frame)
            print(f"Snapshot saved: snapshot_{frame_number:06d}.jpg")

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Processed {frame_number} frames")


def run_enhanced_mode(args, video_source, fps=30, mode="auto", pixel_size=1.0):
    """Run enhanced AMR system"""
    # Load configuration
    config = None
    if Path(args.config).exists():
        try:
            config = SystemConfig.load(args.config)
            print(f"✓ Configuration loaded from {args.config}")
        except Exception as e:
            print(f"⚠ Error loading config: {e}")

    # Initialize enhanced system
    system = EnhancedAMRSystem(
        config=config,
        detector_type=args.detector,
        tracker_type=args.tracker,
        pixel_size=pixel_size,
    )

    # Create sequence loader based on mode
    if mode == "camera":
        cap = create_camera_device_loader(int(video_source))
    elif mode == "video":
        cap = create_video_file_loader(video_source)
    elif mode == "sequence":
        cap = create_image_sequence_loader(video_source, fps)
    else:  # auto mode
        cap = create_sequence_loader(video_source, fps)

    if cap is None:
        print("Error: Cannot create sequence loader")
        return

    frame_number = 0
    print(f"\nEnhanced AMR System running...")
    print(f"Press 'q' to quit, 's' to save snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time()

        # Detect objects
        detections = system.detect_objects(frame, frame_number, timestamp)

        # Track objects
        tracking_results = system.track_objects(frame, detections)

        # Log tracking data
        for result in tracking_results:
            system.data_logger.log_tracking_result(frame_number, result)

        # Visualize results
        vis_frame = system.visualize_results(frame, detections, tracking_results)

        # Add frame info
        if hasattr(cap, "frame_number"):  # Image sequence
            # 이미지 크기에 따라 텍스트 크기 동적 조정
            height, width = vis_frame.shape[:2]
            font_scale = max(0.5, min(2.0, width / 800))  # 800px 기준으로 스케일링
            thickness = max(1, int(font_scale * 2))

            # cv2.putText(
            #     vis_frame,
            #     f"Frame: {frame_number}",
            #     (
            #         int(10 * font_scale),
            #         vis_frame.shape[0] - int(10 * font_scale),
            #     ),  # 맨 아래
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     font_scale,
            #     (255, 255, 255),
            #     thickness,
            # )

        # Print results (removed verbose logging for better performance)

        # 창 크기 조절 (화면이 너무 클 때)
        height, width = vis_frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280 / width, 720 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            vis_frame = cv2.resize(vis_frame, (new_width, new_height))

        # Show results
        cv2.imshow("Enhanced AMR Tracking", vis_frame)

        # Handle key press
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"snapshot_{frame_number:06d}.jpg", vis_frame)
            print(f"Snapshot saved: snapshot_{frame_number:06d}.jpg")

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Enhanced AMR System completed. Processed {frame_number} frames")


if __name__ == "__main__":
    main()
