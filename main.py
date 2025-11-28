import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
from src.utils.sequence_loader import (
    create_sequence_loader,
    create_camera_device_loader,
    create_video_file_loader,
    create_image_sequence_loader,
)

# Import config first (always available)
from config import SystemConfig

# Import tracking modules

from src.core.detection import Detection, YOLODetector
from src.core.measurement.size_measurement import SizeMeasurement
from src.core.tracking import KalmanTracker, MAX_FRAMES_LOST
from src.visualization import Visualizer


class EnhancedAMRTracker:
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
        self.config = config if config else SystemConfig()
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.pixel_size = pixel_size

        # Initialize data logger
        #self.data_logger = TrackingDataLogger()

        # Initialize components
        self.detector = None
        #self.tracker = None
        self.size_measurement = None
        #self.speed_tracker = None
        self.visualizer = None
        self.tracker = KalmanTracker(
            fps=30,
            pixel_size=self.pixel_size,
            track_id=0,
        )

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all system components"""
        logger.info(f"Initializing Enhanced AMR System (detector={self.detector_type}, tracker={self.tracker_type})")

        # Initialize detector
        if self.detector_type == "yolo":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")

                self.detector = YOLODetector(
                    "weights/zoom1/best.pt",
                    confidence_threshold=0.5,
                    device=device,
                    imgsz=1536,
                    target_classes=[0],
                )
                logger.info("YOLO detector initialized")
            except ImportError:
                raise ImportError("ultralytics module is not installed.")
            except FileNotFoundError:
                raise FileNotFoundError("weights file not found. ")
        else:
            raise ValueError(
                f"Unsupported detector type: {self.detector_type}. Only 'yolo' is supported."
            )

        # if self.tracker_type == "speed":
        #     self.speed_tracker = SpeedTracker(max_history=30, max_tracking_distance=500)
        #     logger.info("Speed tracker initialized")
        # else:
        
        #self.trackers = {}
        self.next_track_id = 0
        self.track_id = None
        logger.info("Multi-object Kalman filter tracker initialized")

        # Initialize additional components if using new modules
        if self.config:
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
                    logger.debug(f"Error accessing calibration_data_path: {e}")
                    calibration_path = None
                if calibration_path and Path(calibration_path).exists():
                    with open(calibration_path, "r") as f:
                        calibration_data = json.load(f)

                    self.size_measurement = SizeMeasurement(
                        homography=np.array(calibration_data["homography"]),
                        camera_height=self.config.calibration.camera_height,
                        pixel_size=calibration_data.get("pixel_size", 1.0),
                        calibration_image_size=self.config.calibration.calibration_image_size,
                    )
                    logger.info("Size measurement initialized")

                    self.visualizer = Visualizer(
                        homography=np.array(calibration_data["homography"])
                    )
                    logger.info("Visualizer initialized")
                else:
                    logger.info("No calibration data found - size measurement disabled")
            except Exception as e:
                logger.warning(f"Error initializing modules: {e}")


    def detect_objects(
        self, frame: np.ndarray, frame_number: int = 0, timestamp: float = None
    ) -> List[Dict]:
        """Detect objects in frame using selected detector"""
        if timestamp is None:
            timestamp = time.time()

        detections = []

        # YOLO detection using YOLODetector
        detections = self.detector.detect(
                image=frame, frame_number=frame_number, timestamp=timestamp
        )
        return detections

    def track_objects(
        self, frame: np.ndarray, detections: List[Detection], frame_number: int = 0
    ) -> List[Dict]:
        """Track objects using selected tracker"""
        results = []

        if self.tracker_type == "kalman":
            # Multi-object detection, but track only the first tracked object
            if len(detections) > 0:
                # If no primary tracker exists, create one with the best detection
                best_detection = detections[0]
                #logger.info(f"length of detections: {len(detections)}")

                if self.track_id is None:
                    # Select the best detection (largest area or highest confidence)
                    # best_detection = max(
                    #     detections, key=lambda d: d.get_area() * d.confidence
                    # )

                    # Create primary tracker
                    self.track_id = 0
        
                    self.tracker.initialize_with_detection(best_detection.oriented_box_info["center"], best_detection.oriented_box_info["angle"])
                    self.next_track_id += 1

                    # Update with first detection
                    tracking_result = self.tracker.update(
                        bbox=best_detection.bbox,
                        center=best_detection.oriented_box_info["center"],
                        frame_number=frame_number,
                        theta=best_detection.oriented_box_info["angle"],
                    )
                    tracking_result["detection_type"] = getattr(
                        best_detection, "class_name", "unknown"
                    )
                    tracking_result["track_id"] = self.track_id
                    tracking_result["color"] = self.tracker.color

                    # Add size measurement if available (only store initial size)
                    if self.size_measurement:
                        size_measurement = self.size_measurement.measure(best_detection)
                        tracking_result["size_measurement"] = size_measurement
                        self.tracker.last_size_measurement = size_measurement
                        self.tracker.initial_size_measurement = (
                            size_measurement  # Store initial size
                        )

                    results.append(tracking_result)
                else:
                    # Primary tracker exists - use association to match it with a detection
                    if self.tracker is None:
                        # Primary tracker was lost, reset
                        self.track_id = None
                        return results

                    # Use IoU + Hungarian algorithm for optimal association
                    # But only match with primary tracker
                    # matches, unmatched_detections, unmatched_trackers = (
                    #     associate_detections_to_trackers(
                    #         detections,
                    #         {self.track_id: self.tracker},
                    #         iou_threshold=0.3,
                    #     )
                    # )

                    # Update matched tracker (should be primary tracker)
                    # matched = False
                    # for detection_idx, tracker_id in matches:
                    #     if tracker_id == self.track_id:
                    #         detection = detections[detection_idx]
                    #         tracking_result = self.tracker.update(
                    #             frame=frame,
                    #             bbox=detection.bbox,
                    #             frame_number=detection.frame_number,
                    #             theta=detection.oriented_box_info["angle"],
                    #         )
                    #         tracking_result["detection_type"] = getattr(
                    #             detection, "class_name", "unknown"
                    #         )
                    #         tracking_result["track_id"] = self.track_id
                    #         tracking_result["color"] = self.tracker.color

                    #         # Use initial size measurement (don't update)
                    #         if self.tracker.initial_size_measurement is not None:
                    #             tracking_result["size_measurement"] = (
                    #                 self.tracker.initial_size_measurement
                    #             )

                    #         results.append(tracking_result)
                    #         matched = True
                    #         break

                    # If no match, predict only (don't create new trackers)
                   
                    tracking_result = self.tracker.update(
                            bbox=best_detection.bbox, center=best_detection.oriented_box_info["center"], theta=best_detection.oriented_box_info["angle"], frame_number=frame_number
                    )
                    tracking_result["detection_type"] = "predicted"
                    tracking_result["track_id"] = self.track_id
                    tracking_result["color"] = self.tracker.color
                    results.append(tracking_result)

                    if self.tracker.is_lost(max_frames_lost=MAX_FRAMES_LOST):
                        logger.info(
                                f"Removing lost primary tracker ID: {self.track_id}"
                            )
                        self.tracker.reset()
                        self.visualizer.reset()
                        self.track_id = None
      

            else:
                # No detections: predict with primary tracker
                if self.track_id is not None:
                    if self.tracker is not None:
                        tracking_result = self.tracker.update(
                            bbox=None, center=None, theta=None, frame_number=frame_number
                        )
                        tracking_result["detection_type"] = "predicted"
                        tracking_result["track_id"] = self.track_id
                        tracking_result["color"] = self.tracker.color
                        results.append(tracking_result)

                        # Clean up if tracker is lost
                        if self.tracker.is_lost(max_frames_lost=MAX_FRAMES_LOST):
                            logger.info(
                                f"Removing lost primary tracker ID: {self.track_id}"
                            )
                            self.tracker.reset()
                            self.visualizer.reset()
                            self.track_id = None
                     
 
            # Update speed tracker
           # logger.debug(f"Speed tracker: {len(measurements)} measurements")
            # results = self.speed_tracker.update(measurements)
            # logger.debug(f"Speed tracker: {len(results)} results")
        else:
            raise ValueError(f"Unsupported tracker type: {self.tracker_type}. Only 'kalman' is supported.")
        return results

    def visualize_results(
        self, frame: np.ndarray, detections: List[Dict], tracking_results: List[Dict]
    ) -> np.ndarray:
        """Visualize results using appropriate visualizer"""
            # Use multi-object AMR tracker visualization
        if self.visualizer and self.size_measurement:
                # Use enhanced visualizer if available
                detection_objects = detections
                vis_frame = self.visualizer.draw_single_object(
                    frame, detection_objects, tracking_results
                )
        else:
                # Use basic AMR tracker visualization
                vis_frame = frame.copy()

                # Draw all tracking results
                for i, result in enumerate(tracking_results):
                    if "track_id" in result:
                        if result["track_id"] == 0:
                            vis_frame = self.tracker.draw_visualization(vis_frame, result)

        return vis_frame

            


def load_execution_preset(config_path: str, preset_name: str) -> dict:
    """Load execution preset from config file"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "execution" in config and "presets" in config["execution"]:
            presets = config["execution"]["presets"]
            if preset_name in presets:
                logger.info(f"Loaded preset: {preset_name}")
                return presets[preset_name]
            else:
                available_presets = list(presets.keys())
                logger.warning(f"Preset '{preset_name}' not found. Available: {available_presets}")
                return {}
        else:
            logger.warning("No execution presets found in config file")
            return {}
    except Exception as e:
        logger.warning(f"Error loading preset: {e}")
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
    if args.source1 == "0" and "source1" in preset:
        args.source1 = preset["source1"]
    if args.source2 == "0" and "source2" in preset:
        args.source2 = preset["source2"]
    if args.source3 == "0" and "source3" in preset:
        args.source3 = preset["source3"]
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
        "--source1",
        default="0",
        help="Video source (0 for camera, file path, video file, or image sequence folder)",
    )
    parser.add_argument(
        "--source2",
        default="0",
        help="Video source (0 for camera, file path, video file, or image sequence folder)",
    )
    parser.add_argument(
        "--source3",
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
        default="enhanced",
        help="System mode: basic (AMR tracker only) or enhanced (AGV measurement system)",
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
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    args = parser.parse_args()
    
    # 로깅 설정 (명령줄 인자로 레벨 설정)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(log_level)

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
            logger.warning(f"Error loading config, using default: {e}")
            config = None
    else:
        logger.warning(f"Config file not found: {args.config}")
        config = None

    # Initialize system
    # if args.mode == "basic":
    #     logger.info("Running in basic mode...")
    #     if config:
    #         if isinstance(config, dict):
    #             # dict object
    #             pixel_size = config["measurement"]["pixel_size"]
    #         else:
    #             # SystemConfig object
    #             pixel_size = config.measurement.pixel_size
    #     else:
    #         pixel_size = 1.0
    #     run_basic_mode(
    #         args.source,
    #         args.fps,
    #         args.loader_mode,
    #         pixel_size,
    #     )
    if args.mode == "enhanced":
        logger.info("Running in enhanced mode...")
        if config:
            try:
                if isinstance(config, dict):
                    pixel_size = config["measurement"]["pixel_size"]
                elif hasattr(config, "measurement") and hasattr(config.measurement, "pixel_size"):
                    pixel_size = config.measurement.pixel_size
                else:
                    logger.warning("Config object doesn't have expected structure, using default pixel_size")
                    pixel_size = 1.0
            except (KeyError, AttributeError) as e:
                logger.warning(f"Error accessing pixel_size: {e}")
                pixel_size = 1.0
        else:
            logger.warning("No config loaded, using default pixel_size")
            pixel_size = 1.0
        run_enhanced_mode(
            args,
            args.source1,
            args.fps,
            args.loader_mode,
            pixel_size,
        )
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return


def run_enhanced_mode(args, loader_source, fps=30, loader_mode="auto", pixel_size=1.0):
    """Run enhanced AMR system"""
    # Load configuration
    config = None
    if Path(args.config).exists():
        try:
            config = SystemConfig.load(args.config)
            logger.info(f"Configuration loaded from {args.config}")
        except Exception as e:
            logger.warning(f"Error loading config: {e}")

    # Initialize enhanced system
    amr_tracker = EnhancedAMRTracker(
        config=config,
        detector_type=args.detector,
        tracker_type=args.tracker,
        pixel_size=pixel_size,
    )

    # Create sequence loader based on mode
    if loader_mode == "camera":
        cap = create_camera_device_loader(loader_source)
    elif loader_mode == "video":
        cap = create_video_file_loader(loader_source)
    elif loader_mode == "sequence":
        cap = create_image_sequence_loader(loader_source, fps)
    else:  # auto mode
        cap = create_sequence_loader(loader_source, fps)

    if cap is None:
        logger.error("Cannot create sequence loader")
        return

    frame_number = 0
    logger.info("AMR System running... Press 'q' to quit, 's' to save snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = amr_tracker.detect_objects(
            frame=frame, frame_number=frame_number, timestamp=time.time()
        )

        # Track objects
        tracking_results = amr_tracker.track_objects(frame, detections, frame_number)

        # Log tracking data
        # for result in tracking_results:
        #     amr_tracker.data_logger.log_tracking_result(frame_number, result)

        # Visualize results
        vis_frame = amr_tracker.visualize_results(frame, detections, tracking_results)

        # Add frame info
        if hasattr(cap, "frame_number"):  # Image sequence
            # 이미지 크기에 따라 텍스트 크기 동적 조정
            height, width = vis_frame.shape[:2]
            font_scale = max(0.5, min(2.0, width / 800))  # 800px 기준으로 스케일링
            thickness = max(1, int(font_scale * 2))

        # 창 크기 조절 (화면이 너무 클 때)
        height, width = vis_frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280 / width, 720 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            vis_frame_window = cv2.resize(vis_frame, (new_width, new_height))

        # Show results
        cv2.imshow("AMR Tracking", vis_frame_window)

        # Handle key press
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"snapshot_{frame_number:06d}.png", vis_frame)
            logger.info(f"Snapshot saved: snapshot_{frame_number:06d}.png")

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"AMR System completed. Processed {frame_number} frames")


if __name__ == "__main__":
    main()
