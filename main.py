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
from sequence_loader import (
    SequenceLoader,
    create_sequence_loader,
    create_camera_device_loader,
    create_video_file_loader,
    create_image_sequence_loader,
    LoaderMode,
)

# Import new modules
try:
    from detection.agv_detector import AGVDetector, Detection
    from measurement.size_measurement import SizeMeasurement
    from measurement.speed_tracker import SpeedTracker
    from visualization.display import Visualizer
    from utils.config import SystemConfig

    NEW_MODULES_AVAILABLE = True
except ImportError:
    NEW_MODULES_AVAILABLE = False
    print(
        "Warning: New AGV Measurement System modules not available. Using basic mode."
    )


class AMROBBTracker:
    def __init__(self, fps=30, pixel_to_meter=0.01, track_id=0, stationary_mode=False):
        """
        AMR tracker with position, velocity, and orientation tracking

        Args:
            fps: Camera frame rate
            pixel_to_meter: Conversion ratio (1 pixel = ? meters)
            track_id: Unique ID for this tracker
            stationary_mode: If True, only measure position and orientation, skip velocity
        """
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter
        self.track_id = track_id
        self.stationary_mode = stationary_mode
        self.kf = self.init_kalman()

        # For angle continuity (handle angle wrap-around)
        self.prev_angle = None
        self.angle_offset = 0

        # For debugging/visualization
        self.trajectory = deque(maxlen=100)

        # Color for this tracker (different colors for different objects)
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Light Green
        ]
        self.color = colors[track_id % len(colors)]

        # For tracking lost objects
        self.last_detection_frame = 0
        self.frames_since_detection = 0

    def init_kalman(self):
        """
        Initialize Kalman filter
        State vector: [x, y, theta, vx, vy, omega]
        Measurement vector: [x, y, theta]
        """
        kf = KalmanFilter(dim_x=6, dim_z=3)

        dt = 1.0 / self.fps

        # State transition matrix
        kf.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],  # x = x + vx*dt
                [0, 1, 0, 0, dt, 0],  # y = y + vy*dt
                [0, 0, 1, 0, 0, dt],  # theta = theta + omega*dt
                [0, 0, 0, 1, 0, 0],  # vx = vx
                [0, 0, 0, 0, 1, 0],  # vy = vy
                [0, 0, 0, 0, 0, 1],  # omega = omega
            ]
        )

        # Measurement matrix
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

        # Measurement noise covariance
        kf.R = np.array(
            [
                [10, 0, 0],  # x measurement noise
                [0, 10, 0],  # y measurement noise
                [0, 0, 0.1],  # theta measurement noise (radians)
            ]
        )

        # Process noise covariance
        kf.Q = np.eye(6)
        kf.Q[0, 0] = kf.Q[1, 1] = 0.1  # position process noise
        kf.Q[2, 2] = 0.01  # angle process noise
        kf.Q[3, 3] = kf.Q[4, 4] = 1.0  # velocity process noise
        kf.Q[5, 5] = 0.1  # angular velocity process noise

        # Initial state covariance
        kf.P *= 100

        # Initial state
        kf.x = np.zeros(6)

        return kf

    def detect_orientation_obb(self, image, bbox):
        """
        Detect AMR orientation using Oriented Bounding Box

        Args:
            image: Input frame
            bbox: [x, y, w, h] detection bounding box

        Returns:
            theta: Orientation in radians (-pi to pi)
        """
        x, y, w, h = map(int, bbox)

        # Extract ROI
        roi = image[y : y + h, x : x + w]

        if roi.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to get binary image
        # Use Otsu's method for automatic threshold selection
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get the largest contour (assuming it's the AMR)
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if contour is large enough
        if cv2.contourArea(largest_contour) < 100:  # Minimum area threshold
            return None

        # Fit minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        (center_x, center_y), (width, height), angle = rect

        # OpenCV returns angle in range [-90, 0]
        # Convert to consistent orientation
        if width < height:
            angle = angle + 90

        # Convert to radians
        theta = np.deg2rad(angle)

        # Normalize to [-pi, pi]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        return theta

    def handle_angle_continuity(self, new_angle):
        """
        Handle angle wrap-around for continuous tracking

        Args:
            new_angle: New measured angle in radians

        Returns:
            Continuous angle in radians
        """
        if self.prev_angle is None:
            self.prev_angle = new_angle
            return new_angle

        # Calculate angle difference
        diff = new_angle - self.prev_angle

        # Check for wrap-around
        if diff > np.pi:
            self.angle_offset -= 2 * np.pi
        elif diff < -np.pi:
            self.angle_offset += 2 * np.pi

        continuous_angle = new_angle + self.angle_offset
        self.prev_angle = new_angle

        return continuous_angle

    def update(self, image, detection_bbox, frame_number=0):
        """
        Update tracker with new detection

        Args:
            image: Current frame
            detection_bbox: [x, y, w, h] or None if no detection
            frame_number: Current frame number

        Returns:
            Dictionary with tracking results
        """
        # Prediction step
        self.kf.predict()

        if detection_bbox is not None:
            # Update detection tracking
            self.last_detection_frame = frame_number
            self.frames_since_detection = 0
            # Get center position
            x, y, w, h = detection_bbox
            cx = x + w / 2
            cy = y + h / 2

            # Detect orientation
            theta = self.detect_orientation_obb(image, detection_bbox)

            if theta is not None:
                # Handle angle continuity
                theta = self.handle_angle_continuity(theta)
            else:
                # Use previous angle or default to avoid dimension mismatch
                theta = self.prev_angle if self.prev_angle is not None else 0.0

            # Always use 3D measurement to avoid matrix dimension issues
            z = np.array([[cx], [cy], [theta]], dtype=np.float64)

            # Update step
            self.kf.update(z)

            # Store trajectory point
            self.trajectory.append((cx, cy))
        else:
            # No detection, increment frames since detection
            self.frames_since_detection += 1

        # Extract state
        state = self.kf.x

        # Calculate speeds (skip if in stationary mode)
        if not self.stationary_mode:
            linear_speed_pix = np.sqrt(state[3] ** 2 + state[4] ** 2) * self.fps
            linear_speed_m = linear_speed_pix * self.pixel_to_meter

            angular_speed_rad = abs(state[5])
            angular_speed_deg = np.rad2deg(angular_speed_rad)
        else:
            # In stationary mode, set velocities to 0
            linear_speed_pix = 0
            linear_speed_m = 0
            angular_speed_rad = 0
            angular_speed_deg = 0

        # Prepare output
        results = {
            "position": {
                "x": state[0],
                "y": state[1],
                "x_m": state[0] * self.pixel_to_meter,
                "y_m": state[1] * self.pixel_to_meter,
            },
            "orientation": {
                "theta_rad": state[2],
                "theta_deg": np.rad2deg(state[2]),
                "theta_normalized_deg": np.rad2deg(
                    np.arctan2(np.sin(state[2]), np.cos(state[2]))
                ),
            },
            "velocity": {
                "vx_pix_per_sec": (
                    state[3] * self.fps if not self.stationary_mode else 0
                ),
                "vy_pix_per_sec": (
                    state[4] * self.fps if not self.stationary_mode else 0
                ),
                "vx_m_per_sec": (
                    state[3] * self.fps * self.pixel_to_meter
                    if not self.stationary_mode
                    else 0
                ),
                "vy_m_per_sec": (
                    state[4] * self.fps * self.pixel_to_meter
                    if not self.stationary_mode
                    else 0
                ),
                "linear_speed_pix_per_sec": linear_speed_pix,
                "linear_speed_m_per_sec": linear_speed_m,
            },
            "angular_velocity": {
                "omega_rad_per_sec": state[5] if not self.stationary_mode else 0,
                "omega_deg_per_sec": (
                    np.rad2deg(state[5]) if not self.stationary_mode else 0
                ),
                "angular_speed_rad_per_sec": angular_speed_rad,
                "angular_speed_deg_per_sec": angular_speed_deg,
            },
            "bbox": detection_bbox,
            "trajectory": list(self.trajectory),
            "stationary_mode": self.stationary_mode,
        }

        return results

    def draw_visualization(self, image, results):
        """
        Draw tracking results on image

        Args:
            image: Frame to draw on
            results: Results from update()

        Returns:
            Annotated image
        """
        img_copy = image.copy()

        # Draw bounding box if available
        if results["bbox"] is not None:
            x, y, w, h = map(int, results["bbox"])
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), self.color, 2)

            # Draw orientation arrow
            cx = int(results["position"]["x"])
            cy = int(results["position"]["y"])

            theta = results["orientation"]["theta_rad"]
            arrow_length = 50
            end_x = int(cx + arrow_length * np.cos(theta))
            end_y = int(cy + arrow_length * np.sin(theta))

            cv2.arrowedLine(
                img_copy, (cx, cy), (end_x, end_y), self.color, 3, tipLength=0.3
            )

            # Draw speed instead of ID (or show stationary status)
            if not self.stationary_mode:
                speed_kmh = (
                    results["velocity"]["linear_speed_m_per_sec"] * 3.6
                )  # Convert m/s to km/h
                display_text = f"{speed_kmh:.1f}km/h"
            else:
                # In stationary mode, show position info
                pos_x = results["position"]["x"]
                pos_y = results["position"]["y"]
                display_text = f"({pos_x:.0f},{pos_y:.0f})"

            cv2.putText(
                img_copy,
                display_text,
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color,
                2,
            )

        # Draw trajectory with object's color
        if len(results["trajectory"]) > 1:
            pts = np.array(results["trajectory"], np.int32)
            cv2.polylines(img_copy, [pts], False, self.color, 2)

        # # Add text information
        # info_text = [
        #     f"Speed: {results['velocity']['linear_speed_m_per_sec']:.2f} m/s",
        #     f"Angle: {results['orientation']['theta_normalized_deg']:.1f} deg",
        #     f"Angular: {results['angular_velocity']['angular_speed_deg_per_sec']:.1f} deg/s",
        # ]

        # y_offset = 30
        # for i, text in enumerate(info_text):
        #     cv2.putText(
        #         img_copy,
        #         text,
        #         (10, y_offset + i * 25),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.7,
        #         (0, 255, 255),
        #         2,
        #     )

        return img_copy

    def is_lost(self, max_frames_lost=10):
        """
        Check if tracker has been lost for too long

        Args:
            max_frames_lost: Maximum frames without detection before considering lost

        Returns:
            bool: True if tracker should be removed
        """
        return self.frames_since_detection > max_frames_lost


class TrackingDataLogger:
    """Class to log tracking data to CSV files"""

    def __init__(self, output_dir="tracking_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.output_dir / f"tracking_results_{timestamp}.csv"

        # Create CSV file with headers
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "frame_number",
                    "track_id",
                    "position_x",
                    "position_y",
                    "velocity_x",
                    "velocity_y",
                    "linear_speed_ms",
                    "linear_speed_kmh",
                    "orientation_deg",
                    "detection_type",
                ]
            )

        print(f"✓ Tracking data will be saved to: {self.csv_file}")

    def log_tracking_result(self, frame_number: int, tracking_result: dict):
        """Log a single tracking result to CSV"""
        try:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                timestamp = datetime.now().isoformat()
                track_id = tracking_result.get("track_id", "unknown")

                # Position data
                pos = tracking_result.get("position", {})
                pos_x = pos.get("x", 0)
                pos_y = pos.get("y", 0)

                # Velocity data
                vel = tracking_result.get("velocity", {})
                vel_x = vel.get("vx", 0)
                vel_y = vel.get("vy", 0)
                linear_speed_ms = vel.get("linear_speed_m_per_sec", 0)
                linear_speed_kmh = linear_speed_ms * 3.6

                # Orientation data
                orient = tracking_result.get("orientation", {})
                orientation_deg = orient.get("theta_normalized_deg", 0)

                # Detection type
                detection_type = tracking_result.get("detection_type", "unknown")

                writer.writerow(
                    [
                        timestamp,
                        frame_number,
                        track_id,
                        f"{pos_x:.2f}",
                        f"{pos_y:.2f}",
                        f"{vel_x:.2f}",
                        f"{vel_y:.2f}",
                        f"{linear_speed_ms:.2f}",
                        f"{linear_speed_kmh:.2f}",
                        f"{orientation_deg:.2f}",
                        detection_type,
                    ]
                )

        except Exception as e:
            print(f"Warning: Failed to log tracking data: {e}")


class EnhancedAMRSystem:
    """
    Enhanced AMR tracking system that can use different detectors and trackers
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        detector_type: str = "yolo",
        tracker_type: str = "kalman",
        stationary_mode: bool = False,
    ):
        """
        Initialize enhanced AMR system

        Args:
            config: System configuration
            detector_type: Type of detector ("yolo", "color", "agv")
            tracker_type: Type of tracker ("kalman", "speed")
            stationary_mode: If True, only measure position and orientation, skip velocity
        """
        self.config = config
        self.detector_type = detector_type
        self.tracker_type = tracker_type
        self.stationary_mode = stationary_mode

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
        print(f"New modules: {NEW_MODULES_AVAILABLE}")

        # Initialize detector
        if self.detector_type == "yolo":
            import ultralytics

            self.detector = ultralytics.YOLO("yolov8n.pt")
            print("✓ YOLO detector initialized")
        elif self.detector_type == "color" and NEW_MODULES_AVAILABLE:
            self.detector = AGVDetector(min_area=1000)
            print("✓ Color-based AGV detector initialized")
        elif self.detector_type == "agv" and NEW_MODULES_AVAILABLE:
            self.detector = AGVDetector(min_area=1000)
            print("✓ AGV detector initialized")
        else:
            print("Warning: Using fallback YOLO detector")
            import ultralytics

            self.detector = ultralytics.YOLO("yolov8n.pt")

        # Initialize tracker
        if self.tracker_type == "kalman":
            self.trackers = {}  # Dictionary to store multiple trackers
            self.next_track_id = 0
            print("✓ Multi-object Kalman filter tracker initialized")
        elif self.tracker_type == "speed" and NEW_MODULES_AVAILABLE:
            self.speed_tracker = SpeedTracker(max_history=30, max_tracking_distance=500)
            print("✓ Speed tracker initialized")
        else:
            self.trackers = {}  # Dictionary to store multiple trackers
            self.next_track_id = 0
            print("✓ Fallback multi-object Kalman filter tracker initialized")

        # Initialize additional components if using new modules
        if NEW_MODULES_AVAILABLE and self.config:
            try:
                # Load calibration data if available
                calibration_path = self.config.calibration.calibration_data_path
                if Path(calibration_path).exists():
                    with open(calibration_path, "r") as f:
                        calibration_data = json.load(f)

                    # Always initialize size measurement and visualizer if calibration data exists
                    self.size_measurement = SizeMeasurement(
                        homography=np.array(calibration_data["homography"]),
                        camera_height=calibration_data["camera_height"],
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

        if self.detector_type == "yolo":
            # YOLO detection - filter only cars and trucks
            # COCO class IDs: car=2, truck=7, bus=5
            results = self.detector(
                frame, classes=[2, 7]
            )  # Only detect cars and trucks
            if len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    confidence = results[0].boxes.conf[i].cpu().numpy()

                    # Create a simple detection object for compatibility with size measurement
                    detection_obj = type(
                        "Detection",
                        (),
                        {
                            "bbox": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
                            "confidence": float(confidence),
                            "type": "vehicle",
                        },
                    )()

                    detections.append(
                        {
                            "bbox": bbox,
                            "confidence": float(confidence),
                            "type": "yolo",
                            "detection_obj": detection_obj,
                        }
                    )

        elif self.detector_type in ["color", "agv"] and NEW_MODULES_AVAILABLE:
            # AGV detector
            agv_detections = self.detector.detect(frame, frame_number, timestamp)
            for detection in agv_detections:
                # Convert bbox format
                bbox_points = detection.bbox
                x_coords = bbox_points[:, 0]
                y_coords = bbox_points[:, 1]
                x1, y1 = np.min(x_coords), np.min(y_coords)
                x2, y2 = np.max(x_coords), np.max(y_coords)
                bbox = [x1, y1, x2 - x1, y2 - y1]

                detections.append(
                    {
                        "bbox": bbox,
                        "confidence": detection.confidence,
                        "type": "agv",
                        "detection_obj": detection,
                    }
                )

        return detections

    def track_objects(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Track objects using selected tracker"""
        results = []

        if self.tracker_type == "kalman":
            # Use multi-object Kalman filter tracking
            if detections:
                # Simple association: assign detections to closest trackers
                used_detections = set()

                # Update existing trackers
                for track_id, tracker in list(self.trackers.items()):
                    if not detections:
                        break

                    # Find closest detection
                    best_detection_idx = None
                    best_distance = float("inf")

                    for i, detection in enumerate(detections):
                        if i in used_detections:
                            continue

                        bbox = detection["bbox"]
                        cx = bbox[0] + bbox[2] / 2
                        cy = bbox[1] + bbox[3] / 2

                        # Get tracker's last position
                        tracker_pos = tracker.kf.x[:2]
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
                            frame, detection["bbox"], frame_number=len(results)
                        )
                        tracking_result["detection_type"] = detection.get(
                            "type", "unknown"
                        )
                        tracking_result["track_id"] = track_id
                        tracking_result["color"] = tracker.color

                        # Add size measurement if available
                        if self.size_measurement and "detection_obj" in detection:
                            size_measurement = self.size_measurement.measure(
                                detection["detection_obj"]
                            )
                            tracking_result["size_measurement"] = size_measurement

                        results.append(tracking_result)
                        used_detections.add(best_detection_idx)
                    else:
                        # No detection found, predict only
                        tracking_result = tracker.update(
                            frame, None, frame_number=len(results)
                        )
                        tracking_result["detection_type"] = "predicted"
                        tracking_result["track_id"] = track_id
                        tracking_result["color"] = tracker.color
                        results.append(tracking_result)

                # Create new trackers for unassigned detections
                for i, detection in enumerate(detections):
                    if i not in used_detections:
                        new_tracker = AMROBBTracker(
                            fps=30,
                            pixel_to_meter=0.01,
                            track_id=self.next_track_id,
                            stationary_mode=self.stationary_mode,
                        )
                        self.trackers[self.next_track_id] = new_tracker

                        tracking_result = new_tracker.update(
                            frame, detection["bbox"], frame_number=len(results)
                        )
                        tracking_result["detection_type"] = detection.get(
                            "type", "unknown"
                        )
                        tracking_result["track_id"] = self.next_track_id
                        tracking_result["color"] = new_tracker.color

                        # Add size measurement if available
                        if self.size_measurement and "detection_obj" in detection:
                            size_measurement = self.size_measurement.measure(
                                detection["detection_obj"]
                            )
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

        elif self.tracker_type == "speed" and NEW_MODULES_AVAILABLE:
            # Use speed tracker (multiple objects)
            measurements = []
            for detection in detections:
                if "detection_obj" in detection and self.size_measurement:
                    # Measure size if calibration is available
                    measurement = self.size_measurement.measure(
                        detection["detection_obj"]
                    )
                    measurement["bbox"] = detection["bbox"]
                    measurement["confidence"] = detection["confidence"]
                    measurements.append(measurement)
                else:
                    # Basic measurement
                    bbox = detection["bbox"]
                    cx = bbox[0] + bbox[2] / 2
                    cy = bbox[1] + bbox[3] / 2
                    measurements.append(
                        {
                            "center_world": (cx, cy),
                            "bbox": detection["bbox"],
                            "confidence": detection["confidence"],
                            "timestamp": time.time(),
                        }
                    )

            # Update speed tracker
            results = self.speed_tracker.update(measurements)

        return results

    def visualize_results(
        self, frame: np.ndarray, detections: List[Dict], tracking_results: List[Dict]
    ) -> np.ndarray:
        """Visualize results using appropriate visualizer"""
        if self.tracker_type == "kalman" and tracking_results:
            # Use multi-object AMR tracker visualization
            if self.visualizer and self.size_measurement:
                # Use enhanced visualizer if available
                detection_objects = [
                    d.get("detection_obj") for d in detections if "detection_obj" in d
                ]
                vis_frame = self.visualizer.draw_detections(
                    frame, detection_objects, tracking_results
                )

                # Add summary information
                cv2.putText(
                    vis_frame,
                    f"Objects: {len(tracking_results)} (Enhanced)",
                    (10, vis_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
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
                cv2.putText(
                    vis_frame,
                    f"Objects: {len(tracking_results)}",
                    (10, vis_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            return vis_frame

        elif self.tracker_type == "speed" and NEW_MODULES_AVAILABLE and self.visualizer:
            # Use enhanced visualizer
            detection_objects = [
                d.get("detection_obj") for d in detections if "detection_obj" in d
            ]
            return self.visualizer.draw_detections(
                frame, detection_objects, tracking_results
            )

        else:
            # Basic visualization
            vis_frame = frame.copy()
            for detection in detections:
                bbox = detection["bbox"]
                x, y, w, h = map(int, bbox)
                cv2.putText(
                    vis_frame,
                    f"Conf: {detection['confidence']:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
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
    if not args.stationary_mode and "stationary_mode" in preset:
        args.stationary_mode = preset["stationary_mode"]
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
        choices=["yolo", "color", "agv"],
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

    # Initialize system
    if args.mode == "basic":
        print("Running in basic mode...")
        run_basic_mode(args.source, args.fps, args.loader_mode, args.stationary_mode)
    elif args.mode == "enhanced":
        if not NEW_MODULES_AVAILABLE:
            print("Error: Enhanced mode requires AGV Measurement System modules.")
            print("Please install the required modules or use basic mode.")
            return
        print("Running in enhanced mode...")
        run_enhanced_mode(args, args.source, args.fps, args.loader_mode)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        return


def run_basic_mode(video_source, fps=30, mode="auto", stationary_mode=False):
    """Run basic multi-object AMR tracker"""
    import ultralytics

    detector = ultralytics.YOLO("yolov8n.pt")
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection - filter only cars and trucks
        # COCO class IDs: car=2, truck=7, bus=5
        results = detector(frame, classes=[2, 7])  # Only detect cars and trucks

        # Get all detections
        detections = []
        if len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                bbox = [x1, y1, x2 - x1, y2 - y1]
                confidence = results[0].boxes.conf[i].cpu().numpy()
                detections.append(
                    {"bbox": bbox, "confidence": float(confidence), "type": "vehicle"}
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

                bbox = detection["bbox"]
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2

                # Get tracker's last position
                tracker_pos = tracker.kf.x[:2]
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
                    frame, detection["bbox"], frame_number=frame_number
                )
                tracking_result["detection_type"] = detection.get("type", "unknown")
                tracking_result["track_id"] = track_id
                tracking_result["color"] = tracker.color
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
                new_tracker = AMROBBTracker(
                    fps=fps,
                    pixel_to_meter=0.01,
                    track_id=next_track_id,
                    stationary_mode=stationary_mode,
                )
                trackers[next_track_id] = new_tracker

                tracking_result = new_tracker.update(
                    frame, detection["bbox"], frame_number=frame_number
                )
                tracking_result["detection_type"] = detection.get("type", "unknown")
                tracking_result["track_id"] = next_track_id
                tracking_result["color"] = new_tracker.color
                tracking_results.append(tracking_result)

                next_track_id += 1

                # Clean up lost trackers
        trackers_to_remove = []
        for track_id, tracker in trackers.items():
            if tracker.is_lost(max_frames_lost=10):
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
        cv2.putText(
            vis_frame,
            f"Objects: {len(tracking_results)}",
            (10, vis_frame.shape[0] - 30),  # 위쪽으로 이동
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Add frame info
        if hasattr(cap.loader, "frame_number"):  # Image sequence
            cv2.putText(
                vis_frame,
                f"Frame: {frame_number}",
                (10, vis_frame.shape[0] - 10),  # 맨 아래
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Print results
        if tracking_results:
            print(f"Frame {frame_number}: {len(tracking_results)} objects tracked")
            for result in tracking_results:
                track_id = result.get("track_id", "Unknown")
            print(
                f"  ID {track_id}: Position ({result['position']['x']:.1f}, {result['position']['y']:.1f}), "
                f"Speed: {result['velocity']['linear_speed_m_per_sec']:.2f} m/s, "
                f"Orientation: {result['orientation']['theta_normalized_deg']:.1f}°"
            )

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


def run_enhanced_mode(args, video_source, fps=30, mode="auto"):
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
        stationary_mode=args.stationary_mode,
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
        if hasattr(cap.loader, "frame_number"):  # Image sequence
            cv2.putText(
                vis_frame,
                f"Frame: {frame_number}",
                (10, vis_frame.shape[0] - 10),  # 맨 아래
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Print results (removed verbose logging for better performance)

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
