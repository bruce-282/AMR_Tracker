"""
Kalman Filter Tracker for AMR tracking system.

This module provides a Kalman filter-based tracker for tracking objects
with position, velocity, and orientation.
"""

import logging
from collections import deque
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class KalmanTracker:
    """
    Kalman filter-based tracker for AMR objects

    Tracks position, velocity, and orientation using a Kalman filter.
    """

    def __init__(self, fps=30, pixel_size=1.0, track_id=0):
        """
        Initialize Kalman tracker

        Args:
            fps: Camera frame rate (initial value, can be updated from timestamps)
            pixel_size: Pixel size in mm (1 pixel = ? mm)
            track_id: Unique ID for this tracker
        """
        self.fps = fps
        self.pixel_size = pixel_size
        self.track_id = track_id
        self.kf = self.init_kalman()

        # For angle continuity (handle angle wrap-around)
        self.prev_angle = None
        self.angle_offset = 0

        # For debugging/visualization
        self.trajectory = deque(maxlen=100)
        
        # For timestamp-based FPS calculation
        self.last_timestamp = None
        self.timestamp_history = deque(maxlen=10)  # Store last 10 frame intervals

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
        self.last_bbox = None
        self.last_size_measurement = (
            None  # Store last size measurement for prediction mode
        )
        self.initial_size_measurement = (
            None  # Store initial size measurement (never changes)
        )
        
        # For detecting segmentation failures (sudden center jumps)
        self.last_center = None  # Last valid center position
        self.use_prediction_only = False  # Flag to use prediction only mode

    def init_kalman(self):
        """
        Initialize Kalman filter

        State vector: [x, y, theta, vx, vy, omega]
        - x, y: position
        - theta: orientation (in degrees)
        - vx, vy: linear velocity
        - omega: angular velocity (in degrees/frame)
        """
        kf = cv2.KalmanFilter(6, 3)  # 6 states, 3 measurements

        # State transition matrix (A)
        dt = 1.0 / self.fps
        kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 1, 0, 0],  # x = x + vx*dt
                [0, 1, 0, 0, 1, 0],  # y = y + vy*dt
                [0, 0, 1, 0, 0, 1],  # theta = theta + omega*dt
                [0, 0, 0, 1, 0, 0],  # vx = vx (constant velocity model)
                [0, 0, 0, 0, 1, 0],  # vy = vy (constant velocity model)
                [0, 0, 0, 0, 0, 1],  # omega = omega (constant angular velocity model)
            ],
            dtype=np.float32,
        )
        # 6 x 6 matrix

        # Measurement matrix (H)
        kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # measure x
                [0, 1, 0, 0, 0, 0],  # measure y
                [0, 0, 1, 0, 0, 0],  # measure theta
            ],
            dtype=np.float32,
        )
        # 3 x 6 matrix

        # Process noise covariance (Q) - how much we trust the model
        kf.processNoiseCov = np.eye(6, dtype=np.float32)
        kf.processNoiseCov[0:3, 0:3] *= 0.1  # position/angle noise (low - model is reliable)
        kf.processNoiseCov[3:6, 3:6] *= 0.01  # velocity noise (very low - velocity should be stable)

        # Measurement noise covariance (R) - how much we trust the measurements
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32)
        kf.measurementNoiseCov[0:2, 0:2] *= 0.5  # position noise (lower - trust measurements more)
        kf.measurementNoiseCov[2, 2] *= 5.0  # angle noise (higher - angle is less reliable)

        # Error covariance (P) - initial uncertainty
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 10  # Lower initial uncertainty
        kf.errorCovPost[3:6, 3:6] *= 1.0  # Lower uncertainty for velocities (start with low velocity)

        # Initialize state
        kf.statePre = np.zeros((6, 1), dtype=np.float32)
        kf.statePost = np.zeros((6, 1), dtype=np.float32)

        return kf

    def initialize_with_detection(self, detection_bbox):
        """Initialize tracker with first detection"""
        if detection_bbox is not None:
            x, y, w, h = detection_bbox
            cx = x + w / 2
            cy = y + h / 2

            # Initialize state with detection position
            self.kf.statePre = np.array(
                [[cx], [cy], [0], [0], [0], [0]], dtype=np.float32
            )
            self.kf.statePost = np.array(
                [[cx], [cy], [0], [0], [0], [0]], dtype=np.float32
            )
            
            # Initialize last center for jump detection
            self.last_center = (cx, cy)

            logger.debug(f"Track {self.track_id} initialized at ({cx:.1f}, {cy:.1f}) px")

    def update(
        self,
        frame: np.ndarray,
        bbox: List[float],
        frame_number: int = 0,
        orientation: Optional[float] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Update tracker with new detection

        Args:
            frame: Current frame
            bbox: Bounding box [x, y, w, h]
            frame_number: Current frame number
            orientation: Orientation angle in degrees (from detection mask, optional)

        Returns:
            dict: Tracking results
        """
        # Update FPS from timestamp if available
        if timestamp is not None:
            self._update_fps_from_timestamp(timestamp)
        
        # Prediction step
        self.kf.predict()

        state_pre = self.kf.statePre.flatten()
        oriented_bbox = None

        if bbox is not None:
            # Update detection tracking
            self.last_detection_frame = frame_number
            self.frames_since_detection = 0
            
            # Get center position from bbox
            # Note: If bbox came from mask-based oriented_box_info, this center
            # is already the mask-based center (from minAreaRect)
            x, y, w, h = bbox
            measured_cx = x + w / 2
            measured_cy = y + h / 2
            
            # Get predicted position from Kalman filter (after predict step)
            predicted_cx = state_pre[0]
            predicted_cy = state_pre[1]
            predicted_vx = state_pre[3]
            predicted_vy = state_pre[4]
            
            # Check if center jump is too large (segmentation failure detection)
            # Calculate distance between measured and predicted center
            center_jump_distance = np.sqrt(
                (measured_cx - predicted_cx) ** 2 + (measured_cy - predicted_cy) ** 2
            )
            
            # Use bbox size as reference (average of width and height)
            bbox_size = (w + h) / 2
            jump_threshold = bbox_size * 0.5  # 5% of bbox size
            
            # Determine if we should use measurement or prediction only
            # Each frame is evaluated independently - if measurement is valid, use it
            use_measurement = True
            if self.last_center is not None:
                # Check if measured center jumped too much from predicted position
                if center_jump_distance > jump_threshold:
                    logger.warning(
                        f"Track {self.track_id}: Center jump detected "
                        f"({center_jump_distance:.1f}px > {jump_threshold:.1f}px), using prediction only"
                    )
                    use_measurement = False
                    self.use_prediction_only = True
                else:
                    # Normal detection - measurement is valid, return to normal mode
                    use_measurement = True
                    self.use_prediction_only = False
            else:
                # First frame - always use measurement to initialize
                use_measurement = True
                self.use_prediction_only = False
            
            # Use velocity-based corrected position if measurement is unreliable
            if not use_measurement:
                cx = predicted_cx
                cy = predicted_cy
            else:
                # Use measured position (from mask-based oriented_box_info)
                # This is a valid detection - update last_center and return to normal mode
                cx = measured_cx
                cy = measured_cy
                self.last_center = (cx, cy)
                self.use_prediction_only = False

            # Use orientation from detection mask if available, otherwise detect it
            # orientation comes from detection.get_orientation() which now returns degrees
            if orientation is not None:
                # Orientation from mask-based minAreaRect (already in degrees)
                theta = orientation
                # Handle angle continuity (in degrees)
                theta = self.handle_angle_continuity(theta)
            else:
                # Fallback: Detect orientation using OBB method from frame
                theta = self.detect_orientation_obb(frame, bbox)
                if theta is not None:
                    # Handle angle continuity (in degrees)
                    theta = self.handle_angle_continuity(theta)
                else:
                    # Use previous angle or default to avoid dimension mismatch
                    theta = self.prev_angle if self.prev_angle is not None else 0.0

            # Measurement vector: [cx, cy, theta]
            # cx, cy: either measured (if valid) or predicted (if segmentation failed)
            # theta: orientation in degrees from mask-based oriented_box_info (if mask available)
            # This is the input to Kalman filter correction step
            z = np.array([[cx], [cy], [theta]], dtype=np.float32)

            # Ensure z has correct shape and type
            if z.shape != (3, 1):
                z = z.reshape(3, 1)

            # Update step (only if measurement is reliable, otherwise prediction is already done)
            try:
                if use_measurement:
                    # Normal correction with measurement
                    self.kf.correct(z)
                else:
                    # Don't correct, just use prediction (statePost = statePre)
                    # This maintains prediction-only mode
                    self.kf.statePost = self.kf.statePre.copy()

            except cv2.error as e:
                logger.warning(f"Kalman filter correction failed: {e}")
                # Skip this update if correction fails

            # Apply velocity damping when position is nearly stationary
            # If position change is very small, reduce velocity towards zero
            if use_measurement and self.last_center is not None:
                state_post = self.kf.statePost.flatten()
                # Calculate actual position change
                pos_change_x = abs(measured_cx - self.last_center[0])
                pos_change_y = abs(measured_cy - self.last_center[1])
                pos_change = np.sqrt(pos_change_x**2 + pos_change_y**2)
                
                # If position change is very small (< 0.5 pixels), apply velocity damping
                VELOCITY_DAMPING_THRESHOLD = 0.5  # pixels
                VELOCITY_DAMPING_FACTOR = 0.9  # Multiply velocity by this factor when stationary
                
                if pos_change < VELOCITY_DAMPING_THRESHOLD:
                    # Object is nearly stationary - reduce velocity
                    current_vx = state_post[3]
                    current_vy = state_post[4]
                    current_vel_magnitude = np.sqrt(current_vx**2 + current_vy**2)
                    
                    # Only damp if velocity is already small (avoid damping during actual movement)
                    if current_vel_magnitude < 5.0:  # Only damp velocities < 5 pixels/frame
                        # Apply damping: reduce velocity by damping factor
                        new_vx = current_vx * VELOCITY_DAMPING_FACTOR
                        new_vy = current_vy * VELOCITY_DAMPING_FACTOR
                        
                        # Update velocity in state
                        self.kf.statePost[3, 0] = new_vx
                        self.kf.statePost[4, 0] = new_vy
                        
                        # Also update error covariance for velocity to reflect reduced uncertainty
                        # (velocity is more certain when object is stationary)
                        self.kf.errorCovPost[3, 3] *= 0.9
                        self.kf.errorCovPost[4, 4] *= 0.9

            # Store trajectory point and bbox
            self.trajectory.append((cx, cy))
            self.last_bbox = bbox
        else:
            # No detection, increment frames since detection
            self.frames_since_detection += 1

            state_post = self.kf.statePost.flatten()
            logger.debug(
                f"Track {self.track_id} prediction only: pos=({state_post[0]:.1f}, {state_post[1]:.1f})"
            )

        # Extract state
        state = self.kf.statePost.flatten()

        # Calculate speeds (always calculate)
        if True:
            # state[3], state[4] are velocity in pixels/frame
            # To convert to pixels/sec: multiply by fps
            # Formula: speed = sqrt(vx^2 + vy^2) * fps
            # Explanation:
            #   - state[3] = vx (pixels/frame)
            #   - state[4] = vy (pixels/frame)
            #   - sqrt(vx^2 + vy^2) = velocity magnitude (pixels/frame)
            #   - * fps = convert from per-frame to per-second (pixels/sec)
            linear_speed_pix = (
                np.sqrt(state[3] ** 2 + state[4] ** 2) * self.fps
            )  # pixels/sec
            linear_speed_mm = linear_speed_pix * self.pixel_size  # mm/s

            # Angular speed is already in deg/frame, convert to deg/sec
            angular_speed_deg = abs(state[5]) * self.fps  # deg/sec
            angular_speed_rad = np.deg2rad(angular_speed_deg)  # rad/sec (for compatibility)
        else:
            # In stationary mode, set velocities to 0
            linear_speed_pix = 0
            linear_speed_mm = 0  # mm/s
            angular_speed_rad = 0
            angular_speed_deg = 0

        # Prepare output
        results = {
            "position": {
                "x": state[0],
                "y": state[1],
                "x_mm": state[0] * self.pixel_size,
                "y_mm": state[1] * self.pixel_size,
            },
            "orientation": {
                "theta_deg": state[2],  # Already in degrees
                "theta_rad": np.deg2rad(state[2]),  # Convert to radians for compatibility
                "theta_normalized_deg": self.normalize_angle_deg(state[2]),
            },
            "velocity": {
                "linear_speed_pix_per_sec": linear_speed_pix,
                "linear_speed_mm_per_sec": linear_speed_mm,
                "angular_speed_rad_per_sec": angular_speed_rad,
                "angular_speed_deg_per_sec": angular_speed_deg,
            },
            "bbox": bbox,
            "trajectory": list(self.trajectory),
        }

        # Include initial size measurement if available (always use initial size)
        if self.initial_size_measurement is not None:
            results["size_measurement"] = self.initial_size_measurement

        return results
    
    def _update_fps_from_timestamp(self, timestamp: float):
        """
        Update FPS based on actual frame timestamps.
        
        Args:
            timestamp: Current frame timestamp in seconds
        """
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0:
                # Calculate instantaneous FPS for this frame
                instant_fps = 1.0 / dt
                self.timestamp_history.append(instant_fps)
                
                # Use median of recent FPS values for stability
                if len(self.timestamp_history) >= 3:
                    fps_list = list(self.timestamp_history)
                    fps_list.sort()
                    median_idx = len(fps_list) // 2
                    self.fps = fps_list[median_idx]
                else:
                    # Use average if not enough samples
                    self.fps = sum(self.timestamp_history) / len(self.timestamp_history)
        
        self.last_timestamp = timestamp

    def detect_orientation_obb(self, frame, bbox):
        """
        Detect object orientation using oriented bounding box

        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]

        Returns:
            float: Orientation angle in degrees
        """
        try:
            x, y, w, h = map(int, bbox)

            # Extract ROI
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                return None

            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Find contours
            contours, _ = cv2.findContours(
                gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None

            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit oriented bounding box
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]  # Already in degrees from OpenCV

            # Store for continuity
            self.prev_angle = angle

            return angle

        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return None

    def handle_angle_continuity(self, new_angle):
        """
        Handle angle continuity to avoid jumps (e.g., from 359° to 0°)

        Args:
            new_angle: New angle in degrees

        Returns:
            float: Continuous angle in degrees
        """
        if self.prev_angle is None:
            return new_angle

        # Calculate angle difference
        angle_diff = new_angle - self.prev_angle

        # Normalize to [-180, 180] degrees
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # Apply offset
        continuous_angle = self.prev_angle + angle_diff

        return continuous_angle

    def normalize_angle_deg(self, angle_deg):
        """
        Normalize angle to [0, 360) degrees

        Args:
            angle_deg: Angle in degrees

        Returns:
            float: Normalized angle
        """
        while angle_deg < 0:
            angle_deg += 360
        while angle_deg >= 360:
            angle_deg -= 360
        return angle_deg

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
        
        # Calculate font scale and thickness based on image size (do this first)
        height, width = img_copy.shape[:2]
        base_width = 1920  # 기준 해상도
        base_font_scale = 0.8  # 기준 폰트 크기
        font_scale = max(0.3, min(3.0, (width / base_width) * base_font_scale))
        thickness = max(1, int(font_scale * 2))

        # Draw bounding box if available
        if results["bbox"] is not None:
            x, y, w, h = map(int, results["bbox"])
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), self.color, 2)

            # Draw orientation arrow
            cx = int(results["position"]["x"])
            cy = int(results["position"]["y"])

            theta_deg = results["orientation"]["theta_deg"]
            theta_rad = np.deg2rad(theta_deg)  # Convert to radians for drawing
            arrow_length = 50
            end_x = int(cx + arrow_length * np.cos(theta_rad))
            end_y = int(cy + arrow_length * np.sin(theta_rad))

            cv2.arrowedLine(
                img_copy, (cx, cy), (end_x, end_y), self.color, 3, tipLength=0.3
            )

            # Draw orientation angle text
            angle_text = f"θ: {theta_deg:.1f}°"
            cv2.putText(
                img_copy,
                angle_text,
                (cx + 20, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.7,
                self.color,
                thickness,
            )

            # Draw minAreaRect detection if available
            # if hasattr(self, "last_bbox") and self.last_bbox is not None:
            if True:  # 항상 텍스트 표시
                try:
                    # Extract ROI for orientation detection
                    x, y, w, h = self.last_bbox
                    roi = img_copy[int(y) : int(y + h), int(x) : int(x + w)]
                    if roi.size > 0:
                        # Detect orientation in current frame
                        detected_angle_deg = self.detect_orientation_obb(
                            frame=img_copy, bbox=self.last_bbox
                        )
                        if detected_angle_deg is not None:
                            # Draw detected orientation with different color
                            detected_angle_rad = np.deg2rad(detected_angle_deg)
                            detected_end_x = int(
                                cx + arrow_length * np.cos(detected_angle_rad)
                            )
                            detected_end_y = int(
                                cy + arrow_length * np.sin(detected_angle_rad)
                            )

                            # Draw detected orientation in red
                            cv2.arrowedLine(
                                img_copy,
                                (cx, cy),
                                (detected_end_x, detected_end_y),
                                (0, 0, 255),
                                2,
                                tipLength=0.2,
                            )
                except Exception as e:
                    pass  # Skip if detection fails

            # Show comprehensive tracking info
            speed_mms = results["velocity"]["linear_speed_mm_per_sec"]
            pos_x = results["position"]["x"]
            pos_y = results["position"]["y"]
            track_id = results.get("track_id", "?")

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
