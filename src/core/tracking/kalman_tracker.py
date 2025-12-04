"""
Kalman Filter Tracker for AMR tracking system.

This module provides a Kalman filter-based tracker for tracking objects
with position, velocity, and orientation.
"""

import logging
from collections import deque
from typing import List, Optional, Dict, Tuple
#from sklearn.utils.linear_assignment_ import linear_assignment
import cv2
import numpy as np

MAX_FRAMES_LOST = 500  # Default value, can be overridden via config

logger = logging.getLogger(__name__)

from numba import jit

@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)


# def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
#     """
#     Assigns detections to tracked object (both represented as bounding boxes)

#     Returns 3 lists of matches, unmatched_detections and unmatched_trackers
#     """
#     if(len(trackers)==0):
#         return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
#     iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

#     for d,det in enumerate(detections):
#         for t,trk in enumerate(trackers):
#             iou_matrix[d,t] = iou(det,trk)
#     matched_indices = linear_assignment(-iou_matrix)

#     unmatched_detections = []
#     for d,det in enumerate(detections):
#         if(d not in matched_indices[:,0]):
#             unmatched_detections.append(d)
#     unmatched_trackers = []
#     for t,trk in enumerate(trackers):
#         if(t not in matched_indices[:,1]):
#             unmatched_trackers.append(t)

#     #filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if(iou_matrix[m[0],m[1]]<iou_threshold):
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1,2))
#     if(len(matches)==0):
#         matches = np.empty((0,2),dtype=int)
#     else:
#         matches = np.concatenate(matches,axis=0)

#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class KalmanTracker:
    """
    Kalman filter-based tracker for AMR objects

    Tracks position, velocity, and orientation using a Kalman filter.
    """

    def __init__(self, fps=30, pixel_size=1.0, distance_map_data=None, track_id=0, max_frames_lost=None):
        """
        Initialize Kalman tracker

        Args:
            fps: Camera frame rate (initial value, can be updated from timestamps)
            pixel_size: Pixel size in mm - float or dict with 'x', 'y' keys
                       - float: 단일 값 (기존 형식)
                       - dict: {'x': float, 'y': float} 또는 {'x': float, 'y': float, 'average': float}
            distance_map_data: Distance map data dict from PixelDistanceMapper.load_distance_map() (optional)
            track_id: Unique ID for this tracker
            max_frames_lost: Maximum frames without detection before track is lost (from config)
        """
        self.fps = fps
        self.distance_map_data = distance_map_data
        self.track_id = track_id
        self.max_frames_lost = max_frames_lost if max_frames_lost is not None else MAX_FRAMES_LOST
        
        # pixel_size 처리: dict 형태면 x, y 분리, 아니면 단일 값 사용
        if isinstance(pixel_size, dict):
            self.pixel_size_x = pixel_size.get('x', 1.0)
            self.pixel_size_y = pixel_size.get('y', 1.0)
            self.pixel_size = pixel_size.get('average', (self.pixel_size_x + self.pixel_size_y) / 2)
        else:
            self.pixel_size = pixel_size
            self.pixel_size_x = pixel_size
            self.pixel_size_y = pixel_size
        self.kf = self.init_kalman()

        # For angle continuity (handle angle wrap-around)
        self.prev_angle = None

        # For debugging/visualization
        self.trajectory = deque(maxlen=2000)
        
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
        #sdt = 1.0 / self.fps
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

    def initialize_with_detection(self, center, angle):
        """Initialize tracker with first detection"""
        if center is not None:
            cx, cy = center[0], center[1]

            # Initialize state with detection position
            self.kf.statePre = np.array(
                [[cx], [cy], [angle], [0], [0], [0]], dtype=np.float32
            )
            self.kf.statePost = np.array(
                [[cx], [cy], [angle], [0], [0], [0]], dtype=np.float32
            )
            
            # Initialize last center for jump detection
            self.last_center = (cx, cy)

            logger.debug(f"Track {self.track_id} initialized at ({cx:.1f}, {cy:.1f}) {angle:.1f} deg")

    def update(
        self,
        bbox: List[float],
        center: Optional[Tuple[float, float]] = None,
        theta: Optional[float] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Update tracker with new detection

        Args:
            bbox: Bounding box [x, y, w, h]
            center: Center [x, y]
            theta: Orientation angle in degrees (from detection mask, optional)
            frame_number: Current frame number
            timestamp: Timestamp
            image_size: Image size (width, height) for boundary checking

        Returns:
            dict: Tracking results
        """
        # Update FPS from timestamp if available
        if timestamp is not None:
            self._update_fps_from_timestamp(timestamp)
        
        # Prediction step
        self.kf.predict()

        state_pre = self.kf.statePre.flatten()
        
        # Check if pixel position is outside 70% of image size (reset if too far)
        if image_size is not None:
            img_width, img_height = image_size
            pred_cx = state_pre[0]
            pred_cy = state_pre[1]
            
            # Calculate thresholds (70% of image size)
            x_threshold_min = img_width * 0.3  # 30% from left edge
            x_threshold_max = img_width * 0.7   # 70% from left edge (30% from right)
            y_threshold_min = img_height * 0.3  # 30% from top edge
            y_threshold_max = img_height * 0.7  # 70% from top edge (30% from bottom)
            
            # Check if position is outside the valid region
            if (pred_cx < x_threshold_min or pred_cx > x_threshold_max or
                pred_cy < y_threshold_min or pred_cy > y_threshold_max):
                logger.warning(
                    f"Track {self.track_id}: Position ({pred_cx:.1f}, {pred_cy:.1f}) "
                    f"is outside 70% of image size ({img_width}x{img_height}), resetting tracker"
                )
                self.reset()
                # Return special flag to indicate reset - will need to reinitialize with first detection
                return {
                    "position": {"x": 0, "y": 0, "x_mm": 0, "y_mm": 0},
                    "orientation": {"theta_deg": 0, "theta_rad": 0, "theta_normalized_deg": 0},
                    "velocity": {
                        "linear_speed_pix_per_sec": 0,
                        "linear_speed_mm_per_sec": 0,
                        "angular_speed_rad_per_sec": 0,
                        "angular_speed_deg_per_sec": 0,
                    },
                    "bbox": None,
                    "trajectory": [],
                    "track_id": self.track_id,
                    "reset_required": True,  # Flag to indicate reset
                }

        if bbox is not None:
            # Update detection tracking
            self.last_detection_frame = frame_number
            self.frames_since_detection = 0
            
            # Get center position from bbox
            # Note: If bbox came from mask-based oriented_box_info, this center
            # is already the mask-based center (from minAreaRect)
            measured_cx, measured_cy = center[0], center[1]
            
            # Get predicted position from Kalman filter (after predict step)
            predicted_cx = state_pre[0]
            predicted_cy = state_pre[1]
            predicted_theta = state_pre[2]
            # predicted_vx = state_pre[3]
            # predicted_vy = state_pre[4]
            
            # Default measurement values (will be adjusted based on validation)
            cx = measured_cx
            cy = measured_cy
            theta_value = theta if theta is not None else predicted_theta
            
            # Check if center jump is too large (segmentation failure detection)
            # Calculate distance between measured and predicted center
            # center_jump_distance = np.sqrt(
            #     (measured_cx - predicted_cx) ** 2 + (measured_cy - predicted_cy) ** 2
            # )
            # measured_vx = center_jump_distance / self.fps
            # measured_vy = center_jump_distance / self.fps
            
            # Use bbox size as reference (average of width and height)
           

            
            # Determine if we should use measurement or prediction only
            # Each frame is evaluated independently - if measurement is valid, use it

            
            iou_score = iou((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]), (predicted_cx-bbox[2]/2, predicted_cy-bbox[3]/2, predicted_cx+bbox[2]/2, predicted_cy+bbox[3]/2))

            logger.info(f"measured_cx: {measured_cx}, measured_cy: {measured_cy}, predicted_cx: {predicted_cx}, predicted_cy: {predicted_cy}")
            # if iou_score > 0.5:
            #     use_measurement = True
            # else:
            #     use_measurement = False
            #     self.use_prediction_only = True
            #     logger.warning(
            #             f"Track {self.track_id}: IoU score too low "
            #             f"({iou_score:.1f}), using prediction only"
            #     )
                
                
            use_measurement = True
            if self.last_center is not None:
                # Check if measured center jumped too much from predicted position
                if iou_score < 0.5:
                    logger.warning(
                        f"Track {self.track_id}: IoU score too low "
                        f"({iou_score:.1f}), using prediction only"
                    )
                    use_measurement = False
                    cx = predicted_cx
                    cy = predicted_cy
                    theta_value = predicted_theta
                    self.use_prediction_only = True
                    self.frames_since_detection += 1
                    if self.is_lost(max_frames_lost=self.max_frames_lost):
                        logger.info(f"Track {self.track_id} is lost (frames={self.frames_since_detection} >= {self.max_frames_lost}), resetting")
                        self.reset()
                else:
                    # Normal detection - measurement is valid, return to normal mode
                    cx = measured_cx
                    cy = measured_cy
                    theta_value = theta_value
                    self.last_center = (cx, cy)
                    use_measurement = True
                    self.use_prediction_only = False
            else:
                # First frame - always use measurement to initialize
                cx = measured_cx
                cy = measured_cy
                theta_value = theta_value
                self.last_center = (cx, cy)
                use_measurement = True
                self.use_prediction_only = False
            
            # # Use velocity-based corrected position if measurement is unreliable
            # if not use_measurement:
            #     cx = predicted_cx
            #     cy = predicted_cy
            #     theta = predicted_theta
            # else:
            #     # Use measured position (from mask-based oriented_box_info)
            #     # This is a valid detection - update last_center and return to normal mode
            #     cx = measured_cx
            #     cy = measured_cy
            #     self.last_center = (cx, cy)
            #     self.use_prediction_only = False

            # Use orientation from detection mask if available, otherwise detect it
            # orientation comes from detection.get_orientation() which now returns degrees
            #if theta is not None:
                # Orientation from mask-based minAreaRect (already in degrees)
                # Handle angle continuity (in degrees)
            new_angle = self.handle_angle_continuity(theta_value)
            self.prev_angle = new_angle
            # else:
            #     # Fallback: Detect orientation using OBB method from frame
            #     theta = self.detect_orientation_obb(frame, bbox)
            #     if theta is not None:
            #         # Handle angle continuity (in degrees)
            #         theta = self.handle_angle_continuity(theta)
            #     else:
            #         # Use previous angle or default to avoid dimension mismatch
            #         theta = self.prev_angle if self.prev_angle is not None else 0.0

            # Measurement vector: [cx, cy, theta]
            # cx, cy: either measured (if valid) or predicted (if segmentation failed)
            # theta: orientation in degrees from mask-based oriented_box_info (if mask available)
            # This is the input to Kalman filter correction step
            
            z = np.array([[cx], [cy], [new_angle]], dtype=np.float32)

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
            # if use_measurement and self.last_center is not None:
            #     state_post = self.kf.statePost.flatten()
            #     # Calculate actual position change
            #     pos_change_x = abs(measured_cx - self.last_center[0])
            #     pos_change_y = abs(measured_cy - self.last_center[1])
            #     pos_change = np.sqrt(pos_change_x**2 + pos_change_y**2)
                
            #     # If position change is very small (< 0.5 pixels), apply velocity damping
            #     VELOCITY_DAMPING_THRESHOLD = 0.5  # pixels
            #     VELOCITY_DAMPING_FACTOR = 0.9  # Multiply velocity by this factor when stationary
                
            #     if pos_change < VELOCITY_DAMPING_THRESHOLD:
            #         # Object is nearly stationary - reduce velocity
            #         current_vx = state_post[3]
            #         current_vy = state_post[4]
            #         current_vel_magnitude = np.sqrt(current_vx**2 + current_vy**2)
                    
            #         # Only damp if velocity is already small (avoid damping during actual movement)
            #         if current_vel_magnitude < 5.0:  # Only damp velocities < 5 pixels/frame
            #             # Apply damping: reduce velocity by damping factor
            #             new_vx = current_vx * VELOCITY_DAMPING_FACTOR
            #             new_vy = current_vy * VELOCITY_DAMPING_FACTOR
                        
            #             # Update velocity in state
            #             self.kf.statePost[3, 0] = new_vx
            #             self.kf.statePost[4, 0] = new_vy
                        
            #             # Also update error covariance for velocity to reflect reduced uncertainty
            #             # (velocity is more certain when object is stationary)
            #             self.kf.errorCovPost[3, 3] *= 0.9
            #             self.kf.errorCovPost[4, 4] *= 0.9

            # Store trajectory point and bbox
            self.trajectory.append((cx, cy))
            self.last_bbox = bbox
        else:
            # No detection, increment frames since detection
            self.frames_since_detection += 1

            state_post = self.kf.statePost.flatten()
            # Update trajectory with predicted position even when no detection
            pred_cx = state_post[0]
            pred_cy = state_post[1]
            
            # Check if predicted position is outside 70% of image size (reset if too far)
            if image_size is not None:
                img_width, img_height = image_size
                x_threshold_min = img_width * 0.3
                x_threshold_max = img_width * 0.7
                y_threshold_min = img_height * 0.3
                y_threshold_max = img_height * 0.7
                
                if (pred_cx < x_threshold_min or pred_cx > x_threshold_max or
                    pred_cy < y_threshold_min or pred_cy > y_threshold_max):
                    logger.warning(
                        f"Track {self.track_id}: Predicted position ({pred_cx:.1f}, {pred_cy:.1f}) "
                        f"is outside 70% of image size ({img_width}x{img_height}), resetting tracker"
                    )
                    self.reset()
                    # Return special flag to indicate reset
                    return {
                        "position": {"x": 0, "y": 0, "x_mm": 0, "y_mm": 0},
                        "orientation": {"theta_deg": 0, "theta_rad": 0, "theta_normalized_deg": 0},
                        "velocity": {
                            "linear_speed_pix_per_sec": 0,
                            "linear_speed_mm_per_sec": 0,
                            "angular_speed_rad_per_sec": 0,
                            "angular_speed_deg_per_sec": 0,
                        },
                        "bbox": None,
                        "trajectory": [],
                        "track_id": self.track_id,
                        "reset_required": True,
                    }
            
            self.trajectory.append((pred_cx, pred_cy))
            
            logger.debug(
                f"Track {self.track_id} prediction only: pos=({pred_cx:.1f}, {pred_cy:.1f})"
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
            # Convert pixel speed to mm/s
            if self.distance_map_data:
                # Use distance map: average pixel_size from distance map
                # For speed calculation, use a representative pixel_size from the map
                # (e.g., center of image or average)
                distance_map = self.distance_map_data['distance_map']
                h, w = distance_map.shape
                # Use center pixel as reference
                center_pix_size = distance_map[h//2, w//2] / np.sqrt((h//2)**2 + (w//2)**2) if (h//2)**2 + (w//2)**2 > 0 else self.pixel_size
                linear_speed_mm = linear_speed_pix * center_pix_size  # mm/s
            else:
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

        # Convert pixel position to mm
        if self.distance_map_data:
            # Use distance map to get actual world coordinates
            x_pix, y_pix = int(state[0]), int(state[1])
            dx_map = self.distance_map_data['dx_map']
            dy_map = self.distance_map_data['dy_map']
            h, w = dx_map.shape
            if 0 <= y_pix < h and 0 <= x_pix < w:
                x_mm = float(dx_map[y_pix, x_pix])
                y_mm = float(dy_map[y_pix, x_pix])
            else:
                # Out of bounds, fallback to pixel_size (x, y 별도 적용)
                x_mm = state[0] * self.pixel_size_x
                y_mm = state[1] * self.pixel_size_y
        else:
            # pixel_size_x, pixel_size_y 별도 적용
            x_mm = state[0] * self.pixel_size_x
            y_mm = state[1] * self.pixel_size_y
        
        # Prepare output
        results = {
            "position": {
                "x": state[0],
                "y": state[1],
                "x_mm": x_mm,
                "y_mm": y_mm,
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
            center, (w, h), angle = rect
            
            # Normalize angle so that the LONG axis (major axis) is the reference
            # When long axis is parallel to image horizontal (x-axis), angle should be 0
            if h > w:
                # Height is the long axis, rotate reference by 90 degrees
                angle = angle - 90
            
            # Normalize to -90 ~ 90 range
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180

            # Store for continuity
            self.prev_angle = angle

            return angle

        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return None

    def handle_angle_continuity(self, new_angle):
        """
        Handle angle continuity to avoid jumps in [-90, 90] range.
        
        For long-axis reference angle system:
        - Range is [-90, 90] degrees
        - Jump from 89 to -89 should be interpreted as +2 degrees rotation
        - Jump from -89 to 89 should be interpreted as -2 degrees rotation

        Args:
            new_angle: New angle in degrees (already in -90~90 range)

        Returns:
            float: Continuous angle in degrees
        """
        if self.prev_angle is None:
            return new_angle

        # Calculate angle difference
        angle_diff = new_angle - self.prev_angle

        # Normalize to [-90, 90] range (half period for long-axis symmetry)
        # Since object has 180-degree symmetry (front/back look same), 
        # we use 180-degree period for continuity
        while angle_diff > 90:
            angle_diff -= 180
        while angle_diff < -90:
            angle_diff += 180

        # Apply offset
        continuous_angle = self.prev_angle + angle_diff

        return continuous_angle

    # def _calculate_iou(self, box1, box2):
    #     """
    #     Calculate Intersection over Union (IoU) of two bounding boxes.

    #     Args:
    #         box1: [x, y, w, h] format
    #         box2: [x, y, w, h] format

    #     Returns:
    #         IoU value between 0 and 1
    #     """
    #     x1, y1, w1, h1 = box1
    #     x2, y2, w2, h2 = box2

    #     # Calculate intersection coordinates
    #     x_left = max(x1, x2)
    #     y_top = max(y1, y2)
    #     x_right = min(x1 + w1, x2 + w2)
    #     y_bottom = min(y1 + h1, y2 + h2)

    #     if x_right < x_left or y_bottom < y_top:
    #         return 0.0

    #     intersection_area = (x_right - x_left) * (y_bottom - y_top)

    #     # Calculate union area
    #     box1_area = w1 * h1
    #     box2_area = w2 * h2
    #     union_area = box1_area + box2_area - intersection_area

    #     if union_area == 0:
    #         return 0.0

    #     return intersection_area / union_area


    def normalize_angle_deg(self, angle_deg):
        """
        Normalize angle to [-90, 90] degrees (long axis reference)
        
        When the long axis is horizontal (parallel to x-axis), angle is 0.
        Range: -90 to 90 degrees.

        Args:
            angle_deg: Angle in degrees

        Returns:
            float: Normalized angle in [-90, 90] range
        """
        # Normalize to -90 ~ 90 range
        while angle_deg > 90:
            angle_deg -= 180
        while angle_deg < -90:
            angle_deg += 180
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

    def reset(self):
        """
        Reset tracker
        """
        self.kf = self.init_kalman()
        self.prev_angle = None
        self.trajectory.clear()
        self.last_bbox = None
        self.last_size_measurement = None
        self.initial_size_measurement = None
        self.last_center = None
        self.use_prediction_only = False
        self.frames_since_detection = 0
        self.last_detection_frame = 0
        self.last_timestamp = None
        self.timestamp_history.clear()
        
        # self.fps = 30
        # self.pixel_size = 1.0
        #self.track_id = 0
