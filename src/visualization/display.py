"""Visualization module for displaying results."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches


def get_screen_resolution() -> Tuple[int, int]:
    """
    Get screen resolution. Returns (width, height).
    Falls back to 1920x1080 if detection fails.
    """
    try:
        # Windows
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        return (width, height)
    except Exception:
        pass
    
    # Fallback to common resolution
    return (1920, 1080)


def resize_frame_to_screen(
    frame: np.ndarray,
    max_ratio: float = 0.85,
    screen_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, float]:
    """
    Resize frame to fit within screen resolution.
    
    Args:
        frame: Input frame
        max_ratio: Maximum ratio of screen size to use (default 85%)
        screen_size: Override screen size (width, height). Auto-detected if None.
    
    Returns:
        Tuple of (resized frame, scale factor)
    """
    if screen_size is None:
        screen_size = get_screen_resolution()
    
    screen_w, screen_h = screen_size
    frame_h, frame_w = frame.shape[:2]
    
    # Calculate max allowed dimensions (leaving room for window borders/taskbar)
    max_w = int(screen_w * max_ratio)
    max_h = int(screen_h * max_ratio)
    
    # Calculate scale factor
    scale_w = max_w / frame_w if frame_w > max_w else 1.0
    scale_h = max_h / frame_h if frame_h > max_h else 1.0
    scale = min(scale_w, scale_h)
    
    # Only resize if needed
    if scale < 1.0:
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    return frame, 1.0


class Visualizer:
    """
    Handles visualization of detection and measurement results.

    Provides real-time display of AGV tracking, measurements, and speeds.
    """

    def __init__(self, homography: Optional[np.ndarray] = None):
        """
        Initialize visualizer.

        Args:
            homography: Optional homography matrix for coordinate transformation
        """
        self.homography = homography
        self.colors = self._generate_colors(20)
        self.latest_rect_angles = {}  # Track ID to latest minAreaRect angle (deg)

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for tracking visualization."""
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors

    def _extract_angle_deg(self, orientation) -> float:
        """Extract angle in degrees from various orientation formats.

        Accepts:
        - dict with keys like 'theta_normalized_deg', 'theta_deg', or 'theta_rad'
        - numpy scalar (e.g., np.float32)
        - plain float/int
        - single-element list/tuple/ndarray
        Returns a float angle in degrees.
        """
        # Dict formats from tracker
        if isinstance(orientation, dict):
            if "theta_normalized_deg" in orientation:
                return float(orientation["theta_normalized_deg"])
            if "theta_deg" in orientation:
                return float(orientation["theta_deg"])
            if "theta" in orientation:
                # 'theta' could be degrees; assume degrees
                return float(orientation["theta"])
            if "theta_rad" in orientation:
                return float(np.degrees(float(orientation["theta_rad"])))

        # numpy scalar
        try:
            import numpy as _np

            if isinstance(orientation, (_np.generic,)):
                return float(orientation)
        except Exception:
            pass

        # list/tuple/ndarray with single value
        if isinstance(orientation, (list, tuple)) and len(orientation) == 1:
            return float(orientation[0])
        if isinstance(orientation, np.ndarray) and orientation.size == 1:
            return float(orientation.reshape(()))

        # plain number or fallback
        try:
            return float(orientation)
        except Exception:
            return 0.0

    def _clean_mask(self, mask: np.ndarray, min_area: int = 200) -> np.ndarray:
        m = (mask.astype(np.uint8) > 0).astype(np.uint8)
        if m.sum() == 0:
            return m
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            return np.zeros_like(m)
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[idx, cv2.CC_STAT_AREA] < min_area:
            return np.zeros_like(m)
        out = (lbl == idx).astype(np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return out

    def draw_single_object(
        self, frame: np.ndarray, detections: List[Dict], trackings: List[Dict]
    ) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input image
            detections: List of Detection objects
            measurements: List of measurement results

        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        box_color = (0, 0, 255)
        mask_color = tuple(c // 2 for c in box_color) # half of box color
        detection_color = (0, 255, 0)
        

        for detection, tracking in zip(detections, trackings):
            # Get color for this track
            track_id = tracking.get("track_id", 0)
            if track_id != 0:
                continue

            # Use Kalman tracker position (filtered, handles segmentation failures)
            # instead of raw detection bbox to avoid drawing jumped centers
            if "position" in tracking:
                # Use filtered position from Kalman tracker
                kalman_cx = tracking["position"]["x"]
                kalman_cy = tracking["position"]["y"]
            else:
                # Fallback to detection center if measurement not available
                x, y, w, h = detection.bbox
                kalman_cx = x + w / 2
                kalman_cy = y + h / 2
            
            # Get bbox size from detection (for drawing purposes)
            x, y, w, h = detection.bbox


            # Draw oriented bounding box from mask if available
            if hasattr(detection, "oriented_box_info") and detection.oriented_box_info is not None:
                try:
                    
                    center = detection.oriented_box_info["center"]
                    cv2.circle(vis_frame, (int(center[0]), int(center[1])), 9, detection_color, -1)
                    # Draw polygon mask if available
                    if getattr(detection, "masks", None) is not None:
                        poly = detection.masks
                        poly = np.asarray(poly, dtype=np.float32)
                        if poly.ndim == 2 and poly.shape[1] == 2 and poly.shape[0] >= 3:
                            pts = poly.reshape((-1, 1, 2)).astype(np.int32)
                            cv2.polylines(vis_frame, [pts], True, mask_color, 2)
                    
                    # Draw oriented bounding box from extracted info
                    box_info = detection.oriented_box_info
                    box_points = box_info["box_points"]
                    box_i32 = box_points.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(vis_frame, [box_i32], True, box_color, 2)
                    
                    # Save angle from oriented box (degrees)
                    self.latest_rect_angles[track_id] = float(box_info["angle"])
                except Exception as e:
                    print(f"⚠ Failed to draw oriented box: {e}")
            elif getattr(detection, "masks", None) is not None:
                # Fallback: draw polygon if no oriented box info
                try:
                    poly = detection.masks
                    poly = np.asarray(poly, dtype=np.float32)
                    if poly.ndim == 2 and poly.shape[1] == 2 and poly.shape[0] >= 3:
                        pts = poly.reshape((-1, 1, 2)).astype(np.int32)
                        cv2.polylines(vis_frame, [pts], True, mask_color, 2)
                except Exception:
                    pass


            # Use Kalman tracker center (filtered, handles segmentation failures)
            center = (int(kalman_cx), int(kalman_cy))


            # Draw trajectory from tracker (trajectory is managed by KalmanTracker)
            trajectory = tracking.get("trajectory", [])
            if len(trajectory) >= 2:
                # Limit trajectory to recent points for performance (max 500 points)
                max_trajectory_points = 500
                if len(trajectory) > max_trajectory_points:
                    trajectory = trajectory[-max_trajectory_points:]
                # Convert trajectory points to integer tuples for drawing
                traj_pts = [(int(x), int(y)) for x, y in trajectory]
                for i in range(1, len(traj_pts)):
                    cv2.line(vis_frame, traj_pts[i - 1], traj_pts[i], box_color, 2)
            # 중심점도 표시
            cv2.circle(vis_frame, center, 5, box_color, -1)


        # Bottom overlay: x, y, rotation only
        height, width = vis_frame.shape[:2]
        base_y = height - 20
        font_scale = max(1.0, min(1.6, width / 900))
        thickness = max(2, int(font_scale * 2.2))

        lines = []
        for detection, tracking in zip(detections, trackings):
            track_id = tracking.get("track_id", 0)
            if track_id != 0:
                continue
            
            # Use Kalman tracker position (filtered, handles segmentation failures)
            if "position" in tracking:
                cx = int(tracking["position"]["x"])
                cy = int(tracking["position"]["y"])
            else:
                # Fallback to detection center if measurement not available
                x, y, w, h = detection.bbox
                cx = int(x + w / 2)
                cy = int(y + h / 2)
            # Prefer minAreaRect angle if available
            if track_id in self.latest_rect_angles:
                theta_deg = float(self.latest_rect_angles[track_id])
            else:
                angle = tracking.get("orientation", 0)
                theta_deg = float(self._extract_angle_deg(angle))
            
            # Get mm coordinates if available
            position = tracking.get("position", {})
            x_mm = position.get("x_mm")
            y_mm = position.get("y_mm")
            
            # Build info string
            if x_mm is not None and y_mm is not None:
                lines.append(f"ID {track_id}  px({cx},{cy}) mm({x_mm:.1f},{y_mm:.1f}) yaw={theta_deg:.1f}")
            else:
                lines.append(f"ID {track_id}  px({cx},{cy}) yaw={theta_deg:.1f}")

        overlay_text = " | ".join(lines) if lines else ""
        if overlay_text:
            cv2.putText(
                vis_frame,
                overlay_text,
                (20, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return vis_frame


    # def _draw_measurements(
    #     self,
    #     frame: np.ndarray,
    #     center: Tuple[int, int],
    #     measurement: Dict,
    #     color: Tuple[int, int, int],
    # ):
    #     """Draw measurement information."""
    #     # 이미지 크기에 따라 텍스트 크기 동적 조정 (더 큰 텍스트)
    #     height, width = frame.shape[:2]
    #     font_scale = max(
    #         2.0, min(5.0, width / 200)
    #     )  # 200px 기준으로 스케일링 (더 큰 텍스트)
    #     thickness = max(2, int(font_scale * 3))

    #     y_offset = int(10 * font_scale)

    #     # Size information
    #     if "width" in measurement and "height" in measurement:
    #         size_text = (
    #             f"Size: {measurement['width']:.0f}x{measurement['height']:.0f}mm"
    #         )
    #         # cv2.putText(
    #         #     frame,
    #         #     size_text,
    #         #     (center[0] - int(40 * font_scale), center[1] + y_offset),
    #         #     cv2.FONT_HERSHEY_SIMPLEX,
    #         #     font_scale,
    #         #     color,
    #         #     thickness,
    #         # )
    #         y_offset += int(15 * font_scale)

    #     # Speed information
    #     speed_value = None
    #     if "speed" in measurement:
    #         speed_value = measurement["speed"]
    #     elif (
    #         "velocity" in measurement
    #         and "linear_speed_mm_per_sec" in measurement["velocity"]
    #     ):
    #         speed_value = measurement["velocity"]["linear_speed_mm_per_sec"]

    #     if speed_value is not None:
    #         speed_text = f"Speed: {speed_value:.1f}mm/s"
    #         # Debug: print speed value
    #         print(f"Debug - Speed value: {speed_value}")
    #         # Use smaller font scale for speed text
    #         speed_font_scale = font_scale * 0.7
    #         speed_thickness = max(1, int(speed_font_scale * 2))
    #         # cv2.putText(
    #         #     frame,
    #         #     speed_text,
    #         #     (center[0] - int(40 * speed_font_scale), center[1] + y_offset),
    #         #     cv2.FONT_HERSHEY_SIMPLEX,
    #         #     speed_font_scale,
    #         #     color,
    #         #     speed_thickness,
    #         # )
    #         y_offset += int(15 * speed_font_scale)
    #     else:
    #         # Debug: print measurement structure
    #         print(
    #             f"Debug - No speed found in measurement keys: {list(measurement.keys())}"
    #         )
    #         if "velocity" in measurement:
    #             print(f"Debug - Velocity keys: {list(measurement['velocity'].keys())}")

    #     # Quality indicator
    #     if "rectangularity" in measurement:
    #         quality = measurement["rectangularity"]
    #         quality_color = (0, 255, 0) if quality > 0.95 else (0, 165, 255)
    #         quality_text = f"Q: {quality:.2f}"
    #         # cv2.putText(
    #         #     frame,
    #         #     quality_text,
    #         #     (center[0] - int(40 * font_scale), center[1] + y_offset),
    #         #     cv2.FONT_HERSHEY_SIMPLEX,
    #         #     font_scale,
    #         #     quality_color,
    #         #     thickness,
    #         # )

    # def _draw_speed_vector(
    #     self,
    #     frame: np.ndarray,
    #     center: Tuple[int, int],
    #     measurement: Dict,
    #     color: Tuple[int, int, int],
    # ):
    #     """Draw speed vector arrow."""
    #     # Get speed and direction information
    #     speed_value = None
    #     direction = None

    #     if "speed" in measurement and "direction" in measurement:
    #         speed_value = measurement["speed"]
    #         direction = measurement["direction"]
    #     elif "velocity" in measurement:
    #         velocity = measurement["velocity"]
    #         if "linear_speed_mm_per_sec" in velocity:
    #             speed_value = velocity["linear_speed_mm_per_sec"]
    #         # For direction, we can use orientation from Kalman tracker
    #         if (
    #             "orientation" in measurement
    #             and "theta_deg" in measurement["orientation"]
    #         ):
    #             direction = measurement["orientation"]["theta_deg"]
    #             print(f"Debug - Direction: {direction}")

    #     if speed_value is None or direction is None:
    #         return

    #     # Calculate arrow end point
    #     speed_scale = min(speed_value / 10, 50)  # Scale for visualization
    #     angle_rad = np.radians(direction)

    #     end_x = int(center[0] + speed_scale * np.cos(angle_rad))
    #     end_y = int(center[1] + speed_scale * np.sin(angle_rad))

    #     # Draw arrow
    #     cv2.arrowedLine(frame, center, (end_x, end_y), color, 2, tipLength=0.3)

    # def _draw_statistics(self, frame: np.ndarray, measurements: List[Dict]):
    #     """Draw statistics panel."""
    #     # Create semi-transparent overlay for stats
    #     overlay = frame.copy()
    #     height, width = frame.shape[:2]

    #     # Statistics box
    #     cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
    #     frame_with_overlay = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    #     # Calculate statistics
    #     num_agvs = len(measurements)

    #     # Get speed values from different sources
    #     speeds = []
    #     for m in measurements:
    #         if "speed" in m:
    #             speeds.append(m["speed"])
    #         elif "velocity" in m and "linear_speed_mm_per_sec" in m["velocity"]:
    #             speeds.append(m["velocity"]["linear_speed_mm_per_sec"])

    #     avg_speed = np.mean(speeds) if speeds else 0
    #     total_area = sum(m.get("area", 0) for m in measurements)

    #     # Draw text
    #     stats = [
    #         f"AGVs Detected: {num_agvs}",
    #         f"Avg Speed: {avg_speed:.1f} mm/s",
    #         f"Total Area: {total_area:.0f} mm²",
    #         f"FPS: {cv2.getTickFrequency() / cv2.getTickCount():.1f}",
    #     ]

    #     y_pos = 30
    #     for stat in stats:
    #         # cv2.putText(
    #         #     frame_with_overlay,
    #         #     stat,
    #         #     (20, y_pos),
    #         #     cv2.FONT_HERSHEY_SIMPLEX,
    #         #     1.5,
    #         #     (255, 255, 255),
    #         #     1,
    #         # )
    #         y_pos += 25

    #     return frame_with_overlay

    # def draw_trajectory(
    #     self,
    #     frame: np.ndarray,
    #     trajectory: List[Tuple[float, float]],
    #     color: Tuple[int, int, int] = (0, 255, 0),
    # ) -> np.ndarray:
    #     """
    #     Draw object trajectory on frame.

    #     Args:
    #         frame: Input image
    #         trajectory: List of (x, y) positions in world coordinates
    #         color: Color for trajectory

    #     Returns:
    #         Frame with trajectory
    #     """
    #     if len(trajectory) < 2 or self.homography is None:
    #         return frame

    #     # Convert world coordinates to image coordinates
    #     trajectory_array = np.array(trajectory)
    #     img_points = self._world_to_image(trajectory_array)

    #     # Draw trajectory
    #     pts = img_points.astype(np.int32)
    #     for i in range(1, len(pts)):
    #         cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), color, 2)

    #     # Mark start and end
    #     cv2.circle(frame, tuple(pts[0]), 5, (0, 255, 0), -1)  # Start (green)
    #     cv2.circle(frame, tuple(pts[-1]), 5, (0, 0, 255), -1)  # End (red)

    #     return frame

    def _world_to_image(self, world_points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to image coordinates."""
        if self.homography is None:
            return world_points

        # Inverse homography
        H_inv = np.linalg.inv(self.homography)

        # Convert to homogeneous
        pts_homo = np.column_stack([world_points, np.ones(len(world_points))])

        # Transform
        img_pts_homo = (H_inv @ pts_homo.T).T

        # Normalize
        img_pts = img_pts_homo[:, :2] / img_pts_homo[:, 2:3]

        return img_pts

    def reset(self):
        
        self.latest_rect_angles.clear()