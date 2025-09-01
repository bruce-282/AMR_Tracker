"""Visualization module for displaying results."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches


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
        self.track_colors = {}  # Track ID to color mapping
        
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for tracking visualization."""
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), 
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        measurements: List[Dict]
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
        
        for detection, measurement in zip(detections, measurements):
            # Get color for this track
            track_id = measurement.get('track_id', 0)
            if track_id not in self.track_colors:
                self.track_colors[track_id] = self.colors[
                    track_id % len(self.colors)
                ]
            color = self.track_colors[track_id]
            
            # Draw bounding box
            bbox = detection.bbox.astype(np.int32)
            cv2.polylines(vis_frame, [bbox], True, color, 2)
            
            # Draw track ID
            center = tuple(np.mean(bbox, axis=0).astype(int))
            cv2.putText(
                vis_frame, f"ID: {track_id}",
                (center[0] - 20, center[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw measurements
            self._draw_measurements(vis_frame, center, measurement, color)
            
            # Draw speed vector
            if 'speed' in measurement and measurement['speed'] > 0:
                self._draw_speed_vector(vis_frame, center, measurement, color)
        
        # Draw statistics panel
        self._draw_statistics(vis_frame, measurements)
        
        return vis_frame
    
    def _draw_measurements(
        self, 
        frame: np.ndarray, 
        center: Tuple[int, int],
        measurement: Dict,
        color: Tuple[int, int, int]
    ):
        """Draw measurement information."""
        y_offset = 10
        
        # Size information
        if 'width' in measurement and 'height' in measurement:
            size_text = f"Size: {measurement['width']:.0f}x{measurement['height']:.0f}mm"
            cv2.putText(
                frame, size_text,
                (center[0] - 40, center[1] + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
            y_offset += 15
        
        # Speed information
        if 'speed' in measurement:
            speed_text = f"Speed: {measurement['speed']:.1f}mm/s"
            cv2.putText(
                frame, speed_text,
                (center[0] - 40, center[1] + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
            y_offset += 15
        
        # Quality indicator
        if 'rectangularity' in measurement:
            quality = measurement['rectangularity']
            quality_color = (0, 255, 0) if quality > 0.95 else (0, 165, 255)
            quality_text = f"Q: {quality:.2f}"
            cv2.putText(
                frame, quality_text,
                (center[0] - 40, center[1] + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1
            )
    
    def _draw_speed_vector(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        measurement: Dict,
        color: Tuple[int, int, int]
    ):
        """Draw speed vector arrow."""
        if 'direction' not in measurement:
            return
        
        # Calculate arrow end point
        speed_scale = min(measurement['speed'] / 10, 50)  # Scale for visualization
        angle_rad = np.radians(measurement['direction'])
        
        end_x = int(center[0] + speed_scale * np.cos(angle_rad))
        end_y = int(center[1] + speed_scale * np.sin(angle_rad))
        
        # Draw arrow
        cv2.arrowedLine(
            frame, center, (end_x, end_y),
            color, 2, tipLength=0.3
        )
    
    def _draw_statistics(self, frame: np.ndarray, measurements: List[Dict]):
        """Draw statistics panel."""
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Statistics box
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        frame_with_overlay = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Calculate statistics
        num_agvs = len(measurements)
        avg_speed = np.mean([m.get('speed', 0) for m in measurements]) if measurements else 0
        total_area = sum(m.get('area', 0) for m in measurements)
        
        # Draw text
        stats = [
            f"AGVs Detected: {num_agvs}",
            f"Avg Speed: {avg_speed:.1f} mm/s",
            f"Total Area: {total_area:.0f} mm²",
            f"FPS: {cv2.getTickFrequency() / cv2.getTickCount():.1f}"
        ]
        
        y_pos = 30
        for stat in stats:
            cv2.putText(
                frame_with_overlay, stat,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_pos += 25
        
        return frame_with_overlay
    
    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[float, float]],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw object trajectory on frame.
        
        Args:
            frame: Input image
            trajectory: List of (x, y) positions in world coordinates
            color: Color for trajectory
            
        Returns:
            Frame with trajectory
        """
        if len(trajectory) < 2 or self.homography is None:
            return frame
        
        # Convert world coordinates to image coordinates
        trajectory_array = np.array(trajectory)
        img_points = self._world_to_image(trajectory_array)
        
        # Draw trajectory
        pts = img_points.astype(np.int32)
        for i in range(1, len(pts)):
            cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), color, 2)
        
        # Mark start and end
        cv2.circle(frame, tuple(pts[0]), 5, (0, 255, 0), -1)  # Start (green)
        cv2.circle(frame, tuple(pts[-1]), 5, (0, 0, 255), -1)  # End (red)
        
        return frame
    
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
    
    def create_bird_eye_view(
        self,
        measurements: List[Dict],
        world_size: Tuple[float, float] = (5000, 5000),
        scale: float = 0.2
    ) -> np.ndarray:
        """
        Create bird's eye view visualization.
        
        Args:
            measurements: List of measurements with world coordinates
            world_size: Size of world to visualize in mm
            scale: Scale factor for visualization
            
        Returns:
            Bird's eye view image
        """
        # Create blank canvas
        width = int(world_size[0] * scale)
        height = int(world_size[1] * scale)
        bird_eye = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw grid
        grid_spacing = int(500 * scale)  # 500mm grid
        for x in range(0, width, grid_spacing):
            cv2.line(bird_eye, (x, 0), (x, height), (200, 200, 200), 1)
        for y in range(0, height, grid_spacing):
            cv2.line(bird_eye, (0, y), (width, y), (200, 200, 200), 1)
        
        # Draw AGVs
        for measurement in measurements:
            if 'center_world' not in measurement:
                continue
            
            # Convert world coordinates to canvas coordinates
            cx, cy = measurement['center_world']
            cx_canvas = int(cx * scale + width / 2)
            cy_canvas = int(cy * scale + height / 2)
            
            # Get dimensions
            width_agv = int(measurement.get('width', 100) * scale)
            height_agv = int(measurement.get('height', 100) * scale)
            
            # Get color for track
            track_id = measurement.get('track_id', 0)
            color = self.colors[track_id % len(self.colors)]
            
            # Draw AGV rectangle
            angle = measurement.get('orientation', 0)
            rect = ((cx_canvas, cy_canvas), (width_agv, height_agv), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.fillPoly(bird_eye, [box], color)
            
            # Draw ID
            cv2.putText(
                bird_eye, str(track_id),
                (cx_canvas - 10, cy_canvas),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        return bird_eye
