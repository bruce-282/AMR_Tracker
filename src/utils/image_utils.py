"""Image and visualization utilities for AMR Tracking System."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from src.core.detection import Detection


def draw_trajectory_on_frame(
    frame: np.ndarray, 
    trajectory_data: List[Dict],
    line_thickness: int = 2,
    start_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    end_color: Tuple[int, int, int] = (0, 0, 255),    # Red
    show_labels: bool = True,
    show_info: bool = True
) -> np.ndarray:
    """
    Draw trajectory on frame using trajectory data.
    
    Args:
        frame: Input frame
        trajectory_data: List of trajectory points with x_pix, y_pix
        line_thickness: Line thickness for trajectory
        start_color: Color for start point (BGR)
        end_color: Color for end point (BGR)
        show_labels: Whether to show Start/End labels
        show_info: Whether to show trajectory info text
    
    Returns:
        Frame with trajectory drawn
    """
    vis_frame = frame.copy()
    
    if len(trajectory_data) < 2:
        return vis_frame
    
    # Extract pixel coordinates
    points = []
    for point in trajectory_data:
        x_pix = point.get("x_pix")
        y_pix = point.get("y_pix")
        if x_pix is not None and y_pix is not None:
            points.append((int(x_pix), int(y_pix)))
    
    if len(points) < 2:
        return vis_frame
    
    # Draw trajectory line with gradient color
    for i in range(1, len(points)):
        # Color gradient from blue (old) to red (new)
        ratio = i / len(points)
        color = (
            int(255 * (1 - ratio)),  # B: blue at start
            0,  # G
            int(255 * ratio)  # R: red at end
        )
        cv2.line(vis_frame, points[i-1], points[i], color, line_thickness)
    
    # Draw start point
    cv2.circle(vis_frame, points[0], 8, start_color, -1)
    if show_labels:
        cv2.putText(vis_frame, "Start", (points[0][0] + 10, points[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, start_color, 2)
    
    # Draw end point
    cv2.circle(vis_frame, points[-1], 8, end_color, -1)
    if show_labels:
        cv2.putText(vis_frame, "End", (points[-1][0] + 10, points[-1][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, end_color, 2)
    
    # Draw trajectory info
    if show_info:
        cv2.putText(vis_frame, f"Trajectory: {len(points)} points", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return vis_frame


def transform_point_with_homography(
    point: Tuple[float, float], 
    homography: np.ndarray
) -> Tuple[float, float]:
    """
    Transform a single point with homography.
    
    Args:
        point: (x, y) point
        homography: 3x3 homography matrix
    
    Returns:
        Transformed (x, y) point
    """
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed_pt = cv2.perspectiveTransform(pt, homography)
    return (float(transformed_pt[0, 0, 0]), float(transformed_pt[0, 0, 1]))


def transform_bbox_with_homography(
    bbox: Tuple[float, float, float, float], 
    homography: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box with homography.
    
    Args:
        bbox: (x, y, w, h) bounding box
        homography: 3x3 homography matrix
    
    Returns:
        Transformed (x, y, w, h) bounding box
    """
    x, y, w, h = bbox
    # Get corner points
    corners = np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]]
    ], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners, homography)
    
    # Get bounding box of transformed corners
    xs = transformed_corners[:, 0, 0]
    ys = transformed_corners[:, 0, 1]
    new_x = float(min(xs))
    new_y = float(min(ys))
    new_w = float(max(xs) - new_x)
    new_h = float(max(ys) - new_y)
    
    return (new_x, new_y, new_w, new_h)


def transform_polygon_with_homography(
    polygon: List[List[float]], 
    homography: np.ndarray
) -> List[List[float]]:
    """
    Transform polygon points with homography.
    
    Args:
        polygon: List of [x, y] points
        homography: 3x3 homography matrix
    
    Returns:
        Transformed polygon points
    """
    try:
        poly_arr = np.array(polygon, dtype=np.float32)
        if poly_arr.ndim == 2 and poly_arr.shape[1] == 2:
            poly_reshaped = poly_arr.reshape(-1, 1, 2)
            transformed_poly = cv2.perspectiveTransform(poly_reshaped, homography)
            return transformed_poly.reshape(-1, 2).tolist()
    except Exception:
        pass
    return polygon


def transform_detection_with_homography(
    detection: Detection, 
    homography: np.ndarray
) -> Detection:
    """
    Transform detection coordinates with homography.
    
    Args:
        detection: Detection object
        homography: 3x3 homography matrix
    
    Returns:
        New Detection object with transformed coordinates
    """
    # Transform bbox
    new_bbox = transform_bbox_with_homography(detection.bbox, homography)
    
    # Transform masks (polygon points) if available
    new_masks = None
    if detection.masks is not None:
        new_masks = transform_polygon_with_homography(detection.masks, homography)
    
    # Transform oriented_box_info if available
    new_oriented_box_info = None
    if hasattr(detection, 'oriented_box_info') and detection.oriented_box_info is not None:
        try:
            obi = detection.oriented_box_info
            new_oriented_box_info = obi.copy()
            
            # Transform center directly (same method as tracking position)
            if "center" in obi and obi["center"] is not None:
                center = obi["center"]
                new_center = transform_point_with_homography((center[0], center[1]), homography)
                new_oriented_box_info["center"] = new_center
            
            # Transform box_points if available (for visualization)
            if "box_points" in obi and obi["box_points"] is not None:
                box_pts = np.array(obi["box_points"], dtype=np.float32).reshape(-1, 1, 2)
                transformed_box = cv2.perspectiveTransform(box_pts, homography)
                new_oriented_box_info["box_points"] = transformed_box.reshape(-1, 2)
        except Exception:
            pass
    
    # Create new detection with transformed coordinates
    new_detection = Detection(
        bbox=new_bbox,
        confidence=detection.confidence,
        class_id=detection.class_id,
        class_name=detection.class_name,
        frame_number=detection.frame_number,
        timestamp=detection.timestamp,
        masks=new_masks
    )
    
    if new_oriented_box_info:
        new_detection.oriented_box_info = new_oriented_box_info
    
    return new_detection


def transform_tracking_result_with_homography(
    tracking_result: Dict, 
    homography: np.ndarray
) -> Dict:
    """
    Transform tracking result coordinates with homography.
    
    Args:
        tracking_result: Tracking result dictionary
        homography: 3x3 homography matrix
    
    Returns:
        New tracking result with transformed coordinates
    """
    result = tracking_result.copy()
    
    # Transform position
    if "position" in result:
        pos = result["position"]
        x_pix, y_pix = pos.get("x", 0), pos.get("y", 0)
        new_x, new_y = transform_point_with_homography((x_pix, y_pix), homography)
        result["position"] = pos.copy()
        result["position"]["x"] = new_x
        result["position"]["y"] = new_y
    
    # Transform trajectory
    if "trajectory" in result and len(result["trajectory"]) > 0:
        traj = result["trajectory"]
        traj_arr = np.array(traj, dtype=np.float32).reshape(-1, 1, 2)
        transformed_traj = cv2.perspectiveTransform(traj_arr, homography)
        result["trajectory"] = transformed_traj.reshape(-1, 2).tolist()
    
    # Transform bbox if present
    if "bbox" in result and result["bbox"] is not None:
        result["bbox"] = transform_bbox_with_homography(result["bbox"], homography)
    
    return result


def transform_trajectory_data_with_homography(
    trajectory_data: List[Dict],
    homography: np.ndarray,
    pixel_size: float = 1.0
) -> List[Dict]:
    """
    Transform trajectory data points with homography.
    
    Args:
        trajectory_data: List of trajectory points with x_pix, y_pix
        homography: 3x3 homography matrix
        pixel_size: Pixel size for recalculating mm values
    
    Returns:
        Transformed trajectory data
    """
    transformed_trajectory = []
    
    for point in trajectory_data:
        x_pix = point.get("x_pix", 0)
        y_pix = point.get("y_pix", 0)
        
        # Apply homography to point
        new_x_pix, new_y_pix = transform_point_with_homography((x_pix, y_pix), homography)
        
        transformed_point = point.copy()
        transformed_point["x_pix"] = round(new_x_pix, 1)
        transformed_point["y_pix"] = round(new_y_pix, 1)
        transformed_point["x"] = round(new_x_pix * pixel_size, 3)
        transformed_point["y"] = round(new_y_pix * pixel_size, 3)
        transformed_trajectory.append(transformed_point)
    
    return transformed_trajectory


def warp_frame_with_homography(
    frame: np.ndarray,
    homography: np.ndarray,
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT
) -> np.ndarray:
    """
    Apply homography transformation to frame.
    
    Args:
        frame: Input frame
        homography: 3x3 homography matrix
        flags: Interpolation flags
        border_mode: Border mode
    
    Returns:
        Warped frame
    """
    h, w = frame.shape[:2]
    return cv2.warpPerspective(frame, homography, (w, h), flags=flags, borderMode=border_mode)


def save_image(
    image: np.ndarray,
    path: str,
    create_dirs: bool = True
) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        path: Output file path
        create_dirs: Whether to create parent directories
    
    Returns:
        True if saved successfully, False otherwise
    """
    from pathlib import Path
    
    try:
        if create_dirs:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(path), image)
    except Exception:
        return False

