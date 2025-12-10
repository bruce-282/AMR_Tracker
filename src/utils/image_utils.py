"""Image and visualization utilities for AMR Tracking System."""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from src.core.detection import Detection

logger = logging.getLogger(__name__)


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


def refine_box_with_edges(
    frame: np.ndarray,
    oriented_box_info: Dict,
    search_range_px: int = 5,
    debug_image_path: Optional[str] = None
) -> Optional[Dict]:
    """
    Refine oriented box by searching for sharpest edges along each side.
    
    Args:
        frame: Input image (grayscale or BGR)
        oriented_box_info: Dictionary with rect, center, width, height, angle, box_points
        search_range_px: Pixel range to search along each edge (±search_range_px)
        debug_image_path: Optional path to save debug image showing before/after refinement
    
    Returns:
        Refined oriented_box_info dictionary, or None if refinement failed
    """
    try:
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Get Canny edges for edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Get original rect
        rect = oriented_box_info.get("rect")
        if rect is None:
            return None
        
        center, (w, h), angle = rect
        angle_rad = np.deg2rad(angle)
        
        # Get box points for the 4 sides
        box_points = cv2.boxPoints(rect).astype(np.float32)
        
        # Define the 4 sides (each side is a line segment)
        sides = [
            (box_points[0], box_points[1]),  # Side 0
            (box_points[1], box_points[2]),  # Side 1
            (box_points[2], box_points[3]),  # Side 2
            (box_points[3], box_points[0]),  # Side 3
        ]
        
        refined_points = []
        for side_idx, (p1, p2) in enumerate(sides):
            # Calculate perpendicular direction (normal to the edge, pointing outward)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                refined_points.append(p1)
                continue
            
            # Normalize direction vector
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Perpendicular direction (rotate 90 degrees)
            perp_x = -dy_norm
            perp_y = dx_norm
            
            # Sample points along the edge
            num_samples = max(int(length), 10)
            best_offset = 0.0
            max_edge_strength = 0.0
            
            # Search along perpendicular direction
            for offset in np.arange(-search_range_px, search_range_px + 0.5, 0.5):
                edge_strength = 0.0
                valid_samples = 0
                
                for i in range(num_samples):
                    t = i / max(num_samples - 1, 1)
                    # Point along the edge
                    px = p1[0] + t * dx
                    py = p1[1] + t * dy
                    
                    # Move perpendicular to the edge
                    search_x = int(px + offset * perp_x)
                    search_y = int(py + offset * perp_y)
                    
                    # Check bounds
                    if (0 <= search_y < edges.shape[0] and 
                        0 <= search_x < edges.shape[1]):
                        edge_strength += float(edges[search_y, search_x])
                        valid_samples += 1
                
                if valid_samples > 0:
                    avg_strength = edge_strength / valid_samples
                    if avg_strength > max_edge_strength:
                        max_edge_strength = avg_strength
                        best_offset = offset
            
            # Apply best offset to both endpoints
            refined_p1 = (
                p1[0] + best_offset * perp_x,
                p1[1] + best_offset * perp_y
            )
            refined_p2 = (
                p2[0] + best_offset * perp_x,
                p2[1] + best_offset * perp_y
            )
            
            # Store refined points (each side contributes its second point)
            if side_idx == 0:
                refined_points.append(refined_p1)  # First point of first side
            refined_points.append(refined_p2)  # Second point of each side
        
        # Recalculate minAreaRect from refined points
        refined_points_arr = np.array(refined_points, dtype=np.float32)
        refined_rect = cv2.minAreaRect(refined_points_arr)
        refined_center, (refined_w, refined_h), refined_angle = refined_rect
        
        # Normalize angle (same logic as extract_box_from_mask)
        if refined_h > refined_w:
            normalized_angle = refined_angle - 90
            long_axis = refined_h
            short_axis = refined_w
        else:
            normalized_angle = refined_angle
            long_axis = refined_w
            short_axis = refined_h
        
        # Normalize angle to -90 ~ 90 range
        while normalized_angle > 90:
            normalized_angle -= 180
        while normalized_angle < -90:
            normalized_angle += 180
        
        # Get refined box points
        refined_box_points = cv2.boxPoints(refined_rect).astype(np.float32)
        
        # Create debug image if requested
        if debug_image_path:
            try:
                logger.debug(f"Attempting to save refinement debug image to {debug_image_path}")
                # Convert to BGR if grayscale
                if len(frame.shape) == 2:
                    debug_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    debug_frame = frame.copy()
                
                # Draw original box (blue)
                original_box_points = oriented_box_info.get("box_points", box_points)
                original_box_i32 = original_box_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(debug_frame, [original_box_i32], True, (255, 0, 0), 2)  # Blue
                
                # Draw refined box (green)
                refined_box_i32 = refined_box_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(debug_frame, [refined_box_i32], True, (0, 255, 0), 2)  # Green
                
                # Draw original center (blue circle)
                original_center = oriented_box_info.get("center", center)
                cv2.circle(debug_frame, (int(original_center[0]), int(original_center[1])), 5, (255, 0, 0), -1)
                
                # Draw refined center (green circle)
                cv2.circle(debug_frame, (int(refined_center[0]), int(refined_center[1])), 5, (0, 255, 0), -1)
                
                # Add text labels
                cv2.putText(debug_frame, "Original (Blue)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(debug_frame, "Refined (Green)", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save debug image
                debug_path_obj = Path(debug_image_path)
                debug_path_obj.parent.mkdir(parents=True, exist_ok=True)
                success = cv2.imwrite(debug_image_path, debug_frame)
                if success:
                    logger.info(f"Refinement debug image saved: {debug_image_path}")
                else:
                    logger.warning(f"Failed to save refinement debug image (cv2.imwrite returned False): {debug_image_path}")
            except Exception as e:
                logger.error(f"Failed to save refinement debug image: {e}", exc_info=True)
        
        # Create refined oriented_box_info
        refined_info = oriented_box_info.copy()
        refined_info["center"] = refined_center
        refined_info["width"] = long_axis
        refined_info["height"] = short_axis
        refined_info["angle"] = normalized_angle
        refined_info["angle_rad"] = np.deg2rad(normalized_angle)
        refined_info["box_points"] = refined_box_points
        refined_info["rect"] = refined_rect
        
        return refined_info
        
    except Exception as e:
        print(f"⚠ Failed to refine box with edges: {e}")
        return None


def transform_detection_with_homography(
    detection: Detection, 
    homography: np.ndarray,
    transformed_image_size: Optional[Tuple[int, int]] = None,
    frame: Optional[np.ndarray] = None,
    debug_base_path: Optional[Path] = None,
    camera_id: Optional[int] = None
) -> Detection:
    """
    Transform detection coordinates with homography.
    
    Args:
        detection: Detection object
        homography: 3x3 homography matrix
        transformed_image_size: Optional image size (width, height) of transformed frame.
                                If None, will be estimated from transformed mask bounds.
    
    Returns:
        New Detection object with transformed coordinates
    """
    # Transform bbox
    new_bbox = transform_bbox_with_homography(detection.bbox, homography)
    
    # Transform masks (polygon points) if available
    new_masks = None
    if detection.masks is not None:
        new_masks = transform_polygon_with_homography(detection.masks, homography)
    
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
    
    # Re-extract oriented_box_info from transformed mask if available
    # This ensures accurate box_points, center, width, height, and angle after homography transformation
    if new_masks is not None:
        # Determine image size for mask extraction
        if transformed_image_size is None:
            # Estimate image size from transformed mask bounds
            poly_arr = np.asarray(new_masks, dtype=np.float32)
            if poly_arr.ndim == 2 and poly_arr.shape[1] == 2:
                max_x = int(np.max(poly_arr[:, 0])) + 100
                max_y = int(np.max(poly_arr[:, 1])) + 100
                # Use common image sizes as fallback, but ensure it's large enough
                estimated_width = max(max_x, 1920)
                estimated_height = max(max_y, 1080)
                image_size = (estimated_width, estimated_height)
            else:
                image_size = (1920, 1080)  # Default fallback
        else:
            image_size = transformed_image_size
        
        # Extract oriented_box_info from transformed mask
        new_oriented_box_info = Detection.extract_box_from_mask(new_masks, image_size)
        if new_oriented_box_info:
            # Apply edge-based refinement if rect is available
            if "rect" in new_oriented_box_info and frame is not None:
                # Prepare debug image path if requested
                debug_image_path = None
                if debug_base_path is not None and camera_id is not None:
                    debug_image_path = str(debug_base_path / f"cam_{camera_id}_refinement_debug.png")
                    logger.debug(f"Creating refinement debug image at {debug_image_path}")
                else:
                    logger.debug(f"Debug image not created: debug_base_path={debug_base_path}, camera_id={camera_id}")
                
                refined_oriented_box_info = refine_box_with_edges(
                    frame, new_oriented_box_info, search_range_px=10, debug_image_path=debug_image_path
                )
                if refined_oriented_box_info:
                    new_oriented_box_info = refined_oriented_box_info
            
            new_detection.oriented_box_info = new_oriented_box_info
            # Recalculate bbox from re-extracted oriented_box_info (same as Detection.__init__)
            center = new_oriented_box_info["center"]
            width = new_oriented_box_info["width"]
            height = new_oriented_box_info["height"]
            x = center[0] - width / 2
            y = center[1] - height / 2
            new_detection.bbox = [x, y, width, height]
    
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
    pixel_size: Union[float, Dict[str, float]] = 1.0
) -> List[Dict]:
    """
    Transform trajectory data points with homography.
    
    Args:
        trajectory_data: List of trajectory points with x_pix, y_pix
        homography: 3x3 homography matrix
        pixel_size: Pixel size for recalculating mm values - float or dict with 'x', 'y' keys
    
    Returns:
        Transformed trajectory data
    """
    # Extract pixel_size_x and pixel_size_y
    if isinstance(pixel_size, dict):
        pixel_size_x = pixel_size.get('x', 1.0)
        pixel_size_y = pixel_size.get('y', 1.0)
    else:
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
    
    transformed_trajectory = []
    
    for point in trajectory_data:
        x_pix = point.get("x_pix", 0)
        y_pix = point.get("y_pix", 0)
        
        # Apply homography to point
        new_x_pix, new_y_pix = transform_point_with_homography((x_pix, y_pix), homography)
        
        transformed_point = point.copy()
        transformed_point["x_pix"] = round(new_x_pix, 1)
        transformed_point["y_pix"] = round(new_y_pix, 1)
        # Use pixel_size_x and pixel_size_y separately
        transformed_point["x"] = round(new_x_pix * pixel_size_x, 3)
        transformed_point["y"] = round(new_y_pix * pixel_size_y, 3)
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

