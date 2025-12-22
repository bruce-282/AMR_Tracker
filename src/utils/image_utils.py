"""Image and visualization utilities for AMR Tracking System."""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from src.core.detection import Detection

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def _find_edge_offset_in_roi(
    gray: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    search_range_px: int = 5
) -> float:
    """
    ROI를 crop하고 수평으로 회전시켜 edge가 가장 강한 위치(offset)를 찾음.
    
    Args:
        gray: Grayscale image
        p1, p2: Side의 두 끝점
        search_range_px: 탐색 범위 (±px)
    
    Returns:
        best_offset: 원래 side 위치 기준 최적 offset (양수=바깥쪽)
    """
    # Side 벡터 및 수직 벡터 계산
    side_vec = p2 - p1
    side_length = np.linalg.norm(side_vec)
    if side_length < 1:
        return 0.0
    
    side_dir = side_vec / side_length
    perp_dir = np.array([-side_dir[1], side_dir[0]])  # 90도 회전 (수직 방향)
    
    # Side 중심점
    side_center = (p1 + p2) / 2
    
    # ROI 크기: side 길이 x (search_range * 2)
    roi_width = int(side_length)
    roi_height = search_range_px * 2
    
    if roi_width < 3 or roi_height < 3:
        return 0.0
    
    # ROI의 4개 코너 계산 (side 중심 기준)
    half_w = side_length / 2
    half_h = search_range_px
    
    # ROI 코너 (side 방향으로 ±half_w, 수직 방향으로 ±half_h)
    roi_corners = np.array([
        side_center - half_w * side_dir - half_h * perp_dir,
        side_center + half_w * side_dir - half_h * perp_dir,
        side_center + half_w * side_dir + half_h * perp_dir,
        side_center - half_w * side_dir + half_h * perp_dir,
    ], dtype=np.float32)
    
    # Destination points (수평으로 정렬된 직사각형)
    dst_corners = np.array([
        [0, 0],
        [roi_width - 1, 0],
        [roi_width - 1, roi_height - 1],
        [0, roi_height - 1],
    ], dtype=np.float32)
    
    # Perspective transform으로 ROI 추출 (회전된 영역을 수평으로)
    M = cv2.getPerspectiveTransform(roi_corners, dst_corners)
    roi = cv2.warpPerspective(gray, M, (roi_width, roi_height))
    
    # Sobel Y (수직 방향 edge) - ROI가 수평이므로 수직 edge가 side에 해당
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs = np.abs(sobel_y)
    
    # 각 행(y)별로 edge 강도 합산 → 가장 강한 행 찾기
    row_sums = np.sum(sobel_abs, axis=1)
    
    if len(row_sums) == 0:
        return 0.0
    
    best_row = np.argmax(row_sums)
    
    # best_row를 offset으로 변환 (ROI 중심이 offset=0)
    best_offset = best_row - search_range_px
    
    return float(best_offset)


def _line_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    """
    두 직선의 교점 계산.
    직선1: p1 + t * d1
    직선2: p2 + s * d2
    """
    # 2x2 행렬로 풀기: [d1, -d2] * [t, s]^T = p2 - p1
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = p2 - p1
    
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-10:
        return None  # 평행선
    
    t = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
    return p1 + t * d1


def refine_box_with_edges(
    frame: np.ndarray,
    oriented_box_info: Dict,
    search_range_px: int = 5,
    debug_image_path: Optional[str] = None
) -> Optional[Dict]:
    """
    각 side를 ROI로 crop하고 edge가 가장 강한 위치로 side를 갱신하여 박스를 정밀화.
    
    Args:
        frame: Input image (grayscale or BGR)
        oriented_box_info: Dictionary with rect, center, width, height, angle, box_points
        search_range_px: 탐색 범위 (±px)
        debug_image_path: Optional path to save debug image
    
    Returns:
        Refined oriented_box_info dictionary, or None if refinement failed
    """
    try:
        # Grayscale 변환
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        rect = oriented_box_info.get("rect")
        if rect is None:
            return None
        
        box_points = cv2.boxPoints(rect).astype(np.float32)
        
        # 4개 sides 정의
        sides = [
            (box_points[0], box_points[1]),  # Side 0
            (box_points[1], box_points[2]),  # Side 1
            (box_points[2], box_points[3]),  # Side 2
            (box_points[3], box_points[0]),  # Side 3
        ]
        
        # 각 side에 대해 best offset 찾고 refined side 저장
        refined_sides = []  # [(point_on_line, direction_vector), ...]
        
        for p1, p2 in sides:
            p1 = np.array(p1, dtype=np.float64)
            p2 = np.array(p2, dtype=np.float64)
            
            # Edge 기반 최적 offset 찾기
            offset = _find_edge_offset_in_roi(gray, p1, p2, search_range_px)
            
            # Side 방향 및 수직 방향
            side_vec = p2 - p1
            side_length = np.linalg.norm(side_vec)
            if side_length < 1:
                refined_sides.append((p1, np.array([1.0, 0.0])))
                continue
            
            side_dir = side_vec / side_length
            perp_dir = np.array([-side_dir[1], side_dir[0]])
            
            # Offset 적용하여 refined side 위치 계산
            refined_p1 = p1 + offset * perp_dir
            refined_sides.append((refined_p1, side_dir))
        
        # 인접한 side들의 교점 계산 → refined box points
        refined_box_points = []
        for i in range(4):
            line1 = refined_sides[i]
            line2 = refined_sides[(i + 1) % 4]
            
            intersection = _line_intersection(line1[0], line1[1], line2[0], line2[1])
            if intersection is None:
                # 교점 계산 실패 시 원래 점 사용
                refined_box_points.append(box_points[(i + 1) % 4])
            else:
                refined_box_points.append(intersection)
        
        refined_box_points = np.array(refined_box_points, dtype=np.float32)
        
        # 중심 계산
        refined_center = np.mean(refined_box_points, axis=0)
        
        # minAreaRect로 angle, width, height 재계산
        refined_rect = cv2.minAreaRect(refined_box_points)
        _, (refined_w, refined_h), refined_angle = refined_rect
        
        # Normalize angle
        if refined_h > refined_w:
            normalized_angle = refined_angle - 90
            long_axis = refined_h
            short_axis = refined_w
        else:
            normalized_angle = refined_angle
            long_axis = refined_w
            short_axis = refined_h
        
        while normalized_angle > 90:
            normalized_angle -= 180
        while normalized_angle < -90:
            normalized_angle += 180
        
        # Debug image
        if debug_image_path:
            try:
                if len(frame.shape) == 2:
                    debug_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    debug_frame = frame.copy()
                
                # Original box (blue)
                original_box_i32 = box_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(debug_frame, [original_box_i32], True, (255, 0, 0), 2)
                
                # Refined box (green)
                refined_box_i32 = refined_box_points.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(debug_frame, [refined_box_i32], True, (0, 255, 0), 2)
                
                # Centers
                original_center = oriented_box_info.get("center", tuple(np.mean(box_points, axis=0)))
                cv2.circle(debug_frame, (int(original_center[0]), int(original_center[1])), 5, (255, 0, 0), -1)
                cv2.circle(debug_frame, (int(refined_center[0]), int(refined_center[1])), 5, (0, 255, 0), -1)
                
                cv2.putText(debug_frame, "Original (Blue)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(debug_frame, "Refined (Green)", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                debug_path_obj = Path(debug_image_path)
                debug_path_obj.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(debug_image_path, debug_frame)
                logger.info(f"Refinement debug image saved: {debug_image_path}")
            except Exception as e:
                logger.error(f"Failed to save refinement debug image: {e}")
        
        # Create refined oriented_box_info
        refined_info = oriented_box_info.copy()
        refined_info["center"] = tuple(refined_center)
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
    camera_id: Optional[int] = None,
    enable_edge_refinement: bool = True,
    edge_search_range_px: int = 10
) -> Detection:
    """
    Transform detection coordinates with homography.
    
    Args:
        detection: Detection object
        homography: 3x3 homography matrix
        transformed_image_size: Optional image size (width, height) of transformed frame.
                                If None, will be estimated from transformed mask bounds.
        frame: Optional frame for edge refinement
        debug_base_path: Optional path for debug images
        camera_id: Optional camera ID for debug logging
        enable_edge_refinement: Whether to apply edge-based box refinement (default: True)
        edge_search_range_px: Search range in pixels for edge detection (default: 10)
    
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
    
    # Re-extract oriented_box_info from transformed mask or OBB box_points
    # This ensures accurate box_points, center, width, height, and angle after homography transformation
    new_oriented_box_info = None
    
    if new_masks is not None:
        # Case 1: Mask available - extract oriented_box_info from transformed mask
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
        
    elif detection.oriented_box_info is not None and "box_points" in detection.oriented_box_info:
        # Case 2: OBB model - transform box_points with homography and recalculate rect
        original_box_points = np.array(detection.oriented_box_info["box_points"], dtype=np.float32)
        
        # Transform box_points with homography
        transformed_points = cv2.perspectiveTransform(
            original_box_points.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        
        # Recalculate minAreaRect from transformed points
        rect = cv2.minAreaRect(transformed_points)
        center, (w, h), angle = rect
        
        # Normalize angle and determine long/short axis
        if h > w:
            normalized_angle = angle - 90
            long_axis = h
            short_axis = w
        else:
            normalized_angle = angle
            long_axis = w
            short_axis = h
        
        while normalized_angle > 90:
            normalized_angle -= 180
        while normalized_angle < -90:
            normalized_angle += 180
        
        new_oriented_box_info = {
            "center": tuple(center),
            "width": long_axis,
            "height": short_axis,
            "angle": normalized_angle,
            "angle_rad": np.deg2rad(normalized_angle),
            "box_points": transformed_points,
            "rect": rect
        }
    
    # Apply edge-based refinement if oriented_box_info and frame are available
    if new_oriented_box_info is not None:
        if enable_edge_refinement and "rect" in new_oriented_box_info and frame is not None:
            # Prepare debug image path if requested
            debug_image_path = None
            if debug_base_path is not None and camera_id is not None:
                debug_image_path = str(debug_base_path / f"cam_{camera_id}_refinement_debug.png")
                print(f"[DEBUG] Creating refinement debug image at {debug_image_path}")
            
            print(f"[DEBUG] Calling refine_box_with_edges for camera {camera_id} (search_range={edge_search_range_px}px)...")
            refined_oriented_box_info = refine_box_with_edges(
                frame, new_oriented_box_info, search_range_px=edge_search_range_px, debug_image_path=debug_image_path
            )
            if refined_oriented_box_info:
                new_oriented_box_info = refined_oriented_box_info
                print(f"[DEBUG] Camera {camera_id}: Edge refinement applied successfully")
            else:
                print(f"[DEBUG] Camera {camera_id}: Edge refinement returned None")
        elif not enable_edge_refinement:
            print(f"[DEBUG] Camera {camera_id}: Edge refinement disabled by config")
        
        new_detection.oriented_box_info = new_oriented_box_info
        # Recalculate bbox from oriented_box_info
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
        # Store transformed pixel coordinates (for internal use, not sent in response)
        transformed_point["x_pix"] = round(new_x_pix, 1)
        transformed_point["y_pix"] = round(new_y_pix, 1)
        # Calculate mm values from transformed pixel coordinates
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

