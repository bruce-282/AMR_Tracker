import numpy as np
import cv2
import json
import argparse
import os
from typing import Tuple, List, Optional, Dict
from functools import wraps
from enum import Enum
from pathlib import Path


def requires_calibration(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.H is None:
            raise RuntimeError(f"{method.__name__}() 호출 전 calibrate 필수!")
        return method(self, *args, **kwargs)
    return wrapper


class PointType(Enum):
    CORNER = 1   # 모서리
    HOLE = 2     # 볼트 구멍


class PointSelector:
    def __init__(self, image: np.ndarray, search_radius: int = 15):
        self.image = image.copy()
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.display = image.copy()
        self.points = []
        self.circle_radii = []  # HOLE 타입 점의 원 반지름 저장
        self.current_pos = [image.shape[1]//2, image.shape[0]//2]
        self.zoom_level = 4
        self.point_type = PointType.CORNER
        self.search_radius = search_radius
        self.display_scale = 1.0  # 리사이즈 스케일 저장

    def _refine_corner(self, x: int, y: int, search_radius: int = 15) -> Tuple[float, float]:
        """모서리 서브픽셀 보정"""
        pt = np.array([[[x, y]]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(self.gray, pt, (search_radius, search_radius), (-1, -1), criteria)
        return float(refined[0, 0, 0]), float(refined[0, 0, 1])
    
    def _refine_hole(self, x: int, y: int, search_radius: int = 40) -> Tuple[float, float, Optional[float]]:
        """볼트 구멍 원 중심 검출"""
        h, w = self.image.shape[:2]
        
        x1, x2 = max(0, x-search_radius), min(w, x+search_radius)
        y1, y2 = max(0, y-search_radius), min(h, y+search_radius)
        crop = self.gray[y1:y2, x1:x2]
        crop_blur = cv2.GaussianBlur(crop, (5, 5), 0)
        
        circles = cv2.HoughCircles(
            crop_blur, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=search_radius-20, maxRadius=search_radius+20
        )
        
        if circles is not None:
            circles = circles[0]
            cx, cy = x - x1, y - y1
            distances = np.sqrt((circles[:, 0] - cx)**2 + (circles[:, 1] - cy)**2)
            best = circles[np.argmin(distances)]
            return float(x1 + best[0]), float(y1 + best[1]), float(best[2])
        
        print("    (원 검출 실패, 원래 위치 사용)")
        return float(x), float(y), None
    
    def _update_display(self):
        """메인 이미지에 현재 위치 표시"""
        disp = self.display.copy()
        x, y = self.current_pos
        cv2.drawMarker(disp, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        # 이미지가 화면보다 크면 리사이즈
        height, width = disp.shape[:2]
        max_width, max_height = 1920, 1080
        if width > max_width or height > max_height:
            self.display_scale = min(max_width / width, max_height / height)
            new_width = int(width * self.display_scale)
            new_height = int(height * self.display_scale)
            # 마커 위치도 스케일에 맞게 조정
            x_scaled = int(x * self.display_scale)
            y_scaled = int(y * self.display_scale)
            disp = cv2.resize(disp, (new_width, new_height))
            # 리사이즈된 이미지에 마커 다시 그리기
            cv2.drawMarker(disp, (x_scaled, y_scaled), (0, 0, 255), cv2.MARKER_CROSS, 
                          max(5, int(20 * self.display_scale)), max(1, int(2 * self.display_scale)))
        else:
            self.display_scale = 1.0
        
        cv2.imshow("Select Points", disp)
    
    def _update_zoom(self):
        """줌 윈도우 업데이트"""
        x, y = self.current_pos
        h, w = self.image.shape[:2]
        
        radius = 50
        x1, x2 = max(0, x-radius), min(w, x+radius)
        y1, y2 = max(0, y-radius), min(h, y+radius)
        
        crop = self.image[y1:y2, x1:x2].copy()
        zoom = cv2.resize(crop, None, fx=self.zoom_level, fy=self.zoom_level, 
                         interpolation=cv2.INTER_NEAREST)
        
        zh, zw = zoom.shape[:2]
        # 십자선
        cv2.line(zoom, (zw//2, 0), (zw//2, zh), (0, 255, 0), 1)
        cv2.line(zoom, (0, zh//2), (zw, zh//2), (0, 255, 0), 1)

        radius_scaled = int(self.search_radius * self.zoom_level)
        cv2.circle(zoom, (zw//2, zh//2), radius_scaled, (255, 255, 0), 2)
        
        # 정보 표시
        type_str = "CORNER" if self.point_type == PointType.CORNER else "HOLE"
        cv2.putText(zoom, f"({x}, {y})", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(zoom, f"[{type_str}]", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Zoom", zoom)
    
    def select(self, n_points: int = 4) -> Tuple[np.ndarray, List[PointType]]:
        """
        점 선택
        Returns: (image_points, point_types)
        """
        cv2.namedWindow("Select Points")
        cv2.namedWindow("Zoom")
        
        self._update_display()
        self._update_zoom()
        
        print(f"\n=== {n_points}개 이상 선택 ===")
        print("  마우스: 대략적 위치 이동")
        print("  방향키: 1픽셀 미세조정")
        print("  Tab: 타입 전환 (CORNER <-> HOLE)")
        print("  Space: 점 확정 (자동 보정)")
        print("  Enter: 완료 / ESC: 취소")
        
        point_types = []
        
        def mouse_cb(event, mx, my, flags, param):
            if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
                # 리사이즈된 좌표를 원본 좌표로 변환
                if self.display_scale != 1.0:
                    mx_original = int(mx / self.display_scale)
                    my_original = int(my / self.display_scale)
                else:
                    mx_original = mx
                    my_original = my
                self.current_pos = [mx_original, my_original]
                self._update_display()
                self._update_zoom()
        
        cv2.setMouseCallback("Select Points", mouse_cb)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return np.array([]), []
            
            # 방향키
            elif key in [81, 2, ord('a')]:  # Left
                self.current_pos[0] -= 1
            elif key in [83, 3, ord('d')]:  # Right
                self.current_pos[0] += 1
            elif key in [82, 0, ord('w')]:  # Up
                self.current_pos[1] -= 1
            elif key in [84, 1, ord('s')]:  # Down
                self.current_pos[1] += 1
            
            # Tab: 타입 전환
            elif key == 9:
                if self.point_type == PointType.CORNER:
                    self.point_type = PointType.HOLE
                else:
                    self.point_type = PointType.CORNER
                print(f"  타입 변경: {self.point_type.name}")
            
            # Space: 점 확정
            elif key == 32:
                x, y = self.current_pos
                
                # 자동 보정
                if self.point_type == PointType.CORNER:
                    rx, ry = self._refine_corner(x, y, search_radius=self.search_radius)
                    radius = None
                else:
                    rx, ry, radius = self._refine_hole(x, y, search_radius=self.search_radius)
                
                self.points.append((rx, ry))
                self.circle_radii.append(radius)
                point_types.append(self.point_type)
                
                # 표시
                cv2.circle(self.display, (int(rx), int(ry)), 3, (0, 255, 0), -1)
                
                # HOLE 타입이고 원을 찾았으면 실제 원도 그리기
                if self.point_type == PointType.HOLE and radius is not None:
                    cv2.circle(self.display, (int(rx), int(ry)), int(radius), (255, 0, 0), 2)
                
                cv2.putText(self.display, str(len(self.points)), 
                           (int(rx)+10, int(ry)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if radius is not None:
                    print(f"  점 {len(self.points)} [{self.point_type.name}]: "
                          f"({x},{y}) -> ({rx:.1f}, {ry:.1f}), 반지름: {radius:.1f}px")
                else:
                    print(f"  점 {len(self.points)} [{self.point_type.name}]: "
                          f"({x},{y}) -> ({rx:.1f}, {ry:.1f})")
                
                self._update_display()
            
            # Enter: 완료
            elif key == 13:
                if len(self.points) >= n_points:
                    break
                print(f"  최소 {n_points}개 필요! 현재 {len(self.points)}개")
            
            self._update_display()
            self._update_zoom()
        
        cv2.destroyAllWindows()
        return np.array(self.points), point_types


class PixelDistanceMapper:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.H = None
        self.H_inv = None
        self.reference_world = np.array([0.0, 0.0])
        # 미리 계산된 distance map
        self.distance_map = None
        self.dx_map = None
        self.dy_map = None
        self.image_shape = None
    
    def calibrate_with_known_points(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        캘리브레이션 수행 (undistortion된 이미지의 포인트 사용)
        
        Args:
            image_points: 이미지 포인트 (undistortion된 이미지의 좌표)
            world_points: 실제 world 좌표 (mm)
            image_shape: 이미지 크기 (h, w)
        """
        if len(image_points) < 4:
            print("최소 4개 점 필요!")
            return False
        
        # 이미지 포인트는 이미 undistortion된 이미지의 좌표이므로 그대로 사용
        imgp = image_points.astype(np.float32)
        
        self.H, mask = cv2.findHomography(imgp, world_points.astype(np.float32))
        self.H_inv = np.linalg.inv(self.H)
        self.reference_world = np.array([0.0, 0.0])
        
        # Reprojection error
        projected = cv2.perspectiveTransform(
            imgp.reshape(-1, 1, 2), self.H
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - world_points, axis=1)
        
        print(f"캘리브레이션 완료!")
        print(f"  Reprojection error: mean={errors.mean():.2f}mm, max={errors.max():.2f}mm")
        
        # Distance map 미리 계산
        if image_shape is not None:
            self.initialize_distance_map(image_shape)
        
        return True
    
    @requires_calibration
    def initialize_distance_map(self, image_shape: Tuple[int, int]):
        """Distance map을 미리 계산하여 저장 (undistortion된 이미지의 픽셀 좌표를 직접 변환)"""
        h, w = image_shape
        self.image_shape = image_shape
        
        print(f"Distance map 계산 중... ({w}x{h})")
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Undistortion된 이미지의 모든 픽셀 좌표를 직접 world 좌표로 변환
        pixel_coords_u = u_coords.flatten().astype(np.float32)
        pixel_coords_v = v_coords.flatten().astype(np.float32)
        
        # Undistortion된 좌표를 world 좌표로 변환
        world_x, world_y = self.pixel_to_world(pixel_coords_u, pixel_coords_v)
        
        # X, Y 거리 맵
        self.dx_map = (world_x - self.reference_world[0]).reshape(h, w)
        self.dy_map = (world_y - self.reference_world[1]).reshape(h, w)
        
        # 전체 거리 맵
        self.distance_map = np.sqrt(self.dx_map**2 + self.dy_map**2)
        print("Distance map 계산 완료!")
    
    def _pixel_to_world_single(self, u: float, v: float) -> np.ndarray:
        pt = np.array([[[u, v]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, self.H)
        return world_pt[0, 0]
    
    @requires_calibration
    def pixel_to_world(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.atleast_1d(u).astype(np.float32)
        v = np.atleast_1d(v).astype(np.float32)
        pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
        world_pts = cv2.perspectiveTransform(pts, self.H)
        return world_pts[:, 0, 0], world_pts[:, 0, 1]
    
    @requires_calibration
    def get_distance(self, u: int, v: int) -> float:
        """픽셀 좌표에서 기준점까지의 거리 (mm) - undistortion된 이미지의 좌표 사용"""
        if self.distance_map is not None:
            # 미리 계산된 맵 사용
            if 0 <= v < self.distance_map.shape[0] and 0 <= u < self.distance_map.shape[1]:
                return float(self.distance_map[v, u])
            else:
                # 범위 밖이면 직접 계산
                world_pt = self._pixel_to_world_single(u, v)
                return float(np.linalg.norm(world_pt - self.reference_world))
        else:
            # 맵이 없으면 직접 계산
            world_pt = self._pixel_to_world_single(u, v)
            return float(np.linalg.norm(world_pt - self.reference_world))
    
    @requires_calibration
    def get_xy_distance(self, u: int, v: int) -> Tuple[float, float]:
        """픽셀 좌표에서 기준점까지의 X, Y 거리 (mm) - undistortion된 이미지의 좌표 사용"""
        if self.dx_map is not None and self.dy_map is not None:
            # 미리 계산된 맵 사용
            if 0 <= v < self.dx_map.shape[0] and 0 <= u < self.dx_map.shape[1]:
                return float(self.dx_map[v, u]), float(self.dy_map[v, u])
            else:
                # 범위 밖이면 직접 계산
                world_pt = self._pixel_to_world_single(u, v)
                diff = world_pt - self.reference_world
                return float(diff[0]), float(diff[1])
        else:
            # 맵이 없으면 직접 계산
            world_pt = self._pixel_to_world_single(u, v)
            diff = world_pt - self.reference_world
            return float(diff[0]), float(diff[1])
    
    @requires_calibration
    def create_distance_map(self, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Distance map 생성 (이미 계산되어 있으면 반환)"""
        if image_shape is None:
            if self.distance_map is not None:
                return self.distance_map
            else:
                raise ValueError("image_shape가 필요하거나 먼저 initialize_distance_map()을 호출하세요.")
        
        # 이미 같은 크기로 계산되어 있으면 반환
        if self.distance_map is not None and self.image_shape == image_shape:
            return self.distance_map
        
        # 새로 계산
        self.initialize_distance_map(image_shape)
        return self.distance_map
    
    @requires_calibration
    def save_distance_map(self, filepath: str) -> bool:
        """
        Distance map을 .npz 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로 (.npz 확장자 권장)
        
        Returns:
            저장 성공 여부
        """
        if self.distance_map is None:
            print("오류: Distance map이 계산되지 않았습니다. 먼저 initialize_distance_map()을 호출하세요.")
            return False
        
        try:
            # 메타데이터와 함께 저장
            np.savez_compressed(
                filepath,
                distance_map=self.distance_map,
                dx_map=self.dx_map,
                dy_map=self.dy_map,
                image_shape=np.array(self.image_shape),
                reference_world=self.reference_world,
                camera_matrix=self.K,
                dist_coeffs=self.dist,
                homography=self.H
            )
            print(f"Distance map 저장 완료: {filepath}")
            print(f"  이미지 크기: {self.image_shape}")
            print(f"  기준점: ({self.reference_world[0]:.2f}, {self.reference_world[1]:.2f}) mm")
            return True
        except Exception as e:
            print(f"Distance map 저장 실패: {e}")
            return False
    
    @staticmethod
    def load_distance_map(filepath: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Distance map을 .npz 파일에서 로드
        
        Args:
            filepath: 로드할 파일 경로
        
        Returns:
            로드된 데이터 딕셔너리 또는 None (실패 시)
            {
                'distance_map': np.ndarray,
                'dx_map': np.ndarray,
                'dy_map': np.ndarray,
                'image_shape': np.ndarray,
                'reference_world': np.ndarray,
                'camera_matrix': np.ndarray (선택적),
                'dist_coeffs': np.ndarray (선택적),
                'homography': np.ndarray (선택적)
            }
        """
        try:
            data = np.load(filepath)
            result = {
                'distance_map': data['distance_map'],
                'dx_map': data['dx_map'],
                'dy_map': data['dy_map'],
                'image_shape': tuple(data['image_shape']),
                'reference_world': data['reference_world']
            }
            
            # 선택적 데이터
            if 'camera_matrix' in data:
                result['camera_matrix'] = data['camera_matrix']
            if 'dist_coeffs' in data:
                result['dist_coeffs'] = data['dist_coeffs']
            if 'homography' in data:
                result['homography'] = data['homography']
            
            print(f"Distance map 로드 완료: {filepath}")
            print(f"  이미지 크기: {result['image_shape']}")
            print(f"  기준점: ({result['reference_world'][0]:.2f}, {result['reference_world'][1]:.2f}) mm")
            return result
        except Exception as e:
            print(f"Distance map 로드 실패: {e}")
            return None
    
    @requires_calibration
    def load_distance_map_to_self(self, filepath: str) -> bool:
        """
        Distance map을 로드하여 현재 인스턴스에 설정
        
        Args:
            filepath: 로드할 파일 경로
        
        Returns:
            로드 성공 여부
        """
        data = self.load_distance_map(filepath)
        if data is None:
            return False
        
        self.distance_map = data['distance_map']
        self.dx_map = data['dx_map']
        self.dy_map = data['dy_map']
        self.image_shape = data['image_shape']
        self.reference_world = data['reference_world']
        
        # 선택적 데이터 설정
        if 'camera_matrix' in data:
            self.K = data['camera_matrix']
        if 'dist_coeffs' in data:
            self.dist = data['dist_coeffs']
        if 'homography' in data:
            self.H = data['homography']
            self.H_inv = np.linalg.inv(self.H)
        
        return True


def detect_aruco_board_center(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    ArUco 보드의 중심을 찾습니다.
    
    Args:
        image: 입력 이미지 (undistortion된 이미지 권장)
        camera_matrix: 카메라 내부 파라미터
        dist_coeffs: 왜곡 계수
    
    Returns:
        (center_x, center_y) 또는 None (검출 실패 시)
    """
    # 이미지 undistortion
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # ArUco 딕셔너리 생성 (기본 DICT_4X4_50 사용, 필요시 변경 가능)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # 마커 검출
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None or len(ids) == 0:
        print("  ArUco 마커를 찾을 수 없습니다.")
        return None
    
    # 모든 마커의 중심점 계산
    centers = []
    for i, corner in enumerate(corners):
        # corner는 (1, 4, 2) 형태의 배열
        corner_pts = corner[0]  # (4, 2) 형태
        center = np.mean(corner_pts, axis=0)  # 4개 코너의 평균
        centers.append(center)
        marker_id = ids[i][0]
        print(f"  마커 ID {marker_id} 중심: ({center[0]:.1f}, {center[1]:.1f})")
    
    # 모든 마커 중심의 평균을 보드 중심으로 사용
    board_center = np.mean(centers, axis=0)
    print(f"  보드 중심: ({board_center[0]:.1f}, {board_center[1]:.1f})")
    
    return float(board_center[0]), float(board_center[1])


def detect_aruco_from_single_image(image_path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, min_markers: int = 4) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    한 이미지에서 여러 개의 ArUco 마커 중심을 찾습니다.
    
    Args:
        image_path: 이미지 파일 경로
        camera_matrix: 카메라 내부 파라미터
        dist_coeffs: 왜곡 계수
        min_markers: 최소 필요한 마커 개수 (기본: 4)
    
    Returns:
        (image_points, corners, ids, undistorted_image) 튜플 또는 None (실패 시)
        - image_points: 마커 중심 좌표 (Nx2 numpy array)
        - corners: 마커 코너 좌표
        - ids: 마커 ID 배열
        - undistorted_image: undistortion된 이미지
    """
    print(f"\n=== ArUco 마커 검출 모드 ===")
    print(f"이미지: {image_path}")
    print(f"최소 {min_markers}개의 ArUco 마커를 검출합니다...")
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"  오류: 이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    # 이미지 undistortion
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    
    # ArUco 딕셔너리 생성 (기본 DICT_4X4_50 사용, 필요시 변경 가능)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 마커 검출 (OpenCV 버전에 따라 다른 API 사용)
    try:
        # OpenCV 4.7+ 방식
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        try:
            # OpenCV 4.5-4.6 방식
            aruco_params = cv2.aruco.DetectorParameters()
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        except AttributeError:
            # OpenCV 3.x 방식
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is None or len(ids) == 0:
        print(f"  오류: ArUco 마커를 찾을 수 없습니다.")
        return None
    
    if len(ids) < min_markers:
        print(f"  오류: 최소 {min_markers}개의 마커가 필요합니다. 현재 {len(ids)}개만 검출됨")
        return None
    
    # 모든 마커의 중심점 계산
    image_points = []
    for i, corner in enumerate(corners):
        # corner는 (1, 4, 2) 형태의 배열
        corner_pts = corner[0]  # (4, 2) 형태
        center = np.mean(corner_pts, axis=0)  # 4개 코너의 평균
        image_points.append(center)
        marker_id = ids[i][0]
        print(f"  마커 ID {marker_id} 중심: ({center[0]:.1f}, {center[1]:.1f})")
    
    print(f"\n=== 검출 완료 ===")
    print(f"  총 {len(image_points)}개의 마커 검출됨")
    for i, pt in enumerate(image_points):
        print(f"  점 {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    return np.array(image_points), corners, ids, undistorted


def visualize_aruco_detection(
    image: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    image_points: np.ndarray,
    output_path: str,
    highlight_indices: Optional[np.ndarray] = None,
    point_order: Optional[np.ndarray] = None
) -> None:
    """
    ArUco 마커 검출 결과를 시각화하여 저장합니다.
    
    Args:
        image: undistortion된 이미지
        corners: 마커 코너 좌표
        ids: 마커 ID 배열
        image_points: 마커 중심 좌표
        output_path: 저장할 이미지 경로
        highlight_first_n: 처음 N개 마커를 다른 색으로 강조 (기본: 4)
    """
    result_image = image.copy()
    
    # 모든 마커 그리기
    cv2.aruco.drawDetectedMarkers(result_image, corners, ids)
    
    # 재정렬된 순서 매핑 생성 (point_order가 있으면 사용)
    if point_order is not None and len(point_order) == len(image_points):
        # point_order: 재정렬된 순서의 원본 인덱스 배열
        # 예: point_order = [2, 0, 3, 1] -> 원본 인덱스 2가 1번, 0이 2번, 3이 3번, 1이 4번
        # 중복 확인
        if len(set(point_order)) != len(point_order):
            print(f"  ⚠ 경고: point_order에 중복이 있습니다. 원본 순서 사용")
            display_numbers = [i + 1 for i in range(len(image_points))]
        else:
            order_map = {orig_idx: new_order + 1 for new_order, orig_idx in enumerate(point_order)}
            # 모든 원본 인덱스에 대해 재정렬된 번호 할당
            display_numbers = [order_map.get(i, i + 1) for i in range(len(image_points))]
    else:
        display_numbers = [i + 1 for i in range(len(image_points))]
    
    # 마커 중심점과 번호 표시
    for i, (corner, marker_id, center) in enumerate(zip(corners, ids, image_points)):
        center_int = (int(center[0]), int(center[1]))
        
        # 선택된 인덱스는 빨간색, 나머지는 초록색
        if highlight_indices is not None and i in highlight_indices:
            color = (0, 0, 255)  # 빨간색
            thickness = 3
            radius = 15
        else:
            color = (0, 255, 0)  # 초록색
            thickness = 2
            radius = 10
        
        # 중심점 원 그리기
        cv2.circle(result_image, center_int, radius, color, thickness)
        
        # 마커 ID와 번호 표시 (재정렬된 순서 사용)
        label = f"#{display_numbers[i]} (ID:{marker_id[0]})"
        font_scale = 1.2  # 글씨 크기 증가
        font_thickness = 3  # 글씨 두께 증가
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # 텍스트 배경
        cv2.rectangle(
            result_image,
            (center_int[0] - 5, center_int[1] - text_height - baseline - 5),
            (center_int[0] + text_width + 5, center_int[1] + baseline + 5),
            (255, 255, 255),
            -1
        )
        
        # 텍스트
        cv2.putText(
            result_image,
            label,
            (center_int[0], center_int[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness
        )
    
    # 범례 추가
    legend_y = 40
    legend_font_scale = 0.8  # 범례 글씨 크기
    legend_thickness = 2
    if highlight_indices is not None:
        cv2.putText(result_image, f"Red: Selected {len(highlight_indices)} markers (used for coordinate system)", 
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 0, 255), legend_thickness)
    else:
        cv2.putText(result_image, "Red: First 4 markers (used for coordinate system)", 
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 0, 255), legend_thickness)
    cv2.putText(result_image, "Green: Additional markers", 
                (10, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (0, 255, 0), legend_thickness)
    
    # 저장
    cv2.imwrite(output_path, result_image)
    print(f"  검출 결과 이미지 저장: {output_path}")
    
    # 창으로 표시
    cv2.namedWindow("ArUco Detection Result", cv2.WINDOW_NORMAL)
    cv2.imshow("ArUco Detection Result", result_image)
    print(f"  이미지 창 표시 (아무 키나 누르면 계속)")
    cv2.waitKey(0)


def select_farthest_4_points(image_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    가장 멀리 떨어진 4개 점을 선택합니다.
    
    Args:
        image_points: 모든 점의 좌표 (Nx2)
    
    Returns:
        (selected_points, selected_indices) 튜플
        - selected_points: 선택된 4개 점 (4x2)
        - selected_indices: 선택된 점의 원본 인덱스 (4,)
    """
    n = len(image_points)
    if n <= 4:
        return image_points, np.arange(n)
    
    # 모든 점 쌍 간의 거리 계산
    max_dist = 0
    best_pair = (0, 1)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(image_points[i] - image_points[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (i, j)
    
    # 가장 먼 두 점 선택
    idx1, idx2 = best_pair
    p1, p2 = image_points[idx1], image_points[idx2]
    
    # 나머지 점들 중에서 p1, p2와 함께 가장 큰 영역을 만드는 2개 점 찾기
    remaining_indices = [i for i in range(n) if i != idx1 and i != idx2]
    
    best_area = 0
    best_indices = [idx1, idx2, remaining_indices[0], remaining_indices[1]]
    
    # 모든 가능한 조합 시도
    for i in range(len(remaining_indices)):
        for j in range(i + 1, len(remaining_indices)):
            idx3, idx4 = remaining_indices[i], remaining_indices[j]
            pts = [p1, p2, image_points[idx3], image_points[idx4]]
            
            # 4개 점으로 만든 사각형의 넓이 계산 (또는 convex hull 넓이)
            # 간단하게: 두 대각선의 곱의 절반 사용
            # 또는: 4개 점의 bounding box 넓이
            pts_arr = np.array(pts)
            min_x, min_y = pts_arr.min(axis=0)
            max_x, max_y = pts_arr.max(axis=0)
            area = (max_x - min_x) * (max_y - min_y)
            
            if area > best_area:
                best_area = area
                best_indices = [idx1, idx2, idx3, idx4]
    
    # 선택된 4개 점을 정렬 (좌상, 우상, 좌하, 우하 순서로)
    selected_pts = image_points[best_indices]
    selected_indices = np.array(best_indices)
    
    # 점들을 정렬: x+y가 작은 순서대로 (좌상부터)
    # 또는: x 좌표로 먼저 정렬, 그 다음 y 좌표로 정렬
    # 하지만 world 좌표 계산을 위해 점1을 기준으로 점2, 점3, 점4를 선택해야 함
    # 점1: 가장 왼쪽 위 (또는 첫 번째 점)
    # 점2: 점1과 가장 먼 점 중 하나
    # 점3: 점1과 다른 방향으로 가장 먼 점
    
    # 간단하게: 선택된 점들을 원래 순서대로 유지하되, 점1을 기준으로 재배열
    # 점1을 첫 번째로, 나머지는 거리 순으로 정렬
    center = selected_pts.mean(axis=0)
    distances = [np.linalg.norm(pt - center) for pt in selected_pts]
    sorted_idx = np.argsort(distances)[::-1]  # 중심에서 먼 순서
    
    # 가장 먼 점을 점1로
    final_indices = [best_indices[sorted_idx[0]]]
    remaining = [best_indices[i] for i in sorted_idx[1:]]
    
    # 점1과 가장 먼 점을 점2로
    p1_final = image_points[final_indices[0]]
    dists_to_p1 = [np.linalg.norm(image_points[i] - p1_final) for i in remaining]
    idx2 = remaining[np.argmax(dists_to_p1)]
    final_indices.append(idx2)
    remaining.remove(idx2)
    
    # 나머지 두 점 중 점1과 다른 방향으로 가장 먼 점을 점3으로
    if len(remaining) == 2:
        # 점1->점2 벡터에 수직인 방향으로 가장 먼 점 찾기
        vec_12 = image_points[idx2] - p1_final
        vec_12_norm = vec_12 / np.linalg.norm(vec_12) if np.linalg.norm(vec_12) > 0 else np.array([0, 1])
        perp_vec = np.array([-vec_12_norm[1], vec_12_norm[0]])  # 수직 벡터
        
        dists_perp = []
        for idx in remaining:
            vec_to_pt = image_points[idx] - p1_final
            proj = np.abs(np.dot(vec_to_pt, perp_vec))
            dists_perp.append(proj)
        
        idx3 = remaining[np.argmax(dists_perp)]
        final_indices.append(idx3)
        remaining.remove(idx3)
        final_indices.append(remaining[0])
    
    final_points = image_points[final_indices]
    final_indices_arr = np.array(final_indices)
    
    return final_points, final_indices_arr


def input_world_coordinates(n_points: int) -> np.ndarray:
    print(f"\n=== 각 점의 실제 좌표 입력 (mm) ===")
    world_points = []
    
    for i in range(n_points):
        while True:
            try:
                coord = input(f"  점 {i+1} (x,y): ").strip()
                x, y = map(float, coord.replace(" ", ",").split(","))
                world_points.append([x, y])
                break
            except:
                print("    형식: x,y (예: 100,50)")
    
    return np.array(world_points)



def main(args):
    """메인 함수: Pixel Distance Mapper 실행"""
    
    # ===== 설정 =====
    IMAGE_PATH = args.image
    CAMERA_CONFIG_PATH = args.camera_config
    
    # ===== 카메라 캘리브레이션 데이터 로드 =====
    try:
        with open(CAMERA_CONFIG_PATH, 'r', encoding='utf-8') as f:
            camera_config = json.load(f)
        
        calibration = camera_config.get("calibration", {})
        camera_matrix = np.array(calibration.get("CameraMatrix", []), dtype=np.float64)
        dist_coeffs_list = calibration.get("DistortionCoefficients", [])
        
        # DistortionCoefficients는 중첩 리스트일 수 있으므로 평탄화
        if dist_coeffs_list and isinstance(dist_coeffs_list[0], list):
            dist_coeffs = np.array(dist_coeffs_list[0], dtype=np.float64)
        else:
            dist_coeffs = np.array(dist_coeffs_list, dtype=np.float64)
        
        print(f"카메라 설정 로드: {CAMERA_CONFIG_PATH}")
        print(f"  Camera Matrix shape: {camera_matrix.shape}")
        print(f"  Distortion Coefficients shape: {dist_coeffs.shape}")
        
        if camera_matrix.size == 0 or dist_coeffs.size == 0:
            raise ValueError("카메라 매트릭스 또는 왜곡 계수가 비어있습니다.")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"카메라 설정 파일을 찾을 수 없습니다: {CAMERA_CONFIG_PATH}")

    except Exception as e:
        raise Exception(f"카메라 설정 파일 로드 실패: {e}")

    search_radius = 8
    
    # ArUco 모드용 변수 초기화
    output_image_path = None
    corners = None
    ids = None
    
    # ===== 특징점 선택 =====
    if args.mode == "aruco":
        # ArUco 자동 검출 모드 - 한 이미지에서 여러 개의 마커 검출
        result = detect_aruco_from_single_image(IMAGE_PATH, camera_matrix, dist_coeffs, min_markers=args.min_markers)
        if result is None:
            print("ArUco 마커 검출 실패")
            exit(1)
        
        image_points, corners, ids, image = result
        
        n_points = len(image_points)
        if n_points < args.min_markers:
            print(f"오류: 최소 {args.min_markers}개의 마커가 필요합니다. 현재 {n_points}개만 검출됨")
            exit(1)
        
        h, w = image.shape[:2]
        
        # 검출 결과 시각화 및 저장 (초기 - 모든 점 표시)
        output_dir = Path(IMAGE_PATH).parent
        output_image_path = output_dir / f"{Path(IMAGE_PATH).stem}_aruco_detection.png"
        print(f"\n=== ArUco 마커 검출 결과 시각화 ===")
        # 일단 None으로 호출 (나중에 선택된 인덱스로 다시 시각화)
        visualize_aruco_detection(image, corners, ids, image_points, str(output_image_path), highlight_indices=None)
        
        point_types = [PointType.CORNER] * n_points  # ArUco 모드는 모두 CORNER로 처리
        
    else:
        # 수동 선택 모드
        # 원본 이미지를 로드하고 내부에서 undistortion 수행
        image_original = cv2.imread(IMAGE_PATH)
        if image_original is None:
            raise FileNotFoundError(f"이미지 없음: {IMAGE_PATH}")
        
        h, w = image_original.shape[:2]
        print(f"이미지: {IMAGE_PATH} ({w}x{h})")
        print("원본 이미지를 undistortion 처리 중...")
        
        # Undistortion 수행
        image = cv2.undistort(image_original, camera_matrix, dist_coeffs)
        print("✓ Undistortion 완료")
        
        selector = PointSelector(image, search_radius=search_radius)
        image_points, point_types = selector.select(n_points=4)
        
        if len(image_points) == 0:
            print("취소됨")
            exit(1)
    
    # ===== World 좌표 자동 계산 =====
    n_points = len(image_points)
    
    if n_points >= 4:
        # 4개 이상의 점이 있으면 가장 멀리 떨어진 4개 선택
        if n_points > 4:
            print(f"\n⚠ {n_points}개의 마커가 검출되었습니다. 가장 멀리 떨어진 4개를 선택하여 기준 좌표계를 만듭니다.")
            selected_points, selected_indices = select_farthest_4_points(image_points)
            print(f"  선택된 마커 인덱스: {selected_indices + 1} (원본 순서 기준)")
            
            # 선택되지 않은 점들은 나중에 처리하기 위해 저장
            all_indices = set(range(n_points))
            unselected_indices = list(all_indices - set(selected_indices))
            unselected_points = image_points[unselected_indices] if unselected_indices else np.array([])
            
            # 선택된 4개 점을 ArUco 마커 방향을 이용하여 재정렬
            # 첫 번째 마커의 방향을 기준으로 X축(오른쪽), Y축(아래) 정의
            
            # 먼저 왼쪽 상단 마커 찾기 (x+y 최소)
            x_coords = selected_points[:, 0]
            y_coords = selected_points[:, 1]
            idx1_in_selected = np.argmin(x_coords + y_coords)
            p1 = selected_points[idx1_in_selected]
            idx1_original = selected_indices[idx1_in_selected]
            
            # 첫 번째 마커의 corners를 이용하여 방향 계산
            if corners is not None and len(corners) > idx1_original:
                # ArUco 마커의 corners는 시계방향: [왼상, 오상, 오하, 왼하]
                marker_corners = corners[idx1_original][0]  # (4, 2)
                # 왼쪽 상단 -> 오른쪽 상단 벡터 (X축 방향)
                vec_x = marker_corners[1] - marker_corners[0]
                # 왼쪽 상단 -> 왼쪽 하단 벡터 (Y축 방향)
                vec_y = marker_corners[3] - marker_corners[0]
                
                # 정규화
                vec_x_norm = vec_x / np.linalg.norm(vec_x) if np.linalg.norm(vec_x) > 0 else np.array([1, 0])
                vec_y_norm = vec_y / np.linalg.norm(vec_y) if np.linalg.norm(vec_y) > 0 else np.array([0, 1])
            else:
                # corners 정보가 없으면 기본 방향 사용
                vec_x_norm = np.array([1, 0])
                vec_y_norm = np.array([0, 1])
            
            # 나머지 점들을 첫 번째 마커 기준 좌표계로 변환
            remaining_indices_in_selected = [i for i in range(len(selected_points)) if i != idx1_in_selected]
            remaining_points = selected_points[remaining_indices_in_selected]
            
            # 각 점을 첫 번째 마커 기준으로 상대 위치 계산
            rel_vectors = remaining_points - p1
            
            # X축과 Y축으로 투영
            proj_x = np.dot(rel_vectors, vec_x_norm)
            proj_y = np.dot(rel_vectors, vec_y_norm)
            
            # 오른쪽 상단: X축 양수, Y축 음수 또는 작음
            # 왼쪽 하단: X축 음수 또는 작음, Y축 양수
            # 오른쪽 하단: X축 양수, Y축 양수
            
            # 오른쪽 상단 찾기 (X축 투영이 가장 큰 것 중 Y축 투영이 가장 작은 것)
            candidates_rt = [i for i in range(len(remaining_points)) if proj_x[i] > 0]
            if len(candidates_rt) > 0:
                idx2_in_remaining = min(candidates_rt, key=lambda i: proj_y[i])
            else:
                idx2_in_remaining = np.argmax(proj_x)
            idx2_in_selected = remaining_indices_in_selected[idx2_in_remaining]
            p2 = selected_points[idx2_in_selected]
            
            # 왼쪽 하단 찾기 (Y축 투영이 가장 큰 것 중 X축 투영이 가장 작은 것)
            remaining_after_rt = [i for i in range(len(remaining_points)) if i != idx2_in_remaining]
            if len(remaining_after_rt) > 0:
                candidates_lb = [i for i in remaining_after_rt if proj_y[i] > 0]
                if len(candidates_lb) > 0:
                    idx3_in_remaining = min(candidates_lb, key=lambda i: proj_x[i])
                else:
                    idx3_in_remaining = np.argmax(proj_y)
                idx3_in_selected = remaining_indices_in_selected[idx3_in_remaining]
                p3 = selected_points[idx3_in_selected]
            else:
                idx3_in_selected = remaining_indices_in_selected[0]
                p3 = selected_points[idx3_in_selected]
            
            # 오른쪽 하단: 나머지 점
            remaining_final = [i for i in range(len(selected_points)) if i not in [idx1_in_selected, idx2_in_selected, idx3_in_selected]]
            idx4_in_selected = remaining_final[0]
            p4 = selected_points[idx4_in_selected]
            
            # 재정렬된 점들
            image_points_4 = np.array([p1, p2, p3, p4])
            
            # 원본 인덱스도 재정렬
            reordered_indices = [
                selected_indices[idx1_in_selected],
                selected_indices[idx2_in_selected],
                selected_indices[idx3_in_selected],
                selected_indices[idx4_in_selected]
            ]
            selected_indices = np.array(reordered_indices)
        else:
            # 4개 점일 때도 ArUco 마커 방향을 이용하여 재정렬
            if corners is not None and len(corners) >= 4:
                # 첫 번째 마커의 방향 계산
                marker_corners = corners[0][0]  # (4, 2)
                vec_x = marker_corners[1] - marker_corners[0]  # X축 방향
                vec_y = marker_corners[3] - marker_corners[0]  # Y축 방향
                vec_x_norm = vec_x / np.linalg.norm(vec_x) if np.linalg.norm(vec_x) > 0 else np.array([1, 0])
                vec_y_norm = vec_y / np.linalg.norm(vec_y) if np.linalg.norm(vec_y) > 0 else np.array([0, 1])
                
                # 왼쪽 상단 찾기
                idx1 = np.argmin(image_points[:, 0] + image_points[:, 1])
                p1 = image_points[idx1]
                
                # 나머지 점들을 첫 번째 마커 기준 좌표계로 변환
                remaining_indices = [i for i in range(4) if i != idx1]
                remaining_points = image_points[remaining_indices]
                rel_vectors = remaining_points - p1
                proj_x = np.dot(rel_vectors, vec_x_norm)
                proj_y = np.dot(rel_vectors, vec_y_norm)
                
                # 오른쪽 상단, 왼쪽 하단, 오른쪽 하단 찾기
                candidates_rt = [i for i in range(len(remaining_points)) if proj_x[i] > 0]
                if len(candidates_rt) > 0:
                    idx2_in_remaining = min(candidates_rt, key=lambda i: proj_y[i])
                else:
                    idx2_in_remaining = np.argmax(proj_x)
                idx2 = remaining_indices[idx2_in_remaining]
                
                remaining_after_rt = [i for i in range(len(remaining_points)) if i != idx2_in_remaining]
                if len(remaining_after_rt) > 0:
                    candidates_lb = [i for i in remaining_after_rt if proj_y[i] > 0]
                    if len(candidates_lb) > 0:
                        idx3_in_remaining = min(candidates_lb, key=lambda i: proj_x[i])
                    else:
                        idx3_in_remaining = np.argmax(proj_y)
                    idx3 = remaining_indices[idx3_in_remaining]
                else:
                    idx3 = remaining_indices[0]
                
                idx4 = [i for i in range(4) if i not in [idx1, idx2, idx3]][0]
                
                image_points_4 = np.array([image_points[idx1], image_points[idx2], image_points[idx3], image_points[idx4]])
                selected_indices = np.array([idx1, idx2, idx3, idx4])
            else:
                # corners 정보가 없으면 기본 정렬
                image_points_4 = image_points
                selected_indices = np.arange(4)
            unselected_points = np.array([])
        
        # 재정렬된 순서 계산 (world 좌표 매핑에 필요)
        if n_points > 4:
            all_indices = set(range(n_points))
            unselected_indices_set = all_indices - set(selected_indices)
            # unselected_indices를 원본 순서대로 정렬
            unselected_indices_list = sorted(list(unselected_indices_set))
            point_order = np.concatenate([selected_indices, unselected_indices_list])
        else:
            point_order = selected_indices.copy()
        
        # 모든 인덱스가 포함되었는지 확인
        if len(set(point_order)) != n_points or len(point_order) != n_points:
            print(f"  ⚠ 경고: point_order 계산 실패. 원본 순서 사용")
            point_order = np.arange(n_points)
        
        # World 좌표 로드 또는 자동 계산
        if args.world_coords and os.path.exists(args.world_coords):
            # JSON 파일에서 모든 점의 실제 좌표 로드
            print(f"\n=== World 좌표 로드 ===")
            print(f"  파일: {args.world_coords}")
            try:
                with open(args.world_coords, 'r', encoding='utf-8') as f:
                    world_data = json.load(f)
                
                world_points_list = world_data.get("points", [])
                if len(world_points_list) != n_points:
                    print(f"  ⚠ 경고: 파일에 {len(world_points_list)}개 점이 있지만, 검출된 점은 {n_points}개입니다.")
                    print(f"  처음 {min(len(world_points_list), n_points)}개만 사용합니다.")
                
                # JSON 파일의 points는 재정렬된 순서로 되어 있음
                # point_order를 사용하여 원본 인덱스에 매핑
                world_points_all = np.zeros((n_points, 2))
                for reordered_idx, original_idx in enumerate(point_order):
                    if reordered_idx < len(world_points_list):
                        world_points_all[original_idx] = world_points_list[reordered_idx]
                
                # 사용자가 지정한 각도만큼 회전 적용
                if args.rotate != 0.0 and len(selected_indices) >= 4:
                    # 첫 번째 점(재정렬된 순서의 첫 번째)을 중심으로 회전
                    first_point_idx = selected_indices[0] if len(selected_indices) > 0 else 0
                    p1_world = world_points_all[first_point_idx].copy()
                    
                    # 각도를 라디안으로 변환 (시계방향이 양수)
                    angle_rad = np.radians(-args.rotate)  # 시계방향이 양수이므로 반대 방향으로 회전
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    
                    # 모든 world 좌표를 첫 번째 점 기준으로 회전
                    for i in range(n_points):
                        rel_pos = world_points_all[i] - p1_world
                        rotated_rel = rotation_matrix @ rel_pos
                        world_points_all[i] = p1_world + rotated_rel
                    
                    print(f"  회전 적용: {args.rotate:.2f}° (첫 번째 점 중심)")
                
                world_points = world_points_all
                print(f"  ✓ {len(world_points_list)}개 점의 실제 좌표 로드 완료")
                print(f"  재정렬된 순서: {point_order + 1} (원본 인덱스 기준)")
                
            except Exception as e:
                print(f"  ✗ World 좌표 파일 로드 실패: {e}")
                print(f"  자동 계산 방식으로 전환합니다.")
                args.world_coords = None  # 자동 계산으로 전환
        
        if not args.world_coords or not os.path.exists(args.world_coords):
            # 자동 계산 방식 (기존 로직)
            # 점 1을 기준으로, 점 2, 3, 4가 직사각형이 되도록 world 좌표 생성
            p1 = image_points_4[0]  # 점 1 (기준점)
            dist_x = np.linalg.norm(image_points_4[1] - image_points_4[0])  # 점1-점2 거리
            dist_y = np.linalg.norm(image_points_4[2] - image_points_4[0])  # 점1-점3 거리
            
            # 처음 4개 점의 world 좌표 (직사각형)
            world_points = [
                [p1[0], p1[1]],                    # 점 1: 내가 찍은 위치 그대로
                [p1[0] + dist_x, p1[1]],           # 점 2: 오른쪽
                [p1[0], p1[1] + dist_y],           # 점 3: 아래
                [p1[0] + dist_x, p1[1] + dist_y]   # 점 4: 대각선
            ]
            
            # 선택되지 않은 점들이 있으면, 상대적 위치로 world 좌표 계산
            if len(unselected_points) > 0:
                # 기준점 (점 1)을 원점으로 하는 상대 좌표계 사용
                # 점 1-2와 점 1-3의 방향 벡터를 사용하여 나머지 점들의 world 좌표 계산
                vec_12 = image_points_4[1] - image_points_4[0]  # 점1->점2 벡터
                vec_13 = image_points_4[2] - image_points_4[0]  # 점1->점3 벡터
                
                # 단위 벡터 계산
                dir_x = vec_12 / dist_x if dist_x > 0 else np.array([1, 0])
                dir_y = vec_13 / dist_y if dist_y > 0 else np.array([0, 1])
                
                for unselected_pt in unselected_points:
                    # 이미지 좌표에서 점 1을 기준으로 한 상대 위치
                    rel_vec = unselected_pt - p1
                    # dir_x와 dir_y 방향으로의 투영 (픽셀 단위 거리)
                    proj_x = np.dot(rel_vec, dir_x)  # 픽셀 단위
                    proj_y = np.dot(rel_vec, dir_y)  # 픽셀 단위
                    # world 좌표 계산 (점1을 기준으로 상대 위치를 그대로 사용)
                    world_x = p1[0] + proj_x
                    world_y = p1[1] + proj_y
                    world_points.append([world_x, world_y])
            
            # 모든 점의 world 좌표를 원본 순서대로 재배열
            world_points_all = np.zeros((n_points, 2))
            world_points_all[selected_indices] = np.array(world_points[:4])
            if len(unselected_points) > 0:
                unselected_world = np.array(world_points[4:])
                for i, orig_idx in enumerate(unselected_indices):
                    world_points_all[orig_idx] = unselected_world[i]
            
            world_points = world_points_all
        
        # 픽셀 거리 계산 (world 좌표 로드 여부와 관계없이 항상 계산)
        dist_x = np.linalg.norm(image_points_4[1] - image_points_4[0])
        dist_y = np.linalg.norm(image_points_4[2] - image_points_4[0])
        
        if args.world_coords and os.path.exists(args.world_coords):
            print(f"\n=== World 좌표 (파일에서 로드) ===")
        else:
            print(f"\n=== World 좌표 (자동 계산) ===")
            if n_points > 4:
                print(f"  기준 좌표계: 마커 {selected_indices + 1} (가장 멀리 떨어진 4개)")
            print(f"  점1-점2 픽셀 거리: {dist_x:.1f}px")
            print(f"  점1-점3 픽셀 거리: {dist_y:.1f}px")
        print(f"  총 {n_points}개 점 사용")
        for i, (img_pt, world_pt) in enumerate(zip(image_points, world_points)):
            marker_type = "★" if i in selected_indices else " "
            print(f"  {marker_type} 점 {i+1}: 이미지({img_pt[0]:.1f}, {img_pt[1]:.1f}) -> World({world_pt[0]:.1f}, {world_pt[1]:.1f})")
        
        # ArUco 모드일 때 선택된 인덱스로 다시 시각화 (재정렬된 순서 반영)
        if args.mode == "aruco" and output_image_path is not None:
            print(f"\n=== 재정렬된 순서로 시각화 업데이트 ===")
            print(f"  재정렬된 순서: {point_order + 1} (원본 인덱스 기준)")
            
            visualize_aruco_detection(image, corners, ids, image_points, str(output_image_path), 
                                     highlight_indices=selected_indices, point_order=point_order)
    else:
        print(f"오류: 최소 4개 점이 필요합니다. 현재 {n_points}개")
        exit(1)
    
    # ===== Pixel Size 계산 =====
    pixel_size_x = None
    pixel_size_y = None
    pixel_size_avg = None
    real_dist_x = None
    real_dist_y = None
    pixel_dist_x = None
    pixel_dist_y = None
    
    # World 좌표가 로드된 경우 자동으로 pixel size 계산
    if args.world_coords and os.path.exists(args.world_coords) and len(selected_indices) >= 4:
        # 선택된 4개 점의 world 좌표로부터 실제 거리 계산
        world_pts_4 = world_points[selected_indices]
        real_dist_x = np.linalg.norm(world_pts_4[1] - world_pts_4[0])  # 점1-점2 world 거리
        real_dist_y = np.linalg.norm(world_pts_4[2] - world_pts_4[0])  # 점1-점3 world 거리
        pixel_dist_x = np.linalg.norm(image_points_4[1] - image_points_4[0])  # 점1-점2 픽셀 거리
        pixel_dist_y = np.linalg.norm(image_points_4[2] - image_points_4[0])  # 점1-점3 픽셀 거리
        
        if pixel_dist_x > 0 and pixel_dist_y > 0:
            pixel_size_x = real_dist_x / pixel_dist_x  # mm/pixel
            pixel_size_y = real_dist_y / pixel_dist_y  # mm/pixel
            pixel_size_avg = (pixel_size_x + pixel_size_y) / 2
            
            print(f"\n=== Pixel Size 계산 (World 좌표에서 자동 계산) ===")
            print(f"  점1-점2 실제 거리: {real_dist_x:.1f}mm")
            print(f"  점1-점3 실제 거리: {real_dist_y:.1f}mm")
            print(f"  점1-점2 픽셀 거리: {pixel_dist_x:.1f}px")
            print(f"  점1-점3 픽셀 거리: {pixel_dist_y:.1f}px")
            print(f"  Pixel Size (X축): {pixel_size_x:.6f} mm/pixel")
            print(f"  Pixel Size (Y축): {pixel_size_y:.6f} mm/pixel")
            print(f"  Pixel Size (평균): {pixel_size_avg:.6f} mm/pixel")
            
            if abs(pixel_size_x - pixel_size_y) / pixel_size_avg > 0.1:
                print(f"  ⚠ 경고: X/Y pixel size 차이가 10% 이상입니다. 호모그래피 적용 후에는 균일해집니다.")
    elif args.real_dist_x is not None and args.real_dist_y is not None:
        # 명시적으로 제공된 경우
        real_dist_x = args.real_dist_x
        real_dist_y = args.real_dist_y
        pixel_dist_x = dist_x
        pixel_dist_y = dist_y
        pixel_size_x = args.real_dist_x / dist_x  # mm/pixel
        pixel_size_y = args.real_dist_y / dist_y  # mm/pixel
        pixel_size_avg = (pixel_size_x + pixel_size_y) / 2
        
        print(f"\n=== Pixel Size 계산 ===")
        print(f"  점1-점2 실제 거리: {args.real_dist_x:.1f}mm")
        print(f"  점1-점3 실제 거리: {args.real_dist_y:.1f}mm")
        print(f"  Pixel Size (X축): {pixel_size_x:.6f} mm/pixel")
        print(f"  Pixel Size (Y축): {pixel_size_y:.6f} mm/pixel")
        print(f"  Pixel Size (평균): {pixel_size_avg:.6f} mm/pixel")
        
        if abs(pixel_size_x - pixel_size_y) / pixel_size_avg > 0.1:
            print(f"  ⚠ 경고: X/Y pixel size 차이가 10% 이상입니다. 호모그래피 적용 후에는 균일해집니다.")
    elif args.real_dist_x is not None or args.real_dist_y is not None:
        print(f"\n=== Pixel Size 계산 ===")
        print(f"  ⚠ --real-dist-x와 --real-dist-y를 모두 입력해야 pixel size를 계산할 수 있습니다.")
    
    # ===== 캘리브레이션 =====
    mapper = PixelDistanceMapper(camera_matrix, dist_coeffs)
    success = mapper.calibrate_with_known_points(image_points, world_points, image_shape=(h, w))
    
    if not success:
        exit(1)
    
    # # ===== Distance Map 저장 (선택적) =====
    # # 저장할 경로 설정 (None이면 저장 안 함)
    # DISTANCE_MAP_PATH = "data/distance_map.npz"
    
    # if DISTANCE_MAP_PATH:
    #     print(f"\n=== Distance Map 저장 ===")
    #     success = mapper.save_distance_map(DISTANCE_MAP_PATH)
    #     if success:
    #         print(f"✓ 저장 완료: {DISTANCE_MAP_PATH}")
    #     else:
    #         print(f"✗ 저장 실패: {DISTANCE_MAP_PATH}")
    # else:
    #     print(f"\n=== Distance Map 저장 건너뜀 ===")
    #     print(f"  저장하려면 DISTANCE_MAP_PATH를 설정하세요.")
    #     print(f"  예: DISTANCE_MAP_PATH = 'config/distance_map_camera1.npz'")
    
    # ===== 호모그래피 Warp 이미지 저장 =====
    print(f"\n=== 호모그래피 Warp 이미지 생성 ===")
    # 오프셋 정보 초기화 (JSON 저장에 사용)
    offset_x = None
    offset_y = None
    margin = None
    
    try:
        # 이미지의 4개 모서리를 호모그래피로 변환하여 world 좌표 범위 계산
        corners_original = np.array([
            [[0, 0]],      # 좌상
            [[w, 0]],      # 우상
            [[w, h]],      # 우하
            [[0, h]]       # 좌하
        ], dtype=np.float32)
        
        corners_world = cv2.perspectiveTransform(corners_original, mapper.H)
        corners_world = corners_world.reshape(-1, 2)
        
        # World 좌표 범위 계산
        min_x = corners_world[:, 0].min()
        min_y = corners_world[:, 1].min()
        max_x = corners_world[:, 0].max()
        max_y = corners_world[:, 1].max()
        
        # 마진 추가 (첫 번째 점이 (0,0)이 아닌 적절한 위치에 오도록)
        margin = args.warp_margin  # 픽셀 단위 마진
        offset_x = -min_x + margin
        offset_y = -min_y + margin
        
        # 변환된 이미지 크기 계산
        warped_width = int(max_x - min_x + 2 * margin)
        warped_height = int(max_y - min_y + 2 * margin)
        
        # 오프셋을 적용한 호모그래피 행렬 생성
        translation = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ], dtype=np.float64)
        
        H_with_offset = translation @ mapper.H
        
        # 호모그래피로 undistortion된 이미지를 world 좌표계로 변환
        warped_image = cv2.warpPerspective(image, H_with_offset, (warped_width, warped_height), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 결과 이미지 저장 경로
        output_image_path = Path(IMAGE_PATH).parent / f"{Path(IMAGE_PATH).stem}_warped.jpg"
        success = cv2.imwrite(str(output_image_path), warped_image)
        
        if success:
            print(f"✓ 호모그래피 Warp 이미지 저장 완료: {output_image_path}")
            print(f"  출력 이미지 크기: {warped_image.shape[1]}x{warped_image.shape[0]} 픽셀")
            print(f"  World 좌표 범위: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
            print(f"  오프셋 적용: X={offset_x:.1f}, Y={offset_y:.1f} (마진: {margin}px)")
            
            # 창으로 표시
            cv2.namedWindow("Homography Warped Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Homography Warped Image", warped_image)
            print(f"  이미지 창 표시 (아무 키나 누르면 계속)")
            cv2.waitKey(0)
        else:
            print(f"✗ 호모그래피 Warp 이미지 저장 실패: {output_image_path}")
    except Exception as e:
        print(f"✗ 호모그래피 Warp 이미지 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== 호모그래피와 Pixel Size를 별도 JSON 파일에 저장 =====
    # 출력 파일명: camera1_config.json -> camera1_homography.json
    homography_output_path = Path(CAMERA_CONFIG_PATH).parent / f"{Path(CAMERA_CONFIG_PATH).stem}_homography.json"
    
    print(f"\n=== 호모그래피 캘리브레이션 데이터 저장 ===")
    try:
        # 호모그래피 행렬을 리스트로 변환
        homography_list = mapper.H.tolist()
        
        # 별도 파일에 저장할 데이터 (순서: PixelSize -> Homography -> 기타)
        homography_data = {}
        
        # Pixel Size가 계산되었으면 먼저 저장
        if pixel_size_avg is not None:
            homography_data["PixelSize"] = {
                "x": float(pixel_size_x),
                "y": float(pixel_size_y),
                "average": float(pixel_size_avg),
                "unit": "mm/pixel"
            }
            # RealDistance와 PixelDistance는 사용되지 않으므로 저장하지 않음
        
        # Homography 저장
        homography_data["Homography"] = homography_list
        
        # 기타 정보 저장
        homography_data["image_path"] = str(IMAGE_PATH)
        homography_data["camera_config_path"] = str(CAMERA_CONFIG_PATH)
        
        # Warp 이미지 생성 시 사용된 오프셋 정보 저장 (있는 경우)
        if offset_x is not None and offset_y is not None:
            homography_data["WarpOffset"] = {
                "x": float(offset_x),
                "y": float(offset_y),
                "margin": margin,
                "unit": "pixel",
                "_description": "호모그래피 warp 이미지 생성 시 적용된 오프셋. 첫 번째 점이 (0,0)이 아닌 적절한 위치에 오도록 조정"
            }
        
        # 저장
        with open(homography_output_path, 'w', encoding='utf-8') as f:
            json.dump(homography_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 저장 완료: {homography_output_path}")
        print(f"  Homography (3x3):")
        for row in homography_list:
            print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
        
        if pixel_size_avg is not None:
            print(f"  PixelSize:")
            print(f"    X: {pixel_size_x:.6f} mm/pixel")
            print(f"    Y: {pixel_size_y:.6f} mm/pixel")
            print(f"    Average: {pixel_size_avg:.6f} mm/pixel")
    except Exception as e:
        print(f"✗ 저장 실패: {e}")
    
    # ===== 테스트 모드 =====
    #print("\n=== 테스트: 클릭하면 거리 표시 (ESC 종료) ===")
    
    # def test_callback(event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         dist = mapper.get_distance(x, y)
    #         dx, dy = mapper.get_xy_distance(x, y)
    #         print(f"  ({x}, {y}) -> 거리: {dist:.1f}mm (X: {dx:.1f}, Y: {dy:.1f})")
    
    #cv2.namedWindow("Test")
    #cv2.setMouseCallback("Test", test_callback)
    #cv2.imshow("Test", image)
    
    # while cv2.waitKey(1) & 0xFF != 27:
    #     pass
    
    # 모든 창 닫기
    print(f"\n=== 완료 ===")
    cv2.destroyAllWindows()


# ArUco 모드로 실행 예시
# python scripts/pixel_distance_mapper.py \
#     --mode aruco \
#     --images img1.jpg img2.jpg img3.jpg img4.jpg \
#     --camera-config config/camera1_config.json \
#     --real-dist-x 900 \
#     --real-dist-y 450
def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Pixel Distance Mapper - 이미지 픽셀과 실제 거리 매핑\n"
                    "원본 이미지를 입력하면 내부에서 자동으로 undistortion을 수행합니다."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/aruco/cam3/cam3.png",
        help="입력 이미지 경로 (원본 이미지 가능 - 내부에서 undistortion 수행)"
    )
    parser.add_argument(
        "--camera-config",
        type=str,
        default="config/cam3_config.json",
        help="카메라 설정 파일 경로"
    )
    parser.add_argument(
        "--world-coords",
        type=str,
        default="data/cam3_world_coords.json",
        help="모든 점의 실제 world 좌표 JSON 파일 경로 (정확한 보정용). 형식: {\"points\": [[x1,y1], [x2,y2], ...]} (mm 단위)"
    )
    parser.add_argument(
        "--rotate",
        type=float,
        default=-30.0,
        help="World 좌표를 회전할 각도 (도 단위, 시계방향이 양수). 첫 번째 점을 중심으로 회전합니다."
    )
    parser.add_argument(
        "--warp-margin",
        type=int,
        default=0,
        help="호모그래피 warp 이미지 생성 시 적용할 마진 (픽셀 단위, 기본: 100). 첫 번째 점이 (0,0)이 아닌 적절한 위치에 오도록 조정"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/homography.npz",
        help="호모그래피 및 distance map 저장 경로"
    )
    parser.add_argument(
        "--real-dist-x",
        type=float,
        default=None,
        help="점1-점2 간의 실제 물리적 거리 (mm)"
    )
    parser.add_argument(
        "--real-dist-y",
        type=float,
        default=None,
        help="점1-점3 간의 실제 물리적 거리 (mm)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "aruco"],
        default="aruco",
        help="포인트 선택 모드: 'manual' (수동 선택) 또는 'aruco' (ArUco 자동 검출)"
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=4,
        help="ArUco 모드에서 최소 필요한 마커 개수 (기본: 4)"
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    main(args)