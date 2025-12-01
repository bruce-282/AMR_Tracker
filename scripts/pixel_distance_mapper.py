import numpy as np
import cv2
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
        if len(image_points) < 4:
            print("최소 4개 점 필요!")
            return False
        
        imgp = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2).astype(np.float32),
            self.K, self.dist, P=self.K
        ).reshape(-1, 2)
        
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
        """Distance map을 미리 계산하여 저장"""
        h, w = image_shape
        self.image_shape = image_shape
        
        print(f"Distance map 계산 중... ({w}x{h})")
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        world_x, world_y = self.pixel_to_world(
            u_coords.flatten().astype(np.float32),
            v_coords.flatten().astype(np.float32)
        )
        
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
        """픽셀 좌표에서 기준점까지의 거리 (mm)"""
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
        """픽셀 좌표에서 기준점까지의 X, Y 거리 (mm)"""
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


if __name__ == "__main__":
    # ===== 설정 =====
    IMAGE_PATH = "data/250926_HN_AGV_Calib/7.jpg"
    camera_matrix = np.array([
        [23706.897, 0, 2144.215],
        [0, 23707.981, 1114.160],
        [0, 0, 1.0]
    ])
    dist_coeffs = np.array([ -3.969228,  285.347899,  -0.041028,  -0.010927, 0.000000 ]) 

    search_radius = 29
    
    # ===== World 좌표 설정 (mm) =====
    # 이미지에서 선택한 점 순서대로 실제 좌표를 입력하세요
    # 예시:
    # WORLD_COORDINATES = np.array([
    #     [0, 0],      # 점 1: 기준점
    #     [300, 0],    # 점 2: 점1에서 오른쪽으로 300mm
    #     [0, 200],    # 점 3: 점1에서 아래로 200mm
    #     [300, 200]   # 점 4: 대각선
    # ])
    WORLD_COORDINATES = np.array([
        [0, 0],
        [300, 0],
        [0, 200],
        [300, 200]
    ])
    # ===== 이미지 로드 =====
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"이미지 없음: {IMAGE_PATH}")
    
    h, w = image.shape[:2]
    print(f"이미지: {IMAGE_PATH} ({w}x{h})")
    
    # ===== 특징점 선택 =====
    selector = PointSelector(image, search_radius=search_radius)
    image_points, point_types = selector.select(n_points=4)
    
    if len(image_points) == 0:
        print("취소됨")
        exit(1)
    
    # ===== World 좌표 확인 =====
    n_points = len(image_points)
    if len(WORLD_COORDINATES) != n_points:
        print(f"오류: WORLD_COORDINATES의 점 개수({len(WORLD_COORDINATES)})가 선택한 점 개수({n_points})와 일치하지 않습니다!")
        print("코드에서 WORLD_COORDINATES를 수정하세요.")
        exit(1)
    
    world_points = WORLD_COORDINATES[:n_points]
    print(f"\n=== World 좌표 (mm) ===")
    for i, (img_pt, world_pt) in enumerate(zip(image_points, world_points)):
        print(f"  점 {i+1}: 이미지({img_pt[0]:.1f}, {img_pt[1]:.1f}) -> World({world_pt[0]:.1f}, {world_pt[1]:.1f})")
    
    # ===== 캘리브레이션 =====
    mapper = PixelDistanceMapper(camera_matrix, dist_coeffs)
    success = mapper.calibrate_with_known_points(image_points, world_points, image_shape=(h, w))
    
    if not success:
        exit(1)
    
    # ===== Distance Map 저장 (선택적) =====
    # 저장할 경로 설정 (None이면 저장 안 함)
    DISTANCE_MAP_PATH = "data/distance_map.npz"
    
    if DISTANCE_MAP_PATH:
        print(f"\n=== Distance Map 저장 ===")
        success = mapper.save_distance_map(DISTANCE_MAP_PATH)
        if success:
            print(f"✓ 저장 완료: {DISTANCE_MAP_PATH}")
        else:
            print(f"✗ 저장 실패: {DISTANCE_MAP_PATH}")
    else:
        print(f"\n=== Distance Map 저장 건너뜀 ===")
        print(f"  저장하려면 DISTANCE_MAP_PATH를 설정하세요.")
        print(f"  예: DISTANCE_MAP_PATH = 'config/distance_map_camera1.npz'")
    
    # ===== 테스트 모드 =====
    print("\n=== 테스트: 클릭하면 거리 표시 (ESC 종료) ===")
    
    def test_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            dist = mapper.get_distance(x, y)
            dx, dy = mapper.get_xy_distance(x, y)
            print(f"  ({x}, {y}) -> 거리: {dist:.1f}mm (X: {dx:.1f}, Y: {dy:.1f})")
    
    cv2.namedWindow("Test")
    cv2.setMouseCallback("Test", test_callback)
    cv2.imshow("Test", image)
    
    while cv2.waitKey(1) & 0xFF != 27:
        pass
    
    cv2.destroyAllWindows()
