import cv2
import numpy as np
import json
from pathlib import Path


def calculate_pixel_size(image_path, camera_matrix, dist_coeffs, square_size_mm, pattern_size):
    """
    체커보드 이미지에서 pixel size (mm/pixel) 계산
    """
    # 이미지 로드 및 undistort
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # 코너 검출
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    
    if not ret:
        raise ValueError("체커보드 코너 검출 실패")
    
    # 서브픽셀 정밀도
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # 인접 코너 간 pixel 거리 계산
    pixel_distances = []
    for i in range(pattern_size[1]):
        for j in range(pattern_size[0] - 1):
            idx = i * pattern_size[0] + j
            dist = np.linalg.norm(corners[idx] - corners[idx + 1])
            pixel_distances.append(dist)
    
    avg_pixel_dist = np.mean(pixel_distances)
    pixel_size = square_size_mm / avg_pixel_dist
    
    return pixel_size, avg_pixel_dist, corners


def verify_pixel_size(corners, pixel_size, square_size_mm, pattern_size):
    """
    계산된 pixel size 검증
    """
    total_pixels = np.linalg.norm(corners[0] - corners[pattern_size[0] - 1])
    total_mm = total_pixels * pixel_size
    expected_mm = (pattern_size[0] - 1) * square_size_mm
    error = abs(total_mm - expected_mm)
    
    return total_mm, expected_mm, error


def load_camera_config(config_path: str = "config/camera1_config.json"):
    """
    카메라 설정 파일에서 calibration 데이터 로드
    
    Args:
        config_path: 설정 파일 경로 (기본값: config/camera1_config.json)
    
    Returns:
        camera_matrix: 카메라 행렬 (numpy array)
        dist_coeffs: 왜곡 계수 (numpy array)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if "calibration" not in config:
        raise ValueError(f"설정 파일에 'calibration' 섹션이 없습니다: {config_path}")
    
    calibration = config["calibration"]
    
    # CameraMatrix를 numpy array로 변환
    if "CameraMatrix" not in calibration:
        raise ValueError("calibration에 'CameraMatrix'가 없습니다")
    camera_matrix = np.array(calibration["CameraMatrix"], dtype=np.float64)
    
    # DistortionCoefficients를 numpy array로 변환
    if "DistortionCoefficients" not in calibration:
        raise ValueError("calibration에 'DistortionCoefficients'가 없습니다")
    dist_coeffs = np.array(calibration["DistortionCoefficients"], dtype=np.float64)
    
    return camera_matrix, dist_coeffs


def main():
    # ===== 설정 (본인 값으로 수정) =====
    image_path = 'calibration_image.png'
    square_size_mm = 20.0
    pattern_size = (9, 6)
    
    # Config 파일에서 calibration 데이터 로드
    config_path = "config/camera1_config.json"  # 필요시 다른 카메라 config로 변경 가능
    
    camera_matrix, dist_coeffs = load_camera_config(config_path)
    print(f"설정 파일에서 calibration 데이터 로드 완료: {config_path}")
    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Distortion Coefficients: {dist_coeffs}")

    # ===== 계산 =====
    try:
        pixel_size, avg_pixel_dist, corners = calculate_pixel_size(
            image_path, camera_matrix, dist_coeffs, square_size_mm, pattern_size
        )
        
        print(f"평균 코너 간 거리: {avg_pixel_dist:.2f} pixels")
        print(f"Pixel size: {pixel_size:.4f} mm/pixel")
        
        # 검증
        total_mm, expected_mm, error = verify_pixel_size(
            corners, pixel_size, square_size_mm, pattern_size
        )
        print(f"검증 - 측정: {total_mm:.2f} mm, 실제: {expected_mm:.2f} mm, 오차: {error:.4f} mm")
        
    except Exception as e:
        print(f"에러: {e}")


if __name__ == '__main__':
    main()