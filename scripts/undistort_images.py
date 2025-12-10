"""
이미지 언디스토션 스크립트
폴더 내 모든 이미지를 카메라 캘리브레이션 파라미터를 사용하여 왜곡 보정합니다.
"""

import cv2
import numpy as np
import os
import argparse
import json
from pathlib import Path


# 기본 카메라 파라미터 (cam1)

# DEFAULT_INTRINSIC = np.array([
#     [3629.20, 0, 2069.28],
#     [0, 3622.66, 996.82],
#     [0, 0, 1]
# ], dtype=np.float64)
# DEFAULT_DISTORTION = np.array([-0.090631, 0.079203, 0.001397, 0.001820], dtype=np.float64)


def load_intrinsic(filepath: str) -> np.ndarray:
    """Intrinsic 파라미터 파일 로드"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    matrix = []
    for line in lines:
        line = line.strip()
        if line:
            values = [float(x) for x in line.split()]
            matrix.append(values)
    
    return np.array(matrix, dtype=np.float64)


def load_distortion(filepath: str) -> np.ndarray:
    """Distortion 파라미터 파일 로드 (4개면 k3=0 추가하여 5개로)"""
    with open(filepath, 'r') as f:
        line = f.readline().strip()
    
    values = [float(x) for x in line.split()]
    
    # OpenCV 표준: 5개 (k1, k2, p1, p2, k3)
    # 4개만 있으면 k3=0 추가
    if len(values) == 4:
        values.append(0.0)
    
    return np.array(values, dtype=np.float64)


def load_calibration_from_json(json_path: str) -> tuple:
    """
    JSON 카메라 설정 파일에서 calibration 파라미터 로드
    
    Args:
        json_path: cam1_config.json 같은 카메라 설정 파일 경로
    
    Returns:
        (camera_matrix, dist_coeffs) 튜플
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if 'calibration' not in config:
        raise ValueError(f"calibration 섹션이 없습니다: {json_path}")
    
    calib = config['calibration']
    
    # Camera Matrix 로드
    if 'CameraMatrix' not in calib:
        raise ValueError(f"CameraMatrix가 없습니다: {json_path}")
    camera_matrix = np.array(calib['CameraMatrix'], dtype=np.float64)
    
    # Distortion Coefficients 로드
    if 'DistortionCoefficients' not in calib:
        raise ValueError(f"DistortionCoefficients가 없습니다: {json_path}")
    dist_coeffs = np.array(calib['DistortionCoefficients'], dtype=np.float64).flatten()
    
    return camera_matrix, dist_coeffs


def undistort_image(image: np.ndarray, 
                    camera_matrix: np.ndarray, 
                    dist_coeffs: np.ndarray,
                    use_optimal_matrix: bool = True) -> np.ndarray:
    """
    이미지 왜곡 보정
    
    Args:
        image: 입력 이미지
        camera_matrix: 카메라 내부 파라미터 (3x3)
        dist_coeffs: 왜곡 계수
        use_optimal_matrix: 최적 카메라 행렬 사용 여부 (True: 이미지 크롭 없음)
    
    Returns:
        왜곡 보정된 이미지
    """
    h, w = image.shape[:2]
    
    if use_optimal_matrix:
        # 최적 카메라 행렬 계산 (모든 픽셀 유지)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    else:
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    return undistorted


def process_folder(input_folder: str, 
                   output_folder: str = None,
                   camera_config: str = None,
                   use_optimal_matrix: bool = True):
    """
    폴더 내 모든 이미지 언디스토션 처리
    
    Args:
        input_folder: 입력 이미지 폴더
        output_folder: 출력 폴더 (None이면 input_folder/undistorted)
        camera_config: 카메라 설정 JSON 파일 (cam1_config.json 등)
        use_optimal_matrix: 최적 카메라 행렬 사용 여부
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"[ERROR] 입력 폴더가 존재하지 않습니다: {input_folder}")
        return
    
    # 출력 폴더 설정
    if output_folder is None:
        output_path = input_path / "undistorted"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 카메라 파라미터 로드
    camera_matrix = None
    dist_coeffs = None
    
    # 1. JSON 카메라 설정 파일 우선
    if camera_config and os.path.exists(camera_config):
        try:
            camera_matrix, dist_coeffs = load_calibration_from_json(camera_config)
            print(f"[INFO] 카메라 설정 로드: {camera_config}")
        except Exception as e:
            print(f"[ERROR] JSON 로드 실패: {e}")
    
    print(f"\n[INFO] Camera Matrix:\n{camera_matrix}")
    print(f"[INFO] Distortion Coefficients: {dist_coeffs}")
    print(f"\n[INFO] 입력 폴더: {input_path}")
    print(f"[INFO] 출력 폴더: {output_path}")
    print()
    
    # 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 이미지 파일 목록
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("[WARNING] 처리할 이미지가 없습니다.")
        return
    
    print(f"[INFO] 총 {len(image_files)}개 이미지 처리 시작...")
    print("-" * 50)
    
    success_count = 0
    fail_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        try:
            # 이미지 로드
            image = cv2.imread(str(img_file))
            
            if image is None:
                print(f"[FAIL] {img_file.name} - 이미지 로드 실패")
                fail_count += 1
                continue
            
            # 언디스토션
            undistorted = undistort_image(image, camera_matrix, dist_coeffs, use_optimal_matrix)
            
            # 저장
            output_file = output_path / f"{img_file.stem}_undistorted{img_file.suffix}"
            cv2.imwrite(str(output_file), undistorted)
            
            print(f"[{i}/{len(image_files)}] {img_file.name} -> {output_file.name} 완료")
            success_count += 1
            
        except Exception as e:
            print(f"[FAIL] {img_file.name} - {e}")
            fail_count += 1
    
    print("-" * 50)
    print(f"[완료] 성공: {success_count}, 실패: {fail_count}")
    print(f"[INFO] 결과 저장 위치: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="이미지 언디스토션 스크립트")
    parser.add_argument("-i", "--input_folder", help="입력 이미지 폴더 경로")
    parser.add_argument("-o", "--output", help="출력 폴더 경로 (기본: input_folder/undistorted)")
    parser.add_argument("-c", "--camera-config", default="config/cam3_config.json", help="카메라 설정 JSON 파일 경로 (calibration 포함)")
    parser.add_argument("-n", "--no-optimal", action="store_true", help="최적 카메라 행렬 사용 안함 (이미지 크롭될 수 있음)")
    
    args = parser.parse_args()
    
    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output,
        camera_config=args.camera_config,
        use_optimal_matrix=not args.no_optimal
    )


if __name__ == "__main__":
    main()

