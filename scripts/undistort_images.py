"""
이미지 언디스토션 스크립트
폴더 내 모든 이미지를 카메라 캘리브레이션 파라미터를 사용하여 왜곡 보정합니다.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# 기본 카메라 파라미터 (cam1)
DEFAULT_INTRINSIC = np.array([
    [3505.72, 0, 2140.16],
    [0, 3496.98, 1058.46],
    [0, 0, 1]
], dtype=np.float64)

DEFAULT_DISTORTION = np.array([-0.090755, 0.185086, 0.001742, 0.002785], dtype=np.float64)


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
                   intrinsic_file: str = None,
                   distortion_file: str = None,
                   use_optimal_matrix: bool = True):
    """
    폴더 내 모든 이미지 언디스토션 처리
    
    Args:
        input_folder: 입력 이미지 폴더
        output_folder: 출력 폴더 (None이면 input_folder/undistorted)
        intrinsic_file: Intrinsic 파라미터 파일 경로
        distortion_file: Distortion 파라미터 파일 경로
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
    if intrinsic_file and os.path.exists(intrinsic_file):
        camera_matrix = load_intrinsic(intrinsic_file)
        print(f"[INFO] Intrinsic 파라미터 로드: {intrinsic_file}")
    else:
        camera_matrix = DEFAULT_INTRINSIC
        print("[INFO] 기본 Intrinsic 파라미터 사용")
    
    if distortion_file and os.path.exists(distortion_file):
        dist_coeffs = load_distortion(distortion_file)
        print(f"[INFO] Distortion 파라미터 로드: {distortion_file}")
    else:
        dist_coeffs = DEFAULT_DISTORTION
        print("[INFO] 기본 Distortion 파라미터 사용")
    
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
    parser.add_argument("--input_folder", help="입력 이미지 폴더 경로")
    parser.add_argument("-o", "--output", help="출력 폴더 경로 (기본: input_folder/undistorted)")
    parser.add_argument("-i", "--intrinsic", 
                        default=r"C:\Users\user\Documents\cal_1202\cam1\cam1Intrinsic.txt",
                        help="Intrinsic 파라미터 파일 경로")
    parser.add_argument("-d", "--distortion",
                        default=r"C:\Users\user\Documents\cal_1202\cam1\cam1Distortion.txt", 
                        help="Distortion 파라미터 파일 경로")
    parser.add_argument("--no-optimal", action="store_true",
                        help="최적 카메라 행렬 사용 안함 (이미지 크롭될 수 있음)")
    
    args = parser.parse_args()
    
    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output,
        intrinsic_file=args.intrinsic,
        distortion_file=args.distortion,
        use_optimal_matrix=not args.no_optimal
    )


if __name__ == "__main__":
    main()

