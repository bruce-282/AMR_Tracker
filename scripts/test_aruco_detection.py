import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from scripts.pixel_distance_mapper import detect_aruco_board_center

# Argument parser
parser = argparse.ArgumentParser(description="ArUco 마커 검출 및 보정")
parser.add_argument("--image", type=str, default="scripts/novitec_manual_1765263482.png", help="입력 이미지 경로")
parser.add_argument("--camera-config", type=str, default="config/cam1_config.json", help="카메라 설정 파일 경로")
parser.add_argument("--marker-size", type=float, help="ArUco 마커의 실제 크기 (mm) - 전체 마커의 한 변 길이")
parser.add_argument("--cell-size", type=float, help="ArUco 마커 셀의 실제 크기 (mm) - 한 셀의 크기")
parser.add_argument("--marker-type", type=str, default="4x4", choices=["4x4", "5x5", "6x6", "7x7"], help="마커 타입 (기본: 4x4)")
args = parser.parse_args()

# 이미지 경로
image_path = args.image

# 카메라 설정 로드
camera_config_path = args.camera_config

print(f"이미지 로드: {image_path}")
image = cv2.imread(image_path)
if image is None:
    print(f"오류: 이미지를 읽을 수 없습니다: {image_path}")
    exit(1)

print(f"이미지 크기: {image.shape[1]}x{image.shape[0]}")

# 카메라 설정 로드
try:
    with open(camera_config_path, 'r', encoding='utf-8') as f:
        camera_config = json.load(f)
    
    calibration = camera_config.get("calibration", {})
    camera_matrix = np.array(calibration.get("CameraMatrix", []), dtype=np.float64)
    dist_coeffs_list = calibration.get("DistortionCoefficients", [])
    
    if dist_coeffs_list and isinstance(dist_coeffs_list[0], list):
        dist_coeffs = np.array(dist_coeffs_list[0], dtype=np.float64)
    else:
        dist_coeffs = np.array(dist_coeffs_list, dtype=np.float64)
    
    print(f"\n카메라 설정 로드: {camera_config_path}")
    print(f"  Camera Matrix shape: {camera_matrix.shape}")
    print(f"  Distortion Coefficients shape: {dist_coeffs.shape}")
    
except Exception as e:
    print(f"카메라 설정 로드 실패: {e}")
    print("기본값 사용...")
    # 기본값 (임시)
    h, w = image.shape[:2]
    camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)

# 여러 ArUco 딕셔너리 시도
aruco_dicts = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
    ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
]

print("\n=== ArUco 마커 검출 시도 ===")
found = False

for dict_name, dict_id in aruco_dicts:
    print(f"\n딕셔너리: {dict_name}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    
    # 원본 이미지에서 먼저 검출
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 마커 검출 (OpenCV 버전에 따라 다른 API 사용)
    try:
        # OpenCV 4.7+ 방식
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, rejected = detector.detectMarkers(gray_original)
    except AttributeError:
        try:
            # OpenCV 4.5-4.6 방식
            aruco_params = cv2.aruco.DetectorParameters()
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_original, aruco_dict, parameters=aruco_params)
        except AttributeError:
            # OpenCV 3.x 방식
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_original, aruco_dict, parameters=aruco_params)
    
    # Undistortion
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    
    # 코너 좌표를 undistortion 변환
    if ids is not None and len(ids) > 0:
        corners_undistorted = []
        for corner in corners:
            # corner는 (1, 4, 2) 형태
            corner_pts = corner[0].reshape(-1, 1, 2).astype(np.float32)  # (4, 1, 2)
            # Undistort points
            corner_undist = cv2.undistortPoints(corner_pts, camera_matrix, dist_coeffs, P=camera_matrix)
            corners_undistorted.append(corner_undist.reshape(1, 4, 2))
        corners = corners_undistorted
    
    if ids is not None and len(ids) > 0:
        print(f"  ✓ {len(ids)}개 마커 검출됨!")
        found = True
        
        # Undistortion된 이미지에 표시
        result_image = undistorted.copy()
        cv2.aruco.drawDetectedMarkers(result_image, corners, ids)
        
        # 마커 크기 계산 및 보정
        for i, corner in enumerate(corners):
            corner_pts = corner[0]  # (4, 2) 형태
            center = np.mean(corner_pts, axis=0)
            marker_id = ids[i][0]
            
            # 마커의 픽셀 크기 계산 (4개 코너로부터)
            # 마커는 정사각형이므로 대각선 길이 또는 변 길이 측정
            side1 = np.linalg.norm(corner_pts[0] - corner_pts[1])  # 상단 변
            side2 = np.linalg.norm(corner_pts[1] - corner_pts[2])  # 우측 변
            side3 = np.linalg.norm(corner_pts[2] - corner_pts[3])  # 하단 변
            side4 = np.linalg.norm(corner_pts[3] - corner_pts[0])  # 좌측 변
            
            avg_side_pixels = np.mean([side1, side2, side3, side4])
            
            print(f"\n  마커 ID {marker_id}:")
            print(f"    중심 (undistorted): ({center[0]:.1f}, {center[1]:.1f})")
            print(f"    픽셀 크기 (평균 변 길이): {avg_side_pixels:.2f} px")
            
            # 실제 크기 입력이 있으면 스케일 계산
            if args.marker_size is not None:
                pixel_size = args.marker_size / avg_side_pixels  # mm/pixel
                print(f"    실제 마커 크기: {args.marker_size:.2f} mm")
                print(f"    계산된 Pixel Size: {pixel_size:.6f} mm/pixel")
            elif args.cell_size is not None:
                # 셀 크기로부터 전체 마커 크기 계산
                if args.marker_type == "4x4":
                    num_cells = 4
                elif args.marker_type == "5x5":
                    num_cells = 5
                elif args.marker_type == "6x6":
                    num_cells = 6
                elif args.marker_type == "7x7":
                    num_cells = 7
                else:
                    num_cells = 4
                
                actual_marker_size = args.cell_size * num_cells
                pixel_size = actual_marker_size / avg_side_pixels  # mm/pixel
                print(f"    셀 크기: {args.cell_size:.2f} mm")
                print(f"    마커 타입: {args.marker_type} ({num_cells}x{num_cells})")
                print(f"    계산된 전체 마커 크기: {actual_marker_size:.2f} mm")
                print(f"    계산된 Pixel Size: {pixel_size:.6f} mm/pixel")
            else:
                print(f"    ⚠ 실제 크기 정보 없음 (--marker-size 또는 --cell-size 입력 필요)")
            
            # 결과 이미지에 표시
            cv2.circle(result_image, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
            cv2.putText(result_image, f"ID:{marker_id}", 
                       (int(center[0])+15, int(center[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 마커 크기 정보 표시
            if args.marker_size is not None or args.cell_size is not None:
                info_text = f"{avg_side_pixels:.1f}px"
                cv2.putText(result_image, info_text,
                           (int(center[0])+15, int(center[1])+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 원본 이미지에도 표시 (비교용)
        result_image_original = image.copy()
        cv2.aruco.drawDetectedMarkers(result_image_original, corners, ids)
        for i, corner in enumerate(corners):
            corner_pts = corner[0]
            center = np.mean(corner_pts, axis=0)
            marker_id = ids[i][0]
            cv2.circle(result_image_original, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
            cv2.putText(result_image_original, f"ID:{marker_id}", 
                       (int(center[0])+15, int(center[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 결과 이미지 저장
        output_path_undistorted = f"scripts/aruco_result_{dict_name}_undistorted.png"
        output_path_original = f"scripts/aruco_result_{dict_name}_original.png"
        cv2.imwrite(output_path_undistorted, result_image)
        cv2.imwrite(output_path_original, result_image_original)
        print(f"\n  결과 이미지 저장:")
        print(f"    Undistorted: {output_path_undistorted}")
        print(f"    Original: {output_path_original}")
        
        # 보드 중심 계산 (undistorted 좌표)
        centers = [np.mean(corner[0], axis=0) for corner in corners]
        board_center = np.mean(centers, axis=0)
        print(f"\n  보드 중심 (undistorted): ({board_center[0]:.1f}, {board_center[1]:.1f})")
        
        break
    else:
        print(f"  ✗ 마커 없음")

if not found:
    print("\n⚠ 모든 딕셔너리에서 마커를 찾을 수 없습니다.")
    print("이미지에 ArUco 마커가 없거나, 다른 형식의 마커일 수 있습니다.")

