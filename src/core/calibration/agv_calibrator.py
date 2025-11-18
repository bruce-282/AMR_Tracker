"""Main application for AGV measurement system."""

import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

# Import all modules
from config import SystemConfig
from src.core.calibration.camera_calibrator import CameraCalibrator
from src.core.calibration.homography_calibrator import HomographyCalibrator
from src.core.measurement.size_measurement import SizeMeasurement
from src.core.tracking.speed_tracker import SpeedTracker
from src.visualization.display import Visualizer


class AGVCalibrator:
    """
    AGV 캘리브레이션 및 측정 시스템

    카메라 캘리브레이션부터 정밀 측정까지의 완전한 파이프라인을 제공합니다.
    """

    def __init__(self, config: SystemConfig):
        """
        AGV 캘리브레이션 시스템 초기화

        Args:
            config: 시스템 설정
        """
        self.config = config
        self.calibration_data = None

        # 체스보드 크기를 내부 코너로 변환 (8,7) -> (7,6)
        checkerboard_size = self.config.calibration.checkerboard_size
        self.inner_corners = (checkerboard_size[0] - 1, checkerboard_size[1] - 1)

        self.camera_calibrator = CameraCalibrator(
            self.inner_corners,
            self.config.calibration.square_size,
        )

        self.homography_calibrator = None
        self.detector = None
        self.size_measurement = None
        self.speed_tracker = None
        self.visualizer = None

    def run_calibration(self, image_path: str = None):
        """
        완전한 캘리브레이션 절차 실행

        Args:
            image_path: 이미지 폴더 경로 (None이면 실시간 카메라 사용)
        """
        print("=" * 50)
        print("AGV 캘리브레이션 시스템 - 캘리브레이션 모드")
        print("=" * 50)

        # Step 1: 카메라 캘리브레이션
        print("\n[Step 1] 카메라 내부 캘리브레이션")
        print("-" * 40)
        self._calibrate_camera(image_path)

        # Step 2: 지면 평면 캘리브레이션
        print("\n[Step 2] 지면 평면 캘리브레이션")
        print("-" * 40)
        self._calibrate_ground_plane(image_path)

        # 캘리브레이션 저장
        self._save_calibration()
        print("\n✓ 캘리브레이션 완료 및 저장!")

    def _calibrate_camera(self, image_path: str = None):
        """카메라 내부 캘리브레이션 수행"""
        # camera_calibrator는 __init__에서 이미 초기화됨

        if image_path:
            # 이미지 폴더에서 로드
            self._calibrate_from_images(image_path)
        else:
            # 실시간 카메라 사용
            raise ValueError("실시간 카메라 사용은 지원하지 않습니다")

    def _calibrate_from_images(self, image_path: str):
        """이미지 폴더에서 캘리브레이션 수행"""
        import glob
        import os

        # camera_calibrator가 초기화되었는지 확인
        if self.camera_calibrator is None:
            print("❌ camera_calibrator가 초기화되지 않았습니다!")
            return

        # 이미지 파일 찾기
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))

        # 중복 제거
        image_files = list(set(image_files))
        image_files.sort()

        print(f"발견된 이미지: {len(image_files)}개")

        if len(image_files) == 0:
            print("❌ 이미지 파일을 찾을 수 없습니다!")
            return

        # 이미지 처리
        processed_count = 0
        last_image = None

        for i, image_file in enumerate(image_files):
            print(f"처리 중: {os.path.basename(image_file)} ({i+1}/{len(image_files)})")

            image = cv2.imread(image_file)
            if image is None:
                print(f"❌ 이미지 로드 실패: {image_file}")
                continue

            # 이미지 크기 조정
            height, width = image.shape[:2]

            # 체스보드 감지 시도
            success = self.camera_calibrator.find_corners(image)
            if success:
                print(f"✓ 체스보드 감지됨: {os.path.basename(image_file)}")
                last_image = image  # 마지막 성공한 이미지 저장
            else:
                print(f"❌ 체스보드 감지 실패: {os.path.basename(image_file)}")
            if width > 1920 or height > 1080:
                scale = min(1920 / width, 1080 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_image = cv2.resize(image, (new_width, new_height))
            # 이미지 표시
            cv2.imshow("Calibration Image", resized_image)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break

            processed_count += 1

        cv2.destroyAllWindows()

        # 캘리브레이션 수행
        if processed_count > 3 and last_image is not None:
            # 이미지 크기 가져오기 (마지막 성공한 이미지 기준)
            height, width = last_image.shape[:2]
            if self.camera_calibrator.calibrate((width, height)):
                print("✓ 카메라 캘리브레이션 성공!")
            else:
                print("❌ 카메라 캘리브레이션 실패!")
        else:
            print("❌ 캘리브레이션 이미지가 3개 미만입니다.")

    def _calibrate_from_camera(self):
        """실시간 카메라에서 캘리브레이션 수행"""
        print(
            f"체스보드 패턴을 {self.config.calibration.num_calibration_images}개 "
            f"다양한 각도에서 촬영해주세요."
        )
        print("SPACE: 촬영, ESC: 조기 종료")

        cap = cv2.VideoCapture(0)
        captured_images = []

        while len(captured_images) < self.config.calibration.num_calibration_images:
            ret, frame = cap.read()
            if not ret:
                continue

            # 프레임 표시
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"촬영됨: {len(captured_images)}/{self.config.calibration.num_calibration_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("캘리브레이션 - SPACE로 촬영", display_frame)

            key = cv2.waitKey(1)
            if key == ord(" "):
                # 캘리브레이션 이미지 추가 시도
                if self.camera_calibrator.find_corners(frame):
                    captured_images.append(frame)
                    print(f"✓ 이미지 {len(captured_images)} 촬영 성공")
                else:
                    print("✗ 체스보드를 찾을 수 없습니다. 다시 시도해주세요")
            elif key == 27:  # ESC
                if len(captured_images) >= 3:
                    break
                else:
                    print("캘리브레이션을 위해 최소 3개 이미지가 필요합니다")

        cap.release()
        cv2.destroyAllWindows()

        # 캘리브레이션 수행
        h, w = captured_images[0].shape[:2]
        print(f"캘리브레이션 이미지 크기: {w}x{h}")
        calibration_result = self.camera_calibrator.calibrate((w, h))

        print(f"\n✓ 카메라 캘리브레이션 성공!")
        print(f"  재투영 오차: {calibration_result['reprojection_error']:.3f} 픽셀")

    def _calibrate_ground_plane(self, image_path: str = None):
        """지면 평면 호모그래피 캘리브레이션 수행"""
        if self.camera_calibrator is None:
            raise ValueError("먼저 카메라를 캘리브레이션해야 합니다")

        if image_path:
            # 이미지 폴더에서 지면 이미지 찾기
            ground_image = self._find_ground_image(image_path)
        else:
            # 실시간 카메라 사용
            raise ValueError("실시간 카메라 사용은 지원하지 않습니다")

        if ground_image is None:
            print("❌ 지면 이미지를 찾을 수 없습니다!")
            return

        self.homography_calibrator = HomographyCalibrator(
            self.camera_calibrator.camera_matrix, self.camera_calibrator.dist_coeffs
        )

        homography_result = self.homography_calibrator.calibrate_ground_plane(
            ground_image,
            self.inner_corners,
            self.config.calibration.square_size,
        )

        print(f"\n✓ 지면 평면 캘리브레이션 성공!")
        print(f"  픽셀 크기: {homography_result['pixel_size']:.3f} mm/pixel")
        print(f"  인라이어 수: {homography_result['inliers']}")

    def _find_ground_image(self, image_path: str):
        """이미지 폴더에서 지면 이미지 찾기"""
        import glob
        import os

        # 이미지 파일 찾기
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))

        # 중복 제거
        image_files = list(set(image_files))
        image_files.sort()

        print(f"지면 이미지 찾기: {len(image_files)}개 이미지 중에서...")

        for image_file in image_files:
            print(f"확인 중: {os.path.basename(image_file)}")

            image = cv2.imread(image_file)
            if image is None:
                continue

            # 원본 이미지 크기 사용 (리사이즈 제거)

            # 체스보드 감지 시도
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.inner_corners, None)

            if ret:
                print(f"✓ 체스보드 발견: {os.path.basename(image_file)}")
                return image
            else:
                print(f"❌ 체스보드 감지 실패: {os.path.basename(image_file)}")

        return None

    def _capture_ground_image(self):
        """실시간 카메라에서 지면 이미지 캡처"""
        print("\n체스보드를 지면에 놓고 SPACE를 눌러 촬영하세요")

        cap = cv2.VideoCapture(0)
        ground_image = None

        while ground_image is None:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow("지면 평면 캘리브레이션 - SPACE를 누르세요", frame)

            if cv2.waitKey(1) == ord(" "):
                ground_image = frame
                break

        cap.release()
        cv2.destroyAllWindows()
        return ground_image

    def _save_calibration(self):
        """캘리브레이션 데이터 저장"""
        calibration_data = {
            "camera_matrix": self.camera_calibrator.camera_matrix.tolist(),
            "dist_coeffs": self.camera_calibrator.dist_coeffs.tolist(),
            "homography": self.homography_calibrator.homography.tolist(),
            "pixel_size": float(self.homography_calibrator.pixel_size),
            "camera_height": 0.0,  # AGV 위에 캘판을 놓았으므로 높이 0
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 캘리브레이션 파일 저장
        calibration_file = Path(self.config.calibration.calibration_data_path)
        with open(calibration_file, "w") as f:
            json.dump(calibration_data, f, indent=2)

        print(f"✓ 캘리브레이션 데이터 저장됨: {calibration_file}")

        # 캘리브레이션 데이터를 인스턴스에 저장
        self.calibration_data = calibration_data

    def load_calibration(self, calibration_file: str = "calibration_data.json"):
        """저장된 캘리브레이션 데이터 로드"""
        try:
            with open(calibration_file, "r") as f:
                self.calibration_data = json.load(f)

            # NumPy 배열로 변환
            self.calibration_data["camera_matrix"] = np.array(
                self.calibration_data["camera_matrix"]
            )
            self.calibration_data["dist_coeffs"] = np.array(
                self.calibration_data["dist_coeffs"]
            )
            self.calibration_data["homography"] = np.array(
                self.calibration_data["homography"]
            )

            print(f"✓ 캘리브레이션 데이터 로드됨: {calibration_file}")
            return True

        except FileNotFoundError:
            print(f"✗ 캘리브레이션 파일을 찾을 수 없습니다: {calibration_file}")
            return False
        except Exception as e:
            print(f"✗ 캘리브레이션 데이터 로드 실패: {e}")
            return False

    def initialize_components(self):
        """측정을 위한 모든 구성 요소 초기화"""
        if self.calibration_data is None:
            if not self.load_calibration(self.config.calibration.calibration_data_path):
                raise ValueError("캘리브레이션 데이터가 필요합니다")

        # AGV 감지기는 YOLO를 사용하므로 별도 초기화 불필요

        # 크기 측정 초기화
        self.size_measurement = SizeMeasurement(
            homography=self.calibration_data["homography"],
            camera_height=None,
        )

        # 속도 추적기 초기화
        self.speed_tracker = SpeedTracker(
            max_tracking_distance=self.config.measurement.max_tracking_distance
        )

        # 시각화 도구 초기화
        self.visualizer = Visualizer(homography=self.calibration_data["homography"])

        print("✓ 모든 구성 요소 초기화 완료")

    def run_measurement(self, source: Optional[str] = None):
        """
        비디오 스트림에서 측정 실행

        Args:
            source: 비디오 소스 (파일 경로 또는 카메라 인덱스)
        """
        print("=" * 50)
        print("AGV 캘리브레이션 시스템 - 측정 모드")
        print("=" * 50)

        # 구성 요소 초기화
        self.initialize_components()

        # 비디오 소스 열기
        if source is None:
            cap = cv2.VideoCapture(0)
        elif source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"비디오 소스를 열 수 없습니다: {source}")

        # 비디오 녹화용 (선택사항)
        out = None
        if self.config.record_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(
                self.config.output_video_path, fourcc, fps, (width, height)
            )

        frame_number = 0
        print("\n측정 시작... 'q'로 종료, 's'로 스냅샷 저장")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.time()

            # AGV 감지
            detections = self.detector.detect(frame, frame_number, timestamp)

            # 크기 측정
            measurements = []
            for detection in detections:
                # 크기 기반으로 높이 추정 (단순화)
                height = self._estimate_agv_height(detection)

                # 측정
                measurement = self.size_measurement.measure(detection, height)
                measurement["frame_number"] = frame_number
                measurements.append(measurement)

            # 속도 추적
            measurements = self.speed_tracker.update(measurements)

            # 시각화
            vis_frame = self.visualizer.draw_detections(frame, detections, measurements)

            # 조감도 표시
            # bird_eye = self.visualizer.create_bird_eye_view(measurements)

            # 결과 표시
            cv2.imshow("AGV 측정 시스템", vis_frame)
            # cv2.imshow("조감도", bird_eye)

            # 측정 결과 출력
            for i, m in enumerate(measurements):
                if frame_number % 30 == 0:  # 1초마다 출력
                    print(f"\nAGV {m.get('track_id', i)}:")
                    print(f"  크기: {m['width']:.0f} x {m['height']:.0f} mm")
                    print(f"  속도: {m.get('speed', 0):.1f} mm/s")
                    print(
                        f"  위치: ({m['center_world'][0]:.0f}, "
                        f"{m['center_world'][1]:.0f}) mm"
                    )

            # 비디오 녹화
            if out is not None:
                out.write(vis_frame)

            # 키 입력 처리
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                # 스냅샷 저장
                snapshot_path = f"snapshot_{frame_number}.jpg"
                cv2.imwrite(snapshot_path, vis_frame)
                print(f"스냅샷 저장됨: {snapshot_path}")

            frame_number += 1

        # 정리
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        print(f"\n✓ 측정 완료. 처리된 프레임: {frame_number}")

    def _estimate_agv_height(self, detection) -> Optional[float]:
        """AGV 높이 추정 (단순화된 방법)"""
        # 실제로는 더 정교한 방법이 필요
        # 여기서는 고정값 사용
        return None  # mm


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AGV 캘리브레이션 및 측정 시스템")
    parser.add_argument(
        "--config", default="tracker_config.json", help="설정 파일 경로"
    )
    parser.add_argument(
        "--mode",
        choices=["calibration", "measurement", "interactive"],
        default="interactive",
        help="실행 모드",
    )
    parser.add_argument("--source", help="비디오 소스 (측정 모드에서)")

    args = parser.parse_args()

    # 설정 로드
    config = SystemConfig.load(args.config)

    # 시스템 초기화
    calibrator = AGVCalibrator(config)

    # 모드에 따라 실행
    if args.mode == "calibration":
        calibrator.run_calibration()
    elif args.mode == "measurement":
        calibrator.run_measurement(args.source)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        return


if __name__ == "__main__":
    main()
