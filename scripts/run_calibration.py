#!/usr/bin/env python3
"""
AGV 캘리브레이션 실행 스크립트
"""

import sys
import argparse
from pathlib import Path
from config import SystemConfig
from src.core.calibration.agv_calibrator import AGVCalibrator


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="AGV 캘리브레이션 실행")
    parser.add_argument(
        "--image-path",
        default=None,
        help="이미지 폴더 경로 (지정하지 않으면 실시간 카메라 사용)",
    )
    parser.add_argument(
        "--config", default="tracker_config.json", help="설정 파일 경로"
    )

    args = parser.parse_args()

    # 설정 로드
    import json

    with open(args.config, "r") as f:
        config_data = json.load(f)

    # SystemConfig 객체 생성
    config = SystemConfig()
    config.calibration.checkerboard_size = tuple(
        config_data["calibration"]["checkerboard_size"]
    )
    config.calibration.square_size = config_data["calibration"]["square_size"]
    config.calibration.num_calibration_images = config_data["calibration"][
        "num_calibration_images"
    ]
    config.calibration.camera_height = config_data["calibration"]["camera_height"]
    config.calibration.calibration_data_path = config_data["calibration"]["calibration_data_path"]

    print(f"설정 로드됨: {config.calibration.checkerboard_size}")

    # AGV 캘리브레이터 초기화
    calibrator = AGVCalibrator(config)

    # 캘리브레이션 실행
    calibrator.run_calibration(args.image_path)


if __name__ == "__main__":
    main()
