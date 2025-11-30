import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Fix OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

logger = logging.getLogger(__name__)
from src.utils.sequence_loader import (
    create_sequence_loader,
    create_camera_device_loader,
    create_video_file_loader,
    create_image_sequence_loader,
)

# Import config first (always available)
from config import SystemConfig

# Import tracking modules
from src.core.amr_tracker import EnhancedAMRTracker


def load_execution_preset(config_path: str, preset_name: str) -> dict:
    """Load execution preset from config file"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "execution" in config and "presets" in config["execution"]:
            presets = config["execution"]["presets"]
            if preset_name in presets:
                logger.info(f"Loaded preset: {preset_name}")
                return presets[preset_name]
            else:
                available_presets = list(presets.keys())
                logger.warning(
                    f"Preset '{preset_name}' not found. Available: {available_presets}"
                )
                return {}
        else:
            logger.warning("No execution presets found in config file")
            return {}
    except Exception as e:
        logger.warning(f"Error loading preset: {e}")
        return {}


def apply_preset_to_args(args: argparse.Namespace, preset: dict) -> argparse.Namespace:
    """Apply preset values to args if not explicitly set"""
    if not preset:
        return args

    # Apply preset values only if args are not explicitly set

    # Handle camera configuration in new format (camera_1, camera_2, camera_3)
    if not hasattr(args, "camera_configs"):
        args.camera_configs = {}
    if not hasattr(args, "camera_measurements"):
        args.camera_measurements = {}

    if "camera_1" in preset and isinstance(preset["camera_1"], dict):
        if "id" in preset["camera_1"]:
            args.camera_1 = preset["camera_1"]["id"]
        if "config" in preset["camera_1"]:
            args.camera_configs["1"] = preset["camera_1"]["config"]
        if "measurement" in preset["camera_1"]:
            args.camera_measurements["1"] = preset["camera_1"]["measurement"]

    if "camera_2" in preset and isinstance(preset["camera_2"], dict):
        if "id" in preset["camera_2"]:
            args.camera_2 = preset["camera_2"]["id"]
        if "config" in preset["camera_2"]:
            args.camera_configs["2"] = preset["camera_2"]["config"]
        if "measurement" in preset["camera_2"]:
            args.camera_measurements["2"] = preset["camera_2"]["measurement"]

    if "camera_3" in preset and isinstance(preset["camera_3"], dict):
        if "id" in preset["camera_3"]:
            args.camera_3 = preset["camera_3"]["id"]
        if "config" in preset["camera_3"]:
            args.camera_configs["3"] = preset["camera_3"]["config"]
        if "measurement" in preset["camera_3"]:
            args.camera_measurements["3"] = preset["camera_3"]["measurement"]

    return args


def main():
    """Enhanced main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced AMR Tracking System")

    parser.add_argument(
        "--config",
        default="tracker_config.json",
        help="Configuration file path (enhanced mode only)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        help="Load execution preset from config file (e.g., 'camera_tracking', 'video_tracking', 'sequence_tracking')",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for image sequences (default: 30)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    args = parser.parse_args()

    # 로깅 설정 (명령줄 인자로 레벨 설정)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(log_level)

    # Load preset if provided
    if args.preset:
        preset_config = load_execution_preset(args.config, args.preset)
        args = apply_preset_to_args(args, preset_config)

    # Load configuration for pixel_size
    # pixel_size = 1.0  # default value (1 pixel = 1mm)
    if Path(args.config).exists():
        try:
            config = SystemConfig.load(args.config)
        except Exception as e:
            logger.warning(f"Error loading config, using default: {e}")
            config = None
    else:
        logger.warning(f"Config file not found: {args.config}")
        config = None

    logger.info("Running in enhanced mode...")

    # Get preset configuration if available
    loader_mode = None
    source = None
    fps = args.fps
    pixel_size = 1.0

    if config and config.execution:
        # Get preset name from args or config
        preset_name = args.preset
        if not preset_name and config.execution.get("use_preset"):
            preset_name = config.execution.get("use_preset")

        if preset_name and config.execution.get("presets"):
            presets = config.execution.get("presets", {})
            preset = presets.get(preset_name, {})
            if preset:
                loader_mode = preset.get("loader_mode", "auto")
                if "camera_1" in preset and isinstance(preset["camera_1"], dict):
                    source = preset["camera_1"].get("id")
                    if "measurement" in preset["camera_1"]:
                        measurement = preset["camera_1"]["measurement"]
                        pixel_size = measurement.get("pixel_size", 1.0)
                        fps = measurement.get("fps", args.fps)
                logger.info(
                    f"Using preset: {preset_name} (loader_mode={loader_mode}, source={source})"
                )

    # Fallback to config.measurement.pixel_size if not set from preset
    if pixel_size == 1.0 and config:
        try:
            pixel_size = config.measurement.pixel_size
        except AttributeError:
            pass

    # Fallback to args.camera_1 if source not set
    if not source and hasattr(args, "camera_1"):
        source = args.camera_1

    if not source:
        logger.error("No source specified. Please provide --preset or --camera_1")
        return

    run_enhanced_mode(
        args,
        source,
        loader_mode or "auto",
        fps,
        pixel_size,
    )


def run_enhanced_mode(args, source, loader_mode="auto", fps=30, pixel_size=1.0):
    """Run enhanced AMR system"""
    # Load configuration
    config = None
    if Path(args.config).exists():
        try:
            config = SystemConfig.load(args.config)
            logger.info(f"Configuration loaded from {args.config}")
        except Exception as e:
            logger.warning(f"Error loading config: {e}")

    # Initialize enhanced system
    amr_tracker = EnhancedAMRTracker(
        config=config,
        pixel_size=pixel_size,
    )

    # Create sequence loader based on mode
    if loader_mode == "camera":
        cap = create_camera_device_loader(source)
    elif loader_mode == "video":
        cap = create_video_file_loader(source)
    elif loader_mode == "sequence":
        cap = create_image_sequence_loader(source, fps)
    else:  # auto mode
        cap = create_sequence_loader(source, fps)

    if cap is None:
        logger.error("Cannot create sequence loader")
        return

    frame_number = 0
    logger.info("AMR System running... Press 'q' to quit, 's' to save snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = amr_tracker.detect_objects(
            frame=frame, frame_number=frame_number, timestamp=time.time()
        )

        # Track objects
        tracking_results = amr_tracker.track_objects(frame, detections, frame_number)

        # Log tracking data
        # for result in tracking_results:
        #     amr_tracker.data_logger.log_tracking_result(frame_number, result)

        # Visualize results
        vis_frame = amr_tracker.visualize_results(frame, detections, tracking_results)

        # Add frame info
        if hasattr(cap, "frame_number"):  # Image sequence
            # 이미지 크기에 따라 텍스트 크기 동적 조정
            height, width = vis_frame.shape[:2]
            font_scale = max(0.5, min(2.0, width / 800))  # 800px 기준으로 스케일링
            thickness = max(1, int(font_scale * 2))

        # 창 크기 조절 (화면이 너무 클 때)
        height, width = vis_frame.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280 / width, 720 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            vis_frame_window = cv2.resize(vis_frame, (new_width, new_height))
        else:
            vis_frame_window = vis_frame

        # Show results
        cv2.imshow("AMR Tracking", vis_frame_window)

        # Handle key press
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"snapshot_{frame_number:06d}.png", vis_frame)
            logger.info(f"Snapshot saved: snapshot_{frame_number:06d}.png")

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"AMR System completed. Processed {frame_number} frames")


if __name__ == "__main__":
    main()
