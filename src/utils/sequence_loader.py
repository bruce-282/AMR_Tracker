"""
Sequence Loader Module

This module provides classes for loading different types of video sequences.
"""

import cv2
import glob
import os
import sys
from pathlib import Path
from typing import Optional, Union, List
from enum import Enum
import time

# Novitec Camera import
try:
    # submodule 경로 설정
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    novitec_src_path = project_root / "submodules" / "novitec_camera_module" / "src"

    if novitec_src_path.exists():
        novitec_src_str = str(novitec_src_path.resolve())
        if novitec_src_str not in sys.path:
            sys.path.insert(0, novitec_src_str)

        from crp_camera.cam.novitec.novitec_camera import NovitecCamera

        NOVITEC_AVAILABLE = True
    else:
        NovitecCamera = None
        NOVITEC_AVAILABLE = False
except ImportError:
    NovitecCamera = None
    NOVITEC_AVAILABLE = False


class LoaderMode(Enum):
    """Enumeration of available loader modes"""

    CAMERA_DEVICE = "camera_device"
    VIDEO_FILE = "video_file"
    IMAGE_SEQUENCE = "image_sequence"


class BaseLoader:
    """Base class for all sequence loaders"""

    def __init__(self):
        self.frame_number = 0
        self.is_connected = False

    def read(self):
        """Read next frame"""
        raise NotImplementedError

    def release(self):
        """Release resources"""
        raise NotImplementedError

    def is_opened(self):
        """Check if loader is opened"""
        raise NotImplementedError

    def check_connection(self) -> bool:
        """Check if loader is connected"""
        return self.is_connected

    def get_frame_number(self):
        return self.frame_number


class VideoFileLoader(BaseLoader):
    """Loader for video files"""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {file_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_connected = True

        print(f"[OK] Video file opened: {file_path}")
        print(f"  Total frames: {self.total_frames}, FPS: {self.fps:.2f}")

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_number += 1
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()

    def is_opened(self):
        return self.cap and self.cap.isOpened()

    def get_fps(self):
        return self.fps

    def get_total_frames(self):
        return self.total_frames


class ImageSequenceLoader(BaseLoader):
    """Loader for image sequences"""

    def __init__(self, sequence_path: str, fps: float = 30.0):
        super().__init__()
        self.sequence_path = Path(sequence_path)
        self.fps = fps
        self.frame_duration = 1.0 / fps

        # Find image files
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        self.image_files = []

        for ext in extensions:
            self.image_files.extend(glob.glob(str(self.sequence_path / ext)))
            self.image_files.extend(glob.glob(str(self.sequence_path / ext.upper())))

        self.image_files.sort()

        if not self.image_files:
            raise RuntimeError(f"No image files found in: {sequence_path}")

        self.current_index = 0
        self.last_frame_time = time.time()
        self.is_connected = True

        print(f"[OK] Image sequence opened: {sequence_path}")
        print(f"  Found {len(self.image_files)} images, FPS: {fps}")

    def read(self):
        if self.current_index >= len(self.image_files):
            return False, None

        # Frame rate control
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_duration:
            return False, None

        # Load image
        image_path = self.image_files[self.current_index]
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Warning: Could not load image: {image_path}")
            self.current_index += 1
            return False, None

        self.current_index += 1
        self.frame_number += 1
        self.last_frame_time = current_time

        return True, frame

    def release(self):
        pass

    def is_opened(self):
        return self.current_index < len(self.image_files)

    def get_fps(self):
        return self.fps

    def get_total_frames(self):
        return len(self.image_files)


class NovitecCameraLoader(BaseLoader):
    """Loader for Novitec industrial cameras"""

    def __init__(self, device_id: str, config: Optional[dict] = None):
        """
        Initialize Novitec camera loader.

        Args:
            device_id: Device ID - serial number of the camera
            config: Optional configuration dictionary for camera parameters
        """
        super().__init__()

        if not NOVITEC_AVAILABLE:
            raise RuntimeError(
                "Novitec Camera Module not available. Please ensure submodule is initialized."
            )

        self.device_id = device_id
        self.config = config or {}
        self.camera: Optional[NovitecCamera] = None
        self.initialized = False

        self._initialize()

    def _initialize(self):
        """Initialize Novitec camera"""

        # Create NovitecCamera instance
        self.camera = NovitecCamera(
            device_id=self.device_id,
            device_ip=None,  # Not used by Novitec
            config=self.config,
        )
        if not self.camera:
            raise RuntimeError(f"Failed to create Novitec camera: {self.device_id}")

        # Connect to camera
        if not self.camera.connect():
            raise RuntimeError(f"Failed to connect to Novitec camera: {self.device_id}")

        self.initialized = True
        self.is_connected = True
        print(f"[OK] Novitec camera initialized and streaming: {self.device_id}")

    def read(self):
        """
        Read frame from Novitec camera.

        Returns:
            tuple: (ret, frame) where ret is bool and frame is numpy array (BGR format)
        """
        if not self.initialized or not self.camera:
            return False, None

        try:
            # Capture image using NovitecCamera API
            data = self.camera.capture(output_formats=["image"])

            if not data or "image" not in data:
                return False, None

            frame = data["image"]

            # Ensure frame is in correct format (BGR for OpenCV)
            if frame is not None and len(frame.shape) == 3:
                self.frame_number += 1
                return True, frame
            else:
                return False, None

        except Exception as e:
            print(f"Error reading from Novitec camera: {e}")
            return False, None

    def release(self):
        """Release Novitec camera resources"""
        try:
            if self.camera:
                # Stop stream if running
                if self.camera._is_streaming:
                    self.camera.stop_stream()
                # Disconnect
                self.camera.disconnect()
                self.camera = None
                print("[OK] Novitec camera released")

            self.initialized = False
            self.is_connected = False

        except Exception as e:
            print(f"Error releasing Novitec camera: {e}")

    def is_opened(self):
        """Check if camera is opened and connected"""
        if not self.camera:
            return False
        return self.initialized and self.camera.check_connection()


def create_sequence_loader(
    source: Union[str, int], 
    fps: float = 30.0, 
    loader_mode: str = "auto",
    config: Optional[dict] = None
) -> Optional[BaseLoader]:
    """
    Create appropriate sequence loader based on source and mode

    Args:
        source: Video source (file path, device ID, or sequence path)
        fps: Frame rate for image sequences
        loader_mode: Loader mode ("auto", "camera_device", "video_file", "image_sequence")
        config: Optional configuration dictionary (for camera loaders)

    Returns:
        Appropriate loader instance or None if failed
    """
    try:
        if loader_mode == "camera":
            return create_camera_device_loader(source=source, config=config)
        elif loader_mode == "video":
            return create_video_file_loader(source=source)
        elif loader_mode == "sequence":
            return create_image_sequence_loader(source=source, fps=fps)
        else:
            print(f"Error: Unknown loader mode: {loader_mode}")
            return None

    except Exception as e:
        print(f"Error creating sequence loader: {e}")
        return None


def create_camera_device_loader(
    source: str, config: Optional[dict] = None
) -> Optional[BaseLoader]:
    """Create camera device loader with Novitec fallback"""

    try:
        loader = NovitecCameraLoader(device_id=source, config=config)
        print("[OK] Novitec camera loader created")
        return loader
    except Exception as e:
        print(f"Novitec camera loader failed: {e}")
        return None


def create_video_file_loader(source: str) -> Optional[BaseLoader]:
    """Create video file loader"""
    try:
        loader = VideoFileLoader(file_path=source)
        print("[OK] Video file loader created")
        return loader
    except Exception as e:
        print(f"Video file loader failed: {e}")
        return None

def create_image_sequence_loader(
    source: str, fps: float = 30.0
) -> Optional[BaseLoader]:
    """Create image sequence loader"""
    try:
        loader = ImageSequenceLoader(sequence_path=source, fps=fps)
        print("[OK] Image sequence loader created")
        return loader
    except Exception as e:
        print(f"Error creating image sequence loader: {e}")
        return None

