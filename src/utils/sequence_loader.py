"""
Sequence Loader Module

This module provides classes for loading different types of video sequences.
"""

import cv2
import glob
import os
from pathlib import Path
from typing import Optional, Union, List
from enum import Enum
import time

# Try to import Novitec camera module
try:
    from .novitec_camera_loader import NOVITEC_AVAILABLE, nvt

    if NOVITEC_AVAILABLE:
        print("✓ Novitec Camera Module available")
    else:
        print("⚠ Novitec Camera Module not available")
        print("  Using fallback camera loader")
except ImportError as e:
    NOVITEC_AVAILABLE = False
    print(f"⚠ Novitec Camera Module not available: {e}")
    print("  Using fallback camera loader")


class LoaderMode(Enum):
    """Enumeration of available loader modes"""

    CAMERA_DEVICE = "camera_device"
    VIDEO_FILE = "video_file"
    IMAGE_SEQUENCE = "image_sequence"


class BaseLoader:
    """Base class for all sequence loaders"""

    def __init__(self):
        self.frame_number = 0

    def read(self):
        """Read next frame"""
        raise NotImplementedError

    def release(self):
        """Release resources"""
        raise NotImplementedError

    def is_opened(self):
        """Check if loader is opened"""
        raise NotImplementedError


class CameraDeviceLoader(BaseLoader):
    """Loader for camera devices (webcam, USB camera, etc.)"""

    def __init__(self, device_id: int = 0):
        super().__init__()
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device_id}")

        print(f"✓ Camera device {device_id} opened")

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

        print(f"✓ Video file opened: {file_path}")
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

        print(f"✓ Image sequence opened: {sequence_path}")
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


class NovitecCameraLoader(BaseLoader):
    """Loader for Novitec industrial cameras"""

    def __init__(self, device_id: int = 0):
        super().__init__()
        if not NOVITEC_AVAILABLE:
            raise RuntimeError("Novitec Camera Module not available")

        self.device_id = device_id
        self.device_manager = None
        self.camera = None
        self.initialized = False

        self._initialize()

    def _initialize(self):
        """Initialize Novitec camera"""
        try:
            # Initialize device manager
            self.device_manager = nvt.DeviceManager()
            print("✓ Novitec DeviceManager initialized")

            # Update device list
            err = self.device_manager.Update()
            if err != nvt.NVT_OK:
                print(f"DeviceManager Update 실패: {err}")
                raise RuntimeError(f"Failed to update device list: {err}")
            print("✓ DeviceManager updated successfully")

            # Get available devices
            device_count = self.device_manager.GetDeviceCount()
            print(f"✓ Found {device_count} Novitec devices")

            if device_count == 0:
                raise RuntimeError("No Novitec devices found")

            # Connect to first available device
            self.camera = self.device_manager.GetDevice(0)
            if self.camera is None:
                raise RuntimeError("Failed to get camera device")

            # Connect to camera
            err = self.camera.Connect()
            if err != nvt.NVT_OK:
                print(f"Camera Connect 실패: {err}")
                raise RuntimeError(f"Failed to connect to camera: {err}")
            print("✓ Novitec camera connected")

            self.initialized = True

        except Exception as e:
            print(f"Novitec camera initialization failed: {e}")
            raise

    def read(self):
        """Read frame from Novitec camera"""
        if not self.initialized:
            return False, None

        try:
            # Capture image
            image = self.camera.CaptureImage()
            if image is None:
                print("Failed to capture image from Novitec camera")
                return False, None

            # Convert to OpenCV format
            frame = self._convert_to_opencv(image)
            if frame is None:
                print("Failed to convert Novitec image to OpenCV format")
                return False, None

            self.frame_number += 1
            return True, frame

        except Exception as e:
            print(f"Error reading from Novitec camera: {e}")
            return False, None

    def _convert_to_opencv(self, image):
        """Convert Novitec image to OpenCV format"""
        try:
            # Try direct data access first
            if hasattr(image, "data") and image.data is not None:
                # Get image properties
                width = image.GetWidth()
                height = image.GetHeight()
                format_type = image.GetFormat()

                print(
                    f"이미지 획득 성공! 크기: {width} x {height}, 포맷: {format_type}"
                )

                # Try to access image data directly
                try:
                    import numpy as np

                    # Convert based on format
                    if format_type == nvt.NVT_IMAGE_FORMAT_RGB24:
                        # RGB24 format
                        data = np.frombuffer(image.data, dtype=np.uint8)
                        if len(data) >= width * height * 3:
                            frame = data[: width * height * 3].reshape(
                                (height, width, 3)
                            )
                            # Convert RGB to BGR for OpenCV
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            return frame
                    elif format_type == nvt.NVT_IMAGE_FORMAT_YUV420_NV12:
                        # YUV420_NV12 format
                        data = np.frombuffer(image.data, dtype=np.uint8)
                        if len(data) >= width * height * 3 // 2:
                            # Convert YUV to BGR
                            yuv = data[: width * height * 3 // 2].reshape(
                                (height * 3 // 2, width)
                            )
                            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                            return frame

                except Exception as e:
                    print(f"직접 데이터 접근 실패: {e}")
                    print("파일 저장 방식으로 폴백")

            # Fallback: Save as JPEG and reload
            temp_path = "temp_novitec_image.jpg"
            err = image.SaveAsJPEG(temp_path)
            if err == nvt.NVT_OK:
                frame = cv2.imread(temp_path)
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                return frame
            else:
                print(f"JPEG 저장 실패: {err}")
                return None

        except Exception as e:
            print(f"이미지 변환 실패: {e}")
            return None

    def release(self):
        """Release Novitec camera resources"""
        try:
            if self.camera:
                self.camera.Disconnect()
                print("✓ Novitec camera disconnected")
            if self.device_manager:
                self.device_manager = None
                print("✓ Novitec DeviceManager released")
            print("Novitec 카메라 리소스 해제 완료")
        except Exception as e:
            print(f"Error releasing Novitec camera: {e}")

    def is_opened(self):
        return self.initialized and self.camera is not None


def create_sequence_loader(
    source: Union[str, int], fps: float = 30.0, loader_mode: str = "auto"
) -> Optional[BaseLoader]:
    """
    Create appropriate sequence loader based on source and mode

    Args:
        source: Video source (file path, device ID, or sequence path)
        fps: Frame rate for image sequences
        loader_mode: Loader mode ("auto", "camera_device", "video_file", "image_sequence")

    Returns:
        Appropriate loader instance or None if failed
    """
    try:
        if loader_mode == "auto":
            # Auto-detect based on source type
            if isinstance(source, int):
                # Integer source - try camera device
                return create_camera_device_loader(source)
            elif isinstance(source, str):
                if os.path.isfile(source):
                    # File path - try video file
                    return create_video_file_loader(source)
                elif os.path.isdir(source):
                    # Directory path - try image sequence
                    return create_image_sequence_loader(source, fps)
                else:
                    print(f"Error: Source path does not exist: {source}")
                    return None
            else:
                print(f"Error: Invalid source type: {type(source)}")
                return None

        elif loader_mode == "camera_device":
            return create_camera_device_loader(int(source))
        elif loader_mode == "video_file":
            return create_video_file_loader(str(source))
        elif loader_mode == "image_sequence":
            return create_image_sequence_loader(str(source), fps)
        else:
            print(f"Error: Unknown loader mode: {loader_mode}")
            return None

    except Exception as e:
        print(f"Error creating sequence loader: {e}")
        return None


def create_camera_device_loader(device_id: int = 0) -> Optional[BaseLoader]:
    """Create camera device loader with Novitec fallback"""
    try:
        # Try Novitec camera first
        if NOVITEC_AVAILABLE:
            try:
                loader = NovitecCameraLoader(device_id)
                print("✓ Novitec camera loader created")
                return loader
            except Exception as e:
                print(f"Novitec 카메라 실패, 일반 카메라로 폴백: {e}")

        else:
            raise RuntimeError("Novitec Camera Module not available")

    except Exception as e:
        print(f"Error creating camera device loader: {e}")
        return None


def create_video_file_loader(file_path: str) -> Optional[BaseLoader]:
    """Create video file loader"""
    try:
        loader = VideoFileLoader(file_path)
        print("✓ Video file loader created")
        return loader
    except Exception as e:
        print(f"Error creating video file loader: {e}")
        return None


def create_image_sequence_loader(
    sequence_path: str, fps: float = 30.0
) -> Optional[BaseLoader]:
    """Create image sequence loader"""
    try:
        loader = ImageSequenceLoader(sequence_path, fps)
        print("✓ Image sequence loader created")
        return loader
    except Exception as e:
        print(f"Error creating image sequence loader: {e}")
        return None


# Legacy function names for backward compatibility
def create_sequence_loader_legacy(source, fps=30, mode="auto"):
    """Legacy function for backward compatibility"""
    return create_sequence_loader(source, fps, mode)
