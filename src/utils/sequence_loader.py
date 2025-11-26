"""
Sequence Loader Module

This module provides classes for loading different types of video sequences.
"""

import cv2
import glob
import logging
import os
from pathlib import Path
from typing import Optional, Union, List
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Try to import Novitec camera module
try:
    from .novitec_camera_loader import (
        NOVITEC_AVAILABLE,
        NovitecCameraLoader,
        create_novitec_camera_loader,
        list_novitec_cameras
    )
    if NOVITEC_AVAILABLE:
        from .novitec_camera_loader import nvt
        logger.info("Novitec Camera Module available")
    else:
        logger.warning("Novitec Camera Module not available - using fallback camera loader")
except ImportError as e:
    NOVITEC_AVAILABLE = False
    logger.warning(f"Novitec Camera Module not available: {e} - using fallback camera loader")


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

        logger.info(f"Video file opened: {file_path} (frames={self.total_frames}, fps={self.fps:.2f})")

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
        self.is_connected = True

        logger.info(f"Image sequence opened: {sequence_path} (images={len(self.image_files)}, fps={fps})")

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
            logger.warning(f"Could not load image: {image_path}")
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
                    logger.error(f"Source path does not exist: {source}")
                    return None
            else:
                logger.error(f"Invalid source type: {type(source)}")
                return None

        elif loader_mode == "camera_device":
            return create_camera_device_loader(int(source))
        elif loader_mode == "video_file":
            return create_video_file_loader(str(source))
        elif loader_mode == "image_sequence":
            return create_image_sequence_loader(str(source), fps)
        else:
            logger.error(f"Unknown loader mode: {loader_mode}")
            return None

    except Exception as e:
        logger.error(f"Error creating sequence loader: {e}")
        return None


def create_camera_device_loader(device_id: int = 0) -> Optional[BaseLoader]:
    """Create camera device loader with Novitec fallback"""
    try:
        # Try Novitec camera first
        if NOVITEC_AVAILABLE:
            try:
                loader = create_novitec_camera_loader(device_id)
                if loader:
                    logger.info("Novitec camera loader created")
                    return loader
                else:
                    logger.warning("Novitec camera initialization failed, falling back to standard camera")
            except Exception as e:
                logger.warning(f"Novitec camera failed, falling back to standard camera: {e}")

        # Fallback to standard OpenCV camera
        loader = NovitecCameraLoader(device_id)
        logger.info("Standard camera loader created")
        return loader

    except Exception as e:
        logger.error(f"Error creating camera device loader: {e}")
        return None


def create_video_file_loader(file_path: str) -> Optional[BaseLoader]:
    """Create video file loader"""
    try:
        loader = VideoFileLoader(file_path)
        logger.info("Video file loader created")
        return loader
    except Exception as e:
        logger.error(f"Error creating video file loader: {e}")
        return None


def create_image_sequence_loader(
    sequence_path: str, fps: float = 30.0
) -> Optional[BaseLoader]:
    """Create image sequence loader"""
    try:
        loader = ImageSequenceLoader(sequence_path, fps)
        logger.info("Image sequence loader created")
        return loader
    except Exception as e:
        logger.error(f"Error creating image sequence loader: {e}")
        return None


# Legacy function names for backward compatibility
def create_sequence_loader_legacy(source, fps=30, mode="auto"):
    """Legacy function for backward compatibility"""
    return create_sequence_loader(source, fps, mode)
