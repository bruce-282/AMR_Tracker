"""
Sequence Loader Module

This module provides classes for loading different types of video sequences.
"""

import cv2
import glob
import os
from pathlib import Path
from typing import Optional, Union, Tuple
from enum import Enum


class LoaderMode(Enum):
    """Enumeration for different loader modes."""

    CAMERA_DEVICE = "camera_device"
    VIDEO_FILE = "video_file"
    IMAGE_SEQUENCE = "image_sequence"


class SequenceLoader:
    """
    A unified loader for different types of video sequences.

    Supports:
    - Webcam
    - Video files (mp4, avi, etc.)
    - Image sequences (png, jpg, etc.)
    """

    def __init__(
        self, source: Union[str, int], fps: int = 30, mode: Optional[LoaderMode] = None
    ):
        """
        Initialize the sequence loader.

        Args:
            source: Video source (webcam index, video file path, or image folder path)
            fps: Frames per second for image sequences
            mode: Explicit mode specification (optional, auto-detected if not provided)
        """
        self.source = source
        self.fps = fps
        self.mode = mode
        self.loader = None
        self.total_frames = 0
        self.current_frame = 0

        self._initialize_loader()

    def _initialize_loader(self):
        """Initialize the appropriate loader based on source type or explicit mode."""
        if self.mode is not None:
            # Use explicit mode
            if self.mode == LoaderMode.CAMERA_DEVICE:
                if not isinstance(self.source, int):
                    raise ValueError(
                        f"Camera device mode requires integer source, got {type(self.source)}"
                    )
                self.loader = CameraDeviceLoader(self.source)
            elif self.mode == LoaderMode.VIDEO_FILE:
                if not isinstance(self.source, str) or not Path(self.source).is_file():
                    raise ValueError(
                        f"Video file mode requires valid file path, got {self.source}"
                    )
                self.loader = VideoFileLoader(self.source)
            elif self.mode == LoaderMode.IMAGE_SEQUENCE:
                if not isinstance(self.source, str) or not Path(self.source).is_dir():
                    raise ValueError(
                        f"Image sequence mode requires valid directory path, got {self.source}"
                    )
                self.loader = ImageSequenceLoader(self.source, self.fps)
        else:
            # Auto-detect mode
            if isinstance(self.source, int):
                # Camera device
                self.mode = LoaderMode.CAMERA_DEVICE
                self.loader = CameraDeviceLoader(self.source)
            elif isinstance(self.source, str):
                source_path = Path(self.source)
                if source_path.is_file():
                    # Video file
                    self.mode = LoaderMode.VIDEO_FILE
                    self.loader = VideoFileLoader(self.source)
                elif source_path.is_dir():
                    # Image sequence folder
                    self.mode = LoaderMode.IMAGE_SEQUENCE
                    self.loader = ImageSequenceLoader(self.source, self.fps)
                else:
                    raise ValueError(f"Invalid source: {self.source}")
            else:
                raise ValueError(f"Unsupported source type: {type(self.source)}")

        if self.loader:
            self.total_frames = self.loader.get_total_frames()
            print(f"✓ Initialized {self.mode.value} loader")

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, frame)
        """
        if self.loader is None:
            return False, None

        ret, frame = self.loader.read()
        if ret:
            self.current_frame += 1
        return ret, frame

    def get_total_frames(self) -> int:
        """Get total number of frames."""
        return self.total_frames

    def get_current_frame(self) -> int:
        """Get current frame number."""
        return self.current_frame

    def get_fps(self) -> float:
        """Get frames per second."""
        return self.loader.get_fps() if self.loader else self.fps

    def get_mode(self) -> LoaderMode:
        """Get current loader mode."""
        return self.mode

    def is_camera_device(self) -> bool:
        """Check if this is a camera device loader."""
        return self.mode == LoaderMode.CAMERA_DEVICE

    def is_video_file(self) -> bool:
        """Check if this is a video file loader."""
        return self.mode == LoaderMode.VIDEO_FILE

    def is_image_sequence(self) -> bool:
        """Check if this is an image sequence loader."""
        return self.mode == LoaderMode.IMAGE_SEQUENCE

    def release(self):
        """Release resources."""
        if self.loader:
            self.loader.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class CameraDeviceLoader:
    """Loader for camera device input."""

    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        print(f"✓ Camera device {camera_index} initialized")

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        return self.cap.read()

    def get_total_frames(self) -> int:
        return -1  # Infinite for webcam

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self):
        if self.cap:
            self.cap.release()


class VideoFileLoader:
    """Loader for video files."""

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        self.video_path = video_path
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"✓ Video file loaded: {video_path}")
        print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        return self.cap.read()

    def get_total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def release(self):
        if self.cap:
            self.cap.release()


class ImageSequenceLoader:
    """Loader for image sequences."""

    def __init__(self, folder_path: str, fps: int = 30):
        self.folder_path = Path(folder_path)
        self.fps = fps
        self.current_frame_idx = 0

        # Find all image files
        self.image_files = self._find_image_files()

        if not self.image_files:
            raise RuntimeError(f"No image files found in {folder_path}")

        print(
            f"✓ Image sequence loaded: {len(self.image_files)} images from {folder_path}"
        )
        print(f"  FPS: {fps}, Duration: {len(self.image_files)/fps:.2f} seconds")

    def _find_image_files(self) -> list:
        """Find and sort image files in the folder."""
        # Common image extensions
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]

        image_files = []
        for ext in extensions:
            files = glob.glob(str(self.folder_path / ext))
            files.extend(glob.glob(str(self.folder_path / ext.upper())))
            image_files.extend(files)

        # Sort by filename (works for numbered sequences like 000000.png, 000001.png)
        image_files.sort(key=lambda x: os.path.basename(x))

        return image_files

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read the next frame from image sequence."""
        if self.current_frame_idx >= len(self.image_files):
            return False, None

        image_path = self.image_files[self.current_frame_idx]
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Warning: Could not read image {image_path}")
            self.current_frame_idx += 1
            return self.read()  # Try next image

        self.current_frame_idx += 1
        return True, frame

    def get_total_frames(self) -> int:
        return len(self.image_files)

    def get_fps(self) -> float:
        return self.fps

    def release(self):
        # No resources to release for image sequences
        pass

    def reset(self):
        """Reset sequence to beginning."""
        self.current_frame_idx = 0

    @property
    def frame_number(self) -> int:
        """Get current frame number (for compatibility)."""
        return self.current_frame_idx


def create_sequence_loader(
    source: Union[str, int], fps: int = 30, mode: Optional[LoaderMode] = None
) -> Optional[SequenceLoader]:
    """
    Factory function to create appropriate sequence loader.

    Args:
        source: Video source (webcam index, video file path, or image folder path)
        fps: Frames per second for image sequences
        mode: Explicit mode specification (optional, auto-detected if not provided)

    Returns:
        SequenceLoader instance or None if failed
    """
    try:
        return SequenceLoader(source, fps, mode)
    except Exception as e:
        print(f"Error creating sequence loader: {e}")
        return None


def create_camera_device_loader(camera_index: int = 0) -> Optional[SequenceLoader]:
    """Create a camera device loader explicitly."""
    return create_sequence_loader(camera_index, mode=LoaderMode.CAMERA_DEVICE)


def create_video_file_loader(video_path: str) -> Optional[SequenceLoader]:
    """Create a video file loader explicitly."""
    return create_sequence_loader(video_path, mode=LoaderMode.VIDEO_FILE)


def create_image_sequence_loader(
    folder_path: str, fps: int = 30
) -> Optional[SequenceLoader]:
    """Create an image sequence loader explicitly."""
    return create_sequence_loader(folder_path, fps, mode=LoaderMode.IMAGE_SEQUENCE)
