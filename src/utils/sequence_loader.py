"""
Sequence Loader Module

This module provides classes for loading different types of video sequences.
"""

import cv2
import glob
import os
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict
from enum import Enum
import time
import numpy as np

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

    def __init__(self, enable_undistortion: bool = False, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        self.frame_number = 0
        self.is_connected = False
        self.enable_undistortion = enable_undistortion
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Pre-compute undistortion maps for performance
        self.undistort_map1 = None
        self.undistort_map2 = None
        self.undistort_image_size = None

    def _initialize_undistort_maps(self, image_width: int, image_height: int):
        """Initialize undistortion maps (called once during initialization)."""
        if not self.enable_undistortion or self.camera_matrix is None or self.dist_coeffs is None:
            return
        
        if self.undistort_map1 is None:
            self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None, self.camera_matrix, (image_width, image_height), cv2.CV_16SC2
            )
            self.undistort_image_size = (image_width, image_height)
    
    def _undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply undistortion to frame if enabled."""
        if not self.enable_undistortion or self.camera_matrix is None or self.dist_coeffs is None:
            return frame
        
        if frame is None:
            return frame
        
        # Initialize maps on first frame if not already initialized
        if self.undistort_map1 is None:
            h, w = frame.shape[:2]
            self._initialize_undistort_maps(w, h)
        
        # Apply undistortion using pre-computed maps
        return cv2.remap(frame, self.undistort_map1, self.undistort_map2, cv2.INTER_LINEAR)

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
    
    def reset(self):
        """Reset loader to beginning (for video/sequence loaders)."""
        # Default implementation - override in subclasses
        pass


class VideoFileLoader(BaseLoader):
    """Loader for video files"""

    def __init__(self, file_path: str, enable_undistortion: bool = False, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        super().__init__(enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {file_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_connected = True

        # Initialize undistortion maps if enabled (read first frame to get image size)
        if self.enable_undistortion and self.camera_matrix is not None and self.dist_coeffs is not None:
            ret, first_frame = self.cap.read()
            if ret and first_frame is not None:
                h, w = first_frame.shape[:2]
                self._initialize_undistort_maps(w, h)
                # Reset video to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                print(f"[WARN] Could not read first frame to initialize undistortion maps")

        print(f"[OK] Video file opened: {file_path}")
        print(f"  Total frames: {self.total_frames}, FPS: {self.fps:.2f}")

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self._undistort_frame(frame)
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
    
    def reset(self):
        """Reset video to beginning."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_number = 0
            print(f"[OK] Video file reset to beginning: {self.file_path}")


class ImageSequenceLoader(BaseLoader):
    """Loader for image sequences"""

    def __init__(self, sequence_path: str, fps: float = 30.0, enable_undistortion: bool = False, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        super().__init__(enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
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

        # Initialize undistortion maps if enabled (read first image to get image size)
        if self.enable_undistortion and self.camera_matrix is not None and self.dist_coeffs is not None and self.image_files:
            first_image_path = self.image_files[0]
            first_frame = cv2.imread(first_image_path)
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                self._initialize_undistort_maps(w, h)
            else:
                print(f"[WARN] Could not read first image to initialize undistortion maps")

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

        # Apply undistortion if enabled
        frame = self._undistort_frame(frame)
        
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
    
    def reset(self):
        """Reset image sequence to beginning."""
        self.current_index = 0
        self.frame_number = 0
        self.last_frame_time = time.time()
        print(f"[OK] Image sequence reset to beginning: {self.sequence_path}")


class NovitecCameraLoader(BaseLoader):
    """Loader for Novitec industrial cameras"""
    
    # 클래스 레벨에서 모든 인스턴스 추적
    _all_instances: Dict[str, 'NovitecCameraLoader'] = {}
    
    @classmethod
    def _stop_all_other_streams(cls, current_device_id: str):
        """현재 카메라를 제외한 다른 모든 카메라의 스트림 중지 및 disconnect"""
        for device_id, loader in cls._all_instances.items():
            if device_id != current_device_id and getattr(loader, '_stream_started', False):
                try:
                    if loader.camera:
                        print(f"[INFO] Stopping and disconnecting {device_id} (switching to {current_device_id})")
                        # 스트림 중지
                        if hasattr(loader.camera, 'stop_stream'):
                            loader.camera.stop_stream()
                        # 완전히 disconnect
                        if hasattr(loader.camera, 'disconnect'):
                            loader.camera.disconnect()
                        loader._stream_started = False
                        loader.camera._is_streaming = False
                        loader._connected = False
                except Exception as e:
                    print(f"[WARN] Failed to stop/disconnect {device_id}: {e}")

    def __init__(self, device_id: str, config: Optional[dict] = None, enable_undistortion: bool = False, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize Novitec camera loader.

        Args:
            device_id: Device ID - serial number of the camera
            config: Optional configuration dictionary for camera parameters
            enable_undistortion: Whether to enable undistortion
            camera_matrix: Camera intrinsic matrix for undistortion
            dist_coeffs: Distortion coefficients for undistortion
        """
        super().__init__(enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

        if not NOVITEC_AVAILABLE:
            raise RuntimeError(
                "Novitec Camera Module not available. Please ensure submodule is initialized."
            )

        self.device_id = device_id
        self.config = config or {}
        self.camera: Optional[NovitecCamera] = None
        self.initialized = False

        self._initialize()
        
        # 클래스 레벨에서 인스턴스 등록
        NovitecCameraLoader._all_instances[device_id] = self

    def _initialize(self):
        """Initialize Novitec camera"""
        
        print(f"[INFO] NovitecCameraLoader._initialize() - device_id={self.device_id}, id(self)={id(self)}")

        # Create NovitecCamera instance
        self.camera = NovitecCamera(
            device_id=self.device_id,
            device_ip=None,  # Not used by Novitec
            config=self.config,
        )
        if not self.camera:
            raise RuntimeError(f"Failed to create Novitec camera: {self.device_id}")
        
        print(f"[INFO] NovitecCamera created - device_id={self.device_id}, id(camera)={id(self.camera)}, id(_device)={id(self.camera._device) if hasattr(self.camera, '_device') else 'N/A'}")

        # Connect to camera
        if not self.camera.connect():
            raise RuntimeError(f"Failed to connect to Novitec camera: {self.device_id}")
        
        print(f"[INFO] NovitecCamera connected - device_id={self.device_id}, id(_device)={id(self.camera._device) if hasattr(self.camera, '_device') else 'N/A'}")

        # start_stream은 여기서 호출하지 않음 - read() 호출 시 필요할 때만 시작
        # Novitec SDK가 한 번에 하나의 카메라만 스트리밍 지원하기 때문
        self._stream_started = False
        self._connected = True

        self.initialized = True
        self.is_connected = True
        
        # Initialize undistortion maps if enabled (read first frame to get image size)
        if self.enable_undistortion and self.camera_matrix is not None and self.dist_coeffs is not None:
            try:
                data = self.camera.capture(output_formats=["image"])
                if data and "image" in data:
                    first_frame = data["image"]
                    if first_frame is not None and len(first_frame.shape) == 3:
                        h, w = first_frame.shape[:2]
                        self._initialize_undistort_maps(w, h)
                    else:
                        print(f"[WARN] Could not read first frame to initialize undistortion maps")
                else:
                    print(f"[WARN] Could not capture first frame to initialize undistortion maps")
            except Exception as e:
                print(f"[WARN] Error initializing undistortion maps: {e}")
        
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
            # 스트림이 시작되지 않았으면 시작
            if not getattr(self, '_stream_started', False):
                print(f"[INFO] Starting stream for {self.device_id}...")
                # 다른 모든 카메라의 스트림 중지 및 disconnect (Novitec SDK는 한 번에 하나만 지원)
                NovitecCameraLoader._stop_all_other_streams(self.device_id)
                
                # disconnect 상태면 다시 connect
                if not getattr(self, '_connected', True) or not self.camera.is_connected:
                    print(f"[INFO] Re-connecting camera {self.device_id}...")
                    if not self.camera.connect():
                        print(f"[ERROR] Failed to re-connect camera {self.device_id}")
                        return False, None
                    self._connected = True
                    print(f"[INFO] Camera {self.device_id} re-connected successfully")
                
                stream_result = self.camera.start_stream()
                if stream_result:
                    self._stream_started = True
                    self.camera._is_streaming = True
                    print(f"[INFO] Stream started successfully for {self.device_id}")
                else:
                    # 이미 시작된 경우도 처리
                    self._stream_started = True
                    self.camera._is_streaming = True
                    print(f"[WARN] start_stream returned False for {self.device_id}, but continuing...")
            
            # 첫 프레임에서만 로그 출력
            if self.frame_number == 0:
                print(f"[INFO] NovitecCameraLoader.read() FIRST FRAME - device_id={self.device_id}")
            
            # Capture image using NovitecCamera API
            data = self.camera.capture(output_formats=["image"])

            if not data or "image" not in data:
                return False, None

            frame = data["image"]

            # Ensure frame is in correct format (BGR for OpenCV)
            if frame is not None and len(frame.shape) == 3:
                # 첫 프레임을 파일로 저장 (디버그용)
                if self.frame_number == 0:
                    debug_path = f"output/Debug/first_frame_{self.device_id}.png"
                    try:
                        cv2.imwrite(debug_path, frame)
                        print(f"[DEBUG] First frame saved: {debug_path} (device_id={self.device_id})")
                    except Exception as e:
                        print(f"[WARN] Failed to save debug frame: {e}")
                
                # Apply undistortion if enabled
                frame = self._undistort_frame(frame)
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
    config: Optional[dict] = None,
    enable_undistortion: bool = False,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None
) -> Optional[BaseLoader]:
    """
    Create appropriate sequence loader based on source and mode

    Args:
        source: Video source (file path, device ID, or sequence path)
        fps: Frame rate for image sequences
        loader_mode: Loader mode ("auto", "camera_device", "video_file", "image_sequence")
        config: Optional configuration dictionary (for camera loaders)
        enable_undistortion: Whether to enable undistortion
        camera_matrix: Camera intrinsic matrix for undistortion
        dist_coeffs: Distortion coefficients for undistortion

    Returns:
        Appropriate loader instance or None if failed
    """
    try:
        if loader_mode == "camera":
            return create_camera_device_loader(source=source, config=config, enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        elif loader_mode == "video":
            return create_video_file_loader(source=source, enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        elif loader_mode == "sequence":
            return create_image_sequence_loader(source=source, fps=fps, enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        else:
            print(f"Error: Unknown loader mode: {loader_mode}")
            return None

    except Exception as e:
        print(f"Error creating sequence loader: {e}")
        return None


def create_camera_device_loader(
    source: str, config: Optional[dict] = None, enable_undistortion: bool = False, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None
) -> Optional[BaseLoader]:
    """Create camera device loader with Novitec fallback"""

    try:
        loader = NovitecCameraLoader(device_id=source, config=config, enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("[OK] Novitec camera loader created")
        return loader
    except Exception as e:
        print(f"Novitec camera loader failed: {e}")
        return None


def create_video_file_loader(source: str, enable_undistortion: bool = False, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None) -> Optional[BaseLoader]:
    """Create video file loader"""
    try:
        loader = VideoFileLoader(file_path=source, enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("[OK] Video file loader created")
        return loader
    except Exception as e:
        print(f"Video file loader failed: {e}")
        return None

def create_image_sequence_loader(
    source: str, fps: float = 30.0, enable_undistortion: bool = False, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None
) -> Optional[BaseLoader]:
    """Create image sequence loader"""
    try:
        loader = ImageSequenceLoader(sequence_path=source, fps=fps, enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("[OK] Image sequence loader created")
        return loader
    except Exception as e:
        print(f"Error creating image sequence loader: {e}")
        return None

