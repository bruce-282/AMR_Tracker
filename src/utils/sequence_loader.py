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


# ============================================================================
# Threaded Frame Buffer for Real-time Camera Streams
# ============================================================================

import threading
import queue
from dataclasses import dataclass
from typing import Callable


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_number: int


class ThreadedFrameBuffer:
    """
    Producer-Consumer pattern frame buffer for real-time camera streams.
    
    Captures frames in a separate thread and buffers them in a queue.
    When the queue is full, the oldest frames are dropped to maintain real-time performance.
    
    Usage:
        buffer = ThreadedFrameBuffer(capture_func, max_size=30)
        buffer.start()
        
        # In processing loop:
        frame_data = buffer.get()  # Returns FrameData or None
        if frame_data:
            process(frame_data.frame)
        
        buffer.stop()
    """
    
    def __init__(
        self,
        capture_func: Callable[[], tuple],
        max_size: int = 30,
        drop_policy: str = "oldest"
    ):
        """
        Initialize frame buffer.
        
        Args:
            capture_func: Function that returns (ret: bool, frame: np.ndarray)
            max_size: Maximum number of frames to buffer (default: 30 = ~1 second at 30fps)
            drop_policy: Policy when buffer is full:
                - "oldest": Drop oldest frame, add new one (default, maintains real-time)
                - "newest": Reject new frame (preserves history)
        """
        self.capture_func = capture_func
        self.max_size = max_size
        self.drop_policy = drop_policy
        
        # Thread-safe queue
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        
        # Control flags
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._captured_count = 0
        self._dropped_count = 0
        self._error_count = 0
        self._frame_number = 0
        
        # Latency tracking
        self._capture_times: list = []  # Recent capture timestamps for FPS calculation
        self._max_capture_times = 30
    
    def start(self):
        """Start the capture thread."""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            print(f"[INFO] ThreadedFrameBuffer started (max_size={self.max_size}, policy={self.drop_policy})")
    
    def stop(self, timeout: float = 2.0):
        """
        Stop the capture thread.
        
        Args:
            timeout: Maximum time to wait for thread to finish
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(f"[WARN] ThreadedFrameBuffer capture thread did not finish within {timeout}s")
        
        # Clear remaining frames
        self._clear_queue()
        
        print(f"[INFO] ThreadedFrameBuffer stopped - captured={self._captured_count}, "
              f"dropped={self._dropped_count}, errors={self._error_count}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        consecutive_errors = 0
        max_consecutive_errors = 50
        
        while self._running:
            try:
                # Capture frame
                capture_start = time.time()
                ret, frame = self.capture_func()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    self._error_count += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[ERROR] ThreadedFrameBuffer: {max_consecutive_errors} consecutive capture failures, stopping")
                        self._running = False
                        break
                    
                    # Brief sleep on error to avoid busy loop
                    time.sleep(0.01)
                    continue
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Create frame data
                self._frame_number += 1
                frame_data = FrameData(
                    frame=frame.copy(),  # Copy to avoid reference issues
                    timestamp=capture_start,
                    frame_number=self._frame_number
                )
                
                # Try to add to queue
                try:
                    self._queue.put_nowait(frame_data)
                    self._captured_count += 1
                except queue.Full:
                    # Queue is full, apply drop policy
                    if self.drop_policy == "oldest":
                        # Remove oldest frame and add new one
                        try:
                            self._queue.get_nowait()
                            self._queue.put_nowait(frame_data)
                            self._captured_count += 1
                            self._dropped_count += 1
                        except queue.Empty:
                            pass
                    else:
                        # Drop new frame
                        self._dropped_count += 1
                
                # Track capture times for FPS calculation
                self._capture_times.append(capture_start)
                if len(self._capture_times) > self._max_capture_times:
                    self._capture_times.pop(0)
                    
            except Exception as e:
                print(f"[ERROR] ThreadedFrameBuffer capture error: {e}")
                self._error_count += 1
                time.sleep(0.01)
    
    def get(self, timeout: float = 0.1) -> Optional[FrameData]:
        """
        Get a frame from the buffer.
        
        Args:
            timeout: Maximum time to wait for a frame (seconds)
        
        Returns:
            FrameData if available, None otherwise
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_nowait(self) -> Optional[FrameData]:
        """
        Get a frame without waiting.
        
        Returns:
            FrameData if available, None if queue is empty
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
    
    def _clear_queue(self):
        """Clear all frames from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
    
    @property
    def is_running(self) -> bool:
        """Check if capture thread is running."""
        return self._running
    
    @property
    def queue_size(self) -> int:
        """Get current number of frames in queue."""
        return self._queue.qsize()
    
    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        # Calculate actual capture FPS
        capture_fps = 0.0
        if len(self._capture_times) >= 2:
            time_span = self._capture_times[-1] - self._capture_times[0]
            if time_span > 0:
                capture_fps = (len(self._capture_times) - 1) / time_span
        
        return {
            "captured": self._captured_count,
            "dropped": self._dropped_count,
            "errors": self._error_count,
            "queue_size": self.queue_size,
            "max_size": self.max_size,
            "capture_fps": round(capture_fps, 2),
            "drop_rate": round(self._dropped_count / max(1, self._captured_count) * 100, 2)
        }
    
    def get_latency(self) -> float:
        """
        Get estimated latency (time between capture and now for oldest frame in queue).
        
        Returns:
            Latency in seconds, or 0 if queue is empty
        """
        if self.is_empty:
            return 0.0
        
        # Peek at oldest frame without removing
        # Note: This is a rough estimate as we can't peek without modifying queue
        return self.queue_size / max(1, self.get_stats()["capture_fps"])


class NovitecCameraLoader(BaseLoader):
    """Loader for Novitec industrial cameras"""
    
    # 클래스 레벨에서 모든 인스턴스 추적
    _all_instances: Dict[str, 'NovitecCameraLoader'] = {}
    
    @classmethod
    def _stop_all_other_streams(cls, current_device_id: str):
        """
        [DEPRECATED] No longer needed since each camera uses a separate DLL.
        Kept for backward compatibility but does nothing.
        """
        # Each camera now uses a separate DLL (cam1, cam2, cam3), so no need to
        # stop other streams. All cameras can stream independently.
        pass

    def __init__(
        self, 
        device_id: str, 
        config: Optional[dict] = None, 
        enable_undistortion: bool = False, 
        camera_matrix: Optional[np.ndarray] = None, 
        dist_coeffs: Optional[np.ndarray] = None, 
        camera_index: Optional[int] = None,
        enable_buffering: bool = True,
        buffer_size: int = 30,
        buffer_drop_policy: str = "oldest"
    ):
        """
        Initialize Novitec camera loader.

        Args:
            device_id: Device ID - serial number of the camera
            config: Optional configuration dictionary for camera parameters
            enable_undistortion: Whether to enable undistortion
            camera_matrix: Camera intrinsic matrix for undistortion
            dist_coeffs: Distortion coefficients for undistortion
            camera_index: Camera index (1, 2, or 3) to use separate DLL instances.
                         Each camera_index loads a separate DLL to avoid global state conflicts.
            enable_buffering: Enable threaded frame buffering (default: True)
                             When True, frames are captured in a separate thread and buffered.
                             This prevents frame drops when processing is slower than capture.
            buffer_size: Maximum number of frames to buffer (default: 30 = ~1 second at 30fps)
            buffer_drop_policy: Policy when buffer is full:
                               - "oldest": Drop oldest frame (maintains real-time)
                               - "newest": Reject new frame (preserves history)
        """
        super().__init__(enable_undistortion=enable_undistortion, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

        if not NOVITEC_AVAILABLE:
            raise RuntimeError(
                "Novitec Camera Module not available. Please ensure submodule is initialized."
            )

        self.device_id = device_id
        self.config = config or {}
        self.camera_index = camera_index  # Store camera_index for DLL isolation
        self.camera: Optional[NovitecCamera] = None
        self.initialized = False
        
        # Frame buffering settings
        self.enable_buffering = enable_buffering
        self.buffer_size = buffer_size
        self.buffer_drop_policy = buffer_drop_policy
        self._frame_buffer: Optional[ThreadedFrameBuffer] = None
        self._last_frame_data: Optional[FrameData] = None  # For timestamp access
        self._buffer_manually_stopped = False  # Flag to prevent auto-restart after explicit stop

        self._initialize()
        
        # 클래스 레벨에서 인스턴스 등록
        NovitecCameraLoader._all_instances[device_id] = self

    def _initialize(self):
        """Initialize Novitec camera"""
        
        print(f"[INFO] NovitecCameraLoader._initialize() - device_id={self.device_id}, camera_index={self.camera_index}, id(self)={id(self)}")

        # Create NovitecCamera instance with camera_index for DLL isolation
        self.camera = NovitecCamera(
            device_id=self.device_id,
            device_ip=None,  # Not used by Novitec
            config=self.config,
            camera_index=self.camera_index,  # Pass camera_index for separate DLL
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
        
        # 언디스토션 맵 초기화는 read()에서 첫 프레임을 읽을 때 수행
        # (여기서 capture()를 호출하면 start_stream()이 호출되어 마지막 카메라만 활성화됨)
        
        print(f"[OK] Novitec camera initialized (connected, not streaming yet): {self.device_id}")

    def _ensure_stream_started(self) -> bool:
        """Ensure camera stream is started. Returns True if stream is ready."""
        if not self.initialized or not self.camera:
            return False
        
        if not getattr(self, '_stream_started', False):
            print(f"[INFO] Starting stream for {self.device_id} (camera_index={self.camera_index})...")
            
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
        
        return self._stream_started

    def _direct_capture(self) -> tuple:
        """
        Directly capture a frame from the camera (without buffering).
        This is used by ThreadedFrameBuffer for the capture thread.
        
        Returns:
            tuple: (ret, frame) where ret is bool and frame is numpy array (BGR format)
        """
        if not self.initialized or not self.camera:
            return False, None

        try:
            # Ensure stream is started
            if not self._ensure_stream_started():
                return False, None
            
            # capture() 실패 시 재시도 (최대 3회, 빠른 재시도)
            max_retries = 3
            data = None
            
            for retry in range(max_retries):
                try:
                    data = self.camera.capture(output_formats=["image"])
                    if data and "image" in data:
                        break
                except Exception as e:
                    if retry == 0:
                        print(f"[WARN] _direct_capture() exception for {self.device_id}: {e}")
                
                if retry < max_retries - 1:
                    time.sleep(0.005)  # 5ms 대기 후 재시도

            if not data or "image" not in data:
                return False, None

            frame = data["image"]

            # Ensure frame is in correct format (BGR for OpenCV)
            if frame is not None and len(frame.shape) == 3:
                # Apply undistortion if enabled
                frame = self._undistort_frame(frame)
                return True, frame
            else:
                return False, None

        except Exception as e:
            print(f"Error in _direct_capture for {self.device_id}: {e}")
            return False, None

    def start_buffering(self):
        """Start the frame buffer capture thread."""
        self._buffer_manually_stopped = False  # Allow auto-restart again
        
        if not self.enable_buffering:
            print(f"[INFO] Buffering is disabled for {self.device_id}")
            return
        
        if self._frame_buffer is not None and self._frame_buffer.is_running:
            print(f"[WARN] Frame buffer already running for {self.device_id}")
            return
        
        # Ensure stream is started before buffering
        if not self._ensure_stream_started():
            print(f"[ERROR] Cannot start buffering: stream not started for {self.device_id}")
            return
        
        # Create and start frame buffer
        self._frame_buffer = ThreadedFrameBuffer(
            capture_func=self._direct_capture,
            max_size=self.buffer_size,
            drop_policy=self.buffer_drop_policy
        )
        self._frame_buffer.start()
        print(f"[OK] Frame buffering started for {self.device_id}")
    
    def stop_buffering(self):
        """Stop the frame buffer capture thread."""
        self._buffer_manually_stopped = True  # Prevent auto-restart in read()
        if self._frame_buffer is not None:
            self._frame_buffer.stop()
            self._frame_buffer = None
            print(f"[OK] Frame buffering stopped for {self.device_id}")
    
    def get_buffer_stats(self) -> Optional[dict]:
        """Get buffer statistics if buffering is enabled."""
        if self._frame_buffer is not None:
            return self._frame_buffer.get_stats()
        return None

    def read(self):
        """
        Read frame from Novitec camera.
        
        If buffering is enabled, returns frame from buffer.
        Otherwise, captures directly from camera.

        Returns:
            tuple: (ret, frame) where ret is bool and frame is numpy array (BGR format)
        """
        if not self.initialized or not self.camera:
            return False, None

        # === Buffered mode ===
        if self.enable_buffering and self._frame_buffer is not None and self._frame_buffer.is_running:
            frame_data = self._frame_buffer.get(timeout=0.1)
            
            if frame_data is not None:
                self._last_frame_data = frame_data
                self.frame_number = frame_data.frame_number
                
                # Log stats periodically (every 100 frames)
                if self.frame_number % 100 == 0:
                    stats = self._frame_buffer.get_stats()
                    print(f"[INFO] {self.device_id} buffer stats: queue={stats['queue_size']}/{stats['max_size']}, "
                          f"fps={stats['capture_fps']}, dropped={stats['dropped']}")
                
                return True, frame_data.frame
            else:
                # No frame available from buffer
                return False, None
        
        # === Direct capture mode (buffering disabled or buffer not started) ===
        # Auto-start buffering on first read if enabled (but NOT if manually stopped)
        if self.enable_buffering and self._frame_buffer is None and not self._buffer_manually_stopped:
            self.start_buffering()
            # Give buffer time to capture first frame
            time.sleep(0.1)
            return self.read()  # Recursive call to use buffer
        
        # Fallback to direct capture
        ret, frame = self._direct_capture()
        if ret:
            self.frame_number += 1
            
            # 첫 프레임에서만 로그 출력
            if self.frame_number == 1:
                device_obj_id = id(self.camera._device) if hasattr(self.camera, '_device') and self.camera._device else 'N/A'
                print(f"[INFO] NovitecCameraLoader.read() FIRST FRAME (direct) - device_id={self.device_id}, id(_device)={device_obj_id}")
        
        return ret, frame

    def get_timestamp(self) -> Optional[float]:
        """Get timestamp of the last read frame."""
        if self._last_frame_data is not None:
            return self._last_frame_data.timestamp
        return None

    def release(self):
        """Release Novitec camera resources"""
        try:
            # Stop frame buffer first
            if self._frame_buffer is not None:
                self.stop_buffering()
            
            if self.camera:
                # Stop stream if running
                if self.camera._is_streaming:
                    self.camera.stop_stream()
                # Disconnect
                self.camera.disconnect()
                self.camera = None
                print(f"[OK] Novitec camera released: {self.device_id}")

            self.initialized = False
            self.is_connected = False

        except Exception as e:
            print(f"Error releasing Novitec camera {self.device_id}: {e}")

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
    dist_coeffs: Optional[np.ndarray] = None,
    camera_index: Optional[int] = None,
    enable_buffering: bool = True,
    buffer_size: int = 30,
    buffer_drop_policy: str = "oldest"
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
        camera_index: Camera index (1, 2, or 3) for Novitec DLL isolation
        enable_buffering: Enable threaded frame buffering for camera loaders (default: True)
        buffer_size: Maximum number of frames to buffer (default: 30)
        buffer_drop_policy: Policy when buffer is full ("oldest" or "newest")

    Returns:
        Appropriate loader instance or None if failed
    """
    try:
        if loader_mode == "camera":
            return create_camera_device_loader(
                source=source, 
                config=config, 
                enable_undistortion=enable_undistortion, 
                camera_matrix=camera_matrix, 
                dist_coeffs=dist_coeffs, 
                camera_index=camera_index,
                enable_buffering=enable_buffering,
                buffer_size=buffer_size,
                buffer_drop_policy=buffer_drop_policy
            )
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
    source: str, 
    config: Optional[dict] = None, 
    enable_undistortion: bool = False, 
    camera_matrix: Optional[np.ndarray] = None, 
    dist_coeffs: Optional[np.ndarray] = None, 
    camera_index: Optional[int] = None,
    enable_buffering: bool = True,
    buffer_size: int = 30,
    buffer_drop_policy: str = "oldest"
) -> Optional[BaseLoader]:
    """Create camera device loader with Novitec fallback"""

    try:
        loader = NovitecCameraLoader(
            device_id=source, 
            config=config, 
            enable_undistortion=enable_undistortion, 
            camera_matrix=camera_matrix, 
            dist_coeffs=dist_coeffs, 
            camera_index=camera_index,
            enable_buffering=enable_buffering,
            buffer_size=buffer_size,
            buffer_drop_policy=buffer_drop_policy
        )
        buffering_str = f"buffering={'ON' if enable_buffering else 'OFF'}"
        if enable_buffering:
            buffering_str += f", buffer_size={buffer_size}, policy={buffer_drop_policy}"
        print(f"[OK] Novitec camera loader created (camera_index={camera_index}, {buffering_str})")
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

