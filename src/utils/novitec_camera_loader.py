"""
Novitec Camera를 위한 로더 클래스

이 모듈은 Novitec Camera SDK를 사용하여 카메라에서 프레임을 가져오는 기능을 제공합니다.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Novitec 카메라 모듈 경로 설정
current_dir = Path(__file__).parent
novitec_path = current_dir.parent.parent / "submodules" / "novitec_camera_module"
sys.path.insert(0, str(novitec_path))

# Novitec SDK import
try:
    import novitec_camera as nvt
    NOVITEC_AVAILABLE = True
except ImportError:
    NOVITEC_AVAILABLE = False
    nvt = None
    logger.warning("Novitec Camera module not available")


class NovitecCameraLoader:
    """
    Novitec Camera를 위한 로더 클래스

    Novitec Camera SDK를 사용하여 카메라에서 프레임을 가져옵니다.
    """

    def __init__(self, camera_index: int = 0, timeout: int = 2000):
        """
        Novitec Camera 로더 초기화

        Args:
            camera_index: 사용할 카메라 인덱스
            timeout: 이미지 획득 타임아웃 (ms)
        """
        if not NOVITEC_AVAILABLE:
            raise ImportError("Novitec Camera module is not available")

        self.camera_index = camera_index
        self.timeout = timeout
        self.manager = None
        self.camera = None
        self.interface = None
        self.handle = None
        self.connected = False
        self.total_frames = -1  # 실시간 카메라는 무한 프레임
        self.current_frame = 0

    def initialize(self) -> bool:
        """
        카메라 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.manager = nvt.DeviceManager()
            err = self.manager.Update()
            logger.debug(f"DeviceManager Update: {err.GetType()}, {err.GetDescription()}")

            # 카메라 수 확인
            num_cameras = self.manager.GetNumberOfCamerasDirect()
            logger.info(f"Detected {num_cameras} camera(s)")

            if num_cameras == 0:
                logger.error("No cameras detected")
                return False

            if self.camera_index >= num_cameras:
                logger.error(f"Camera index {self.camera_index} out of range (0-{num_cameras-1})")
                return False

            # 카메라 핸들 획득
            self.handle = self.manager.GetCameraHandleDirect(self.camera_index)
            logger.debug(f"Camera handle acquired: {self.handle}")

            # Camera 객체 생성
            self.camera = nvt.Camera()

            # 카메라 연결
            err = self.camera.Connect(self.handle, nvt.AM_CONTROL)
            if err != nvt.NVT_OK:
                logger.error(f"Camera connection failed: {err}")
                return False

            logger.info("Camera connected successfully")

            # 연결 상태 확인
            self.connected = self.camera.IsConnected()
            if not self.connected:
                logger.error("Camera connection verification failed")
                self.camera.Disconnect()
                return False

            # GenICam 인터페이스 설정
            try:
                self.interface = nvt.GenICam(self.camera)

                # 이미지 압축 모드 설정
                try:
                    compression_enum = self.interface.GetFeature_IEnumeration(
                        "ImageCompressionMode"
                    )
                    compression_enum.SetSymbolicValue("JPEG")
                    logger.debug("Image compression mode set to JPEG")
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"Skipping image compression mode setting: {e}")

                # 픽셀 포맷 확인
                try:
                    pixel_format = self.interface.GetFeature_IEnumeration(
                        "PixelFormat"
                    ).GetSymbolicValue()
                    logger.debug(f"Pixel format: {pixel_format}")
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"Skipping pixel format check: {e}")

            except Exception as e:
                logger.warning(f"GenICam interface setup error: {e}")
                self.interface = None

            # 이미지 획득 시작
            err = self.camera.Start()
            if err != nvt.NVT_OK:
                logger.error(f"Image acquisition start failed: {err}")
                self.camera.Disconnect()
                return False

            logger.info(f"Novitec camera {self.camera_index} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        다음 프레임 읽기

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (성공 여부, 프레임)
        """
        if not self.connected or self.camera is None:
            return False, None

        try:
            image = nvt.Image()
            err = self.camera.GetImage(image, self.timeout)

            if err == nvt.NVT_OK:
                logger.debug(f"Image acquired: {image.width}x{image.height}, size={image.dataSize}")
                frame = self._convert_to_opencv(image)
                if frame is not None:
                    self.current_frame += 1
                    return True, frame
                return False, None
            else:
                logger.warning(f"Image acquisition failed: {err}")
                return False, None

        except Exception as e:
            logger.error(f"Frame read error: {e}")
            return False, None

    def _convert_to_opencv(self, image) -> Optional[np.ndarray]:
        """
        Novitec Image 객체를 OpenCV 형식으로 변환

        Args:
            image: Novitec Image 객체

        Returns:
            Optional[np.ndarray]: OpenCV 형식의 이미지 배열
        """
        try:
            # 직접 데이터 접근 시도
            if hasattr(image, "data") and hasattr(image, "width") and hasattr(image, "height"):
                data = image.data
                width = image.width
                height = image.height

                if data and width > 0 and height > 0 and hasattr(data, "__len__"):
                    # RGB 형식 처리
                    expected_size = width * height * 3
                    if len(data) >= expected_size:
                        img_array = np.frombuffer(data, dtype=np.uint8)
                        if len(img_array) >= expected_size:
                            frame = img_array[:expected_size].reshape((height, width, 3))
                            return frame

                    # YUV420_NV12 형식 처리
                    expected_size_yuv = width * height * 3 // 2
                    if len(data) >= expected_size_yuv:
                        img_array = np.frombuffer(data, dtype=np.uint8)
                        if len(img_array) >= expected_size_yuv:
                            yuv_frame = img_array[:expected_size_yuv].reshape((height * 3 // 2, width))
                            return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)

            logger.debug("Direct data access failed, falling back to file save")

            # 폴백: JPEG 파일 저장 방식
            temp_filename = f"temp_novitec_{self.current_frame}.jpg"
            try:
                err = image.Save(temp_filename, nvt.JPEG)
                if err == nvt.NVT_OK:
                    frame = cv2.imread(temp_filename)
                    try:
                        os.remove(temp_filename)
                    except OSError:
                        pass
                    if frame is not None:
                        return frame
                    logger.warning("Failed to read JPEG with OpenCV")
                else:
                    logger.warning(f"JPEG save failed: {err}")
            except (IOError, RuntimeError) as e:
                logger.warning(f"JPEG save error: {e}")

            logger.warning("All image conversion methods failed")
            return None

        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            return None

    def get_total_frames(self) -> int:
        """총 프레임 수 반환 (실시간 카메라는 -1)"""
        return self.total_frames

    def get_current_frame(self) -> int:
        """현재 프레임 번호 반환"""
        return self.current_frame

    def get_fps(self) -> float:
        """FPS 반환 (실시간 카메라는 30으로 가정)"""
        return 30.0

    def is_camera_device(self) -> bool:
        """카메라 디바이스 여부 반환"""
        return True

    def is_video_file(self) -> bool:
        """비디오 파일 여부 반환"""
        return False

    def is_image_sequence(self) -> bool:
        """이미지 시퀀스 여부 반환"""
        return False

    def release(self):
        """리소스 해제"""
        try:
            if self.camera is not None:
                if self.connected:
                    self.camera.Stop()
                    self.camera.Disconnect()
                self.camera = None

            self.manager = None
            self.interface = None
            self.handle = None
            self.connected = False
            logger.info("Novitec camera resources released")

        except Exception as e:
            logger.error(f"Resource release error: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

    def get_camera_info(self) -> dict:
        """
        카메라 정보 반환

        Returns:
            dict: 카메라 정보
        """
        if self.manager is None or self.camera_index < 0:
            return {}

        try:
            camera_info = self.manager.GetCameraInfoDirect(self.camera_index)
            return {
                "model_name": camera_info.GetModelName(),
                "serial_number": camera_info.GetSerialNumber(),
                "camera_index": self.camera_index,
                "connected": self.connected,
            }
        except Exception as e:
            logger.error(f"Failed to get camera info: {e}")
            return {}


def create_novitec_camera_loader(
    camera_index: int = 0, timeout: int = 2000
) -> Optional[NovitecCameraLoader]:
    """
    Novitec Camera 로더 생성

    Args:
        camera_index: 카메라 인덱스
        timeout: 타임아웃 (ms)

    Returns:
        Optional[NovitecCameraLoader]: 로더 객체 또는 None
    """
    if not NOVITEC_AVAILABLE:
        logger.warning("Novitec Camera module is not available")
        return None

    try:
        loader = NovitecCameraLoader(camera_index, timeout)
        if loader.initialize():
            return loader
        else:
            loader.release()
            return None
    except Exception as e:
        logger.error(f"Failed to create Novitec Camera loader: {e}")
        return None


def list_novitec_cameras() -> list:
    """
    사용 가능한 Novitec 카메라 목록 반환

    Returns:
        list: 카메라 정보 리스트
    """
    if not NOVITEC_AVAILABLE:
        return []

    try:
        manager = nvt.DeviceManager()
        err = manager.Update()
        if err != nvt.NVT_OK:
            return []

        num_cameras = manager.GetNumberOfCamerasDirect()
        cameras = []

        for i in range(num_cameras):
            try:
                camera_info = manager.GetCameraInfoDirect(i)
                cameras.append({
                    "index": i,
                    "model_name": camera_info.GetModelName(),
                    "serial_number": camera_info.GetSerialNumber(),
                })
            except Exception as e:
                logger.warning(f"Failed to get camera {i} info: {e}")

        return cameras

    except Exception as e:
        logger.error(f"Failed to get camera list: {e}")
        return []


def test_novitec_camera():
    """Novitec 카메라 테스트"""
    logger.info("=== Novitec Camera Test ===")

    cameras = list_novitec_cameras()
    logger.info(f"Detected {len(cameras)} camera(s)")

    for camera in cameras:
        logger.info(f"  {camera['index']}: {camera['model_name']} - {camera['serial_number']}")

    if not cameras:
        logger.warning("No cameras available")
        return

    camera_index = cameras[0]["index"]
    logger.info(f"Testing camera {camera_index}...")

    loader = create_novitec_camera_loader(camera_index)
    if loader is None:
        logger.error("Failed to create camera loader")
        return

    try:
        info = loader.get_camera_info()
        logger.info(f"Camera info: {info}")

        logger.info("Testing 5 frames...")
        for i in range(5):
            ret, frame = loader.read()
            if ret and frame is not None:
                logger.info(f"  Frame {i+1}: {frame.shape}")
            else:
                logger.warning(f"  Frame {i+1}: failed")
            time.sleep(0.1)

        logger.info("Test completed")

    finally:
        loader.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_novitec_camera()
