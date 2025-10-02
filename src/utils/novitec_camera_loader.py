"""
Novitec Camera를 위한 로더 클래스

이 모듈은 Novitec Camera SDK를 사용하여 카메라에서 프레임을 가져오는 기능을 제공합니다.
"""

import sys
import os
import time
import numpy as np
import cv2
from typing import Optional, Tuple, Union
from pathlib import Path

# test_camera_connect.py와 동일한 방식으로 경로 설정
current_dir = Path(__file__).parent
novitec_path = current_dir.parent.parent / "submodules" / "novitec_camera_module"
sys.path.insert(0, str(novitec_path))

# test_camera_connect.py와 동일한 방식으로 import
try:
    import novitec_camera as nvt

    NOVITEC_AVAILABLE = True
except ImportError:
    NOVITEC_AVAILABLE = False
    print("Warning: Novitec Camera module not available")


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
            print(f"DeviceManager Update 결과: {err.GetType()}, {err.GetDescription()}")

            # if err != nvt.NVT_OK:
            #     print(f"DeviceManager Update 실패: {err}")
            #     return False
            # else:
            #     print("DeviceManager Update 성공!")

            # 2. 카메라 수 확인 (test_camera_connect.py와 동일)
            num_cameras = self.manager.GetNumberOfCamerasDirect()
            print(f"감지된 카메라 수: {num_cameras}")

            if num_cameras == 0:
                print("감지된 카메라가 없습니다")
                return False

            if self.camera_index >= num_cameras:
                print(
                    f"카메라 인덱스 {self.camera_index}가 범위를 벗어났습니다 (0-{num_cameras-1})"
                )
                return False

            # 3. 카메라 핸들 획득 (test_camera_connect.py와 동일)
            self.handle = self.manager.GetCameraHandleDirect(self.camera_index)
            print(f"카메라 핸들 획득: {self.handle}")

            # 4. Camera 객체 생성 (test_camera_connect.py와 동일)
            self.camera = nvt.Camera()
            print(f"Camera 객체: {self.camera}")

            # 5. 카메라 연결 (test_camera_connect.py와 동일)
            err = self.camera.Connect(self.handle, nvt.AM_CONTROL)
            print(f"Connect 결과: {err}")

            if err != nvt.NVT_OK:
                print(f"카메라 연결 실패: {err}")
                return False

            print("카메라 연결 성공!")

            # 6. 연결 상태 확인 (test_camera_connect.py와 동일)
            self.connected = self.camera.IsConnected()
            print(f"연결 상태: {self.connected}")

            if not self.connected:
                print("카메라가 연결되지 않았습니다")
                self.camera.Disconnect()
                return False

            print("카메라가 연결되었습니다!")

            # 7. GenICam 인터페이스 설정 (test_camera_connect.py와 동일)
            try:
                self.interface = nvt.GenICam(self.camera)

                # 이미지 압축 모드 설정
                try:
                    compression_enum = self.interface.GetFeature_IEnumeration(
                        "ImageCompressionMode"
                    )
                    compression_enum.SetSymbolicValue("JPEG")
                    print("이미지 압축 모드를 JPEG로 설정했습니다")
                except:
                    print("이미지 압축 모드 설정을 건너뜁니다")

                # 픽셀 포맷 확인
                try:
                    pixel_format = self.interface.GetFeature_IEnumeration(
                        "PixelFormat"
                    ).GetSymbolicValue()
                    print(f"픽셀 포맷: {pixel_format}")
                except:
                    print("픽셀 포맷 확인을 건너뜁니다")

            except Exception as e:
                print(f"GenICam 설정 중 오류: {e}")
                self.interface = None

            # 8. 이미지 획득 시작 (test_camera_connect.py와 동일)
            err = self.camera.Start()
            if err != nvt.NVT_OK:
                print(f"이미지 획득 시작 실패: {err}")
                self.camera.Disconnect()
                return False

            print("이미지 획득 시작 성공!")

            print(f"Novitec 카메라 {self.camera_index} 초기화 완료")
            return True

        except Exception as e:
            print(f"카메라 초기화 중 오류: {e}")
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
            # 이미지 객체 생성 (test_camera_connect.py와 동일)
            image = nvt.Image()

            # 이미지 획득 (test_camera_connect.py와 동일)
            err = self.camera.GetImage(image, self.timeout)

            if err == nvt.NVT_OK:
                print(
                    f"이미지 획득 성공! 크기: {image.width} x {image.height}, 데이터 크기: {image.dataSize}"
                )

                # 이미지 데이터를 NumPy 배열로 변환
                frame = self._convert_to_opencv(image)
                if frame is not None:
                    self.current_frame += 1
                    return True, frame
                else:
                    return False, None
            else:
                print(f"이미지 획득 실패: {err}")
                return False, None

        except Exception as e:
            print(f"프레임 읽기 중 오류: {e}")
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
            # 파일 저장 없이 직접 데이터 접근 시도
            try:
                # 이미지 데이터를 직접 NumPy 배열로 변환 시도
                if (
                    hasattr(image, "data")
                    and hasattr(image, "width")
                    and hasattr(image, "height")
                ):
                    # 이미지 데이터를 직접 읽기
                    data = image.data
                    width = image.width
                    height = image.height

                    if data and width > 0 and height > 0:
                        # 데이터를 NumPy 배열로 변환
                        if hasattr(data, "__len__"):
                            # 데이터 길이 확인
                            expected_size = width * height * 3  # RGB
                            if len(data) >= expected_size:
                                # RGB 형식으로 변환
                                img_array = np.frombuffer(data, dtype=np.uint8)
                                if len(img_array) >= expected_size:
                                    img_array = img_array[:expected_size]
                                    frame = img_array.reshape((height, width, 3))
                                    return frame

                            # YUV420_NV12 형식 처리
                            expected_size_yuv = width * height * 3 // 2  # YUV420_NV12
                            if len(data) >= expected_size_yuv:
                                img_array = np.frombuffer(data, dtype=np.uint8)
                                if len(img_array) >= expected_size_yuv:
                                    img_array = img_array[:expected_size_yuv]
                                    # YUV420_NV12를 RGB로 변환
                                    yuv_frame = img_array.reshape(
                                        (height * 3 // 2, width)
                                    )
                                    frame = cv2.cvtColor(
                                        yuv_frame, cv2.COLOR_YUV2BGR_NV12
                                    )
                                    return frame

                print("직접 데이터 접근 실패, 파일 저장 방식으로 폴백")

            except Exception as e:
                print(f"직접 데이터 접근 중 오류: {e}")

            # 폴백: JPEG만 시도 (가장 성공률이 높음)
            temp_filename = f"temp_novitec_{self.current_frame}.jpg"

            try:
                err = image.Save(temp_filename, nvt.JPEG)
                if err == nvt.NVT_OK:
                    # OpenCV로 읽기
                    frame = cv2.imread(temp_filename)
                    if frame is not None:
                        # 임시 파일 삭제
                        try:
                            os.remove(temp_filename)
                        except:
                            pass
                        return frame
                    else:
                        print("OpenCV로 JPEG 읽기 실패")
                else:
                    print(f"JPEG 저장 실패: {err}")
            except Exception as e:
                print(f"JPEG 저장 중 오류: {e}")

            print("모든 변환 방법 실패")
            return None

        except Exception as e:
            print(f"이미지 변환 중 오류: {e}")
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

            print("Novitec 카메라 리소스 해제 완료")

        except Exception as e:
            print(f"리소스 해제 중 오류: {e}")

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
            print(f"카메라 정보 획득 중 오류: {e}")
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
        print("Novitec Camera module is not available")
        return None

    try:
        loader = NovitecCameraLoader(camera_index, timeout)
        if loader.initialize():
            return loader
        else:
            loader.release()
            return None
    except Exception as e:
        print(f"Novitec Camera 로더 생성 실패: {e}")
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
                cameras.append(
                    {
                        "index": i,
                        "model_name": camera_info.GetModelName(),
                        "serial_number": camera_info.GetSerialNumber(),
                    }
                )
            except Exception as e:
                print(f"카메라 {i} 정보 획득 실패: {e}")

        return cameras

    except Exception as e:
        print(f"카메라 목록 획득 실패: {e}")
        return []


# 테스트 함수
def test_novitec_camera():
    """Novitec 카메라 테스트"""
    print("=== Novitec Camera 테스트 ===")

    # 사용 가능한 카메라 목록
    cameras = list_novitec_cameras()
    print(f"감지된 카메라 수: {len(cameras)}")

    for camera in cameras:
        print(
            f"  {camera['index']}: {camera['model_name']} - {camera['serial_number']}"
        )

    if not cameras:
        print("사용 가능한 카메라가 없습니다")
        return

    # 첫 번째 카메라로 테스트
    camera_index = cameras[0]["index"]
    print(f"\n카메라 {camera_index}로 테스트 시작...")

    loader = create_novitec_camera_loader(camera_index)
    if loader is None:
        print("카메라 로더 생성 실패")
        return

    try:
        # 카메라 정보 출력
        info = loader.get_camera_info()
        print(f"카메라 정보: {info}")

        # 몇 개의 프레임 테스트
        print("\n프레임 테스트 (5개)...")
        for i in range(5):
            ret, frame = loader.read()
            if ret and frame is not None:
                print(f"  프레임 {i+1}: {frame.shape}")
            else:
                print(f"  프레임 {i+1}: 실패")
            time.sleep(0.1)

        print("테스트 완료")

    finally:
        loader.release()


if __name__ == "__main__":
    test_novitec_camera()
