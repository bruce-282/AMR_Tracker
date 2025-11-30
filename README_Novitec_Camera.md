# Novitec Camera Loader

Novitec Camera SDK를 사용하여 카메라에서 실시간 프레임을 가져오는 Python 모듈입니다.

## 📋 요구사항

### 하드웨어 요구사항
- **Novitec Camera**: USB3 카메라 디바이스
- **운영체제**: Windows (USB3 드라이버 필요)
- **포트**: USB 3.0 포트

### 소프트웨어 요구사항
- **Python**: 3.7+
- **Novitec Camera SDK**: `submodules/novitec_camera_module/` 경로에 설치
- **필수 라이브러리**:
  ```bash
  numpy>=1.19.0
  opencv-python>=4.5.0
  ```

### 드라이버 설치
Windows에서 USB3 카메라 드라이버가 필요합니다:
```
submodules/novitec_camera_module/novitec_camera/drivers/USB3 Camera/
├── x64/
│   ├── cyusb3.cat
│   ├── cyusb3.inf
│   └── cyusb3.sys
└── x86/
    ├── cyusb3.cat
    ├── cyusb3.inf
    └── cyusb3.sys
```

## 🚀 설치 및 설정

### 1. Novitec Camera SDK 설정
```bash
# submodules 디렉토리로 이동
cd submodules/novitec_camera_module

# Windows에서 드라이버 설치
setup_novitec.bat

# 또는 수동으로 드라이버 설치
# Device Manager에서 USB3 Camera 드라이버 설치
```

### 2. Python 환경 설정
```bash
# 프로젝트 루트에서
pip install -r requirements.txt

# 또는 uv 사용
uv sync
```

### 3. 카메라 연결 확인
```bash
# 카메라 테스트 실행
python src/utils/sequence_loader.py
```

## 📖 사용법

### 기본 사용법 (Sequence Loader)
```python
from src.utils.sequence_loader import create_sequence_loader

# Novitec 카메라 로더 생성
loader = create_sequence_loader(
    loader_mode="camera",
    source=0,  # 카메라 인덱스
    config=None,  # 카메라 설정 파일 경로 (선택사항)
    enable_undistortion=False,  # 이미지 왜곡 보정
    camera_matrix=None,  # 카메라 매트릭스
    dist_coeffs=None  # 왜곡 계수
)

if loader:
    try:
        # 프레임 읽기
        ret, frame = loader.read()
        if ret and frame is not None:
            print(f"프레임 크기: {frame.shape}")
            # OpenCV로 이미지 처리
            cv2.imshow("Novitec Camera", frame)
            cv2.waitKey(1)
    finally:
        loader.release()
```

### Vision Server에서 사용
Vision Server는 설정 파일을 통해 Novitec 카메라를 자동으로 사용합니다:

```json
{
  "execution": {
    "use_preset": "camera_tracking",
    "presets": {
      "camera_tracking": {
        "loader_mode": "camera",
        "camera_1": {
          "id": 0,
          "config": "config/camera1_config.json",
          "measurement": {
            "fps": 30.0,
            "pixel_size": 0.1
          }
        }
      }
    }
  }
}
```

### 카메라 정보 확인
```python
from src.utils.sequence_loader import create_camera_device_loader

# Novitec 카메라 로더 생성
loader = create_camera_device_loader(
    camera_index=0,
    config="config/camera1_config.json"  # 선택사항
)

if loader:
    info = loader.get_camera_info()
    print(f"카메라 모델: {info.get('model_name')}")
    print(f"시리얼 번호: {info.get('serial_number')}")
    loader.release()
```

## 🔧 API 참조

### Sequence Loader 통합

Novitec 카메라는 `BaseLoader` 인터페이스를 통해 통합되어 있습니다:

```python
from src.utils.sequence_loader import BaseLoader, create_sequence_loader

# 로더 생성
loader = create_sequence_loader(
    loader_mode="camera",
    source=0,
    config="config/camera1_config.json"
)

# BaseLoader 인터페이스 사용
if isinstance(loader, BaseLoader):
    ret, frame = loader.read()
    if ret:
        # 프레임 처리
        process_frame(frame)
    
    # 리셋 (비디오/이미지 시퀀스의 경우)
    if hasattr(loader, 'reset'):
        loader.reset()
    
    # 해제
    loader.release()
```

### 이미지 왜곡 보정

Novitec 카메라 로더는 이미지 왜곡 보정을 지원합니다:

```python
import json
from pathlib import Path

# 카메라 설정 파일 읽기
with open("config/camera1_config.json", 'r') as f:
    camera_config = json.load(f)

camera_matrix = np.array(camera_config["CameraMatrix"])
dist_coeffs = np.array(camera_config["DistortionCoefficients"])

# 왜곡 보정 활성화
loader = create_sequence_loader(
    loader_mode="camera",
    source=0,
    enable_undistortion=True,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)
```

## 🎯 지원되는 이미지 포맷

- **JPEG**: 압축된 이미지 (기본값)
- **RGB**: 24비트 RGB 이미지
- **YUV420_NV12**: YUV 형식 (자동 변환)

## ⚠️ 주의사항

### 1. 카메라 연결
- USB 3.0 포트에 연결해야 합니다
- 카메라가 다른 프로그램에서 사용 중이면 연결이 실패할 수 있습니다

### 2. 타임아웃 설정
- 타임아웃은 Novitec SDK 내부에서 관리됩니다
- 기본값은 대부분의 경우에 적합합니다

### 3. 리소스 관리
- 사용 후 반드시 `release()` 메서드를 호출하세요
- 카메라 연결을 해제하지 않으면 다른 프로그램에서 사용할 수 없습니다

### 4. 에러 처리
```python
try:
    loader = create_sequence_loader(loader_mode="camera", source=0)
    if loader is None:
        print("카메라 연결 실패")
        return
    
    ret, frame = loader.read()
    if not ret:
        print("프레임 읽기 실패")
        
except Exception as e:
    print(f"오류 발생: {e}")
finally:
    if loader:
        loader.release()
```

## 🐛 문제 해결

### 카메라가 감지되지 않는 경우
1. USB 3.0 포트에 연결했는지 확인
2. 드라이버가 올바르게 설치되었는지 확인
3. Device Manager에서 카메라가 인식되는지 확인

### 이미지 획득 실패
1. 다른 프로그램에서 카메라를 사용 중인지 확인
2. 카메라 케이블 연결 상태 확인
3. 카메라 전원 상태 확인

### SDK 모듈을 찾을 수 없는 경우
```python
# 경로 확인
import sys
from pathlib import Path
novitec_path = Path(__file__).parent.parent / "submodules" / "novitec_camera_module"
print(f"Novitec 경로: {novitec_path}")
print(f"경로 존재: {novitec_path.exists()}")
```

### 이미지 왜곡 보정 오류
1. `camera1_config.json` 파일이 올바른지 확인
2. `CameraMatrix`와 `DistortionCoefficients` 형식 확인
3. `enable_undistortion`이 `true`로 설정되었는지 확인

## 📁 파일 구조

```
src/utils/sequence_loader.py         # 통합 시퀀스 로더
├── BaseLoader                       # 기본 로더 인터페이스
├── NovitecCameraLoader             # Novitec 카메라 로더
├── VideoFileLoader                 # 비디오 파일 로더
└── ImageSequenceLoader             # 이미지 시퀀스 로더

submodules/novitec_camera_module/   # Novitec SDK
├── novitec_camera/                 # SDK 라이브러리
│   ├── *.dll                      # Windows DLL 파일들
│   └── drivers/                   # USB3 드라이버
└── src/novitec_camera_binding.cpp  # Python 바인딩

config/
├── camera1_config.json            # 카메라 1 설정 (왜곡 보정용)
└── zoom1.json                     # 제품 모델 설정
```

## 🔗 관련 파일

- `main.py`: AMR 트래킹 시스템 메인 (Standalone 모드)
- `run_server.py`: TCP/IP 서버 실행
- `src/server/vision_server.py`: Vision Server 메인
- `src/server/camera_manager.py`: 카메라 관리
- `src/core/amr_tracker.py`: EnhancedAMRTracker (통합 추적 시스템)
- `src/core/detection/`: 객체 감지 모듈
- `src/core/tracking/`: 객체 추적 모듈
- `config/`: 설정 파일들

## 📝 설정 파일 예시

### camera1_config.json
```json
{
  "CameraMatrix": [
    [1000.0, 0.0, 640.0],
    [0.0, 1000.0, 360.0],
    [0.0, 0.0, 1.0]
  ],
  "DistortionCoefficients": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

### zoom1.json (일부)
```json
{
  "execution": {
    "use_preset": "camera_tracking",
    "image_undistortion": true,
    "presets": {
      "camera_tracking": {
        "loader_mode": "camera",
        "camera_1": {
          "id": 0,
          "config": "config/camera1_config.json",
          "measurement": {
            "fps": 30.0,
            "pixel_size": 0.1
          }
        }
      }
    }
  }
}
```
