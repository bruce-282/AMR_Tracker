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
python src/utils/novitec_camera_loader.py
```

## 📖 사용법

### 기본 사용법
```python
from src.utils.novitec_camera_loader import create_novitec_camera_loader

# 카메라 로더 생성
loader = create_novitec_camera_loader(camera_index=0, timeout=2000)

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

### Context Manager 사용
```python
from src.utils.novitec_camera_loader import create_novitec_camera_loader

with create_novitec_camera_loader() as loader:
    if loader:
        ret, frame = loader.read()
        if ret:
            # 프레임 처리
            process_frame(frame)
```

### 카메라 정보 확인
```python
from src.utils.novitec_camera_loader import list_novitec_cameras

# 사용 가능한 카메라 목록
cameras = list_novitec_cameras()
for camera in cameras:
    print(f"카메라 {camera['index']}: {camera['model_name']} - {camera['serial_number']}")
```

## 🔧 API 참조

### NovitecCameraLoader 클래스

#### 초기화
```python
loader = NovitecCameraLoader(camera_index=0, timeout=2000)
```

**매개변수:**
- `camera_index` (int): 사용할 카메라 인덱스 (기본값: 0)
- `timeout` (int): 이미지 획득 타임아웃 (ms, 기본값: 2000)

#### 주요 메서드

##### `initialize() -> bool`
카메라를 초기화하고 연결합니다.
```python
success = loader.initialize()
```

##### `read() -> Tuple[bool, Optional[np.ndarray]]`
다음 프레임을 읽습니다.
```python
ret, frame = loader.read()
if ret and frame is not None:
    # frame은 (height, width, 3) 형태의 BGR 이미지
    process_frame(frame)
```

##### `get_camera_info() -> dict`
카메라 정보를 반환합니다.
```python
info = loader.get_camera_info()
# {'model_name': '...', 'serial_number': '...', 'camera_index': 0, 'connected': True}
```

##### `release()`
리소스를 해제합니다.
```python
loader.release()
```

### 유틸리티 함수

#### `create_novitec_camera_loader(camera_index=0, timeout=2000)`
카메라 로더를 생성하고 초기화합니다.
```python
loader = create_novitec_camera_loader(camera_index=0)
```

#### `list_novitec_cameras() -> list`
사용 가능한 카메라 목록을 반환합니다.
```python
cameras = list_novitec_cameras()
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
- `timeout` 값이 너무 작으면 이미지 획득이 실패할 수 있습니다
- 기본값 2000ms는 대부분의 경우에 적합합니다

### 3. 리소스 관리
- 사용 후 반드시 `release()` 메서드를 호출하거나 context manager를 사용하세요
- 카메라 연결을 해제하지 않으면 다른 프로그램에서 사용할 수 없습니다

### 4. 에러 처리
```python
try:
    loader = create_novitec_camera_loader()
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
1. `timeout` 값을 늘려보세요 (3000ms 이상)
2. 다른 프로그램에서 카메라를 사용 중인지 확인
3. 카메라 케이블 연결 상태 확인

### SDK 모듈을 찾을 수 없는 경우
```python
# 경로 확인
import sys
from pathlib import Path
novitec_path = Path(__file__).parent.parent / "submodules" / "novitec_camera_module"
print(f"Novitec 경로: {novitec_path}")
print(f"경로 존재: {novitec_path.exists()}")
```

## 📁 파일 구조
```
src/utils/novitec_camera_loader.py    # 메인 모듈
submodules/novitec_camera_module/     # Novitec SDK
├── novitec_camera/                   # SDK 라이브러리
│   ├── *.dll                        # Windows DLL 파일들
│   └── drivers/                     # USB3 드라이버
└── src/novitec_camera_binding.cpp   # Python 바인딩
```

## 🔗 관련 파일
- `main.py`: AMR 트래킹 시스템 메인
- `src/core/detection/`: 객체 감지 모듈
- `src/core/tracking/`: 객체 추적 모듈
- `config/`: 설정 파일들
