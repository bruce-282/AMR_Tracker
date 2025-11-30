# AMR Tracker - 자율주행 로봇 추적 및 측정 시스템

## 📋 개요

AMR Tracker는 자율주행 로봇(AGV/AMR)을 실시간으로 감지, 추적, 측정하는 컴퓨터 비전 시스템입니다. 칼만 필터를 활용한 다중 객체 추적과 호모그래피 변환을 통한 정확한 물리적 측정을 제공합니다.

## 🏗️ 시스템 구조

### 프로젝트 구조

```
AMR_Tracker/
├── main.py                          # 메인 실행 파일
├── run_server.py                    # TCP/IP 서버 실행 파일
├── config/                          # 설정 파일 디렉토리
│   ├── zoom1.json                   # 제품 모델 설정 (zoom1)
│   ├── zoom2.json                   # 제품 모델 설정 (zoom2)
│   ├── camera1_config.json          # 카메라 1 설정
│   └── model_config.json            # 모델 목록 설정
├── src/
│   ├── core/                        # 핵심 추적 시스템
│   │   ├── amr_tracker.py           # EnhancedAMRTracker (통합 추적 시스템)
│   │   ├── detection/               # 객체 감지 모듈
│   │   │   ├── yolo_detector.py    # YOLO 기반 감지
│   │   │   ├── binary_detector.py  # 이진화 기반 감지
│   │   │   └── detection.py        # Detection 클래스
│   │   ├── tracking/                # 객체 추적 모듈
│   │   │   ├── kalman_tracker.py   # 칼만 필터 추적
│   │   │   └── association.py      # 객체 연결
│   │   ├── measurement/             # 측정 모듈
│   │   │   └── size_measurement.py # 크기 측정
│   │   └── calibration/             # 보정 모듈
│   │       ├── camera_calibrator.py # 카메라 보정
│   │       └── homography_calibrator.py # 호모그래피 보정
│   ├── server/                      # TCP/IP 서버 모듈
│   │   ├── vision_server.py        # VisionServer (메인 서버)
│   │   ├── camera_manager.py        # 카메라 관리
│   │   ├── tracking_manager.py      # 추적 관리
│   │   ├── response_builder.py     # 응답 생성
│   │   ├── model_config.py          # 모델 설정 관리
│   │   └── protocol.py             # 프로토콜 처리
│   ├── utils/                       # 유틸리티 모듈
│   │   ├── sequence_loader.py      # 비디오 소스 로더
│   │   └── config_loader.py        # 설정 로더
│   └── visualization/               # 시각화 모듈
│       └── display.py              # 결과 표시
└── submodules/                      # 서브모듈
    └── novitec_camera_module/       # Novitec 카메라 SDK
```

## 🔧 주요 기능

### 1. 다중 객체 추적 (Multi-Object Tracking)
- **칼만 필터 기반 추적**: 위치, 속도, 방향을 동시에 추적
- **다중 객체 지원**: 여러 AGV를 동시에 독립적으로 추적
- **ID 유지**: 객체가 일시적으로 가려져도 ID 유지

### 2. 물리적 측정 (Physical Measurement)
- **크기 측정**: 픽셀 단위를 실제 mm 단위로 변환
- **속도 측정**: 실시간 속도 및 평균 속도 계산
- **방향 추적**: 객체의 회전 각도 추적

### 3. 카메라 보정 (Camera Calibration)
- **내부 파라미터 보정**: 카메라 매트릭스 및 왜곡 계수
- **지면 평면 보정**: 호모그래피 변환을 통한 좌표계 변환
- **정확한 측정**: 픽셀 좌표를 실제 세계 좌표로 변환

### 4. 다양한 카메라 지원
- **웹캠**: 일반 USB 카메라
- **비디오 파일**: MP4, AVI, MKV 등
- **이미지 시퀀스**: PNG, JPG 등
- **Novitec 카메라**: 산업용 고성능 카메라 (자동 감지)

### 5. 다양한 감지 방법
- **YOLO 기반**: 딥러닝 객체 감지 (YOLODetector)
- **이진화 기반**: Adaptive threshold를 활용한 객체 감지 (BinaryDetector)
  - 어두운 객체/밝은 객체 모두 지원
  - 조명 불균일 환경에 적합

## 🚀 사용 방법

### 기본 모드 (Standalone)
```bash
# 웹캠 사용
python main.py --source 0

# 비디오 파일 사용
python main.py --source "data/video.mp4"

# 이미지 시퀀스 사용
python main.py --source "data/images/" --loader-mode image_sequence
```

### TCP/IP 서버 모드
```bash
# 서버 실행 (기본 포트 10000)
python run_server.py

# 특정 호스트/포트로 실행
python run_server.py --host 0.0.0.0 --port 10000

# 특정 프리셋 사용
python run_server.py --preset video_tracking
```

### 서버 클라이언트 통신
서버는 TCP/IP 소켓을 통해 JSON 프로토콜로 통신합니다:
- `START_VISION`: 추적 시작
- `STOP_VISION`: 추적 중지
- `GET_STATUS`: 상태 조회
- `GET_TRACKING_DATA`: 추적 데이터 조회

## 📊 시스템 동작 방식

### 1. 감지 단계 (Detection)
```python
# YOLO 기반 감지
from src.core.detection import YOLODetector
detector = YOLODetector(model_path="weights/best.pt")
detections = detector.detect(frame, frame_number, timestamp)

# 이진화 기반 감지
from src.core.detection import BinaryDetector
detector = BinaryDetector(
    threshold=100,
    use_adaptive=True,
    adaptive_block_size=11,
    adaptive_c=20.0,
    inverse=True  # True: 어두운 객체, False: 밝은 객체
)
detections = detector.detect(frame, frame_number, timestamp)
```

### 2. 추적 단계 (Tracking)
```python
# EnhancedAMRTracker 사용 (통합 시스템)
from src.core.amr_tracker import EnhancedAMRTracker

tracker = EnhancedAMRTracker(
    detector_type="yolo",  # 또는 "binary"
    detector_config={...},
    tracking_config={...},
    pixel_size=0.1,
    fps=30.0
)

# 객체 감지 및 추적
detections = tracker.detect_objects(frame, frame_number)
tracking_results = tracker.track_objects(frame, detections, frame_number)
```

### 3. 측정 단계 (Measurement)
```python
# 호모그래피 변환을 통한 좌표 변환
world_coords = homography @ image_coords

# 물리적 크기 계산
width_mm = pixel_width * pixels_per_mm
height_mm = pixel_height * pixels_per_mm
```

### 4. 시각화 단계 (Visualization)
- 실시간 바운딩 박스 표시
- 속도 벡터 화살표
- 궤적 표시
- 측정 정보 오버레이
- 이진화 디버그 이미지 (BinaryDetector 사용 시)

## ⚙️ 설정 파일

### 제품 모델 설정 (config/zoom1.json)
```json
{
  "detector": {
    "detector_type": "binary",
    "threshold": 100,
    "min_area": 700,
    "width_height_ratio_min": 0.8,
    "width_height_ratio_max": 1.2,
    "mask_area_ratio": 0.9,
    "inverse": true,
    "use_adaptive": true,
    "adaptive_block_size": 11,
    "adaptive_c": 20.0
  },
  "tracker": {
    "speed_threshold_pix_per_frame": 5.0,
    "max_frames_lost": 500
  },
  "execution": {
    "use_preset": "video_tracking",
    "image_undistortion": true,
    "result_base_path": "C:/CMES_AI/Result",
    "summary_base_path": "C:/CMES_AI/Summary",
    "debug_base_path": "C:/CMES_AI/Debug"
  }
}
```

### 모델 설정 (config/model_config.json)
```json
{
  "model_list": ["zoom1", "zoom2"],
  "selected_model": "zoom1"
}
```

### 카메라 설정 (config/camera1_config.json)
```json
{
  "CameraMatrix": [[...], [...], [...]],
  "DistortionCoefficients": [...]
}
```

## 📈 성능 특징

### 추적 정확도
- **위치 정확도**: ±2 픽셀 (보정 후)
- **속도 정확도**: ±5% (안정된 추적 시)
- **각도 정확도**: ±2도 (명확한 객체 경계 시)

### 처리 성능
- **실시간 처리**: 30 FPS (1080p 해상도)
- **다중 객체**: 최대 10개 객체 동시 추적
- **메모리 사용량**: ~500MB (기본 설정)

## 🔧 설치 및 의존성

### PyTorch 설치 (권장)

YOLO 기반 감지를 위해 PyTorch가 필요합니다. GPU 가속을 사용하려면 CUDA 지원 버전을 설치하세요.

#### CPU 버전
```bash
pip install torch torchvision torchaudio
```

#### CUDA 지원 버전 (GPU 가속)
```bash
# CUDA 12.8 (CUDA 12.9 호환)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4 (CUDA 12.x 호환)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**참고**: 설치된 CUDA 버전에 맞는 PyTorch를 설치하세요. `nvcc --version`으로 CUDA 버전을 확인할 수 있습니다.

### 필수 패키지
```bash
pip install opencv-python>=4.8.0
pip install numpy>=1.21.0
pip install filterpy>=1.4.5
pip install ultralytics>=8.0.0
pip install matplotlib>=3.5.0
```

**참고**: `ultralytics`는 PyTorch를 자동으로 설치하지만, GPU 가속을 위해서는 위의 CUDA 지원 PyTorch를 먼저 설치하는 것을 권장합니다.

### UV 패키지 매니저 사용
```bash
uv sync
```

### Novitec 카메라 모듈 설치
```bash
# Windows
submodules/novitec_camera_module/setup_novitec.bat

# 또는 수동으로
git submodule update --init --recursive
uv pip install -e submodules/novitec_camera_module
```

자세한 내용은 [README_Novitec_Camera.md](README_Novitec_Camera.md)를 참조하세요.

## 📁 데이터 구조

### 입력 데이터
- **비디오 파일**: MP4, AVI, MKV 등
- **이미지 시퀀스**: PNG, JPG 등
- **웹캠**: 실시간 카메라 입력
- **Novitec 카메라**: 산업용 고성능 카메라

### 출력 데이터
- **결과 이미지**: `C:/CMES_AI/Result/cam_{camera_id}_result.png`
- **요약 데이터**: `C:/CMES_AI/Summary/`
- **디버그 이미지**: `C:/CMES_AI/Debug/cam_{camera_id}_binary_debug.png` (BinaryDetector 사용 시)

## 🎯 사용 사례

### 1. 창고 자동화
- AGV 위치 추적
- 작업 효율성 분석
- 충돌 방지 시스템

### 2. 제조업
- 로봇 팔 추적
- 품질 검사 자동화
- 생산 라인 모니터링

### 3. 연구 개발
- 로봇 동작 분석
- 알고리즘 성능 평가
- 데이터 수집

## 🛠️ 고급 설정

### 이진화 디텍터 튜닝
```json
{
  "detector": {
    "detector_type": "binary",
    "use_adaptive": true,
    "adaptive_block_size": 11,  // 홀수 (3, 5, 7, 11, 15, 21 등)
    "adaptive_c": 30.0,          // 완전 검은 물체: 30-50, 일반: 2-10
    "inverse": true,              // true: 어두운 객체, false: 밝은 객체
    "threshold": 100,             // use_adaptive=false일 때만 사용
    "min_area": 700
  }
}
```

### 칼만 필터 튜닝
칼만 필터 파라미터는 `config/zoom1.json`의 `tracker` 섹션에서 설정할 수 있습니다:
- `speed_threshold_pix_per_frame`: 속도 임계값
- `max_frames_lost`: 최대 손실 프레임 수
- `detection_loss_threshold_frames`: 감지 손실 임계값

### 감지 임계값 조정
```json
{
  "detector": {
    "min_area": 1000,              // 최소 객체 크기 (픽셀)
    "width_height_ratio_min": 0.8,  // 최소 가로/세로 비율
    "width_height_ratio_max": 1.2, // 최대 가로/세로 비율
    "mask_area_ratio": 0.9         // 마스크/바운딩박스 비율
  }
}
```

## 🐛 문제 해결

### 일반적인 문제
1. **카메라 연결 실패**: 카메라 인덱스 확인
2. **보정 실패**: 체커보드 크기 및 품질 확인
3. **추적 실패**: 조명 조건 및 객체 크기 확인
4. **이진화 검출 실패**: `adaptive_c` 값 조정 (완전 검은 물체: 30-50)

### 성능 최적화
- GPU 가속 사용 (CUDA 지원)
- 해상도 조정
- 프레임 스킵 옵션

### OpenMP 오류
```
OMP: Error #15: Initializing libiomp5md.dll
```
이 오류는 자동으로 처리되지만, 수동으로 설정하려면:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

## 📞 지원

기술적 지원이나 질문이 있으시면 이슈를 생성해 주세요.

## 🔗 관련 문서

- [Novitec 카메라 사용 가이드](README_Novitec_Camera.md)
