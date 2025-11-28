# AMR Tracker - 자율주행 로봇 추적 및 측정 시스템

## 📋 개요

AMR Tracker는 자율주행 로봇(AGV/AMR)을 실시간으로 감지, 추적, 측정하는 컴퓨터 비전 시스템입니다. 칼만 필터를 활용한 다중 객체 추적과 호모그래피 변환을 통한 정확한 물리적 측정을 제공합니다.

## 🏗️ 시스템 구조

### 핵심 컴포넌트

```
AMR_Tracker/
├── main.py                    # 메인 실행 파일 (기본/고급 모드)
├── agv_system.py              # AGV 측정 시스템 (고급 모드)
├── sequence_loader.py         # 다양한 비디오 소스 로더
├── utils/
│   └── config.py             # 시스템 설정 관리
├── detection/
│   └── agv_detector.py       # AGV 감지 모듈
├── measurement/
│   ├── size_measurement.py   # 물리적 크기 측정
│   └── speed_tracker.py      # 속도 및 궤적 추적
├── calibration/
│   ├── camera_calibrator.py  # 카메라 내부 파라미터 보정
│   └── homography_calibrator.py # 지면 평면 보정
└── visualization/
    └── display.py            # 결과 시각화
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
- **YOLO 기반**: 딥러닝 객체 감지
- **색상 기반**: HSV 색상 공간을 활용한 AGV 감지
- **컨투어 기반**: 형태 분석을 통한 객체 감지

## 🚀 사용 방법

### 기본 모드 (Basic Mode)
```bash
# 웹캠 사용
python main.py --mode basic --source 0

# 비디오 파일 사용
python main.py --mode basic --source "data/video.mp4"

# 카메라 사용 (Novitec 카메라가 있으면 자동으로 사용, 없으면 일반 카메라)
python main.py --mode basic --source 0 --loader-mode camera

# 정지 모드 (속도 측정 없음)
python main.py --mode basic --stationary-mode
```

### 고급 모드 (Enhanced Mode)
```bash
# 보정 실행
python agv_system.py calibrate

# 측정 실행
python agv_system.py measure --source "data/video.mp4"
```

### 프리셋 사용
```bash
# 고급 정지 모드
python main.py --preset enhanced_stationary

# 고급 추적 모드
python main.py --preset enhanced_tracking

# 기본 추적 모드
python main.py --preset basic_tracking
```

## 📊 시스템 동작 방식

### 1. 감지 단계 (Detection)
```python
# YOLO 기반 감지
detector = ultralytics.YOLO("yolov8n.pt")
results = detector(frame, classes=[2, 7])  # 자동차, 트럭만 감지

# 색상 기반 감지
detector = AGVDetector(min_area=1000)
detections = detector.detect(frame, frame_number, timestamp)
```

### 2. 추적 단계 (Tracking)
```python
# 칼만 필터 초기화
kf = KalmanFilter(dim_x=6, dim_z=3)  # [x, y, θ, vx, vy, ω]
# 상태 벡터: [위치x, 위치y, 각도, 속도x, 속도y, 각속도]

# 예측 및 업데이트
kf.predict()
kf.update(measurement)
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

## ⚙️ 설정 파일

### tracker_config.json
```json
{
  "calibration": {
    "checkerboard_size": [9, 6],
    "square_size": 25.0,
    "camera_height": 2000.0
  },
  "measurement": {
    "min_agv_area": 1000,
    "fps": 30
  },
  "execution": {
    "presets": {
      "enhanced_stationary": {
        "mode": "enhanced",
        "detector": "yolo",
        "tracker": "kalman",
        "stationary_mode": true
      }
    }
  }
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

## 📁 데이터 구조

### 입력 데이터
- **비디오 파일**: MP4, AVI, MKV 등
- **이미지 시퀀스**: PNG, JPG 등
- **웹캠**: 실시간 카메라 입력

### 출력 데이터
- **CSV 로그**: 추적 결과 저장
- **비디오 녹화**: 측정 과정 녹화
- **스냅샷**: 특정 프레임 저장

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

### 칼만 필터 튜닝
```python
# 프로세스 노이즈 조정
kf.Q[0, 0] = kf.Q[1, 1] = 0.1  # 위치 노이즈
kf.Q[3, 3] = kf.Q[4, 4] = 1.0  # 속도 노이즈

# 측정 노이즈 조정
kf.R[0, 0] = kf.R[1, 1] = 10   # 위치 측정 노이즈
kf.R[2, 2] = 0.1              # 각도 측정 노이즈
```

### 감지 임계값 조정
```python
# 최소 객체 크기
min_area = 1000  # 픽셀

# 최대 추적 거리
max_distance = 500  # 픽셀

# 신뢰도 임계값
confidence_threshold = 0.5
```

## 🐛 문제 해결

### 일반적인 문제
1. **카메라 연결 실패**: 카메라 인덱스 확인
2. **보정 실패**: 체커보드 크기 및 품질 확인
3. **추적 실패**: 조명 조건 및 객체 크기 확인

### 성능 최적화
- GPU 가속 사용 (CUDA 지원)
- 해상도 조정
- 프레임 스킵 옵션

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

## 📞 지원

기술적 지원이나 질문이 있으시면 이슈를 생성해 주세요.