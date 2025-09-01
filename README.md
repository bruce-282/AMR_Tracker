# AGV Measurement System

AGV(Automated Guided Vehicle) 측정 시스템으로, 카메라 캘리브레이션, 객체 감지, 크기 측정, 속도 추적을 통합한 완전한 시스템입니다.

## 기능

- **카메라 캘리브레이션**: 내부 파라미터 및 호모그래피 캘리브레이션
- **AGV 감지**: 색상 기반 및 딥러닝 기반 객체 감지
- **크기 측정**: 실시간 물리적 크기 측정 (mm 단위)
- **속도 추적**: 다중 객체 속도 및 궤적 추적
- **시각화**: 실시간 결과 표시 및 Bird's Eye View
- **설정 관리**: JSON 기반 설정 파일 관리

## 프로젝트 구조

```
agv_measurement_system/
├── calibration/
│   ├── __init__.py
│   ├── camera_calibrator.py      # 카메라 내부 파라미터 캘리브레이션
│   └── homography_calibrator.py  # 지면 평면 호모그래피 캘리브레이션
├── measurement/
│   ├── __init__.py
│   ├── size_measurement.py       # 크기 측정 모듈
│   └── speed_tracker.py          # 속도 추적 모듈
├── detection/
│   ├── __init__.py
│   └── agv_detector.py           # AGV 감지 모듈
├── visualization/
│   ├── __init__.py
│   └── display.py                # 시각화 모듈
├── utils/
│   ├── __init__.py
│   └── config.py                 # 설정 관리
├── sequence_loader.py             # 비디오 소스 로더 (웹캠, 비디오, 이미지 시퀀스)
├── main.py                       # 통합 AMR Tracker (Basic/Enhanced 모드)
├── agv_system.py                 # AGV Measurement System
└── README.md
```

## 설치

uv를 사용한 설치:

```bash
# 가상환경 생성 및 의존성 설치
uv sync

# 또는 직접 실행
uv run agv_system.py calibrate
```

## 사용법

### 1. 캘리브레이션 모드

먼저 카메라와 지면 평면을 캘리브레이션해야 합니다:

```bash
# 캘리브레이션 실행
uv run agv_system.py calibrate

# 또는
python agv_system.py calibrate
```

캘리브레이션 과정:
1. 체커보드를 다양한 각도에서 촬영 (최소 3장)
2. 지면에 체커보드를 놓고 촬영
3. 캘리브레이션 데이터가 `calibration_data.json`에 저장됨

### 2. 측정 모드

캘리브레이션 완료 후 측정을 실행:

```bash
# 웹캠 사용
uv run agv_system.py measure

# 특정 카메라 사용
uv run agv_system.py measure --source 1

# 비디오 파일 사용
uv run agv_system.py measure --source video.mp4

# 설정 파일 지정
uv run agv_system.py measure --config custom_config.json
```

### 3. 통합 AMR Tracker 사용 (main.py)

#### Basic Mode (기본 AMR Tracker)
```bash
# 기본 모드 (AMR Tracker만)
uv run main.py --mode basic

# 또는 기본 실행 (자동으로 basic 모드 선택)
uv run main.py

# 정지 모드 (위치와 방향만 측정, 속도 계산 없음)
uv run main.py --mode basic --stationary-mode

# 다양한 로더 모드
uv run main.py --mode basic --loader-mode camera --source 0
uv run main.py --mode basic --loader-mode video --source "video.mp4"
uv run main.py --mode basic --loader-mode sequence --source "data/0010_fixed" --fps 10
```

#### Enhanced Mode (AGV Measurement System)
```bash
# 향상된 모드 (AGV system 기능 포함)
uv run main.py --mode enhanced --config config.json

# 다양한 감지기와 추적기 선택
uv run main.py --mode enhanced --detector yolo --tracker kalman --config config.json
uv run main.py --mode enhanced --detector color --tracker speed --config config.json
uv run main.py --mode enhanced --detector agv --tracker speed --config config.json

# 정지 모드와 함께 사용
uv run main.py --mode enhanced --stationary-mode --config config.json

# 다양한 입력 소스와 함께
uv run main.py --mode enhanced --source "data/0010_fixed" --fps 30 --config config.json
uv run main.py --mode enhanced --source "video.mp4" --detector yolo --tracker kalman --config config.json
```

#### 로더 모드 (Loader Modes)
- **`--loader-mode auto`**: 소스 타입 자동 감지 (기본값)
- **`--loader-mode camera`**: 카메라 장치 (웹캠)
- **`--loader-mode video`**: 비디오 파일
- **`--loader-mode sequence`**: 이미지 시퀀스 폴더

#### 정지 모드 (Stationary Mode)
- **`--stationary-mode`**: 속도 계산을 건너뛰고 위치(x, y)와 방향(yaw)만 측정
- 정지된 차량의 정확한 위치와 방향 측정에 유용

## 설정

`config.json` 파일에서 시스템 설정을 관리할 수 있습니다:

```json
{
  "calibration": {
    "checkerboard_size": [9, 6],
    "square_size": 25.0,
    "num_calibration_images": 15,
    "camera_height": 2000.0
  },
  "measurement": {
    "min_agv_area": 1000,
    "max_tracking_distance": 500,
    "fps": 30
  },
  "display_scale": 0.5,
  "record_video": false,
  "output_video_path": "output.mp4"
}
```

## 의존성

- **opencv-python**: 컴퓨터 비전 및 이미지 처리
- **numpy**: 수치 계산
- **filterpy**: Kalman 필터 구현
- **ultralytics**: YOLO 객체 감지
- **matplotlib**: 고급 시각화

## 주요 기능

### 시스템 모드 (System Modes)
- **Basic Mode**: 기본 AMR Tracker (Kalman filter 기반 다중 객체 추적)
- **Enhanced Mode**: AGV Measurement System (카메라 캘리브레이션, 크기 측정, 고급 시각화)

### 감지기 (Detectors)
- **YOLO**: 딥러닝 기반 객체 감지 (기본)
- **Color-based**: HSV 색공간 기반 색상 감지
- **AGV-specific**: AGV 전용 감지기 (색상 + 형태 기반)

### 추적기 (Trackers)
- **Kalman Filter**: **다중 객체** 고정밀 추적 (위치, 속도, 방향)
  - 각 객체마다 고유한 색상으로 궤적 표시
  - 자동 객체 연결 및 추적 ID 관리
  - 실시간 다중 차량 추적
  - 정지 모드 지원 (위치와 방향만)
- **Speed Tracker**: 다중 객체 속도 및 궤적 추적

### 비디오 소스 로더 (Sequence Loader)
- **통합 인터페이스**: 웹캠, 비디오 파일, 이미지 시퀀스를 하나의 클래스로 처리
- **자동 감지**: 소스 타입을 자동으로 판단하여 적절한 로더 선택
- **명시적 모드**: `--loader-mode`로 원하는 로더 타입 직접 지정
- **이미지 시퀀스**: PNG 파일들의 순차적 처리 (000000.png, 000001.png, ...)

### 카메라 캘리브레이션
- 체커보드 기반 내부 파라미터 캘리브레이션
- 지면 평면 호모그래피 계산
- 왜곡 보정

### 크기 측정
- 픽셀 좌표를 실제 물리적 크기로 변환 (mm 단위)
- 높이 보정을 통한 정확한 측정
- 품질 지표 제공

### 속도 추적
- 다중 객체 궤적 추적
- 실시간 속도 계산 (mm/s)
- 방향 및 가속도 분석
- 정지 모드에서 속도 계산 건너뛰기

### 시각화
- 실시간 바운딩 박스 및 측정값 표시
- Bird's Eye View 제공 (Enhanced mode)
- 통계 패널 및 속도 벡터 표시
- 향상된 다중 객체 시각화
- 정지 모드에서 위치 좌표 표시

### 입력 소스 지원
- **카메라 장치**: 실시간 카메라 입력 (웹캠)
- **비디오 파일**: MP4, AVI 등 다양한 형식
- **이미지 시퀀스**: PNG 파일들의 순차적 처리
- **폴더 자동 감지**: PNG 파일이 있는 폴더를 자동으로 이미지 시퀀스로 인식

### 호환성
- 기존 AMR Tracker와 완전 호환
- 점진적 업그레이드 가능 (Basic → Enhanced)
- 모듈식 설계로 유연한 구성
- OpenCV VideoCapture와 완전 호환되는 인터페이스
- Sequence Loader를 통한 통합된 비디오 소스 관리

## 사용 예시

### 기본 사용법
```bash
# 1. 기본 AMR Tracker (웹캠)
uv run main.py --mode basic --loader-mode camera --source 0

# 2. 이미지 시퀀스로 다중 객체 추적
uv run main.py --mode basic --loader-mode sequence --source "data/0010_fixed" --fps 10

# 3. 정지 모드로 위치/방향만 측정
uv run main.py --mode basic --loader-mode sequence --source "data/0010_fixed" --fps 10 --stationary-mode
```

### 고급 사용법
```bash
# 1. Enhanced mode로 카메라 캘리브레이션 활용
uv run main.py --mode enhanced --loader-mode sequence --source "data/0010_fixed" --fps 10 --config config.json

# 2. 특정 감지기와 추적기 조합
uv run main.py --mode enhanced --detector yolo --tracker kalman --loader-mode sequence --source "data/0010_fixed" --fps 10 --config config.json

# 3. 정지 모드 + Enhanced mode
uv run main.py --mode enhanced --stationary-mode --loader-mode sequence --source "data/0010_fixed" --fps 10 --config config.json
```

### AGV System 직접 사용
```bash
# 1. 캘리브레이션
uv run agv_system.py calibrate

# 2. 측정
uv run agv_system.py measure --source 0

# 3. 비디오 파일로 측정
uv run agv_system.py measure --source "video.mp4"
```

### Preset 사용법 (간편한 실행)
자주 사용하는 설정을 preset으로 저장하여 간단하게 실행할 수 있습니다:

```bash
# 1. 정지 모드 + Enhanced mode (위치/방향만 측정)
uv run main.py --preset stationary_enhanced

# 2. 기본 추적 모드
uv run main.py --preset basic_tracking

# 3. 웹캠 기본 모드
uv run main.py --preset webcam_basic
```

**사용 가능한 Preset:**
- **`stationary_enhanced`**: 정지 모드 + Enhanced mode (data/0001_fixed, 30fps)
- **`basic_tracking`**: 기본 추적 모드 (data/0001_fixed, 30fps)
- **`webcam_basic`**: 웹캠 기본 모드 (카메라 0번, 30fps)

**Preset 커스터마이징:**
`config.json`의 `execution.presets` 섹션에서 새로운 preset을 추가하거나 기존 preset을 수정할 수 있습니다.

## 주의사항

- **Enhanced Mode**: `config.json` 파일과 AGV Measurement System 모듈이 필요합니다
- **정지 모드**: 속도 계산을 건너뛰므로 정지된 객체의 정확한 위치와 방향 측정에 유용합니다
- **로더 모드**: `auto` 모드가 기본이지만, 특정 소스 타입을 강제하려면 명시적으로 지정하세요
- **이미지 시퀀스**: PNG 파일들이 000000.png, 000001.png 형식으로 정렬되어 있어야 합니다
