# AMR Tracker Vision Server - 에러 코드 문서

이 문서는 `run_server.py`를 통해 실행되는 비전 트래킹 서버 시스템의 에러 코드를 정리합니다.

---

## 목차

1. [TCP/IP 응답 에러 코드](#1-tcpip-응답-에러-코드)
2. [시스템 예외 (Exceptions)](#2-시스템-예외-exceptions)
3. [에러 코드 상세 설명](#3-에러-코드-상세-설명)
4. [문제 해결 가이드](#4-문제-해결-가이드)

---

## 1. TCP/IP 응답 에러 코드

비전 서버는 클라이언트 요청에 대해 JSON 형식으로 응답합니다.
에러 발생 시 `success: false`와 함께 `error_code`와 `error_desc`가 포함됩니다.

### 응답 형식

```json
{
    "cmd": <명령코드>,
    "success": false,
    "error_code": "<에러코드>",
    "error_desc": "<에러설명>"
}
```

### 에러 코드 목록

| 에러 코드 | 발생 위치 | 설명 | 관련 명령 |
|-----------|----------|------|----------|
| `INVALID_CMD` | `vision_server.py` | 알 수 없는 명령어 수신 | 모든 명령 |
| `INTERNAL_ERROR` | `vision_server.py` | 서버 내부 예외 발생 | 모든 명령 |
| `INIT_ERROR` | `vision_server.py` | 시스템/카메라 초기화 실패 | START_VISION (cmd: 1) |
| `CAM_INIT_ERROR` | `vision_server.py` | 특정 카메라 초기화 실패 | START_VISION (cmd: 1) |
| `CONNECTION_FAILED` | `vision_server.py` | 카메라 연결 확인 실패 | START_VISION (cmd: 1) |
| `VISION_NOT_ACTIVE` | `vision_server.py` | 비전 시스템이 시작되지 않은 상태에서 카메라 명령 수신 | START_CAM_* (cmd: 3, 4, 5) |
| `CAM_START_ERROR` | `vision_server.py` | 카메라 트래킹 시작 실패 | START_CAM_* (cmd: 3, 4, 5) |
| `STOP_ERROR` | `vision_server.py` | 비전 시스템 종료 중 오류 | END_VISION (cmd: 2) |
| `MISSING_PARAM` | `vision_server.py` | 필수 파라미터 누락 | CALC_RESULT (cmd: 6) |
| `INVALID_PATH` | `vision_server.py` | 파일 경로 형식 오류 | CALC_RESULT (cmd: 6) |
| `FILE_NOT_FOUND` | `vision_server.py` | 요청된 파일을 찾을 수 없음 | CALC_RESULT (cmd: 6) |
| `INVALID_FORMAT` | `vision_server.py` | CSV 파일 형식 오류 (확장자, 인코딩, 빈 파일) | CALC_RESULT (cmd: 6) |
| `FILE_READ_ERROR` | `vision_server.py` | 파일 읽기 권한 오류 | CALC_RESULT (cmd: 6) |
| `INVALID_CSV_STRUCTURE` | `vision_server.py` | CSV 필수 컬럼 누락 | CALC_RESULT (cmd: 6) |
| `INSUFFICIENT_DATA` | `vision_server.py` | 분석에 필요한 데이터 부족 (최소 2개 trial) | CALC_RESULT (cmd: 6) |
| `INVALID_PARAM` | `vision_server.py` | 파라미터 값 오류 (sampling_interval_mm 등) | CALC_RESULT (cmd: 6) |
| `OUTPUT_DIR_ERROR` | `vision_server.py` | 출력 디렉토리 생성/쓰기 오류 | CALC_RESULT (cmd: 6) |
| `CALC_ERROR` | `vision_server.py` | Trajectory 분석 계산 중 오류 | CALC_RESULT (cmd: 6) |
| `VISION_ENDED` | `vision_server.py` | 비전 시스템 종료로 인한 연결 해제 | NOTIFY_CONNECTION (cmd: 7) |

---

## 2. 시스템 예외 (Exceptions)

`amr_tracker.py` 및 관련 모듈에서 발생할 수 있는 예외입니다.

### EnhancedAMRTracker 초기화 예외

| 예외 유형 | 발생 조건 | 에러 메시지 |
|----------|----------|------------|
| `ValueError` | `pixel_size`가 dict 형태가 아닌 경우 | `pixel_size must be a dict with 'x' and 'y' keys, got {type}` |
| `ImportError` | Detection 모듈 미설치 | `Detection module is not installed.` |
| `FileNotFoundError` | 학습 모델 파일 없음 | `weights file not found: {model_path}` |
| `ValueError` | 지원되지 않는 detector 유형 | `Unsupported detector type: {type}. Supported types: 'yolo', 'binary'` |
| `ValueError` | 지원되지 않는 tracker 유형 | `Unsupported tracker type: {type}. Only 'kalman' is supported.` |

### 설정 관련 예외

| 예외 유형 | 발생 조건 | 에러 메시지 |
|----------|----------|------------|
| `ValueError` | 유효하지 않은 모델 인덱스 | `Invalid model index: {index}. Available models: {model_list}` |
| `ValueError` | 선택된 모델 없음 | `No model selected. Please provide model in request or set in config.` |
| `FileNotFoundError` | 모델 파일 없음 | `Model file not found: {path}` |
| `ValueError` | YOLO detector에 model_path 누락 | `model_path is required for YOLO detector` |

---

## 3. 에러 코드 상세 설명

### 3.1 `INVALID_CMD`

**원인:**
- 클라이언트가 정의되지 않은 명령 코드를 전송

**유효한 명령 코드:**
| cmd | 명령 | 설명 |
|-----|------|------|
| 1 | START_VISION | 비전 시스템 시작 |
| 2 | END_VISION | 비전 시스템 종료 |
| 3 | START_CAM_1 | 카메라 1 트래킹 시작 |
| 4 | START_CAM_2 | 카메라 2 트래킹 시작 |
| 5 | START_CAM_3 | 카메라 3 트래킹 시작 |
| 6 | CALC_RESULT | Trajectory 분석 실행 |
| 7 | NOTIFY_CONNECTION | 연결 상태 알림 (서버→클라이언트) |

**해결 방법:**
- 요청 JSON의 `cmd` 필드가 1~6 범위인지 확인

---

### 3.2 `INTERNAL_ERROR`

**원인:**
- 서버 내부에서 예상치 못한 예외 발생

**해결 방법:**
- 서버 로그 확인 (`C:/CMES_AI/Log/` 또는 설정된 경로)
- `error_desc`에 상세 오류 메시지 포함

---

### 3.3 `INIT_ERROR` / `CAM_INIT_ERROR`

**원인:**
- 카메라 초기화 실패
- 설정 파일 로드 실패
- tracker_config 파일 파싱 오류

**확인 사항:**
1. `config/model_config.json`에서 `selected_model` 확인
2. `config/zoom1.json` (또는 해당 product model 설정) 존재 여부
3. 각 카메라의 `tracker_config` 파일 경로 확인:
   - `config/cam1_tracker_config.json`
   - `config/cam2_tracker_config.json`
   - `config/cam3_tracker_config.json`
4. 카메라 연결 상태 확인 (video mode인 경우 파일 존재 여부)

---

### 3.4 `CONNECTION_FAILED`

**원인:**
- 카메라 연결 확인 실패 (프레임 읽기 불가)

**확인 사항:**
1. 카메라 장치가 연결되어 있는지 확인
2. 다른 프로그램이 카메라를 사용 중인지 확인
3. 비디오 모드인 경우 파일 경로 확인

---

### 3.5 `VISION_NOT_ACTIVE`

**원인:**
- `START_VISION` 명령 없이 `START_CAM_*` 명령 전송

**해결 방법:**
- 카메라 명령 전에 반드시 `START_VISION` (cmd: 1) 먼저 실행

**올바른 명령 순서:**
```
1. START_VISION (cmd: 1)
2. [자동] Camera 1, 2, 3 초기화 및 NOTIFY_CONNECTION 전송
3. [자동 또는 수동] START_CAM_1 (cmd: 3)
4. ... 트래킹 수행 ...
5. END_VISION (cmd: 2)
```

---

### 3.6 CALC_RESULT (cmd: 6) 명령 관련 에러

Trajectory 반복정밀도 분석 명령에서 발생할 수 있는 에러 코드입니다.

| 에러 코드 | 원인 | 해결 방법 |
|----------|------|----------|
| `MISSING_PARAM` | `path_csv` 파라미터 누락 | 요청에 `path_csv` 필드 추가 |
| `INVALID_PATH` | 파일 경로가 유효하지 않음 | 경로가 빈 문자열이 아닌지 확인 |
| `FILE_NOT_FOUND` | CSV 파일이 존재하지 않음 | 파일 경로 및 존재 여부 확인 |
| `INVALID_FORMAT` | CSV 파일 형식 오류 | 확장자(.csv), 인코딩(UTF-8), 파일 내용 확인 |
| `FILE_READ_ERROR` | 파일 읽기 권한 오류 | 파일 읽기 권한 확인 |
| `INVALID_CSV_STRUCTURE` | CSV 필수 컬럼 누락 | 아래 필수 컬럼 확인 |
| `INSUFFICIENT_DATA` | 분석에 필요한 데이터 부족 | 최소 2개 이상의 trial 데이터 필요 |
| `INVALID_PARAM` | 파라미터 값 오류 | `sampling_interval_mm`은 양수 숫자여야 함 |
| `OUTPUT_DIR_ERROR` | 출력 디렉토리 오류 | 출력 디렉토리 쓰기 권한 확인 |
| `CALC_ERROR` | 분석 계산 중 오류 | 데이터 품질 확인, 로그 확인 |

**필수 CSV 컬럼:**
```
Camera 1 정지 위치: cam_1_x, cam_1_y, cam_1_rz
Camera 3 정지 위치: cam_3_x, cam_3_y, cam_3_rz
Camera 2 궤적: cam_2_x_0, cam_2_y_0, cam_2_rz_0 ~ cam_2_x_N, cam_2_y_N, cam_2_rz_N
```

**올바른 요청 형식:**
```json
{
    "cmd": 6,
    "path_csv": "C:/CMES_AI/Result/20251216_raw_data.csv",
    "sampling_interval_mm": 20.0
}
```

**에러 응답 예시:**
```json
{
    "cmd": 6,
    "success": false,
    "error_code": "INVALID_CSV_STRUCTURE",
    "error_desc": "CSV 파일에 필수 컬럼이 누락되었습니다: cam_1_x, cam_1_y. 필요한 컬럼: cam_1_x/y/rz, cam_3_x/y/rz, cam_2_x/y/rz_0~N"
}
```

---

## 4. 문제 해결 가이드

### 4.1 서버 시작 시 발생하는 에러

**문제:** 서버 시작 후 바로 에러 발생

**확인 순서:**
1. `config/model_config.json` 파일 존재 및 형식 확인
2. `selected_model`에 해당하는 설정 파일 확인 (예: `config/zoom1.json`)
3. Detection 학습 모델 파일 확인 (예: `weights/zoom1/best.pt`)

### 4.2 카메라 연결 실패

**문제:** `CONNECTION_FAILED` 또는 `CAM_INIT_ERROR`

**확인 순서:**
1. 카메라 장치 물리적 연결 상태
2. `zoom1.json`의 preset 설정에서 소스 경로 확인:
   ```json
   "camera_tracking": {
       "camera1": {
           "loader_mode": "video",
           "source": "data/video1.mp4"
       }
   }
   ```
3. 카메라 드라이버 설치 상태 (Novitec SDK 등)

### 4.3 트래킹 중 오류

**문제:** 트래킹 도중 에러 발생

**확인 사항:**
1. `tracker_config` 파일의 설정값 범위 확인:
   - `confidence_threshold`: 0.0 ~ 1.0
   - `speed_threshold_pix_per_frame`: 양수
   - `boundary_margin_ratio`: 0.0 ~ 0.5

2. Detection 모델과 입력 이미지 호환성:
   - `imgsz` 설정 확인
   - 모델이 학습된 클래스 확인 (`target_classes`)

### 4.4 로그 확인 방법

서버 로그는 다음 경로에 저장됩니다:
- 기본 경로: `C:/CMES_AI/Log/`
- 설정 경로: `zoom1.json`의 `execution.log_base_path`

로그 레벨은 `execution.log_level`로 설정 (DEBUG, INFO, WARNING, ERROR)

---

## 5. 에러 코드 반환 예시

### 성공 응답

```json
{
    "cmd": 1,
    "success": true
}
```

### 에러 응답

```json
{
    "cmd": 1,
    "success": false,
    "error_code": "CAM_INIT_ERROR",
    "error_desc": "Failed to initialize cameras: [2]. Cannot start tracking."
}
```

### NOTIFY_CONNECTION (카메라 연결 알림)

**연결 성공:**
```json
{
    "cmd": 7,
    "camera_id": 1,
    "is_connected": true
}
```

**연결 실패:**
```json
{
    "cmd": 7,
    "camera_id": 1,
    "is_connected": false,
    "error_code": "CONNECTION_FAILED",
    "error_desc": "Camera connection check failed"
}
```

---

## 6. 관련 설정 파일 구조

```
AMR_Tracker/
├── config/
│   ├── model_config.json          # 모델 선택 및 기본 설정
│   ├── zoom1.json                 # Product model 설정 (zoom1)
│   ├── cam1_tracker_config.json   # Camera 1 전용 detector/tracker 설정
│   ├── cam2_tracker_config.json   # Camera 2 전용 detector/tracker 설정
│   ├── cam3_tracker_config.json   # Camera 3 전용 detector/tracker 설정
│   └── camera*_config.json        # 카메라 캘리브레이션 데이터
├── weights/
│   └── zoom1/
│       └── best.pt                # Detection 모델 가중치
└── src/
    └── server/
        ├── vision_server.py       # 메인 서버 로직
        ├── protocol.py            # 프로토콜 정의 및 에러 코드 생성
        └── ...
```

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2025-12-16 | 초기 문서 작성 |

