#!/usr/bin/env python3
"""
Novitec 1-shot software-trigger capture in an isolated process.

TriggerMode=On + TriggerSource=Software 인 경우 프레임은 TriggerSoftware 직후에만
나오므로, 이 워커는 연속 get_image() 루프를 돌리지 않고 TriggerSoftware + get_image
한 번만 수행하고 종료합니다.

Used by:
- CameraManager: 데몬(novitec_manual_trigger_daemon) 실패 시에만 1회 폴백
- novitec_api_wrapper_test.py 의 't' 키 캡처 (메인이 디바이스를 release 한 뒤 실행)

비전 서버는 START VISION 시 상주 데몬을 먼저 띄워 두고, 수동 시 IPC로 요청합니다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _ensure_novitec_on_path() -> Path:
    script = Path(__file__).resolve()
    repo_root = script.parent.parent
    novitec_src = repo_root / "submodules" / "novitec_camera_module" / "src"
    if novitec_src.is_dir():
        s = str(novitec_src)
        if s not in sys.path:
            sys.path.insert(0, s)
    return repo_root


def main() -> int:
    _ensure_novitec_on_path()
    try:
        from crp_camera.cam.novitec.novitec_camera import NovitecCamera
    except ImportError as e:
        print(f"Import NovitecCamera failed: {e}", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="Novitec manual single-frame capture worker")
    p.add_argument("--device-id", required=True)
    p.add_argument("--camera-index", type=int, required=True)
    p.add_argument("--config-json", default="", help="Optional camera params JSON path")
    p.add_argument("--output", required=True, help="Output PNG path (BGR)")
    args = p.parse_args()

    config: dict = {}
    cfg_path = Path(args.config_json) if args.config_json else None
    if cfg_path and cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            config = json.load(f)

    cam = NovitecCamera(
        device_id=args.device_id,
        device_ip=None,
        config=config,
        camera_index=args.camera_index,
    )
    if not cam.connect():
        print("connect failed", file=sys.stderr)
        return 2

    import cv2

    try:
        if not cam.start_stream():
            print("start_stream failed", file=sys.stderr)
            return 3
        time.sleep(0.12)

        frame = cam.grab_one_frame_software_trigger_then_disarm()
        if frame is None:
            data = cam.capture(output_formats=["image"])
            frame = data.get("image") if data else None
        if frame is None:
            print("no frame after trigger and free-run retry", file=sys.stderr)
            return 4
        if not cv2.imwrite(args.output, frame):
            print("cv2.imwrite failed", file=sys.stderr)
            return 5
        return 0
    finally:
        try:
            if getattr(cam, "_is_streaming", False):
                cam.stop_stream()
        except Exception:
            pass
        try:
            cam.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
