#!/usr/bin/env python3
"""
Novitec 수동 캡처용 **상주 프로세스** (vision 서버가 START VISION 시 기동).

매 요청마다 subprocess.run 으로 인터프리터를 새로 띄우면 연결·DLL 초기화 때문에 수 초 걸릴 수 있어,
이 데몬은 import·환경을 한 번만 로드하고 stdin JSON 한 줄당 connect→트리거 1장→disconnect 만 반복합니다.

프로토콜 (각 줄은 하나의 JSON 객체, UTF-8):
  요청: {"cmd":"capture","device_id":"...","camera_index":1,"config_json":"/path|.","output":"/path/to.png"}
        config_json 가 없거나 빈 문자열이면 설정 파일 없음.
  응답: {"ok": true} 또는 {"ok": false, "err": "..."}

  종료: {"cmd":"quit"}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _ensure_novitec_on_path() -> None:
    script = Path(__file__).resolve()
    repo_root = script.parent.parent
    novitec_src = repo_root / "submodules" / "novitec_camera_module" / "src"
    if novitec_src.is_dir():
        s = str(novitec_src)
        if s not in sys.path:
            sys.path.insert(0, s)


def _run_capture(
    device_id: str,
    camera_index: int,
    config: Dict[str, Any],
    output: str,
) -> Dict[str, Any]:
    import cv2
    from crp_camera.cam.novitec.novitec_camera import NovitecCamera

    cam = NovitecCamera(
        device_id=device_id,
        device_ip=None,
        config=config,
        camera_index=camera_index,
    )
    if not cam.connect():
        return {"ok": False, "err": "connect failed"}
    try:
        if not cam.start_stream():
            return {"ok": False, "err": "start_stream failed"}
        time.sleep(0.12)
        frame = cam.grab_one_frame_software_trigger_then_disarm()
        if frame is None:
            data = cam.capture(output_formats=["image"])
            frame = data.get("image") if data else None
        if frame is None:
            return {"ok": False, "err": "no frame after trigger/free-run"}
        if not cv2.imwrite(output, frame):
            return {"ok": False, "err": "cv2.imwrite failed"}
        return {"ok": True}
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


def main() -> int:
    # stderr는 로그용; stdout은 한 줄 JSON 응답 전용
    _ensure_novitec_on_path()
    try:
        import cv2  # noqa: F401 — 워밍업
        from crp_camera.cam.novitec.novitec_camera import NovitecCamera  # noqa: F401
    except ImportError as e:
        print(json.dumps({"ok": False, "err": f"warmup import: {e}"}), flush=True)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "err": f"bad json: {e}"}), flush=True)
            continue

        cmd = req.get("cmd")
        if cmd == "quit":
            break
        if cmd != "capture":
            print(
                json.dumps({"ok": False, "err": f"unknown cmd: {cmd!r}"}),
                flush=True,
            )
            continue

        try:
            device_id = req["device_id"]
            camera_index = int(req["camera_index"])
            output = req["output"]
            cfg_path = req.get("config_json") or ""
            config: Dict[str, Any] = {}
            if cfg_path and Path(cfg_path).is_file():
                with open(cfg_path, encoding="utf-8") as f:
                    config = json.load(f)
            result = _run_capture(device_id, camera_index, config, output)
        except Exception as e:
            result = {"ok": False, "err": str(e)}
        print(json.dumps(result), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
