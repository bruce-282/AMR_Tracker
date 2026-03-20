"""
Novitec 카메라 1 전용: 스트림·캡처를 **별도 프로세스**에서 수헹하고, 부모는 Queue로 프레임만 수신.

- Windows: multiprocessing spawn
- 자식 프로세스 안에서는 기존 NovitecCameraLoader 로직 그대로 사용 (DLL·버퍼 동일)
"""
from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from multiprocessing import Event, Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sequence_loader import BaseLoader

logger = logging.getLogger(__name__)


def _novitec_cam1_worker_main(
    shutdown_evt: Any,
    cmd_queue: Any,
    frame_queue: Any,
    device_id: str,
    config: Dict[str, Any],
    camera_index: int,
    enable_undistortion: bool,
    camera_matrix_list: Optional[List[List[float]]],
    dist_coeffs_list: Optional[List[float]],
    enable_buffering: bool,
    buffer_size: int,
    buffer_drop_policy: str,
) -> None:
    """자식 프로세스 엔트리 (spawn 가능한 최상위 함수)."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    novitec_src = root / "submodules" / "novitec_camera_module" / "src"
    if novitec_src.is_dir() and str(novitec_src) not in sys.path:
        sys.path.insert(0, str(novitec_src))

    from src.utils.sequence_loader import NovitecCameraLoader

    cm = np.array(camera_matrix_list, dtype=np.float64) if camera_matrix_list else None
    dc = np.array(dist_coeffs_list, dtype=np.float64) if dist_coeffs_list else None

    # Windows 10048 (WSAEADDRINUSE): 이전 프로세스/소켓이 TIME_WAIT 인 채로 재접속하면 발생 가능.
    # 짧게 재시도하면 GigE·Novitec 제어 포트가 풀린 뒤 붙는 경우가 많음.
    loader: Optional[NovitecCameraLoader] = None
    last_err: Optional[Exception] = None
    for attempt in range(1, 6):
        try:
            loader = NovitecCameraLoader(
                device_id=device_id,
                config=config,
                enable_undistortion=enable_undistortion,
                camera_matrix=cm,
                dist_coeffs=dc,
                camera_index=camera_index,
                enable_buffering=enable_buffering,
                buffer_size=buffer_size,
                buffer_drop_policy=buffer_drop_policy,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            loader = None
            print(
                f"[cam1-worker] NovitecCameraLoader init attempt {attempt}/5 failed: {e}",
                flush=True,
            )
            if attempt < 5:
                time.sleep(1.0 * attempt)

    if loader is None:
        print(f"[cam1-worker] giving up after connect/init errors: {last_err}", flush=True)
        return

    try:
        if not loader._ensure_stream_started():
            print("[cam1-worker] stream start failed", flush=True)
            return
        loader.start_buffering()
        fn = 0
        paused = False  # stop_stream 후 read() 하면 NovitecLoader가 스트림 자동 재시작할 수 있음
        while not shutdown_evt.is_set():
            try:
                while True:
                    c = cmd_queue.get_nowait()
                    op = c.get("op")
                    if op == "shutdown":
                        shutdown_evt.set()
                        break
                    if op == "stop_stream":
                        paused = True
                        loader.stop_buffering()
                        if loader.camera and getattr(loader.camera, "_is_streaming", False):
                            loader.camera.stop_stream()
                        loader._stream_started = False
                    elif op == "start_stream":
                        paused = False
                        if loader._ensure_stream_started():
                            loader.start_buffering()
            except queue.Empty:
                pass
            if shutdown_evt.is_set():
                break
            if paused:
                time.sleep(0.02)
                continue
            ret, frame = loader.read()
            if ret and frame is not None:
                fn += 1
                try:
                    frame_queue.put_nowait((fn, time.time(), frame.copy()))
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait((fn, time.time(), frame.copy()))
                    except queue.Full:
                        pass
    except Exception as e:
        print(f"[cam1-worker] fatal: {e}", flush=True)
        import traceback

        traceback.print_exc()
    finally:
        if loader is not None:
            try:
                loader.release()
            except Exception:
                pass
            time.sleep(0.3)


class _SubprocessCam1Proxy:
    """camera_manager 가 기대하는 Novitec camera 객체와 유사한 인터페이스."""

    def __init__(self, owner: "NovitecCamera1SubprocessLoader") -> None:
        self._owner = owner

    @property
    def _is_streaming(self) -> bool:
        return self._owner._streaming

    def start_stream(self) -> bool:
        return self._owner._request_start_stream()

    def stop_stream(self) -> bool:
        return self._owner._request_stop_stream()

    def check_connection(self) -> bool:
        return self._owner.is_opened()


class NovitecCamera1SubprocessLoader(BaseLoader):
    """
    CAM1 Novitec: 부모에는 카메라 핸들 없음. 자식 프로세스만 connect/stream.
    """

    def __init__(
        self,
        device_id: str,
        config: Optional[dict] = None,
        enable_undistortion: bool = False,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        camera_index: int = 1,
        enable_buffering: bool = True,
        buffer_size: int = 30,
        buffer_drop_policy: str = "oldest",
        fps: float = 30.0,
    ) -> None:
        super().__init__(
            enable_undistortion=enable_undistortion,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        self.device_id = device_id
        self.config = dict(config or {})
        self.camera_index = camera_index
        self.enable_buffering = enable_buffering
        self.buffer_size = buffer_size
        self.buffer_drop_policy = buffer_drop_policy
        self.fps = fps

        cm_list: Optional[List[List[float]]] = None
        dc_list: Optional[List[float]] = None
        if camera_matrix is not None:
            cm_list = camera_matrix.tolist()
        if dist_coeffs is not None:
            dc_list = np.asarray(dist_coeffs).flatten().tolist()

        ctx = multiprocessing.get_context("spawn")
        self._shutdown = ctx.Event()
        self._cmd_queue: Queue = ctx.Queue()
        self._frame_queue: Queue = ctx.Queue(maxsize=3)
        self._proc = ctx.Process(
            target=_novitec_cam1_worker_main,
            args=(
                self._shutdown,
                self._cmd_queue,
                self._frame_queue,
                device_id,
                self.config,
                camera_index,
                enable_undistortion,
                cm_list,
                dc_list,
                enable_buffering,
                buffer_size,
                buffer_drop_policy,
            ),
            daemon=False,
            name="NovitecCam1Stream",
        )
        self._proc.start()
        self.initialized = True
        self.is_connected = True
        self._streaming = True
        self._last_ts: Optional[float] = None
        self.camera = _SubprocessCam1Proxy(self)

        if not self._proc.is_alive():
            raise RuntimeError(f"CAM1 subprocess exited immediately (device={device_id})")
        logger.info(
            f"NovitecCamera1SubprocessLoader: child pid={self._proc.pid} device={device_id}"
        )

    def _request_stop_stream(self) -> bool:
        try:
            self._cmd_queue.put_nowait({"op": "stop_stream"})
        except Exception as e:
            logger.warning(f"CAM1 subprocess stop_stream cmd failed: {e}")
            return False
        self._streaming = False
        return True

    def _request_start_stream(self) -> bool:
        try:
            self._cmd_queue.put_nowait({"op": "start_stream"})
        except Exception as e:
            logger.warning(f"CAM1 subprocess start_stream cmd failed: {e}")
            return False
        self._streaming = True
        return True

    def check_connection(self) -> bool:
        return self.is_opened()

    def is_opened(self) -> bool:
        return bool(self._proc.is_alive() and self.initialized)

    def is_stream_process_alive(self) -> bool:
        """자식 캡처 프로세스 생존 여부 (큐 타임아웃으로 read 실패와 구분)."""
        return bool(self._proc.is_alive())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._proc.is_alive():
            return False, None
        try:
            fn, ts, frame = self._frame_queue.get(timeout=0.15)
            self.frame_number = fn
            self._last_ts = ts
            return True, frame
        except queue.Empty:
            return False, None

    def get_timestamp(self) -> Optional[float]:
        return self._last_ts

    def start_buffering(self) -> None:
        """자식 내부에서 이미 버퍼링; 부모는 no-op."""
        pass

    def stop_buffering(self) -> None:
        self._request_stop_stream()

    def get_buffer_stats(self) -> Optional[dict]:
        return None

    def release(self) -> None:
        self.initialized = False
        self.is_connected = False
        self._streaming = False
        self._shutdown.set()
        try:
            self._cmd_queue.put_nowait({"op": "shutdown"})
        except Exception:
            pass
        self._proc.join(timeout=20)
        if self._proc.is_alive():
            logger.warning("CAM1 subprocess did not exit; terminate")
            self._proc.terminate()
            self._proc.join(timeout=5)
        # 짧게 대기: GigE/Novitec 제어 소켓(10048) TIME_WAIT 해소
        time.sleep(0.5)
        self.camera = None  # type: ignore
        logger.info(f"NovitecCamera1SubprocessLoader released device={self.device_id}")
