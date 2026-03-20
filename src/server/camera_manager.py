"""Camera management for vision server."""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2

from src.utils.sequence_loader import create_sequence_loader, BaseLoader
from src.utils.config_loader import (
    get_camera_config, 
    get_camera_pixel_sizes, 
    get_camera_distance_map_paths, 
    get_camera_homographies,
    get_execution_config,
    get_camera_config_from_preset,
    load_tracker_config_file,
    load_product_model_config
)
from src.core.amr_tracker import EnhancedAMRTracker
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

# True: CAM1 Novitec만 자식 프로세스에서 connect/stream (NovitecCamera1SubprocessLoader)
USE_NOVITEC_CAM1_SUBPROCESS_STREAM = True


def _loader_is_novitec_family(loader: Any) -> bool:
    if loader is None:
        return False
    from src.utils.sequence_loader import NovitecCameraLoader
    from src.utils.novitec_cam1_subprocess_loader import NovitecCamera1SubprocessLoader

    return isinstance(loader, (NovitecCameraLoader, NovitecCamera1SubprocessLoader))


class CameraManager:
    """Manages camera initialization, loaders, and AMR trackers."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        system_config: Optional[Any] = None,  # Not used - kept for compatibility
        preset_name: Optional[str] = None
    ):
        """
        Initialize camera manager.
        
        Args:
            model_config: Model configuration manager
            system_config: Not used (deprecated - all configs json)
            preset_name: Preset name override
        """
        self.model_config = model_config
        self.config = None  # Not used - all configs json
        self.preset_name = preset_name
        
        # Camera resources
        self.camera_loaders: Dict[int, BaseLoader] = {}
        self.amr_trackers: Dict[int, EnhancedAMRTracker] = {}
        self.camera_pixel_sizes: Dict[int, float] = {}
        self.camera_distance_map_paths: Dict[int, Optional[str]] = {}
        self.camera_homographies: Dict[int, Optional[Any]] = {}  # 호모그래피 행렬
        self.camera_edge_refinement_config: Dict[int, Dict] = {}  # Edge refinement 설정
        self.frame_numbers: Dict[int, int] = {}
        self.camera_status: Dict[int, bool] = {1: False, 2: False, 3: False}
        
        # Video source cycling: id가 리스트인 경우 회차마다 순환 선택
        self._video_source_lists: Dict[int, List[str]] = {}  # camera_id -> source list
        self._video_source_indices: Dict[int, int] = {}  # camera_id -> current index
        
        # Trackers dict for compatibility with existing code that accesses trackers directly
        # Note: EnhancedAMRTracker manages its own tracker internally
        self.trackers: Dict[int, Dict] = {}  # camera_id -> tracker dict
        self.next_track_ids: Dict[int, int] = {}

        # Novitec 수동 캡처: 상주 데몬(프로세스 1개) — 요청마다 cold subprocess 대신 stdin JSON 1줄
        self._novitec_manual_daemon_lock = threading.Lock()
        self._novitec_manual_daemon_proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _amr_repo_root() -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def ensure_novitec_manual_daemon(self) -> bool:
        """
        `scripts/novitec_manual_trigger_daemon.py` 가 있으면 상주 프로세스를 기동.
        START VISION 직후 호출 권장 (첫 수동 캡처 지연 완화).
        """
        with self._novitec_manual_daemon_lock:
            return self._ensure_novitec_manual_daemon_unlocked()

    def _ensure_novitec_manual_daemon_unlocked(self) -> bool:
        script = self._amr_repo_root() / "scripts" / "novitec_manual_trigger_daemon.py"
        if not script.is_file():
            return False
        proc = self._novitec_manual_daemon_proc
        if proc is not None and proc.poll() is None:
            return True
        self._novitec_manual_daemon_proc = None
        repo_root = self._amr_repo_root()
        novitec_src = repo_root / "submodules" / "novitec_camera_module" / "src"
        py_path = str(repo_root)
        if novitec_src.is_dir():
            py_path = f"{str(novitec_src)}{os.pathsep}{py_path}"
        env = os.environ.copy()
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{py_path}{os.pathsep}{prev}" if prev else py_path
        try:
            self._novitec_manual_daemon_proc = subprocess.Popen(
                [sys.executable, str(script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(repo_root),
            )
        except Exception as e:
            logger.error(f"Novitec manual daemon start failed: {e}")
            self._novitec_manual_daemon_proc = None
            return False
        logger.info("Novitec manual capture daemon started (persistent; warmup for fast manual capture)")
        return True

    def stop_novitec_manual_daemon(self) -> None:
        """END VISION 등에서 데몬 종료."""
        with self._novitec_manual_daemon_lock:
            proc = self._novitec_manual_daemon_proc
            self._novitec_manual_daemon_proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
        logger.info("Novitec manual capture daemon stopped")

    def _novitec_manual_daemon_capture_unlocked(
        self,
        device_id: str,
        camera_index: int,
        config_tmp: str,
        out_png: str,
    ) -> bool:
        proc = self._novitec_manual_daemon_proc
        if proc is None or proc.poll() is not None:
            return False
        req = {
            "cmd": "capture",
            "device_id": device_id,
            "camera_index": camera_index,
            "config_json": config_tmp,
            "output": out_png,
        }
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                logger.error("Novitec manual daemon: empty response (crashed?)")
                self._novitec_manual_daemon_proc = None
                return False
            resp = json.loads(line)
            if not resp.get("ok"):
                logger.error(
                    f"Novitec manual daemon capture failed: {resp.get('err', resp)}"
                )
                return False
            return True
        except Exception as e:
            logger.exception(f"Novitec manual daemon IPC failed: {e}")
            try:
                if proc.poll() is not None:
                    self._novitec_manual_daemon_proc = None
            except Exception:
                self._novitec_manual_daemon_proc = None
            return False

    def get_camera_config(self, camera_id: int, product_model_name: Optional[str] = None) -> Tuple[str, Optional[Any], float, Optional[str]]:
        """
        Get camera configuration (loader_mode, source, fps, config_path).
        
        If config id is a list, the current cycle index is used to select the source.
        Use advance_video_source() to move to the next source in the list.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            product_model_name: Product model name (optional)
        
        Returns:
            Tuple of (loader_mode, source, fps, config_path)
        """
        if product_model_name is None:
            product_model_name = self.model_config.get_selected_model()
        
        # Load execution config json
        exec_config = get_execution_config(product_model_name, None)
        
        # Pass current video_source_index for list id cycling
        video_source_index = self._video_source_indices.get(camera_id)
        
        loader_mode, source, fps, config_path = get_camera_config(
            camera_id=camera_id,
            product_model_name=product_model_name,
            main_config_execution=exec_config,
            preset_name=self.preset_name,
            video_source_index=video_source_index
        )
        
        # Detect and register list sources for cycling
        # (need raw config to check if id is a list before index selection)
        self._register_video_source_list(camera_id, product_model_name, exec_config)
        
        # If camera mode and source is not set, use device ID from product model config
        if loader_mode == "camera" and not source and product_model_name:
            source = self.model_config.get_camera_device_id(product_model_name, camera_id)
            logger.info(f"Camera {camera_id}: Using device_id={source} from product config '{product_model_name}'")
        
        return loader_mode, source, fps, config_path
    
    def _register_video_source_list(self, camera_id: int, product_model_name: Optional[str], exec_config: Optional[Dict]) -> None:
        """Register video source list from preset config if id is a list."""
        if camera_id in self._video_source_lists:
            return  # Already registered
        
        if not exec_config:
            return
        
        preset_name = self.preset_name or exec_config.get("use_preset")
        if not preset_name:
            return
        
        presets = exec_config.get("presets", {})
        preset = presets.get(preset_name, {})
        if not preset:
            return
        
        camera_key = f"camera_{camera_id}"
        camera_config = preset.get(camera_key, {})
        if not isinstance(camera_config, dict):
            return
        
        source_id = camera_config.get("id")
        if isinstance(source_id, list) and len(source_id) > 0:
            self._video_source_lists[camera_id] = source_id
            if camera_id not in self._video_source_indices:
                self._video_source_indices[camera_id] = 0
            logger.info(f"Camera {camera_id}: Registered {len(source_id)} video sources for cycling: {source_id}")
    
    def has_video_source_list(self, camera_id: int) -> bool:
        """Check if a camera has multiple video sources configured."""
        return camera_id in self._video_source_lists and len(self._video_source_lists[camera_id]) > 1
    
    def advance_video_source(self, camera_id: int) -> Optional[str]:
        """
        Advance to the next video source in the cycle for a camera.
        
        Returns:
            The new source path, or None if camera has no source list.
        """
        if camera_id not in self._video_source_lists:
            return None
        
        source_list = self._video_source_lists[camera_id]
        old_idx = self._video_source_indices.get(camera_id, 0)
        new_idx = (old_idx + 1) % len(source_list)
        self._video_source_indices[camera_id] = new_idx
        
        new_source = source_list[new_idx]
        logger.info(f"Camera {camera_id}: Advanced video source index {old_idx} -> {new_idx} "
                    f"({len(source_list)} total), next source: {new_source}")
        return new_source
    
    def get_video_source_info(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """Get current video source cycling info for a camera."""
        if camera_id not in self._video_source_lists:
            return None
        source_list = self._video_source_lists[camera_id]
        idx = self._video_source_indices.get(camera_id, 0)
        return {
            "sources": source_list,
            "current_index": idx,
            "current_source": source_list[idx % len(source_list)],
            "total": len(source_list)
        }
    
    def reset_video_source_indices(self) -> None:
        """Reset all video source cycle indices to 0."""
        for camera_id in self._video_source_indices:
            self._video_source_indices[camera_id] = 0
        self._video_source_lists.clear()
        logger.info("All video source cycle indices reset")
    
    def load_camera_pixel_sizes(self, preset_name: Optional[str] = None, product_model_name: Optional[str] = None):
        """Pre-load pixel sizes for all cameras."""
        if product_model_name is None:
            product_model_name = self.model_config.get_selected_model()
        
        if preset_name is None:
            preset_name = self.preset_name
        
        # Load from tracker_config files only - no SystemConfig needed
        pixel_sizes = get_camera_pixel_sizes(
            product_model_name=product_model_name,
            main_config_execution=get_execution_config(product_model_name, None),
            main_config_measurement=None,  # Not used - kept for compatibility
            preset_name=preset_name
        )
        
        self.camera_pixel_sizes.update(pixel_sizes)
    
    def load_camera_distance_map_paths(self, preset_name: Optional[str] = None, product_model_name: Optional[str] = None):
        """Pre-load distance map paths for all cameras."""
        if product_model_name is None:
            product_model_name = self.model_config.get_selected_model()
        
        if preset_name is None:
            preset_name = self.preset_name
        
        # Load json directly - no SystemConfig needed
        distance_map_paths = get_camera_distance_map_paths(
            product_model_name=product_model_name,
            main_config_execution=get_execution_config(product_model_name, None),
            preset_name=preset_name
        )
        
        self.camera_distance_map_paths.update(distance_map_paths)
    
    def get_distance_map_path(self, camera_id: Optional[int] = None) -> Optional[str]:
        """Get distance map path for a camera."""
        if camera_id and camera_id in self.camera_distance_map_paths:
            return self.camera_distance_map_paths[camera_id]
        return None
    
    def load_camera_homographies(self, preset_name: Optional[str] = None, product_model_name: Optional[str] = None):
        """Load homography matrices from preset config (zoom1.json) or camera config files.
        
        Also loads WarpOffset from tracker_config file and applies it to homography if available.
        """
        import json
        import numpy as np
        
        # 1. 먼저 zoom1.json의 measurement에서 Homography 로드 시도
        if not product_model_name:
            product_model_name = self.model_config.get_selected_model()
        
        homographies_from_preset = get_camera_homographies(
            product_model_name,
            None,  # main_config_execution (not used)
            preset_name or self.preset_name
        )
        
        # Get execution config to find tracker_config paths
        exec_config = get_execution_config(product_model_name, None)
        if exec_config:
            preset_name_actual = preset_name or self.preset_name or exec_config.get("use_preset")
            presets = exec_config.get("presets", {})
            preset = presets.get(preset_name_actual, {}) if preset_name_actual else {}
        else:
            preset = {}
        
        for camera_id in [1, 2, 3]:
            homography_list = homographies_from_preset.get(camera_id)
            
            if homography_list:
                # Preset에서 Homography를 찾음
                homography = np.array(homography_list, dtype=np.float64)
                
                # Try to load WarpOffset from tracker_config file
                camera_key = f"camera_{camera_id}"
                camera_config = preset.get(camera_key, {})
                tracker_config_path = camera_config.get("tracker_config") if isinstance(camera_config, dict) else None
                
                if tracker_config_path:
                    try:
                        tracker_config = load_tracker_config_file(tracker_config_path)
                        if tracker_config and "measurement" in tracker_config:
                            measurement = tracker_config["measurement"]
                            if isinstance(measurement, dict) and "WarpOffset" in measurement:
                                warp_offset = measurement["WarpOffset"]
                                offset_x = warp_offset.get("x", 0.0)
                                offset_y = warp_offset.get("y", 0.0)
                                
                                # Apply offset to homography
                                translation = np.array([
                                    [1, 0, offset_x],
                                    [0, 1, offset_y],
                                    [0, 0, 1]
                                ], dtype=np.float64)
                                
                                homography = translation @ homography
                                logger.info(f"Camera {camera_id}: Homography loaded from preset config with WarpOffset (x={offset_x:.1f}, y={offset_y:.1f})")
                            else:
                                logger.info(f"Camera {camera_id}: Homography loaded from preset config (no WarpOffset)")
                        else:
                            logger.info(f"Camera {camera_id}: Homography loaded from preset config (no WarpOffset)")
                    except Exception as e:
                        logger.debug(f"Camera {camera_id}: Failed to load WarpOffset: {e}, using homography without offset")
                
                self.camera_homographies[camera_id] = homography
                continue
            
            # 2. Preset에 없으면 camera_config.json의 calibration에서 로드 시도 (fallback)
            _, _, _, config_path = self.get_camera_config(camera_id, product_model_name)
            
            if config_path and Path(config_path).exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        camera_config = json.load(f)
                    
                    calibration = camera_config.get("calibration", {})
                    homography_list = calibration.get("Homography")
                    
                    if homography_list:
                        self.camera_homographies[camera_id] = np.array(homography_list, dtype=np.float64)
                        logger.info(f"Camera {camera_id}: Homography loaded from {config_path}")
                    else:
                        self.camera_homographies[camera_id] = None
                        logger.debug(f"Camera {camera_id}: No Homography configured")
                except Exception as e:
                    self.camera_homographies[camera_id] = None
                    logger.warning(f"Camera {camera_id}: Failed to load Homography: {e}")
            else:
                self.camera_homographies[camera_id] = None
    
    def get_homography(self, camera_id: int) -> Optional[Any]:
        """Get homography matrix for a camera."""
        return self.camera_homographies.get(camera_id)
    
    def get_edge_refinement_config(self, camera_id: int) -> Dict:
        """
        Get edge refinement configuration for a camera.
        
        Returns:
            Dict with 'enable' (bool) and 'search_range_px' (int) keys.
            Defaults to enable=True, search_range_px=10 if not configured.
        """
        return self.camera_edge_refinement_config.get(camera_id, {
            "enable": True,
            "search_range_px": 10
        })
    
    def warp_frame(self, camera_id: int, frame) -> Any:
        """Apply homography transformation to frame if available."""
        homography = self.get_homography(camera_id)
        if homography is not None:
            h, w = frame.shape[:2]
            warped = cv2.warpPerspective(frame, homography, (w, h), 
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return warped
        # 호모그래피가 없으면 원본 프레임 반환 (경고 로그 없음)
        return frame
    
    def get_pixel_size(self, camera_id: Optional[int] = None) -> float:
        """Get average pixel size for a camera (for backward compatibility)."""
        if camera_id and camera_id in self.camera_pixel_sizes:
            pixel_size_data = self.camera_pixel_sizes[camera_id]
            if isinstance(pixel_size_data, dict):
                return pixel_size_data.get('average', 1.0)
            return pixel_size_data
        
        # Not used - pixel sizes loaded from tracker_config files per camera
        return 1.0
    
    def get_pixel_size_dict(self, camera_id: Optional[int] = None) -> Dict[str, float]:
        """Get pixel size dict for a camera (x, y, average)."""
        default_dict = {'x': 1.0, 'y': 1.0, 'average': 1.0}
        
        if camera_id and camera_id in self.camera_pixel_sizes:
            pixel_size_data = self.camera_pixel_sizes[camera_id]
            if isinstance(pixel_size_data, dict):
                return pixel_size_data
            # 단일 값이면 dict로 변환
            return {'x': pixel_size_data, 'y': pixel_size_data, 'average': pixel_size_data}
        
        # Not used - pixel sizes loaded from tracker_config files per camera
        return default_dict
    
    def get_fps_from_loader(self, loader: BaseLoader) -> float:
        """Get FPS from loader or config."""
        if loader and hasattr(loader, 'fps'):
            return loader.fps
        # Default FPS if loader doesn't provide it
        return 30.0
    
    def initialize_camera(
        self,
        camera_id: int,
        loader_mode: str,
        source: Optional[Any] = None,
        fps: float = 30.0,
        model_path: Optional[Path] = None,
        detector_config: Optional[Dict] = None,
        tracker_config: Optional[Dict] = None,
        enable_undistortion: bool = False,
        camera_config_path: Optional[str] = None,
        draw_masks: bool = False
    ):
        """
        Initialize camera with loader and AMR tracker.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            loader_mode: Loader mode ("camera", "video", "image_sequence")
            source: Source path or device ID
            fps: Frame rate
            model_path: Path to model file
            detector_config: Detector configuration
            tracker_config: Tracker configuration (boundary_margin_ratio, etc.)
            enable_undistortion: Whether to enable image undistortion
            camera_config_path: Path to camera config file (e.g., camera1_config.json)
        """
        # Default source if not provided
        if source is None:
            if loader_mode == "camera":
                source = camera_id - 1
            else:
                raise ValueError(f"Source must be provided for loader_mode: {loader_mode}")
        
        # Load camera config file if provided (for Novitec camera)
        camera_config = None
        if camera_config_path:
            try:
                config_file = Path(camera_config_path)
                if config_file.exists():
                    import json
                    with open(config_file, 'r', encoding='utf-8') as f:
                        camera_config = json.load(f)
                    logger.info(f"Camera {camera_id}: Loaded config from {camera_config_path}")
                else:
                    logger.warning(f"Camera {camera_id}: Config file not found: {camera_config_path}")
            except Exception as e:
                logger.warning(f"Camera {camera_id}: Failed to load config from {camera_config_path}: {e}")
        
        # Load undistortion parameters from camera config
        camera_matrix = None
        dist_coeffs = None
        
        if enable_undistortion and camera_config:
            # Read calibration parameters from camera config file
            try:
                import numpy as np
                if "calibration" in camera_config:
                    calib = camera_config["calibration"]
                    # CameraMatrix is in camera config
                    if "CameraMatrix" in calib:
                        camera_matrix = np.array(calib["CameraMatrix"])
                    # DistortionCoefficients is in camera config
                    if "DistortionCoefficients" in calib:
                        dist_coeffs = np.array(calib["DistortionCoefficients"])
                    
                    if camera_matrix is not None and dist_coeffs is not None:
                        logger.info(f"Camera {camera_id}: Loaded undistortion parameters from {camera_config_path}")
                    else:
                        logger.warning(f"Camera {camera_id}: Missing calibration parameters in {camera_config_path}")
                else:
                    logger.warning(f"Camera {camera_id}: No calibration section in {camera_config_path}")
            except Exception as e:
                logger.warning(f"Camera {camera_id}: Failed to load calibration from camera config: {e}")
        
        # Create loader with config and undistortion parameters
        # Pass camera_id as camera_index to load separate DLL for each Novitec camera
        # Enable buffering for camera mode (real-time streams) to prevent frame drops
        enable_buffering = (loader_mode == "camera")
        
        # Read buffer settings json -> buffer section
        buffer_size = 40  # Default: ~0.5 second at 30fps (low latency)
        buffer_drop_policy = "oldest"  # Drop oldest frames when buffer is full (maintains real-time)
        
        try:
            product_model_name = self.model_config.get_selected_model() if self.model_config else None
            if product_model_name:
                product_config = load_product_model_config(product_model_name)
                if product_config and "buffer" in product_config:
                    buffer_data = product_config["buffer"]
                    buffer_size = buffer_data.get("size", buffer_size)
                    buffer_drop_policy = buffer_data.get("drop_policy", buffer_drop_policy)
                    logger.debug(f"Loaded buffer config from {product_model_name}.json: size={buffer_size}, policy={buffer_drop_policy}")
        except Exception as e:
            logger.debug(f"Failed to load buffer config from {product_model_name}.json: {e}")
        
        novitec_cam1_sp = (
            USE_NOVITEC_CAM1_SUBPROCESS_STREAM
            and loader_mode == "camera"
            and camera_id == 1
        )
        loader = create_sequence_loader(
            source, 
            fps=fps, 
            loader_mode=loader_mode, 
            config=camera_config,
            enable_undistortion=enable_undistortion,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            camera_index=camera_id,  # Use camera_id as camera_index for DLL isolation
            enable_buffering=enable_buffering,
            buffer_size=buffer_size,
            buffer_drop_policy=buffer_drop_policy,
            novitec_cam1_subprocess=novitec_cam1_sp,
        )
        if loader is None:
            raise RuntimeError(f"Failed to create loader for camera {camera_id} (mode: {loader_mode}, source: {source})")
        
        # Reset loader to start from first frame
        if hasattr(loader, 'cap') and loader.cap is not None:
            loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            loader.frame_number = 0
        elif hasattr(loader, 'frame_number'):
            loader.frame_number = 0
        
        self.camera_loaders[camera_id] = loader
        
        self.trackers[camera_id] = {}  # For compatibility with existing code
        self.next_track_ids[camera_id] = 0
        self.frame_numbers[camera_id] = 0
        self.camera_status[camera_id] = True
        
        # Initialize EnhancedAMRTracker
        pixel_size = self.get_pixel_size_dict(camera_id)
        fps = self.get_fps_from_loader(loader)
        
        model_path_str = str(model_path) if model_path else None
        
        # Get detector_type from detector_config, default to "yolo"
        detector_type = (detector_config or {}).get("detector_type", "aruco")
        
        # Prepare calibration_config with homography from camera_manager (loaded from tracker_config)
        # Note: camera_height, calibration_image_size, and pixel_size are not actually used in SizeMeasurement
        # They are kept for backward compatibility but can be None/default values
        calibration_config_for_tracker = None
        homography = self.get_homography(camera_id)
        if homography is not None:
            # Use homography from camera_manager (loaded from tracker_config)
            # SizeMeasurement only uses homography for transformation, other params are unused
            calibration_config_for_tracker = {
                "homography": homography.tolist() if hasattr(homography, 'tolist') else homography,
                "draw_masks": draw_masks,  # Pass draw_masks from vision_server
            }
            logger.info(f"Camera {camera_id}: Using homography from camera_manager for Visualizer")
        
        self.amr_trackers[camera_id] = EnhancedAMRTracker(
            config=self.config,
            detector_type=detector_type,
            tracker_type="kalman",
            pixel_size=pixel_size,
            model_path=model_path_str,
            detector_config=detector_config or {},
            tracker_config=tracker_config or {},
            calibration_config=calibration_config_for_tracker,
            fps=fps,
        )
        
        # Store edge refinement config from tracker_config
        self.camera_edge_refinement_config[camera_id] = {
            "enable": (tracker_config or {}).get("enable_edge_refinement", True),
            "search_range_px": (tracker_config or {}).get("edge_search_range_px", 10)
        }
        logger.debug(f"Camera {camera_id} edge refinement config: "
                    f"enable={self.camera_edge_refinement_config[camera_id]['enable']}, "
                    f"search_range={self.camera_edge_refinement_config[camera_id]['search_range_px']}px")
        
        logger.info(f"Camera {camera_id} initialized with EnhancedAMRTracker")
    
    def wait_for_novitec_cam1_subprocess_first_frame(
        self,
        timeout_sec: float = 60.0,
        poll_sec: float = 0.05,
    ) -> bool:
        """
        CAM1이 NovitecCamera1SubprocessLoader일 때, 다른 카메라(CAM2/3) SDK를 올리기 전에
        자식 프로세스에서 첫 라이브 프레임이 큐에 도착할 때까지 대기한다.

        - GigE/Novitec가 동일 PC에서 두 프로세스로 동시 discovery 할 때 10048 등이 나기 쉬워
          CAM1 스트림을 먼저 안정화한 뒤 CAM2/CAM3 초기화 순서를 맞춘다.
        - 첫 번째로 읽은 프레임은 소비(consume)되며, 이후 read()는 다음 프레임부터다.
        """
        from src.utils.novitec_cam1_subprocess_loader import NovitecCamera1SubprocessLoader

        loader = self.camera_loaders.get(1)
        if not isinstance(loader, NovitecCamera1SubprocessLoader):
            return True

        deadline = time.time() + timeout_sec
        logger.info(
            "CAM1 subprocess: waiting for first frame before initializing other cameras "
            f"(timeout={timeout_sec:.0f}s)..."
        )
        while time.time() < deadline:
            if not loader.is_stream_process_alive():
                logger.error("CAM1 subprocess: worker exited while waiting for first frame")
                return False
            ret, frame = loader.read()
            if ret and frame is not None:
                logger.info(
                    "CAM1 subprocess: first frame received; continuing with CAM2/CAM3 init"
                )
                return True
            time.sleep(poll_sec)

        logger.error(
            "CAM1 subprocess: timeout waiting for first frame "
            "(GigE conflict / error 10048 / network / device busy)"
        )
        return False

    def check_camera_connection(self, camera_id: int) -> bool:
        """Check if camera is connected."""
        loader = self.camera_loaders.get(camera_id)
        if loader and hasattr(loader, 'check_connection'):
            return loader.check_connection()
        return False
    
    def release_camera(self, camera_id: int):
        """Release camera resources."""
        if camera_id in self.camera_loaders:
            loader = self.camera_loaders[camera_id]
            if hasattr(loader, 'release'):
                loader.release()
            del self.camera_loaders[camera_id]
        
        if camera_id in self.trackers:
            del self.trackers[camera_id]
        
        if camera_id in self.amr_trackers:
            del self.amr_trackers[camera_id]
        
        self.camera_status[camera_id] = False
    
    def release_all_cameras(self):
        """Release all camera resources."""
        for camera_id in list(self.camera_loaders.keys()):
            self.release_camera(camera_id)
    
    def reset_loader_for_cycle(self, camera_id: int) -> bool:
        """
        Reset a camera loader for a new cycle.
        
        - For video loaders: reset to frame 0 (replay the same video)
        - For cameras with list sources: advance index and recreate loader with next video
        - For camera mode (Novitec): no-op (use start/stop_camera_stream instead)
        
        Returns:
            True if loader was reset/recreated successfully, False otherwise.
        """
        loader = self.camera_loaders.get(camera_id)
        if loader is None:
            return False
        
        # For cameras with list sources, advance to next video
        if self.has_video_source_list(camera_id):
            new_source = self.advance_video_source(camera_id)
            if new_source is None:
                return False
            
            # Release old loader
            if hasattr(loader, 'release'):
                loader.release()
            
            # Recreate loader with new source using same parameters
            try:
                loader_mode = "video"
                fps = self.get_fps_from_loader(loader) if loader else 30.0
                
                enable_undistortion = False
                camera_matrix = None
                dist_coeffs = None
                if hasattr(loader, '_enable_undistortion'):
                    enable_undistortion = loader._enable_undistortion
                if hasattr(loader, '_camera_matrix'):
                    camera_matrix = loader._camera_matrix
                if hasattr(loader, '_dist_coeffs'):
                    dist_coeffs = loader._dist_coeffs
                
                new_loader = create_sequence_loader(
                    new_source,
                    fps=fps,
                    loader_mode=loader_mode,
                    enable_undistortion=enable_undistortion,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    camera_index=camera_id,
                    enable_buffering=False,
                )
                if new_loader is None:
                    logger.error(f"Camera {camera_id}: Failed to create loader for new source: {new_source}")
                    return False
                
                self.camera_loaders[camera_id] = new_loader
                self.frame_numbers[camera_id] = 0
                logger.info(f"Camera {camera_id}: Loader recreated with new source: {new_source}")
                return True
            except Exception as e:
                logger.error(f"Camera {camera_id}: Failed to reinitialize loader: {e}")
                return False
        
        # For video loaders with single source: reset to frame 0
        if hasattr(loader, 'cap') and loader.cap is not None:
            loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if hasattr(loader, 'frame_number'):
                loader.frame_number = 0
            self.frame_numbers[camera_id] = 0
            logger.info(f"Camera {camera_id}: Video loader reset to frame 0")
            return True
        
        return False
    
    def prepare_all_cameras_for_next_cycle(self) -> Dict[int, bool]:
        """
        Prepare all video-mode cameras for the next cycle.
        Advances list sources and resets single-source video loaders.
        
        Returns:
            Dict mapping camera_id -> success status for each camera that was reset.
        """
        results = {}
        for camera_id in list(self.camera_loaders.keys()):
            loader = self.camera_loaders.get(camera_id)
            if loader is None:
                continue
            # Only reset video loaders (not Novitec camera loaders)
            if _loader_is_novitec_family(loader):
                continue
            results[camera_id] = self.reset_loader_for_cycle(camera_id)
        
        if results:
            logger.info(f"Cycle preparation results: {results}")
        return results
    
    def stop_camera_stream(self, camera_id: int) -> bool:
        """
        Stop stream for a specific camera (Novitec cameras only).
        Also stops the frame buffer to prevent auto-restart of stream.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            
        Returns:
            True if stream was stopped successfully, False otherwise
        """
        if camera_id not in self.camera_loaders:
            return False
        
        loader = self.camera_loaders[camera_id]
        if loader is None:
            return False
        if _loader_is_novitec_family(loader):
            try:
                # IMPORTANT: Stop frame buffer FIRST to prevent it from restarting the stream
                if hasattr(loader, 'stop_buffering'):
                    loader.stop_buffering()
                    logger.info(f"Camera {camera_id}: Frame buffer stopped")
                
                if loader.camera and hasattr(loader.camera, 'stop_stream'):
                    if loader.camera._is_streaming:
                        logger.info(f"Camera {camera_id}: Stopping stream...")
                        if loader.camera.stop_stream():
                            if hasattr(loader, '_stream_started'):
                                loader._stream_started = False
                            logger.info(f"Camera {camera_id}: Stream stopped successfully")
                            return True
                        else:
                            logger.warning(f"Camera {camera_id}: stop_stream() returned False")
                            return False
                    else:
                        logger.debug(f"Camera {camera_id}: Stream already stopped")
                        return True
            except Exception as e:
                logger.warning(f"Camera {camera_id}: Failed to stop stream: {e}")
                return False
        
        return False
    
    def start_camera_stream(self, camera_id: int) -> bool:
        """
        Start stream for a specific camera (Novitec cameras only).
        Also starts the frame buffer for buffered capture.
        
        Each camera uses a separate DLL instance (cam1, cam2, cam3), so no need to
        disconnect other cameras. Just start the stream for the requested camera.
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            
        Returns:
            True if stream was started successfully, False otherwise
        """
        if camera_id not in self.camera_loaders:
            return False
        
        loader = self.camera_loaders[camera_id]
        if loader is None:
            return False
        if _loader_is_novitec_family(loader):
            try:
                from src.utils.novitec_cam1_subprocess_loader import (
                    NovitecCamera1SubprocessLoader,
                )

                if loader.camera and hasattr(loader.camera, 'start_stream'):
                    if not loader.camera._is_streaming:
                        logger.info(f"Camera {camera_id}: Starting stream...")
                        if loader.camera.start_stream():
                            if hasattr(loader, '_stream_started'):
                                loader._stream_started = True
                            logger.info(f"Camera {camera_id}: Stream started successfully")
                            if (
                                hasattr(loader, 'start_buffering')
                                and getattr(loader, 'enable_buffering', False)
                                and not isinstance(loader, NovitecCamera1SubprocessLoader)
                            ):
                                loader.start_buffering()
                                logger.info(f"Camera {camera_id}: Frame buffer started")
                            return True
                        logger.warning(f"Camera {camera_id}: start_stream() returned False")
                        return False
                    logger.debug(f"Camera {camera_id}: Stream already started")
                    if (
                        hasattr(loader, 'start_buffering')
                        and loader.enable_buffering
                        and not isinstance(loader, NovitecCamera1SubprocessLoader)
                    ):
                        fb = getattr(loader, '_frame_buffer', None)
                        if fb is None or not fb.is_running:
                            loader.start_buffering()
                            logger.info(
                                f"Camera {camera_id}: Frame buffer started (stream was already active)"
                            )
                    return True
            except Exception as e:
                logger.warning(f"Camera {camera_id}: Failed to start stream: {e}")
                import traceback
                traceback.print_exc()
                return False

        return False
    
    def stop_all_camera_streams(self):
        """Stop streams for all Novitec cameras."""
        for camera_id, loader in self.camera_loaders.items():
            if _loader_is_novitec_family(loader):
                self.stop_camera_stream(camera_id)

    def is_novitec_camera(self, camera_id: int) -> bool:
        """CAM이 Novitec 로더인지."""
        loader = self.camera_loaders.get(camera_id)
        return _loader_is_novitec_family(loader)

    def is_camera_stream_active(self, camera_id: int) -> bool:
        """Novitec 스트림(start_stream)이 켜져 있는지."""
        loader = self.camera_loaders.get(camera_id)
        if not _loader_is_novitec_family(loader) or not getattr(loader, 'camera', None):
            return False
        return bool(getattr(loader.camera, "_is_streaming", False))

    def grab_novitec_single_frame_from_stream(self, camera_id: int) -> Optional[Any]:
        """
        스트림이 이미 켜진 상태에서 ``capture()`` 한 장 (연속 모드).
        ``last_frames``가 없을 때 보조용. 언디스토션은 로더 설정을 따름.
        """
        from src.utils.sequence_loader import NovitecCameraLoader
        from src.utils.novitec_cam1_subprocess_loader import NovitecCamera1SubprocessLoader

        loader = self.camera_loaders.get(camera_id)
        if loader is None or not loader.camera:
            return None
        try:
            if isinstance(loader, NovitecCamera1SubprocessLoader):
                ret, frame = loader.read()
                return frame if ret else None
            if not isinstance(loader, NovitecCameraLoader):
                return None
            data = loader.camera.capture(output_formats=["image"])
            if not data or "image" not in data:
                return None
            frame = data["image"]
            return loader._undistort_frame(frame)
        except Exception as e:
            logger.warning(f"Camera {camera_id}: grab_novitec_single_frame_from_stream failed: {e}")
            return None

    def grab_novitec_manual_frame_subprocess(self, camera_id: int) -> Optional[Any]:
        """
        스트림이 꺼진 상태에서 수동 1장: **별도 프로세스**에서 디바이스 독점 후
        소프트웨어 트리거 캡처. 부모는 해당 카메라 로더를 잠시 release 후 재생성.

        기본은 START VISION 때 띄워 둔 **상주 데몬**(`novitec_manual_trigger_daemon`)에
        stdin JSON 한 줄로 요청 (cold subprocess / 매번 Python·DLL 로드 비용 제거).
        데몬 실패 시 `novitec_manual_trigger_worker` 1회 실행으로 폴백.
        """
        loader = self.camera_loaders.get(camera_id)
        if not _loader_is_novitec_family(loader) or not loader.camera:
            return None

        device_id = loader.device_id
        config = dict(loader.config or {})
        camera_index = loader.camera_index
        enable_undistortion = getattr(loader, "enable_undistortion", False)
        camera_matrix = getattr(loader, "camera_matrix", None)
        dist_coeffs = getattr(loader, "dist_coeffs", None)
        enable_buffering = loader.enable_buffering
        buffer_size = loader.buffer_size
        buffer_drop_policy = loader.buffer_drop_policy
        fps = self.get_fps_from_loader(loader)

        repo_root = self._amr_repo_root()
        fallback_script = repo_root / "scripts" / "novitec_manual_trigger_worker.py"

        config_tmp = None
        out_png = None
        frame = None

        with self._novitec_manual_daemon_lock:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as tf:
                    json.dump(config, tf, ensure_ascii=False)
                    config_tmp = tf.name
                out_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

                self._ensure_novitec_manual_daemon_unlocked()

                logger.info(
                    f"Camera {camera_id}: Manual capture (daemon or one-shot worker), "
                    f"device={device_id!r}"
                )
                loader.release()
                self.camera_loaders[camera_id] = None

                novitec_src = repo_root / "submodules" / "novitec_camera_module" / "src"
                py_path = str(repo_root)
                if novitec_src.is_dir():
                    py_path = f"{str(novitec_src)}{os.pathsep}{py_path}"
                env = os.environ.copy()
                prev = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{py_path}{os.pathsep}{prev}" if prev else py_path

                got_frame = False
                if self._novitec_manual_daemon_capture_unlocked(
                    device_id, camera_index, config_tmp, out_png
                ):
                    got_frame = True
                elif fallback_script.is_file():
                    logger.warning(
                        f"Camera {camera_id}: manual daemon failed; falling back to one-shot worker"
                    )
                    cmd = [
                        sys.executable,
                        str(fallback_script),
                        "--device-id",
                        device_id,
                        "--camera-index",
                        str(camera_index),
                        "--config-json",
                        config_tmp,
                        "--output",
                        out_png,
                    ]
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        env=env,
                        cwd=str(repo_root),
                    )
                    if proc.returncode != 0:
                        logger.error(
                            f"Camera {camera_id}: one-shot worker exit={proc.returncode} "
                            f"stderr={proc.stderr!r}"
                        )
                    else:
                        got_frame = True
                else:
                    logger.error(
                        f"Camera {camera_id}: no daemon IPC and no fallback at {fallback_script}"
                    )

                if got_frame:
                    frame = cv2.imread(out_png)
                    if frame is None:
                        logger.error(
                            f"Camera {camera_id}: capture OK but failed to read {out_png}"
                        )
            except subprocess.TimeoutExpired:
                logger.error(f"Camera {camera_id}: manual one-shot worker timed out")
            except Exception as e:
                logger.exception(f"Camera {camera_id}: manual capture failed: {e}")
            finally:
                for p in (config_tmp, out_png):
                    if p:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except Exception:
                            pass

            new_loader = create_sequence_loader(
                device_id,
                fps=fps,
                loader_mode="camera",
                config=config,
                enable_undistortion=enable_undistortion,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                camera_index=camera_index,
                enable_buffering=enable_buffering,
                buffer_size=buffer_size,
                buffer_drop_policy=buffer_drop_policy,
                novitec_cam1_subprocess=(
                    USE_NOVITEC_CAM1_SUBPROCESS_STREAM and camera_index == 1
                ),
            )
            if new_loader is None:
                logger.error(
                    f"Camera {camera_id}: failed to recreate Novitec loader after manual capture"
                )
                return frame
            self.camera_loaders[camera_id] = new_loader
            self.frame_numbers[camera_id] = 0
            logger.info(f"Camera {camera_id}: Novitec loader recreated after manual capture")

        if frame is not None and enable_undistortion and camera_matrix is not None and dist_coeffs is not None:
            try:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            except Exception as e:
                logger.warning(f"Camera {camera_id}: undistort after manual capture failed: {e}")

        return frame

