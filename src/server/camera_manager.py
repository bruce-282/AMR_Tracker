"""Camera management for vision server."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import cv2

from src.utils.sequence_loader import create_sequence_loader, BaseLoader
from src.utils.config_loader import get_camera_config, get_camera_pixel_sizes
from src.core.amr_tracker import EnhancedAMRTracker
from .model_config import ModelConfig
from config import SystemConfig

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera initialization, loaders, and AMR trackers."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        system_config: Optional[SystemConfig] = None,
        preset_name: Optional[str] = None
    ):
        """
        Initialize camera manager.
        
        Args:
            model_config: Model configuration manager
            system_config: System configuration
            preset_name: Preset name override
        """
        self.model_config = model_config
        self.config = system_config
        self.preset_name = preset_name
        
        # Camera resources
        self.camera_loaders: Dict[int, BaseLoader] = {}
        self.amr_trackers: Dict[int, EnhancedAMRTracker] = {}
        self.camera_pixel_sizes: Dict[int, float] = {}
        self.frame_numbers: Dict[int, int] = {}
        self.camera_status: Dict[int, bool] = {1: False, 2: False, 3: False}
        
        # Trackers dict for compatibility with existing code that accesses trackers directly
        # Note: EnhancedAMRTracker manages its own tracker internally
        self.trackers: Dict[int, Dict] = {}  # camera_id -> tracker dict
        self.next_track_ids: Dict[int, int] = {}
    
    def get_camera_config(self, camera_id: int, product_model_name: Optional[str] = None) -> Tuple[str, Optional[Any], float, Optional[str]]:
        """
        Get camera configuration (loader_mode, source, fps, config_path).
        
        Args:
            camera_id: Camera ID (1, 2, or 3)
            product_model_name: Product model name (optional)
        
        Returns:
            Tuple of (loader_mode, source, fps, config_path)
        """
        if product_model_name is None:
            product_model_name = self.model_config.get_selected_model()
        
        loader_mode, source, fps, config_path = get_camera_config(
            camera_id=camera_id,
            product_model_name=product_model_name,
            main_config_execution=self.config.execution if self.config and hasattr(self.config, 'execution') and self.config.execution else None,
            preset_name=self.preset_name
        )
        
        # If camera mode and source is not set, use device ID from product model config
        if loader_mode == "camera" and not source and product_model_name:
            source = self.model_config.get_camera_device_id(product_model_name, camera_id)
            logger.info(f"Camera {camera_id}: Using device_id={source} from product config '{product_model_name}'")
        
        return loader_mode, source, fps, config_path
    
    def load_camera_pixel_sizes(self, preset_name: Optional[str] = None, product_model_name: Optional[str] = None):
        """Pre-load pixel sizes for all cameras."""
        if product_model_name is None:
            product_model_name = self.model_config.get_selected_model()
        
        if preset_name is None:
            preset_name = self.preset_name
        
        pixel_sizes = get_camera_pixel_sizes(
            product_model_name=product_model_name,
            main_config_execution=self.config.execution if self.config and hasattr(self.config, 'execution') and self.config.execution else None,
            main_config_measurement=self.config.measurement if self.config and hasattr(self.config, 'measurement') else None,
            preset_name=preset_name
        )
        
        self.camera_pixel_sizes.update(pixel_sizes)
    
    def get_pixel_size(self, camera_id: Optional[int] = None) -> float:
        """Get pixel size for a camera."""
        if camera_id and camera_id in self.camera_pixel_sizes:
            return self.camera_pixel_sizes[camera_id]
        
        return self.config.measurement.pixel_size if self.config and hasattr(self.config, 'measurement') else 1.0
    
    def get_fps_from_loader(self, loader: BaseLoader) -> float:
        """Get FPS from loader or config."""
        if loader and hasattr(loader, 'fps'):
            return loader.fps
        elif self.config and hasattr(self.config, 'measurement'):
            return self.config.measurement.fps
        return 30.0
    
    def initialize_camera(
        self,
        camera_id: int,
        loader_mode: str,
        source: Optional[Any] = None,
        fps: float = 30.0,
        model_path: Optional[Path] = None,
        detector_config: Optional[Dict] = None,
        enable_undistortion: bool = False,
        camera_config_path: Optional[str] = None
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
        loader = create_sequence_loader(
            source, 
            fps=fps, 
            loader_mode=loader_mode, 
            config=camera_config,
            enable_undistortion=enable_undistortion,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs
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
        pixel_size = self.get_pixel_size(camera_id)
        fps = self.get_fps_from_loader(loader)
        
        model_path_str = str(model_path) if model_path else None
        
        # Get detector_type from detector_config, default to "yolo"
        detector_type = (detector_config or {}).get("detector_type", "yolo")
        
        self.amr_trackers[camera_id] = EnhancedAMRTracker(
            config=self.config,
            detector_type=detector_type,
            tracker_type="kalman",
            pixel_size=pixel_size,
            model_path=model_path_str,
            detector_config=detector_config or {},
            fps=fps,
        )
        
        logger.info(f"Camera {camera_id} initialized with EnhancedAMRTracker")
    
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

