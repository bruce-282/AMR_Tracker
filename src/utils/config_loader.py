"""Configuration loader utilities for product model config files."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from config import TrackingConfig, CalibrationConfig

logger = logging.getLogger(__name__)

# Cache for loaded product model configs (avoids reading same JSON file multiple times)
_product_model_config_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def load_product_model_config(product_model_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load product model configuration file (config/{product_model_name}.json).
    Uses caching to avoid reading the same file multiple times.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        use_cache: Whether to use cached config (default True)
    
    Returns:
        Dictionary with config data, or None if file not found or error
    """
    if not product_model_name:
        return None
    
    # Return cached config if available
    if use_cache and product_model_name in _product_model_config_cache:
        logger.debug(f"Using cached product model config for {product_model_name}")
        return _product_model_config_cache[product_model_name]
    
    product_config_path = Path("config") / f"{product_model_name}.json"
    if not product_config_path.exists():
        logger.debug(f"Product model config file not found: {product_config_path}")
        _product_model_config_cache[product_model_name] = None
        return None
    
    try:
        with open(product_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.debug(f"Loaded product model config from {product_config_path}")
        # Cache the loaded config
        _product_model_config_cache[product_model_name] = config
        return config
    except Exception as e:
        logger.warning(f"Failed to load product model config from {product_config_path}: {e}")
        _product_model_config_cache[product_model_name] = None
        return None


def clear_config_cache():
    """Clear the product model config cache. Useful for testing or config reload."""
    global _product_model_config_cache
    _product_model_config_cache.clear()
    logger.debug("Product model config cache cleared")


def get_execution_config(
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Get execution config with priority: product model config > main config.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        main_config_execution: Execution config from main config file
    
    Returns:
        Execution config dictionary, or None if not found
    """
    # Try product model config first
    if product_model_name:
        product_config = load_product_model_config(product_model_name)
        if product_config and "execution" in product_config:
            logger.debug(f"Using product model config ({product_model_name}.json) for execution")
            return product_config.get("execution", {})
    
    # Fallback to main config
    if main_config_execution:
        logger.debug("Using main config (tracker_config.json) for execution")
        return main_config_execution
    
    return None


def get_camera_config_from_preset(
    camera_id: int,
    preset: Dict[str, Any],
    product_model_name: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], float, Optional[str]]:
    """
    Get camera configuration from preset.
    
    Args:
        camera_id: Camera ID (1, 2, or 3)
        preset: Preset dictionary from execution.presets
        product_model_name: Product model name (for logging)
    
    Returns:
        Tuple of (loader_mode, source, fps, config_path)
    """
    loader_mode = preset.get("loader_mode", "auto")
    
    camera_key = f"camera_{camera_id}"
    camera_config = preset.get(camera_key, {})
    
    if not isinstance(camera_config, dict):
        logger.warning(f"Camera {camera_id}: '{camera_key}' not found or invalid in preset")
        return None, None, 30.0, None
    
    # Get source from camera's id field
    source = camera_config.get("id")
    
    # Get fps from camera's measurement.fps
    measurement = camera_config.get("measurement", {})
    if isinstance(measurement, dict):
        fps = measurement.get("fps", 30.0)
    else:
        fps = 30.0
    
    # Get config file path (for Novitec camera initialization)
    config_path = camera_config.get("config")
    
    return loader_mode, source, fps, config_path


def get_camera_config(
    camera_id: int,
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    preset_name: Optional[str] = None
) -> Tuple[str, Optional[str], float, Optional[str]]:
    """
    Get camera configuration (loader_mode, source, fps, config_path) from config files.
    
    Priority:
    1. Product model config file (config/{product_model_name}.json)
    2. Main config file (tracker_config.json)
    
    Args:
        camera_id: Camera ID (1, 2, or 3)
        product_model_name: Product model name (e.g., "zoom1")
        main_config_execution: Execution config from main config file
        preset_name: Preset name to use
    
    Returns:
        Tuple of (loader_mode, source, fps, config_path)
    """
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        # Fallback defaults if no config
        return "camera", None, 30.0, None
    
    # Get preset name
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    loader_mode = None
    source = None
    fps = 30.0
    config_path = None
    
    if preset_name:
        presets = exec_config.get("presets", {})
        preset = presets.get(preset_name, {})
        if preset:
            loader_mode, source, fps, config_path = get_camera_config_from_preset(
                camera_id, preset, product_model_name
            )
            if loader_mode and source:
                logger.info(
                    f"Camera {camera_id}: Using preset '{preset_name}' "
                    f"(loader_mode={loader_mode}, source={source}, fps={fps}, config={config_path})"
                )
                return loader_mode, source, fps, config_path
        else:
            logger.warning(f"Preset '{preset_name}' not found in config")
    
    # If preset not found or not specified, use default settings
    if isinstance(exec_config, dict):
        loader_mode = exec_config.get("default_loader_mode", "camera")
        source = exec_config.get("default_source", camera_id - 1)
        fps = exec_config.get("default_fps", 30.0)
        config_path = None
    else:
        loader_mode = "camera"
        source = camera_id - 1
        fps = 30.0
        config_path = None
    
    logger.info(f"Camera {camera_id}: Using default settings (loader_mode={loader_mode}, source={source}, fps={fps})")
    return loader_mode, source, fps, config_path


def get_camera_pixel_sizes(
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    main_config_measurement: Optional[Any],
    preset_name: Optional[str] = None
) -> Dict[int, Dict[str, float]]:
    """
    Get pixel sizes for all cameras from preset configuration.
    
    Priority:
    1. Product model config file (config/{product_model_name}.json)
    2. Main config file (tracker_config.json)
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        main_config_execution: Execution config from main config file
        main_config_measurement: Measurement config from main config file
        preset_name: Preset name to use
    
    Returns:
        Dictionary mapping camera_id -> {'x': float, 'y': float, 'average': float}
    """
    result = {}
    default_pixel_size = 1.0
    if main_config_measurement and hasattr(main_config_measurement, 'pixel_size'):
        default_pixel_size = main_config_measurement.pixel_size
    
    default_pixel_size_dict = {'x': default_pixel_size, 'y': default_pixel_size, 'average': default_pixel_size}
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        # No config available, use default
        for camera_id in [1, 2, 3]:
            result[camera_id] = default_pixel_size_dict.copy()
        return result
    
    # Get preset name
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        # No preset, use default
        for camera_id in [1, 2, 3]:
            result[camera_id] = default_pixel_size_dict.copy()
        return result
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    if not preset:
        # Preset not found, use default
        for camera_id in [1, 2, 3]:
            result[camera_id] = default_pixel_size_dict.copy()
        return result
    
    # Load pixel_size for each camera from preset
    for camera_id in [1, 2, 3]:
        camera_key = f"camera_{camera_id}"
        camera_config = preset.get(camera_key, {})
        
        if isinstance(camera_config, dict):
            measurement = camera_config.get("measurement", {})
            if isinstance(measurement, dict):
                # 새 형식: PixelSize.x, PixelSize.y
                if "PixelSize" in measurement:
                    pixel_size_data = measurement.get("PixelSize", {})
                    if isinstance(pixel_size_data, dict):
                        px = pixel_size_data.get("x", default_pixel_size)
                        py = pixel_size_data.get("y", default_pixel_size)
                        avg = pixel_size_data.get("average", (px + py) / 2)
                        result[camera_id] = {'x': px, 'y': py, 'average': avg}
                        logger.debug(f"Camera {camera_id}: pixel_size x={px:.6f}, y={py:.6f}, avg={avg:.6f} from preset '{preset_name}'")
                    else:
                        result[camera_id] = default_pixel_size_dict.copy()
                # 기존 형식: pixel_size (단일 값)
                elif "pixel_size" in measurement:
                    pixel_size = measurement.get("pixel_size", default_pixel_size)
                    result[camera_id] = {'x': pixel_size, 'y': pixel_size, 'average': pixel_size}
                    logger.debug(f"Camera {camera_id}: pixel_size={pixel_size} from preset '{preset_name}'")
                else:
                    result[camera_id] = default_pixel_size_dict.copy()
            else:
                result[camera_id] = default_pixel_size_dict.copy()
        else:
            result[camera_id] = default_pixel_size_dict.copy()
    
    return result


def get_camera_distance_map_paths(
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    preset_name: Optional[str] = None
) -> Dict[int, Optional[str]]:
    """
    Get distance_map_path for all cameras from preset configuration.
    
    Priority:
    1. Product model config file (config/{product_model_name}.json)
    2. Main config file (tracker_config.json)
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        main_config_execution: Execution config from main config file
        preset_name: Preset name to use
    
    Returns:
        Dictionary mapping camera_id -> distance_map_path (None if not set)
    """
    result = {}
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        # No config available
        for camera_id in [1, 2, 3]:
            result[camera_id] = None
        return result
    
    # Get preset name
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        # No preset
        for camera_id in [1, 2, 3]:
            result[camera_id] = None
        return result
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    if not preset:
        # Preset not found
        for camera_id in [1, 2, 3]:
            result[camera_id] = None
        return result
    
    # Load distance_map_path for each camera from preset
    for camera_id in [1, 2, 3]:
        camera_key = f"camera_{camera_id}"
        camera_config = preset.get(camera_key, {})
        
        if isinstance(camera_config, dict):
            measurement = camera_config.get("measurement", {})
            if isinstance(measurement, dict) and "distance_map_path" in measurement:
                distance_map_path = measurement.get("distance_map_path")
                result[camera_id] = distance_map_path
                logger.debug(f"Camera {camera_id}: distance_map_path={distance_map_path} from preset '{preset_name}'")
            else:
                result[camera_id] = None
        else:
            result[camera_id] = None
    
    return result


def get_camera_homographies(
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    preset_name: Optional[str] = None
) -> Dict[int, Optional[List]]:
    """
    Get Homography matrices for all cameras from preset configuration.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        main_config_execution: Execution config from main config file
        preset_name: Preset name to use
    
    Returns:
        Dictionary mapping camera_id -> Homography matrix (as list) or None
    """
    result = {}
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        for camera_id in [1, 2, 3]:
            result[camera_id] = None
        return result
    
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        for camera_id in [1, 2, 3]:
            result[camera_id] = None
        return result
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    if not preset:
        for camera_id in [1, 2, 3]:
            result[camera_id] = None
        return result
    
    # Load Homography for each camera from preset
    for camera_id in [1, 2, 3]:
        camera_key = f"camera_{camera_id}"
        camera_config = preset.get(camera_key, {})
        
        if isinstance(camera_config, dict):
            measurement = camera_config.get("measurement", {})
            if isinstance(measurement, dict) and "Homography" in measurement:
                homography = measurement.get("Homography")
                result[camera_id] = homography
                logger.debug(f"Camera {camera_id}: Homography loaded from preset '{preset_name}'")
            else:
                result[camera_id] = None
        else:
            result[camera_id] = None
    
    return result


def load_tracking_config(
    product_model_name: Optional[str],
    main_config_tracking: Optional[Any]
) -> TrackingConfig:
    """
    Load tracking configuration with priority: product model config > main config > default.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        main_config_tracking: Tracking config from main config file
    
    Returns:
        TrackingConfig instance
    """
    # Try product model config first
    if product_model_name:
        product_config = load_product_model_config(product_model_name)
        if product_config and "tracker" in product_config:
            tracker_data = product_config["tracker"]
            # Filter out comment/description keys (e.g., _comment, _desc_*)
            tracker_data = {k: v for k, v in tracker_data.items() if not k.startswith("_")}
            logger.info(f"Loaded tracking config from product model config ({product_model_name}.json)")
            return TrackingConfig(**tracker_data)
    
    # Fallback to main config
    if main_config_tracking:
        logger.debug("Using tracking config from main config")
        return main_config_tracking
    
    # Fallback to default
    logger.debug("Using default tracking config")
    return TrackingConfig()


def load_calibration_config(
    product_model_name: Optional[str],
    main_config_calibration: Optional[Any]
) -> Optional[Dict[str, Any]]:
    """
    Load calibration configuration with priority: product model config > main config.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        main_config_calibration: Calibration config from main config file
    
    Returns:
        Calibration config dictionary, or None if not found
    """
    # Try product model config first
    if product_model_name:
        product_config = load_product_model_config(product_model_name)
        if product_config and "calibration" in product_config:
            logger.debug(f"Loaded calibration config from product model config ({product_model_name}.json)")
            return product_config["calibration"]
    
    # Fallback to main config
    if main_config_calibration:
        logger.debug("Using calibration config from main config")
        # Convert CalibrationConfig object to dict
        if hasattr(main_config_calibration, '__dict__'):
            return main_config_calibration.__dict__
        return main_config_calibration
    
    return None

