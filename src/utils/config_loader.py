"""Configuration loader utilities for product model config files."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# TrackingConfig removed - using dict from tracker_config files

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
    global _product_model_config_cache, _tracker_config_cache
    _product_model_config_cache.clear()
    _tracker_config_cache.clear()
    logger.debug("Product model config cache cleared")


# Cache for loaded tracker configs (avoids reading same JSON file multiple times)
_tracker_config_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def load_tracker_config_file(tracker_config_path: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load tracker configuration file (e.g., config/cam1_tracker_config.json).
    Uses caching to avoid reading the same file multiple times.
    
    Args:
        tracker_config_path: Path to tracker config file
        use_cache: Whether to use cached config (default True)
    
    Returns:
        Dictionary with config data containing 'detector', 'tracker', 'measurement',
        or None if file not found or error
    """
    if not tracker_config_path:
        return None
    
    # Return cached config if available
    if use_cache and tracker_config_path in _tracker_config_cache:
        logger.debug(f"Using cached tracker config for {tracker_config_path}")
        return _tracker_config_cache[tracker_config_path]
    
    config_path = Path(tracker_config_path)
    if not config_path.exists():
        logger.debug(f"Tracker config file not found: {config_path}")
        _tracker_config_cache[tracker_config_path] = None
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded tracker config from {config_path}")
        # Cache the loaded config
        _tracker_config_cache[tracker_config_path] = config
        return config
    except Exception as e:
        logger.warning(f"Failed to load tracker config from {config_path}: {e}")
        _tracker_config_cache[tracker_config_path] = None
        return None


def get_camera_tracker_config_path(
    camera_id: int,
    preset: Dict[str, Any]
) -> Optional[str]:
    """
    Get tracker_config path for a camera from preset.
    
    Args:
        camera_id: Camera ID (1, 2, or 3)
        preset: Preset dictionary from execution.presets
    
    Returns:
        Path to tracker_config file, or None if not found
    """
    camera_key = f"camera_{camera_id}"
    camera_config = preset.get(camera_key, {})
    
    if isinstance(camera_config, dict):
        return camera_config.get("tracker_config")
    
    return None


def get_camera_tracker_config(
    camera_id: int,
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    preset_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get tracker configuration (detector, tracker, measurement) for a camera from tracker_config file.
    
    Args:
        camera_id: Camera ID (1, 2, or 3)
        product_model_name: Product model name (e.g., "zoom1") (required)
        main_config_execution: Execution config from zoom1.json
        preset_name: Preset name to use
    
    Returns:
        Dictionary with 'detector', 'tracker', 'measurement' keys from tracker_config file
    
    Raises:
        ValueError: If required configs are missing
    """
    if not product_model_name:
        raise ValueError(f"Camera {camera_id}: product_model_name is required")
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        raise ValueError(f"Camera {camera_id}: Failed to load execution config from {product_model_name}.json")
    
    # Get preset name
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        raise ValueError(f"Camera {camera_id}: No preset name found in {product_model_name}.json")
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    
    if not preset:
        raise ValueError(f"Camera {camera_id}: Preset '{preset_name}' not found in {product_model_name}.json")
    
    # Try to get tracker_config path from preset
    tracker_config_path = get_camera_tracker_config_path(camera_id, preset)
    
    if not tracker_config_path:
        raise ValueError(f"Camera {camera_id}: tracker_config path not found in preset '{preset_name}'")
    
    tracker_config = load_tracker_config_file(tracker_config_path)
    if not tracker_config:
        raise ValueError(f"Camera {camera_id}: Failed to load tracker_config from {tracker_config_path}")
    
    logger.info(f"Camera {camera_id}: Loaded tracker config from {tracker_config_path}")
    return tracker_config


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
    
    # No fallback - execution config must be in product_model_name.json
    if not product_model_name:
        raise ValueError("product_model_name is required")
    raise ValueError(f"Failed to load execution config from {product_model_name}.json")


def get_camera_config_from_preset(
    camera_id: int,
    preset: Dict[str, Any],
    product_model_name: Optional[str] = None,
    video_source_index: Optional[int] = None
) -> Tuple[Optional[str], Optional[Any], float, Optional[str]]:
    """
    Get camera configuration from preset.
    
    Args:
        camera_id: Camera ID (1, 2, or 3)
        preset: Preset dictionary from execution.presets
        product_model_name: Product model name (for logging)
        video_source_index: If id is a list, selects source at this index (cycling).
                           None means return the raw id value as-is.
    
    Returns:
        Tuple of (loader_mode, source, fps, config_path)
        When id is a list and video_source_index is provided, source is the selected element.
        When id is a list and video_source_index is None, source is the full list.
    """
    loader_mode = preset.get("loader_mode", "auto")
    
    camera_key = f"camera_{camera_id}"
    camera_config = preset.get(camera_key, {})
    
    if not isinstance(camera_config, dict):
        logger.warning(f"Camera {camera_id}: '{camera_key}' not found or invalid in preset")
        return None, None, 30.0, None
    
    # Get source from camera's id field (supports string or list)
    source = camera_config.get("id")
    
    if isinstance(source, list) and len(source) > 0:
        if video_source_index is not None:
            idx = video_source_index % len(source)
            selected = source[idx]
            logger.info(f"Camera {camera_id}: id is a list ({len(source)} items), "
                       f"selected index {idx}: {selected}")
            source = selected
        else:
            logger.debug(f"Camera {camera_id}: id is a list ({len(source)} items), returning raw list")
    
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
    preset_name: Optional[str] = None,
    video_source_index: Optional[int] = None
) -> Tuple[str, Optional[Any], float, Optional[str]]:
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
        video_source_index: If id is a list in config, selects source at this index (cycling).
    
    Returns:
        Tuple of (loader_mode, source, fps, config_path)
    """
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        raise ValueError(f"Camera {camera_id}: Failed to load execution config from {product_model_name}.json")
    
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
                camera_id, preset, product_model_name,
                video_source_index=video_source_index
            )
            if loader_mode and source:
                logger.info(
                    f"Camera {camera_id}: Using preset '{preset_name}' "
                    f"(loader_mode={loader_mode}, source={source}, fps={fps}, config={config_path})"
                )
                return loader_mode, source, fps, config_path
        else:
            logger.warning(f"Preset '{preset_name}' not found in config")
    
    # Preset is required - no fallback to defaults
    raise ValueError(f"Camera {camera_id}: Preset '{preset_name}' not found or invalid in {product_model_name}.json")


def _extract_pixel_size_from_measurement(measurement: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract pixel size from measurement dictionary in tracker_config file.
    
    Args:
        measurement: Measurement dictionary from tracker_config (must contain PixelSize)
    
    Returns:
        Dictionary with 'x', 'y', 'average' keys
    
    Raises:
        ValueError: If PixelSize is not found in measurement
    """
    if not isinstance(measurement, dict):
        raise ValueError("measurement must be a dictionary")
    
    # 새 형식: PixelSize.x, PixelSize.y (required)
    if "PixelSize" in measurement:
        pixel_size_data = measurement.get("PixelSize", {})
        if isinstance(pixel_size_data, dict):
            px = pixel_size_data.get("x")
            py = pixel_size_data.get("y")
            if px is None or py is None:
                raise ValueError("PixelSize must contain both 'x' and 'y' values")
            avg = pixel_size_data.get("average", (px + py) / 2)
            return {'x': px, 'y': py, 'average': avg}
        else:
            raise ValueError("PixelSize must be a dictionary with 'x' and 'y' keys")
    
    # 기존 형식: pixel_size (단일 값) - deprecated but supported for backward compatibility
    if "pixel_size" in measurement:
        pixel_size = measurement.get("pixel_size")
        if pixel_size is None:
            raise ValueError("pixel_size must have a value")
        return {'x': pixel_size, 'y': pixel_size, 'average': pixel_size}
    
    raise ValueError("measurement must contain either 'PixelSize' (dict with x, y) or 'pixel_size' (single value)")


def get_camera_pixel_sizes(
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    main_config_measurement: Optional[Any] = None,  # Not used - kept for compatibility
    preset_name: Optional[str] = None
) -> Dict[int, Dict[str, float]]:
    """
    Get pixel sizes for all cameras from tracker_config files.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1") (required)
        main_config_execution: Execution config from zoom1.json
        main_config_measurement: Not used (deprecated)
        preset_name: Preset name to use
    
    Returns:
        Dictionary mapping camera_id -> {'x': float, 'y': float, 'average': float}
    
    Raises:
        ValueError: If required configs are missing
    """
    if not product_model_name:
        raise ValueError("product_model_name is required")
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        raise ValueError(f"Failed to load execution config from {product_model_name}.json")
    
    # Get preset name
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        raise ValueError(f"No preset name found in {product_model_name}.json")
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    
    if not preset:
        raise ValueError(f"Preset '{preset_name}' not found in {product_model_name}.json")
    
    result = {}
    
    # Load pixel_size for each camera from tracker_config files
    for camera_id in [1, 2, 3]:
        camera_key = f"camera_{camera_id}"
        camera_config = preset.get(camera_key, {})
        
        if not isinstance(camera_config, dict):
            raise ValueError(f"Camera {camera_id}: Invalid camera_config for {camera_key} in preset '{preset_name}'")
        
        # Load from tracker_config file (required)
        tracker_config_path = camera_config.get("tracker_config")
        if not tracker_config_path:
            raise ValueError(f"Camera {camera_id}: tracker_config path not found in {camera_key} of preset '{preset_name}'")
        
        tracker_config = load_tracker_config_file(tracker_config_path)
        if not tracker_config:
            raise ValueError(f"Camera {camera_id}: Failed to load tracker_config from {tracker_config_path}")
        
        if "measurement" not in tracker_config:
            raise ValueError(f"Camera {camera_id}: 'measurement' section not found in {tracker_config_path}")
        
        pixel_size = _extract_pixel_size_from_measurement(tracker_config["measurement"])
        result[camera_id] = pixel_size
        logger.debug(f"Camera {camera_id}: pixel_size from tracker_config '{tracker_config_path}': {pixel_size}")
    
    return result


def get_camera_distance_map_paths(
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    preset_name: Optional[str] = None
) -> Dict[int, Optional[str]]:
    """
    Get distance map paths for all cameras from tracker_config files.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1") (required)
        main_config_execution: Execution config from zoom1.json
        preset_name: Preset name to use
    
    Returns:
        Dictionary mapping camera_id -> distance_map_path (None if not set in tracker_config)
    
    Raises:
        ValueError: If required configs are missing
    """
    if not product_model_name:
        raise ValueError("product_model_name is required")
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        raise ValueError(f"Failed to load execution config from {product_model_name}.json")
    
    # Get preset name
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        raise ValueError(f"No preset name found in {product_model_name}.json")
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    
    if not preset:
        raise ValueError(f"Preset '{preset_name}' not found in {product_model_name}.json")
    
    result = {}
    
    # Load distance_map_path for each camera from tracker_config files
    for camera_id in [1, 2, 3]:
        camera_key = f"camera_{camera_id}"
        camera_config = preset.get(camera_key, {})
        
        if not isinstance(camera_config, dict):
            raise ValueError(f"Camera {camera_id}: Invalid camera_config for {camera_key} in preset '{preset_name}'")
        
        # Load from tracker_config file
        tracker_config_path = camera_config.get("tracker_config")
        if not tracker_config_path:
            raise ValueError(f"Camera {camera_id}: tracker_config path not found in {camera_key} of preset '{preset_name}'")
        
        tracker_config = load_tracker_config_file(tracker_config_path)
        if not tracker_config:
            raise ValueError(f"Camera {camera_id}: Failed to load tracker_config from {tracker_config_path}")
        
        # distance_map_path is optional - return None if not found
        if "measurement" in tracker_config:
            measurement = tracker_config["measurement"]
            if isinstance(measurement, dict) and "distance_map_path" in measurement:
                distance_map_path = measurement.get("distance_map_path")
                result[camera_id] = distance_map_path
                logger.debug(f"Camera {camera_id}: distance_map_path={distance_map_path} from tracker_config '{tracker_config_path}'")
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
    
    Priority:
    1. Camera-specific tracker_config file
    2. Preset's camera measurement (legacy support)
    
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
    
    # Load Homography for each camera
    for camera_id in [1, 2, 3]:
        camera_key = f"camera_{camera_id}"
        camera_config = preset.get(camera_key, {})
        
        if not isinstance(camera_config, dict):
            result[camera_id] = None
            continue
        
        # Priority 1: Try tracker_config file
        tracker_config_path = camera_config.get("tracker_config")
        if tracker_config_path:
            tracker_config = load_tracker_config_file(tracker_config_path)
            if tracker_config and "measurement" in tracker_config:
                measurement = tracker_config["measurement"]
                if isinstance(measurement, dict) and "Homography" in measurement:
                    result[camera_id] = measurement.get("Homography")
                    logger.debug(f"Camera {camera_id}: Homography loaded from tracker_config '{tracker_config_path}'")
                    continue
        
        # Priority 2: Try preset's camera measurement (legacy)
        measurement = camera_config.get("measurement", {})
        if isinstance(measurement, dict) and "Homography" in measurement:
            result[camera_id] = measurement.get("Homography")
            logger.debug(f"Camera {camera_id}: Homography loaded from preset '{preset_name}'")
        else:
            result[camera_id] = None
    
    return result


def load_tracking_config(
    product_model_name: Optional[str],
    main_config_tracking: Optional[Any]
) -> Optional[Dict[str, Any]]:
    """
    Load tracking configuration from product model config.
    
    Note: This function is deprecated. Use load_camera_tracking_config instead.
    All tracking configs should be loaded from tracker_config files per camera.
    
    Args:
        product_model_name: Product model name (e.g., "zoom1")
        main_config_tracking: Tracking config from main config file (deprecated)
    
    Returns:
        Tracking config dict or None
    """
    # Try product model config
    if product_model_name:
        product_config = load_product_model_config(product_model_name)
        if product_config and "tracker" in product_config:
            tracker_data = product_config["tracker"]
            # Filter out comment/description keys (e.g., _comment, _desc_*)
            tracker_data = {k: v for k, v in tracker_data.items() if not k.startswith("_")}
            logger.info(f"Loaded tracking config from product model config ({product_model_name}.json)")
            return tracker_data
    
    # No fallback - return None
    return None


def load_camera_tracking_config(
    camera_id: int,
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    main_config_tracking: Optional[Any],
    preset_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load tracking configuration for a specific camera from tracker_config file.
    
    Priority:
    1. Camera-specific tracker_config file (required)
    
    Args:
        camera_id: Camera ID (1, 2, or 3)
        product_model_name: Product model name (e.g., "zoom1")
        main_config_execution: Execution config json
        main_config_tracking: Not used (deprecated)
        preset_name: Preset name to use
    
    Returns:
        Tracking config dict from tracker_config file
    """
    if not product_model_name:
        raise ValueError(f"Camera {camera_id}: product_model_name is required")
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        raise ValueError(f"Camera {camera_id}: Failed to load execution config from {product_model_name}.json")
    
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        raise ValueError(f"Camera {camera_id}: No preset name found in {product_model_name}.json")
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    
    if not preset:
        raise ValueError(f"Camera {camera_id}: Preset '{preset_name}' not found in {product_model_name}.json")
    
    camera_key = f"camera_{camera_id}"
    camera_config = preset.get(camera_key, {})
    
    if not isinstance(camera_config, dict):
        raise ValueError(f"Camera {camera_id}: Invalid camera_config for {camera_key} in preset '{preset_name}'")
    
    # Load from tracker_config file (required)
    tracker_config_path = camera_config.get("tracker_config")
    if not tracker_config_path:
        raise ValueError(f"Camera {camera_id}: tracker_config path not found in {camera_key} of preset '{preset_name}'")
    
    tracker_config = load_tracker_config_file(tracker_config_path)
    if not tracker_config:
        raise ValueError(f"Camera {camera_id}: Failed to load tracker_config from {tracker_config_path}")
    
    if "tracker" not in tracker_config:
        raise ValueError(f"Camera {camera_id}: 'tracker' section not found in {tracker_config_path}")
    
    tracker_data = tracker_config["tracker"]
    # Filter out comment/description keys
    tracker_data = {k: v for k, v in tracker_data.items() if not k.startswith("_")}
    logger.info(f"Camera {camera_id}: Loaded tracking config from {tracker_config_path}")
    return tracker_data


def load_camera_detector_config(
    camera_id: int,
    product_model_name: Optional[str],
    main_config_execution: Optional[Dict[str, Any]],
    preset_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load detector configuration for a specific camera from tracker_config file.
    
    Args:
        camera_id: Camera ID (1, 2, or 3)
        product_model_name: Product model name (e.g., "zoom1") (required)
        main_config_execution: Execution config from zoom1.json
        preset_name: Preset name to use
    
    Returns:
        Detector config dictionary from tracker_config file
    
    Raises:
        ValueError: If required configs are missing
    """
    if not product_model_name:
        raise ValueError(f"Camera {camera_id}: product_model_name is required")
    
    exec_config = get_execution_config(product_model_name, main_config_execution)
    
    if not exec_config:
        raise ValueError(f"Camera {camera_id}: Failed to load execution config from {product_model_name}.json")
    
    if not preset_name:
        preset_name = exec_config.get("use_preset")
    
    if not preset_name:
        raise ValueError(f"Camera {camera_id}: No preset name found in {product_model_name}.json")
    
    presets = exec_config.get("presets", {})
    preset = presets.get(preset_name, {})
    
    if not preset:
        raise ValueError(f"Camera {camera_id}: Preset '{preset_name}' not found in {product_model_name}.json")
    
    camera_key = f"camera_{camera_id}"
    camera_config = preset.get(camera_key, {})
    
    if not isinstance(camera_config, dict):
        raise ValueError(f"Camera {camera_id}: Invalid camera_config for {camera_key} in preset '{preset_name}'")
    
    # Load from tracker_config file (required)
    tracker_config_path = camera_config.get("tracker_config")
    if not tracker_config_path:
        raise ValueError(f"Camera {camera_id}: tracker_config path not found in {camera_key} of preset '{preset_name}'")
    
    tracker_config = load_tracker_config_file(tracker_config_path)
    if not tracker_config:
        raise ValueError(f"Camera {camera_id}: Failed to load tracker_config from {tracker_config_path}")
    
    if "detector" not in tracker_config:
        raise ValueError(f"Camera {camera_id}: 'detector' section not found in {tracker_config_path}")
    
    detector_data = tracker_config["detector"]
    # Filter out comment/description keys
    detector_data = {k: v for k, v in detector_data.items() if not k.startswith("_")}
    logger.info(f"Camera {camera_id}: Loaded detector config from {tracker_config_path}")
    return detector_data


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

