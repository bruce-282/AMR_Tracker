"""Model configuration management."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import TrackingConfig

logger = logging.getLogger(__name__)


class ModelConfig:
    """Manages model configuration.
    
    model_list contains product model names (e.g., "zoom1", "zoom2"),
    and actual AI model files are located in folders named after these product models.
    """
    
    DEFAULT_CONFIG_PATH = "model_config.json"
    DEFAULT_WEIGHTS_PATH = "weights/zoom1/best.pt"
    DEFAULT_MODEL_FILE = "best.pt"  # Default model file name in each product model folder
    
    def __init__(self, config_path: Optional[str] = None, weights_path: str = DEFAULT_WEIGHTS_PATH):
        """
        Initialize model configuration.
        
        Args:
            config_path: Path to model config file
            weights_path: Base directory for model weights (default: "weights/zoom1/best.pt")
        """
        self.config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        self.weights_path = Path(weights_path)
        self.model_list: List[str] = []  # Product model names (e.g., "zoom1", "zoom2")
        self.selected_model: Optional[str] = None  # Selected product model name
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.model_list = config.get("model_list", [])
                    self.selected_model = config.get("selected_model")
                    if self.model_list and not self.selected_model:
                        self.selected_model = self.model_list[0]
            except Exception as e:
                print(f"⚠ Failed to load model config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        # Check for existing product model folders
        if self.weights_path.exists():
            # Find folders in weights directory
            folders = [d.name for d in self.weights_path.iterdir() if d.is_dir()]
            if folders:
                self.model_list = sorted(folders)
                self.selected_model = self.model_list[0]
            else:
                # Default fallback
                self.model_list = ["default"]
                self.selected_model = "default"
        else:
            self.model_list = ["default"]
            self.selected_model = "default"
        self.save_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            config = {
                "model_list": self.model_list,
                "selected_model": self.selected_model
            }
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Failed to save model config: {e}")
    
    def get_product_config(self, product_model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get product model configuration from config file.
        
        Args:
            product_model_name: Product model name (e.g., "zoom1"). 
                               If None, uses selected_model.
        
        Returns:
            Dictionary with camera_1, camera_2, camera_3 (device IDs) and detector config
        """
        if product_model_name is None:
            product_model_name = self.selected_model
        
        if not product_model_name:
            # Fallback to default
            return {
                "camera_1": 0,
                "camera_2": 1,
                "camera_3": 2,
                "detector": {
                    "model_path": str(self.weights_path / self.DEFAULT_MODEL_FILE),
                    "confidence_threshold": 0.2,
                    "imgsz": 640,
                    "target_classes": [0]
                }
            }
        
        # Load config from config/{product_model_name}.json
        config_path = Path("config") / f"{product_model_name}.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                    # Get detector config (nested or flat for backward compatibility)
                    detector_config = config.get("detector", {})
                    if not detector_config:
                        # Backward compatibility: if detector not nested, use flat structure
                        detector_config = {
                            "model_path": config.get("model_path", str(self.weights_path / product_model_name / self.DEFAULT_MODEL_FILE)),
                            "confidence_threshold": config.get("confidence_threshold", 0.2),
                            "imgsz": config.get("imgsz", 640),
                            "target_classes": config.get("target_classes", [0])
                        }
                    
                    return {
                        "camera_1": config.get("camera_1", 0),
                        "camera_2": config.get("camera_2", 1),
                        "camera_3": config.get("camera_3", 2),
                        "detector": detector_config
                    }
            except Exception as e:
                print(f"[WARN] Failed to load product config from {config_path}: {e}")
        
        # Fallback to default path structure
        model_path = self.weights_path / product_model_name / self.DEFAULT_MODEL_FILE
        return {
            "camera_1": 0,
            "camera_2": 1,
            "camera_3": 2,
            "detector": {
                "model_path": str(model_path),
                "confidence_threshold": 0.2,
                "imgsz": 1536,
                "target_classes": [0]
            }
        }

    def get_camera_device_id(self, product_model_name: Optional[str] = None, camera_id: int = 1) -> int:
        """
        Get camera device ID for a specific camera.
        
        Args:
            product_model_name: Product model name (e.g., "zoom1")
            camera_id: Camera ID (1, 2, or 3)
        
        Returns:
            Camera device ID (0-based)
        """
        config = self.get_product_config(product_model_name)
        camera_key = f"camera_{camera_id}"
        return config.get(camera_key, camera_id - 1)
    
    def get_model_file_path(self, product_model_name: Optional[str] = None) -> Path:
        """
        Get actual model file path for a product model.
        
        Args:
            product_model_name: Product model name (e.g., "zoom1"). 
                               If None, uses selected_model.
        
        Returns:
            Path to the actual model file (e.g., "weights/zoom1/last.pt")
        """
        config = self.get_product_config(product_model_name)
        detector_config = config.get("detector", {})
        model_path_str = detector_config.get("model_path", str(self.weights_path / self.DEFAULT_MODEL_FILE))
        return Path(model_path_str)
    
    def get_detector_config(self, product_model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detector configuration for a product model.
        
        Args:
            product_model_name: Product model name (e.g., "zoom1"). 
                               If None, uses selected_model.
        
        Returns:
            Dictionary with detector parameters (model_path, confidence_threshold, etc.)
        """
        config = self.get_product_config(product_model_name)
        return config.get("detector", {})
    
    def get_tracking_config(
        self, 
        product_model_name: Optional[str] = None,
        main_config_tracking: Optional[Any] = None
    ) -> TrackingConfig:
        """
        Get tracking configuration for a product model.
        
        Priority: product model config > main config > default
        
        Args:
            product_model_name: Product model name (e.g., "zoom1"). 
                               If None, uses selected_model.
            main_config_tracking: Tracking config from main config file (optional)
        
        Returns:
            TrackingConfig instance
        """
        if product_model_name is None:
            product_model_name = self.selected_model
        
        # Try product model config first
        if product_model_name:
            product_config = self._load_product_model_config_file(product_model_name)
            if product_config and "tracker" in product_config:
                tracker_data = product_config["tracker"]
                logger.info(f"Loaded tracking config from product model config ({product_model_name}.json)")
                return TrackingConfig(**tracker_data)
        
        # Fallback to main config
        if main_config_tracking:
            logger.debug("Using tracking config from main config")
            return main_config_tracking
        
        # Fallback to default
        logger.debug("Using default tracking config")
        return TrackingConfig()
    
    def get_calibration_config(
        self,
        product_model_name: Optional[str] = None,
        main_config_calibration: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get calibration configuration for a product model.
        
        Priority: product model config > main config
        
        Args:
            product_model_name: Product model name (e.g., "zoom1"). 
                               If None, uses selected_model.
            main_config_calibration: Calibration config from main config file (optional)
        
        Returns:
            Calibration config dictionary, or None if not found
        """
        if product_model_name is None:
            product_model_name = self.selected_model
        
        # Try product model config first
        if product_model_name:
            product_config = self._load_product_model_config_file(product_model_name)
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
    
    def _load_product_model_config_file(self, product_model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load product model configuration file (config/{product_model_name}.json).
        
        Args:
            product_model_name: Product model name (e.g., "zoom1")
        
        Returns:
            Dictionary with config data, or None if file not found or error
        """
        if not product_model_name:
            return None
        
        product_config_path = Path("config") / f"{product_model_name}.json"
        if not product_config_path.exists():
            logger.debug(f"Product model config file not found: {product_config_path}")
            return None
        
        try:
            with open(product_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.debug(f"Loaded product model config from {product_config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load product model config from {product_config_path}: {e}")
            return None
    
    def add_product_model(self, product_model_name: str):
        """Add product model to list."""
        if product_model_name not in self.model_list:
            self.model_list.append(product_model_name)
            self.save_config()
    
    def set_selected_model(self, product_model_name: str):
        """Set selected product model."""
        if product_model_name in self.model_list:
            self.selected_model = product_model_name
            self.save_config()
        else:
            # Add to list if not exists
            self.add_product_model(product_model_name)
            self.selected_model = product_model_name
            self.save_config()
    
    def get_model_list(self) -> List[str]:
        """Get product model list."""
        return self.model_list.copy()
    
    def get_selected_model(self) -> Optional[str]:
        """Get selected product model name."""
        return self.selected_model
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_list": self.model_list,
            "selected_model": self.selected_model
        }

