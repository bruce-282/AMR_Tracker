"""Model configuration management."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any


class ModelConfig:
    """Manages model configuration.
    
    model_list contains product model names (e.g., "zoom1", "zoom2"),
    and actual AI model files are located in folders named after these product models.
    """
    
    DEFAULT_CONFIG_PATH = "model_config.json"
    DEFAULT_WEIGHTS_DIR = "weights"
    DEFAULT_MODEL_FILE = "last.pt"  # Default model file name in each product model folder
    
    def __init__(self, config_path: Optional[str] = None, weights_dir: str = "weights"):
        """
        Initialize model configuration.
        
        Args:
            config_path: Path to model config file
            weights_dir: Base directory for model weights (default: "weights")
        """
        self.config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        self.weights_dir = Path(weights_dir)
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
        if self.weights_dir.exists():
            # Find folders in weights directory
            folders = [d.name for d in self.weights_dir.iterdir() if d.is_dir()]
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
                    "model_path": str(self.weights_dir / self.DEFAULT_MODEL_FILE),
                    "confidence_threshold": 0.2,
                    "nms_threshold": 0.4,
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
                            "model_path": config.get("model_path", str(self.weights_dir / product_model_name / self.DEFAULT_MODEL_FILE)),
                            "confidence_threshold": config.get("confidence_threshold", 0.2),
                            "nms_threshold": config.get("nms_threshold", 0.4),
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
        model_path = self.weights_dir / product_model_name / self.DEFAULT_MODEL_FILE
        return {
            "camera_1": 0,
            "camera_2": 1,
            "camera_3": 2,
            "detector": {
                "model_path": str(model_path),
                "confidence_threshold": 0.2,
                "nms_threshold": 0.4,
                "imgsz": 640,
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
        model_path_str = detector_config.get("model_path", str(self.weights_dir / self.DEFAULT_MODEL_FILE))
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

