"""Camera state management for VisionServer."""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    """
    Encapsulates all state related to a single camera.

    This class manages tracking state, speed history, detection loss tracking,
    and trajectory data for individual cameras in the VisionServer.
    """

    camera_id: int

    # Loader and tracking
    loader: Any = None
    tracker: Any = None
    next_track_id: int = 0

    # Frame tracking
    frame_number: int = 0

    # Speed tracking
    speed_history: deque = field(default_factory=lambda: deque(maxlen=30))
    has_reached_speed_threshold: bool = False
    speed_near_zero_frames: int = 0

    # Detection loss tracking
    detection_loss_frames: int = 0
    has_first_detection: bool = False  # Flag to track if first detection has occurred

    # Response tracking
    response_sent: bool = False

    # Trajectory data (for camera 2)
    trajectory: List[Dict] = field(default_factory=list)

    # Latest detection info
    latest_detections: List = field(default_factory=list)

    # Thread management
    tracking_thread: Any = None
    is_tracking: bool = False

    def __post_init__(self):
        """Initialize deque after dataclass creation."""
        if not isinstance(self.speed_history, deque):
            self.speed_history = deque(maxlen=30)

    def reset(self):
        """Reset camera state for new tracking session."""
        self.frame_number = 0
        self.speed_history.clear()
        self.has_reached_speed_threshold = False
        self.speed_near_zero_frames = 0
        self.detection_loss_frames = 0
        self.has_first_detection = False
        self.response_sent = False
        self.trajectory.clear()
        self.latest_detections.clear()
        self.tracker = None
        self.next_track_id = 0
        logger.debug(f"Camera {self.camera_id} state reset")

    def update_speed(self, speed: float) -> None:
        """Update speed history with new measurement."""
        self.speed_history.append(speed)

    def get_average_speed(self, window: int = 5) -> float:
        """Get average speed over recent frames."""
        if not self.speed_history:
            return 0.0
        recent = list(self.speed_history)[-window:]
        return sum(recent) / len(recent) if recent else 0.0

    def increment_detection_loss(self) -> int:
        """Increment detection loss counter and return new value."""
        self.detection_loss_frames += 1
        return self.detection_loss_frames

    def reset_detection_loss(self) -> None:
        """Reset detection loss counter."""
        self.detection_loss_frames = 0

    def add_trajectory_point(self, point: Dict) -> None:
        """Add a point to trajectory history."""
        self.trajectory.append(point)

    def get_trajectory(self) -> List[Dict]:
        """Get copy of trajectory data."""
        return list(self.trajectory)

    def clear_trajectory(self) -> None:
        """Clear trajectory data."""
        self.trajectory.clear()


class CameraStateManager:
    """
    Manages multiple camera states.

    Provides centralized access to camera state objects and
    common operations across all cameras.
    """

    def __init__(self):
        self._states: Dict[int, CameraState] = {}

    def get_or_create(self, camera_id: int) -> CameraState:
        """Get existing camera state or create new one."""
        if camera_id not in self._states:
            self._states[camera_id] = CameraState(camera_id=camera_id)
            logger.debug(f"Created new state for camera {camera_id}")
        return self._states[camera_id]

    def get(self, camera_id: int) -> Optional[CameraState]:
        """Get camera state if exists."""
        return self._states.get(camera_id)

    def remove(self, camera_id: int) -> None:
        """Remove camera state."""
        if camera_id in self._states:
            del self._states[camera_id]
            logger.debug(f"Removed state for camera {camera_id}")

    def reset_all(self) -> None:
        """Reset all camera states."""
        for state in self._states.values():
            state.reset()
        logger.debug("All camera states reset")

    def get_all_ids(self) -> List[int]:
        """Get all camera IDs with active states."""
        return list(self._states.keys())

    def has_camera(self, camera_id: int) -> bool:
        """Check if camera state exists."""
        return camera_id in self._states

    def clear(self) -> None:
        """Remove all camera states."""
        self._states.clear()
        logger.debug("All camera states cleared")
