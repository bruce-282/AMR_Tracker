"""CSV writer utilities for saving cycle results in raw_data.csv format."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, IO

logger = logging.getLogger(__name__)

MAX_TRAJECTORY_POINTS = 100


def generate_csv_header() -> List[str]:
    """Generate CSV header.

    cam1: single detection (x, y, rz)
    cam2, cam3: trajectory (x_0..N, y_0..N, rz_0..N)
    """
    header = [
        "record_time",
        "cam_1_result", "cam_2_result", "cam_3_result",
        "velocity_result", "velocity(mm/s)",
        "cam_1_x(mm)", "cam_1_y(mm)", "cam_1_rz(deg)",
    ]
    for i in range(MAX_TRAJECTORY_POINTS):
        header.extend([f"cam_2_x_{i}", f"cam_2_y_{i}", f"cam_2_rz_{i}"])
    for i in range(MAX_TRAJECTORY_POINTS):
        header.extend([f"cam_3_x_{i}", f"cam_3_y_{i}", f"cam_3_rz_{i}"])
    return header


def extract_position(data: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract (x, y, rz) from server response data.

    Handles both dict (cam1 single detection) and list (cam2/cam3 trajectory).
    For list data, returns the last point.
    """
    if data is None:
        return None, None, None
    if isinstance(data, dict):
        return data.get("x"), data.get("y"), data.get("rz")
    if isinstance(data, list) and len(data) > 0:
        last = data[-1]
        return last.get("x"), last.get("y"), last.get("rz")
    return None, None, None


def _append_trajectory(row: List, data: Any, max_points: int) -> None:
    """Append trajectory points to a CSV row, padding to max_points."""
    points = data if isinstance(data, list) else []
    for i in range(max_points):
        if i < len(points):
            pt = points[i]
            row.extend([pt.get("x", ""), pt.get("y", ""), pt.get("rz", "")])
        else:
            row.extend(["", "", ""])


def build_csv_row(cycle_info: Dict[str, Any]) -> List:
    """Build a CSV row from collected cycle data.

    Args:
        cycle_info: Dict with keys 'timestamp', 'cam1', 'cam2', 'cam3'.
            cam1: dict (single detection) from server response.
            cam2, cam3: list of trajectory points from server response.
    """
    timestamp = cycle_info.get("timestamp", datetime.now().strftime("%Y-%m-%d_%H;%M;%S"))
    cam1_data = cycle_info.get("cam1")
    cam2_data = cycle_info.get("cam2")
    cam3_data = cycle_info.get("cam3")

    cam1_ok = cam1_data is not None
    cam2_ok = cam2_data is not None and isinstance(cam2_data, list)
    cam3_ok = cam3_data is not None and isinstance(cam3_data, list)

    cam1_x, cam1_y, cam1_rz = extract_position(cam1_data)

    row = [
        timestamp,
        str(cam1_ok).lower(),
        str(cam2_ok).lower(),
        str(cam3_ok).lower(),
        "false", "",
        cam1_x if cam1_x is not None else "",
        cam1_y if cam1_y is not None else "",
        cam1_rz if cam1_rz is not None else "",
    ]

    _append_trajectory(row, cam2_data, MAX_TRAJECTORY_POINTS)
    _append_trajectory(row, cam3_data, MAX_TRAJECTORY_POINTS)

    return row


def init_csv_file(csv_path: Path) -> IO:
    """Create CSV file with header and return the opened file handle.

    The caller is responsible for closing the returned file handle.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(generate_csv_header())
    f.flush()
    logger.info(f"CSV file created: {csv_path}")
    return f


def append_csv_row(csv_file: IO, cycle_info: Dict[str, Any]) -> None:
    """Append a completed cycle row to the CSV file."""
    row = build_csv_row(cycle_info)
    writer = csv.writer(csv_file)
    writer.writerow(row)
    csv_file.flush()


def generate_csv_path(model_name: str, base_dir: str = "data") -> Path:
    """Generate a timestamped CSV file path.

    Returns:
        Path like data/20260218-143000_zoom1_raw_data.csv
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(base_dir) / f"{timestamp}_{model_name}_raw_data.csv"
