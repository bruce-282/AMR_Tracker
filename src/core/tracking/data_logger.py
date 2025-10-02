"""
Tracking data logger for AMR tracking system
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class TrackingDataLogger:
    """Class to log tracking data to CSV files"""

    def __init__(self, output_dir="tracking_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.output_dir / f"tracking_results_{timestamp}.csv"

        # Create CSV file with headers
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "frame_number",
                    "track_id",
                    "position_x",
                    "position_y",
                    "velocity_x",
                    "velocity_y",
                    "linear_speed_ms",
                    "linear_speed_kmh",
                    "orientation_deg",
                    "detection_type",
                ]
            )

        print(f"✓ Tracking data will be saved to: {self.csv_file}")

    def log_tracking_result(self, frame_number: int, tracking_result: dict):
        """Log a single tracking result to CSV"""
        try:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                timestamp = datetime.now().isoformat()
                track_id = tracking_result.get("track_id", "unknown")

                # Position data
                pos = tracking_result.get("position", {})
                pos_x = pos.get("x", 0)
                pos_y = pos.get("y", 0)

                # Velocity data
                vel = tracking_result.get("velocity", {})
                vel_x = vel.get("vx", 0)
                vel_y = vel.get("vy", 0)
                linear_speed_ms = vel.get("linear_speed_m_per_sec", 0)
                linear_speed_kmh = linear_speed_ms * 3.6

                # Orientation data
                orient = tracking_result.get("orientation", {})
                orientation_deg = orient.get("theta_normalized_deg", 0)

                # Detection type
                detection_type = tracking_result.get("detection_type", "unknown")

                writer.writerow(
                    [
                        timestamp,
                        frame_number,
                        track_id,
                        f"{pos_x:.2f}",
                        f"{pos_y:.2f}",
                        f"{vel_x:.2f}",
                        f"{vel_y:.2f}",
                        f"{linear_speed_ms:.2f}",
                        f"{linear_speed_kmh:.2f}",
                        f"{orientation_deg:.2f}",
                        detection_type,
                    ]
                )

        except Exception as e:
            print(f"Warning: Failed to log tracking data: {e}")
