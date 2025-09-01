"""Module for tracking object speed and trajectory."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class TrackPoint:
    """Represents a single point in an object's track."""
    position: Tuple[float, float]  # (x, y) in world coordinates
    timestamp: float
    frame_number: int


class SpeedTracker:
    """
    Tracks object movement and calculates speed.
    
    Maintains tracking history for multiple objects and calculates
    instantaneous and average speeds.
    """
    
    def __init__(self, max_history: int = 30, max_tracking_distance: float = 500):
        """
        Initialize speed tracker.
        
        Args:
            max_history: Maximum number of points to keep in history
            max_tracking_distance: Maximum distance (mm) to associate detections
        """
        self.max_history = max_history
        self.max_tracking_distance = max_tracking_distance
        self.tracks = {}  # Dictionary of object ID to track history
        self.next_id = 0
        
    def update(self, measurements: List[Dict]) -> List[Dict]:
        """
        Update tracks with new measurements.
        
        Args:
            measurements: List of measurement dictionaries
            
        Returns:
            Updated measurements with speed information
        """
        current_time = time.time()
        
        # Match measurements to existing tracks
        matched_measurements = self._match_to_tracks(measurements)
        
        results = []
        for measurement in matched_measurements:
            track_id = measurement.get('track_id')
            
            if track_id is None:
                # New track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = deque(maxlen=self.max_history)
                measurement['track_id'] = track_id
            
            # Add point to track
            position = measurement['center_world']
            track_point = TrackPoint(
                position=position,
                timestamp=measurement['timestamp'],
                frame_number=measurement.get('frame_number', 0)
            )
            
            self.tracks[track_id].append(track_point)
            
            # Calculate speed if we have history
            if len(self.tracks[track_id]) >= 2:
                speeds = self._calculate_speed(self.tracks[track_id])
                measurement.update(speeds)
            else:
                measurement.update({
                    'speed': 0.0,
                    'speed_x': 0.0,
                    'speed_y': 0.0,
                    'average_speed': 0.0,
                    'direction': 0.0,
                    'total_distance': 0.0
                })
            
            measurement['track_length'] = len(self.tracks[track_id])
            results.append(measurement)
        
        # Clean up old tracks
        self._cleanup_tracks(current_time)
        
        return results
    
    def _match_to_tracks(self, measurements: List[Dict]) -> List[Dict]:
        """Match new measurements to existing tracks."""
        matched = []
        used_tracks = set()
        
        for measurement in measurements:
            best_track = None
            best_distance = self.max_tracking_distance
            
            # Find closest track
            for track_id, track in self.tracks.items():
                if track_id in used_tracks or len(track) == 0:
                    continue
                
                last_position = track[-1].position
                current_position = measurement['center_world']
                
                distance = np.linalg.norm(
                    np.array(current_position) - np.array(last_position)
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_track = track_id
            
            if best_track is not None:
                measurement['track_id'] = best_track
                used_tracks.add(best_track)
            
            matched.append(measurement)
        
        return matched
    
    def _calculate_speed(self, track: deque) -> Dict:
        """Calculate speed from track history."""
        if len(track) < 2:
            return {'speed': 0.0}
        
        # Instantaneous speed (between last two points)
        p1 = track[-2]
        p2 = track[-1]
        
        dt = p2.timestamp - p1.timestamp
        if dt <= 0:
            return {'speed': 0.0}
        
        dx = p2.position[0] - p1.position[0]
        dy = p2.position[1] - p1.position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        speed = distance / dt  # mm/s
        speed_x = dx / dt
        speed_y = dy / dt
        
        # Direction of movement
        direction = np.degrees(np.arctan2(dy, dx))
        
        # Average speed over history
        if len(track) > 2:
            total_distance = 0
            total_time = track[-1].timestamp - track[0].timestamp
            
            for i in range(1, len(track)):
                d = np.linalg.norm(
                    np.array(track[i].position) - np.array(track[i-1].position)
                )
                total_distance += d
            
            average_speed = total_distance / total_time if total_time > 0 else 0
        else:
            average_speed = speed
            total_distance = distance
        
        return {
            'speed': speed,
            'speed_x': speed_x,
            'speed_y': speed_y,
            'average_speed': average_speed,
            'direction': direction,
            'total_distance': total_distance
        }
    
    def _cleanup_tracks(self, current_time: float, timeout: float = 2.0):
        """Remove tracks that haven't been updated recently."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if len(track) > 0:
                last_update = track[-1].timestamp
                if current_time - last_update > timeout:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_trajectory(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        """Get full trajectory for a specific track."""
        if track_id in self.tracks:
            return [point.position for point in self.tracks[track_id]]
        return None
