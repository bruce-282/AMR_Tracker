"""Main application for AGV measurement system."""

import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

# Import all modules
from utils.config import SystemConfig
from calibration.camera_calibrator import CameraCalibrator
from calibration.homography_calibrator import HomographyCalibrator
from detection.agv_detector import AGVDetector
from measurement.size_measurement import SizeMeasurement
from measurement.speed_tracker import SpeedTracker
from visualization.display import Visualizer


class AGVMeasurementSystem:
    """
    Main system class that integrates all components.
    
    Handles the complete pipeline from calibration to measurement.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the AGV measurement system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.calibration_data = None
        self.camera_calibrator = None
        self.homography_calibrator = None
        self.detector = None
        self.size_measurement = None
        self.speed_tracker = None
        self.visualizer = None
        
    def run_calibration(self):
        """Run complete calibration procedure."""
        print("=" * 50)
        print("AGV Measurement System - Calibration Mode")
        print("=" * 50)
        
        # Step 1: Camera calibration
        print("\n[Step 1] Camera Intrinsic Calibration")
        print("-" * 40)
        self._calibrate_camera()
        
        # Step 2: Ground plane calibration
        print("\n[Step 2] Ground Plane Calibration")
        print("-" * 40)
        self._calibrate_ground_plane()
        
        # Save calibration
        self._save_calibration()
        print("\n✓ Calibration complete and saved!")
        
    def _calibrate_camera(self):
        """Perform camera intrinsic calibration."""
        self.camera_calibrator = CameraCalibrator(
            self.config.calibration.checkerboard_size,
            self.config.calibration.square_size
        )
        
        print(f"Please capture {self.config.calibration.num_calibration_images} "
              f"images of the checkerboard from different angles.")
        print("Press SPACE to capture, ESC to finish early")
        
        cap = cv2.VideoCapture(0)
        captured_images = []
        
        while len(captured_images) < self.config.calibration.num_calibration_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Show frame
            display_frame = frame.copy()
            cv2.putText(
                display_frame, 
                f"Captured: {len(captured_images)}/{self.config.calibration.num_calibration_images}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.imshow("Calibration - Press SPACE to capture", display_frame)
            
            key = cv2.waitKey(1)
            if key == ord(' '):
                # Try to add calibration image
                if self.camera_calibrator.add_calibration_image(frame):
                    captured_images.append(frame)
                    print(f"✓ Image {len(captured_images)} captured successfully")
                else:
                    print("✗ Checkerboard not found, try again")
            elif key == 27:  # ESC
                if len(captured_images) >= 3:
                    break
                else:
                    print("Need at least 3 images for calibration")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Perform calibration
        h, w = captured_images[0].shape[:2]
        calibration_result = self.camera_calibrator.calibrate((w, h))
        
        print(f"\n✓ Camera calibration successful!")
        print(f"  Reprojection error: {calibration_result['reprojection_error']:.3f} pixels")
        
    def _calibrate_ground_plane(self):
        """Perform ground plane homography calibration."""
        if self.camera_calibrator is None:
            raise ValueError("Camera must be calibrated first")
        
        print("\nPlace checkerboard on the ground and press SPACE to capture")
        
        cap = cv2.VideoCapture(0)
        ground_image = None
        
        while ground_image is None:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.imshow("Ground Plane Calibration - Press SPACE", frame)
            
            if cv2.waitKey(1) == ord(' '):
                ground_image = frame
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate homography
        self.homography_calibrator = HomographyCalibrator(
            self.camera_calibrator.camera_matrix,
            self.camera_calibrator.dist_coeffs
        )
        
        homography_result = self.homography_calibrator.calibrate_ground_plane(
            ground_image,
            self.config.calibration.checkerboard_size,
            self.config.calibration.square_size
        )
        
        print(f"✓ Ground plane calibration successful!")
        print(f"  Scale: {homography_result['pixels_per_mm']:.3f} pixels/mm")
        
    def _save_calibration(self):
        """Save calibration data to file."""
        calibration_data = {
            'camera_matrix': self.camera_calibrator.camera_matrix.tolist(),
            'distortion_coeffs': self.camera_calibrator.dist_coeffs.tolist(),
            'homography': self.homography_calibrator.homography.tolist(),
            'pixels_per_mm': self.homography_calibrator.pixels_per_mm,
            'camera_height': self.config.calibration.camera_height,
            'calibration_error': self.camera_calibrator.calibration_error
        }
        
        with open(self.config.calibration.calibration_data_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        self.calibration_data = calibration_data
        
    def load_calibration(self):
        """Load calibration data from file."""
        with open(self.config.calibration.calibration_data_path, 'r') as f:
            self.calibration_data = json.load(f)
        
        # Convert to numpy arrays
        self.calibration_data['camera_matrix'] = np.array(
            self.calibration_data['camera_matrix']
        )
        self.calibration_data['distortion_coeffs'] = np.array(
            self.calibration_data['distortion_coeffs']
        )
        self.calibration_data['homography'] = np.array(
            self.calibration_data['homography']
        )
        
        print("✓ Calibration data loaded successfully")
        
    def initialize_components(self):
        """Initialize all system components."""
        if self.calibration_data is None:
            raise ValueError("Calibration data must be loaded first")
        
        # Initialize detector
        self.detector = AGVDetector(
            min_area=self.config.measurement.min_agv_area
        )
        
        # Initialize measurement
        self.size_measurement = SizeMeasurement(
            homography=self.calibration_data['homography'],
            camera_height=self.calibration_data['camera_height']
        )
        
        # Initialize speed tracker
        self.speed_tracker = SpeedTracker(
            max_tracking_distance=self.config.measurement.max_tracking_distance
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer(
            homography=self.calibration_data['homography']
        )
        
        print("✓ All components initialized")
        
    def run_measurement(self, source: Optional[str] = None):
        """
        Run measurement on video stream.
        
        Args:
            source: Video source (file path or camera index)
        """
        print("=" * 50)
        print("AGV Measurement System - Measurement Mode")
        print("=" * 50)
        
        # Initialize components
        self.initialize_components()
        
        # Open video source
        if source is None:
            cap = cv2.VideoCapture(0)
        elif source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Video writer for recording (optional)
        out = None
        if self.config.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(
                self.config.output_video_path,
                fourcc, fps, (width, height)
            )
        
        frame_number = 0
        print("\nPress 'q' to quit, 's' to save snapshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = time.time()
            
            # Detect AGVs
            detections = self.detector.detect(frame, frame_number, timestamp)
            
            # Measure dimensions
            measurements = []
            for detection in detections:
                # Estimate height based on size (simplified)
                height = self._estimate_agv_height(detection)
                
                # Measure
                measurement = self.size_measurement.measure(detection, height)
                measurement['frame_number'] = frame_number
                measurements.append(measurement)
            
            # Track speed
            measurements = self.speed_tracker.update(measurements)
            
            # Visualize
            vis_frame = self.visualizer.draw_detections(
                frame, detections, measurements
            )
            
            # Display bird's eye view
            bird_eye = self.visualizer.create_bird_eye_view(measurements)
            
            # Show results
            cv2.imshow("AGV Measurement System", vis_frame)
            cv2.imshow("Bird's Eye View", bird_eye)
            
            # Print measurements
            for i, m in enumerate(measurements):
                if frame_number % 30 == 0:  # Print every second
                    print(f"\nAGV {m.get('track_id', i)}:")
                    print(f"  Size: {m['width']:.0f} x {m['height']:.0f} mm")
                    print(f"  Speed: {m.get('speed', 0):.1f} mm/s")
                    print(f"  Position: ({m['center_world'][0]:.0f}, "
                          f"{m['center_world'][1]:.0f}) mm")
            
            # Record video
            if out is not None:
                out.write(vis_frame)
            
            # Handle key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"snapshot_{frame_number}.jpg", vis_frame)
                print(f"Snapshot saved: snapshot_{frame_number}.jpg")
            
            frame_number += 1
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        print("\n✓ Measurement completed")
        
    def _estimate_agv_height(self, detection) -> float:
        """Estimate AGV height based on detection size."""
        # Simplified height estimation
        # In practice, this could use machine learning or known AGV models
        area = cv2.contourArea(detection.bbox)
        
        if area < 5000:
            return self.config.measurement.agv_heights['small']
        elif area < 10000:
            return self.config.measurement.agv_heights['medium']
        else:
            return self.config.measurement.agv_heights['large']


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AGV Measurement System")
    parser.add_argument(
        'mode', 
        choices=['calibrate', 'measure'],
        help="Operating mode"
    )
    parser.add_argument(
        '--source',
        default=None,
        help="Video source for measurement mode"
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = SystemConfig.load(args.config)
    else:
        config = SystemConfig()
        config.save(args.config)
        print(f"Created default configuration: {args.config}")
    
    # Create system
    system = AGVMeasurementSystem(config)
    
    # Run appropriate mode
    if args.mode == 'calibrate':
        system.run_calibration()
    else:  # measure
        # Load calibration first
        try:
            system.load_calibration()
        except FileNotFoundError:
            print("ERROR: No calibration data found!")
            print("Please run calibration first: python agv_system.py calibrate")
            return
        
        system.run_measurement(args.source)


if __name__ == "__main__":
    main()
