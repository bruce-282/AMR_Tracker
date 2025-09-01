"""Camera calibration module for intrinsic parameters."""

import cv2
import numpy as np
import glob
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class CameraCalibrator:
    """
    Performs camera calibration to obtain intrinsic parameters.
    
    This class handles the complete calibration process including:
    - Checkerboard detection
    - Corner refinement
    - Camera matrix calculation
    - Distortion coefficient estimation
    """
    
    def __init__(self, checkerboard_size: Tuple[int, int], square_size: float):
        """
        Initialize the camera calibrator.
        
        Args:
            checkerboard_size: (cols-1, rows-1) inner corners
            square_size: Size of checkerboard square in mm
        """
        self.pattern_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration data
        self.objpoints = []  # 3D points in real world
        self.imgpoints = []  # 2D points in image plane
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None
        
    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add a calibration image with checkerboard.
        
        Args:
            image: Input image containing checkerboard
            
        Returns:
            True if checkerboard was found, False otherwise
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners_refined)
            
            return True
        return False
    
    def calibrate(self, image_size: Tuple[int, int]) -> Dict:
        """
        Perform camera calibration.
        
        Args:
            image_size: (width, height) of the images
            
        Returns:
            Dictionary containing calibration results
        """
        if len(self.objpoints) < 3:
            raise ValueError("Need at least 3 calibration images")
        
        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
        
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        
        # Calculate reprojection error
        self.calibration_error = self._calculate_reprojection_error(rvecs, tvecs)
        
        return {
            'camera_matrix': mtx.tolist(),
            'distortion_coeffs': dist.tolist(),
            'reprojection_error': self.calibration_error,
            'num_images': len(self.objpoints),
            'image_size': image_size
        }
    
    def _calculate_reprojection_error(self, rvecs: List, tvecs: List) -> float:
        """Calculate the mean reprojection error."""
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], 
                self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        return total_error / len(self.objpoints)
    
    def save_calibration(self, filepath: str):
        """Save calibration data to file."""
        if self.camera_matrix is None:
            raise ValueError("No calibration data to save. Run calibrate() first.")
        
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coeffs': self.dist_coeffs.tolist(),
            'calibration_error': self.calibration_error,
            'pattern_size': self.pattern_size,
            'square_size': self.square_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
    
    @staticmethod
    def load_calibration(filepath: str) -> Dict:
        """Load calibration data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        data['camera_matrix'] = np.array(data['camera_matrix'])
        data['distortion_coeffs'] = np.array(data['distortion_coeffs'])
        
        return data
