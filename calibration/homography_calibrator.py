"""Homography calibration for ground plane mapping."""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class HomographyCalibrator:
    """
    Calibrates the homography transformation between image and ground plane.
    
    This class establishes the mapping between pixel coordinates and 
    real-world coordinates on the ground plane.
    """
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Initialize homography calibrator.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.homography = None
        self.pixels_per_mm = None
        
    def calibrate_ground_plane(
        self, 
        image: np.ndarray, 
        pattern_size: Tuple[int, int],
        square_size: float
    ) -> Dict:
        """
        Calibrate homography using checkerboard on ground.
        
        Args:
            image: Image with checkerboard on ground
            pattern_size: Checkerboard pattern size
            square_size: Size of each square in mm
            
        Returns:
            Dictionary with homography and scale information
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if not ret:
            raise ValueError("Checkerboard not found in image")
        
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Create object points on ground plane (Z=0)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Find homography
        obj_points_2d = objp[:, :2]
        img_points_2d = corners.reshape(-1, 2)
        
        self.homography, mask = cv2.findHomography(
            img_points_2d, obj_points_2d, cv2.RANSAC, 5.0
        )
        
        # Calculate scale (pixels per mm)
        self.pixels_per_mm = self._calculate_scale(img_points_2d, obj_points_2d)
        
        return {
            'homography': self.homography.tolist(),
            'pixels_per_mm': self.pixels_per_mm,
            'inliers': np.sum(mask)
        }
    
    def _calculate_scale(self, img_points: np.ndarray, obj_points: np.ndarray) -> float:
        """Calculate the average scale factor."""
        # Calculate distances between consecutive points
        img_dists = []
        obj_dists = []
        
        for i in range(len(img_points) - 1):
            img_dist = np.linalg.norm(img_points[i+1] - img_points[i])
            obj_dist = np.linalg.norm(obj_points[i+1] - obj_points[i])
            
            if obj_dist > 0:  # Avoid division by zero
                img_dists.append(img_dist)
                obj_dists.append(obj_dist)
        
        # Calculate average scale
        scales = [img_d / obj_d for img_d, obj_d in zip(img_dists, obj_dists)]
        return np.mean(scales)
    
    def image_to_world(self, image_points: np.ndarray) -> np.ndarray:
        """
        Transform image points to world coordinates.
        
        Args:
            image_points: Points in image coordinates (N x 2)
            
        Returns:
            Points in world coordinates (N x 2)
        """
        if self.homography is None:
            raise ValueError("Homography not calibrated")
        
        # Convert to homogeneous coordinates
        pts_homo = np.column_stack([image_points, np.ones(len(image_points))])
        
        # Apply homography
        world_pts_homo = (self.homography @ pts_homo.T).T
        
        # Convert back to 2D
        world_pts = world_pts_homo[:, :2] / world_pts_homo[:, 2:3]
        
        return world_pts
    
    def world_to_image(self, world_points: np.ndarray) -> np.ndarray:
        """
        Transform world points to image coordinates.
        
        Args:
            world_points: Points in world coordinates (N x 2)
            
        Returns:
            Points in image coordinates (N x 2)
        """
        if self.homography is None:
            raise ValueError("Homography not calibrated")
        
        # Inverse homography
        H_inv = np.linalg.inv(self.homography)
        
        # Convert to homogeneous coordinates
        pts_homo = np.column_stack([world_points, np.ones(len(world_points))])
        
        # Apply inverse homography
        img_pts_homo = (H_inv @ pts_homo.T).T
        
        # Convert back to 2D
        img_pts = img_pts_homo[:, :2] / img_pts_homo[:, 2:3]
        
        return img_pts
