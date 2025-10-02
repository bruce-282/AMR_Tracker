"""
Association algorithms for tracking system
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Any


def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: [x, y, w, h] format
        box2: [x, y, w, h] format

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked objects using IoU and Hungarian algorithm.

    Args:
        detections: List of detection objects with bbox attribute
        trackers: Dictionary of tracker_id -> tracker objects
        iou_threshold: Minimum IoU for valid association

    Returns:
        matches: List of (detection_idx, tracker_id) pairs
        unmatched_detections: List of unmatched detection indices
        unmatched_trackers: List of unmatched tracker IDs
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    # Get tracker bounding boxes (predicted positions)
    tracker_boxes = []
    tracker_ids = list(trackers.keys())

    for tracker_id in tracker_ids:
        tracker = trackers[tracker_id]
        # Get predicted position from Kalman filter
        state = tracker.kf.statePost.flatten()
        x, y = state[0], state[1]
        # Use actual bbox size from trajectory or default
        if hasattr(tracker, "last_bbox") and tracker.last_bbox is not None:
            w, h = tracker.last_bbox[2], tracker.last_bbox[3]
        else:
            w, h = 50, 50  # Default size
        tracker_boxes.append([x - w / 2, y - h / 2, w, h])

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, detection in enumerate(detections):
        for t, tracker_box in enumerate(tracker_boxes):
            iou_matrix[d, t] = iou(detection.bbox, tracker_box)

    # Use Hungarian algorithm for optimal assignment
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))

    # Filter out matches with low IoU
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))

    for match in matched_indices:
        detection_idx, tracker_idx = match
        if iou_matrix[detection_idx, tracker_idx] >= iou_threshold:
            matches.append((detection_idx, tracker_ids[tracker_idx]))
            unmatched_detections.remove(detection_idx)
            unmatched_trackers.remove(tracker_idx)

    return matches, unmatched_detections, [tracker_ids[i] for i in unmatched_trackers]
