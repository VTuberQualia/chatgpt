"""Feature extraction methods for kickbike movement analysis.

Currently uses simple optical flow to remain robust to camera angles.
"""
from typing import Iterable
import cv2
import numpy as np

# Placeholder for actual pose estimation; this uses optical flow as example

def compute_motion_vectors(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Compute simple frame-to-frame motion vectors."""
    vectors = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            vectors.append(flow.mean())
        prev_gray = gray
    return np.array(vectors)
