
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


def compute_frame_features(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Return center of gravity and speed for each frame."""
    features = []
    prev_center = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = cv2.moments(gray)
        if m["m00"] != 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            h, w = gray.shape
            cx, cy = w / 2, h / 2
        center = np.array([cx, cy])
        if prev_center is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(center - prev_center))
        features.append([cx, cy, speed])
        prev_center = center
    return np.array(features, dtype=np.float32)
