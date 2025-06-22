
from typing import Iterable, List, Tuple
import cv2
import numpy as np

# Basic person detector using HOG. No external models are required so that the
# repository works without additional downloads.  This also serves as a simple
# tracker by matching detected regions between consecutive frames.

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


def _detect_person(frame: np.ndarray, hog: cv2.HOGDescriptor) -> Tuple[int, int, int, int] | None:
    """Return the largest detected person bounding box as ``(x, y, w, h)``."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    if len(rects) == 0:
        return None
    # pick largest
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    return tuple(rects[0])


def _pose_angle(person_roi: np.ndarray) -> float:
    """Very rough pose estimate: average angle of edges inside ROI."""
    edges = cv2.Canny(person_roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=person_roi.shape[0] // 3, maxLineGap=10)
    if lines is None:
        return 0.0
    angles: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        angles.append(np.arctan2(y2 - y1, x2 - x1))
    return float(np.mean(angles)) if angles else 0.0


def compute_frame_features(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Return extended features per frame using detection, tracking and pose."""
    features: List[List[float]] = []
    prev_center = None
    prev_bbox = None
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

        bbox = _detect_person(frame, hog)
        if bbox is None:
            bbox = prev_bbox
        if bbox is not None:
            x, y, w_box, h_box = bbox
            bbox_center = np.array([x + w_box / 2.0, y + h_box / 2.0])
            aspect = w_box / h_box
            area = w_box * h_box
            roi = gray[y : y + h_box, x : x + w_box]
            angle = _pose_angle(roi)
        else:
            bbox_center = np.zeros(2)
            aspect = 0.0
            area = 0.0
            angle = 0.0

        features.append(
            [
                cx,
                cy,
                speed,
                bbox_center[0],
                bbox_center[1],
                aspect,
                area,
                angle,
            ]
        )

        prev_center = center
        prev_bbox = bbox
    return np.array(features, dtype=np.float32)
