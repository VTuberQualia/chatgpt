"""Video loading utilities for the Kickbike Readiness Analyzer."""
from pathlib import Path
import cv2


def load_video_frames(video_path: Path):
    """Load a video file and yield individual frames as numpy arrays."""
    capture = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        yield frame
    capture.release()
