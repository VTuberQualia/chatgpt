from __future__ import annotations

"""Display real-time analysis for a video.

This script streams an input video and overlays pose keypoints, extracted
features and a simple pass/fail percentage for each detected person.
The displayed percentage is computed from heuristic thresholds and is
meant only as a visual aid while inspecting the video.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

# Allow execution both as a module and as a standalone script
try:  # pragma: no cover - import fallback
    from .feature_extractor import (
        _bike_tilt,
        _face_direction,
        _foot_amplitude,
        _kick_period,
        _posture_stability,
        _iou,
    )
except ImportError:  # run as script
    from feature_extractor import (
        _bike_tilt,
        _face_direction,
        _foot_amplitude,
        _kick_period,
        _posture_stability,
        _iou,
    )

# COCO style skeleton pairs used by YOLOv8 pose model
_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


def show(video_path: Path) -> None:
    """Stream the video with analysis overlays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("動画を開けませんでした")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")
    det_model.to(device)
    pose_model.to(device)
    tracker = BYTETracker()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    memory: Dict[int, Dict[str, List]] = defaultdict(lambda: {"centers": [], "foot_y": []})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_results = det_model(frame)[0]
        boxes = det_results.boxes.xyxy.cpu().numpy()
        scores = det_results.boxes.conf.cpu().numpy()
        classes = det_results.boxes.cls.cpu().numpy().astype(int)

        person_boxes = boxes[(classes == 0) & (scores > 0.4)]
        bike_boxes = boxes[(classes == 1) & (scores > 0.4)]

        detections = []
        for p_box in person_boxes:
            for b_box in bike_boxes:
                if _iou(p_box, b_box) > 0.3:
                    detections.append(p_box)
                    break

        tracks = tracker.update(np.array(detections), frame)
        poses = pose_model(frame)[0]

        for track in tracks:
            box = track.tlbr
            tid = track.track_id
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            memory[tid]["centers"].append([cx, cy])

            # find matching pose for the track
            kp = None
            for pose_box, kpt in zip(poses.boxes.xyxy.cpu().numpy(), poses.keypoints.cpu().numpy()):
                if _iou(box, pose_box) > 0.5:
                    kp = kpt[:, :2]
                    break
            if kp is None:
                continue

            memory[tid]["foot_y"].append(float(kp[15, 1]))

            # draw skeleton
            for x, y in kp:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            for i, j in _SKELETON:
                pt1 = tuple(np.int32(kp[i]))
                pt2 = tuple(np.int32(kp[j]))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

            # compute features
            tilt = _bike_tilt(kp)
            posture = _posture_stability(kp)
            face = _face_direction(kp)
            amp = _foot_amplitude(memory[tid]["foot_y"])
            period = _kick_period(memory[tid]["foot_y"], fps)
            centers = np.array(memory[tid]["centers"])
            speed = float(np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0

            # pass/fail percentage based on simple thresholds
            good = 0
            total = 4
            if abs(tilt) <= 20:
                good += 1
            if posture <= 15:
                good += 1
            if face <= 10:
                good += 1
            if amp >= 5:
                good += 1
            pass_pct = int(100 * good / total)
            fail_pct = 100 - pass_pct

            color = (0, 255, 0) if pass_pct >= 50 else (0, 0, 255)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                frame,
                f"合格{pass_pct}% 不合格{fail_pct}%",
                (int(box[0]), int(box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            feat_text = f"tilt:{tilt:.1f} post:{posture:.1f} face:{face:.1f} amp:{amp:.1f} spd:{speed:.1f} per:{period:.2f}"
            cv2.putText(
                frame,
                feat_text,
                (int(box[0]), int(box[3]) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        cv2.imshow("analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kickbike_analysis/live_demo.py <video_path>")
        raise SystemExit(1)
    show(Path(sys.argv[1]))
