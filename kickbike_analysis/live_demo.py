from __future__ import annotations

"""Display real-time analysis for children riding kickbikes only, with color-based re-identification.

This script streams an input video, uses YOLOv8 to detect persons and motorcycles,
combines overlapping person+motorcycle detections into single "child-on-kickbike" objects,
and tracks these combined objects with IDs maintained across frames using both IoU and
appearance features (helmet and bike color). Overlays include detection boxes with IDs,
pose keypoints, and pass/fail percentages.
"""

import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# import feature extractor functions
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from feature_extractor import (
    _bike_tilt,
    _face_direction,
    _foot_amplitude,
    _kick_period,
    _posture_stability,
    _iou,
)

# COCO skeleton pairs for pose model
_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# thresholds for matching
IOU_THRESHOLD = 0.3
COLOR_DIST_THRESHOLD = 40.0  # allowable color distance


def extract_color_features(frame: np.ndarray, box: Tuple[float,float,float,float]) -> Tuple[np.ndarray,np.ndarray]:
    """Extract average BGR color for helmet (top region) and bike (bottom region)."""
    x1,y1,x2,y2 = map(int, box)
    h = y2 - y1
    # helmet region: top 20%
    hy2 = y1 + int(0.2*h)
    helmet_roi = frame[y1:hy2, x1:x2]
    # bike region: bottom 30%
    by1 = y2 - int(0.3*h)
    bike_roi = frame[by1:y2, x1:x2]
    # compute mean color
    helmet_color = cv2.mean(helmet_roi)[:3] if helmet_roi.size else (0,0,0)
    bike_color = cv2.mean(bike_roi)[:3] if bike_roi.size else (0,0,0)
    return np.array(helmet_color), np.array(bike_color)


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    return np.linalg.norm(c1 - c2)


def show(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("動画を開けませんでした")
        return

    cv2.namedWindow("analysis", cv2.WINDOW_NORMAL)
    device = "cpu"

    # detection and pose models
    det_model = YOLO("yolov8n.pt").to(device)
    pose_model = YOLO("yolov8n-pose.pt").to(device)

    # track state: id -> (box, helmet_color, bike_color)
    track_state: Dict[int, Tuple[List[float], np.ndarray, np.ndarray]] = {}
    next_id = 0

    memory: Dict[int, Dict[str, List]] = defaultdict(lambda: {"centers": [], "foot_y": []})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_cpu = frame.copy()

        # raw detection
        result = det_model(frame_cpu)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        # separate person and motorcycle
        person_boxes = [b for b, c, s in zip(boxes, classes, scores) if c==0 and s>0.4]
        moto_boxes = [b for b, c, s in zip(boxes, classes, scores) if c==3 and s>0.4]

        # combine person+moto
        combined_boxes = []
        for p in person_boxes:
            for m in moto_boxes:
                if _iou(p, m) > IOU_THRESHOLD:
                    x1,y1 = min(p[0],m[0]), min(p[1],m[1])
                    x2,y2 = max(p[2],m[2]), max(p[3],m[3])
                    combined_boxes.append([x1,y1,x2,y2])
                    break

        # assign IDs using IoU and color
        new_state = {}
        assignments = []  # list of (box, id)
        used_ids = set()
        for box in combined_boxes:
            box = list(box)
            # extract appearance
            helmet_color, bike_color = extract_color_features(frame, box)
            assigned_id = None
            # try IoU match
            best_iou = IOU_THRESHOLD
            for tid, (prev_box, prev_hc, prev_bc) in track_state.items():
                iou_val = _iou(box, prev_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    assigned_id = tid
            # if no IoU, try color match
            if assigned_id is None:
                best_dist = COLOR_DIST_THRESHOLD
                for tid, (prev_box, prev_hc, prev_bc) in track_state.items():
                    dist = color_distance(helmet_color, prev_hc) + color_distance(bike_color, prev_bc)
                    if dist < best_dist:
                        best_dist = dist
                        assigned_id = tid
            # new ID
            if assigned_id is None:
                assigned_id = next_id
                next_id += 1
            new_state[assigned_id] = (box, helmet_color, bike_color)
            assignments.append((box, assigned_id))

        track_state = new_state

        # pose estimation
        pose_res = pose_model(frame_cpu)[0]
        pose_boxes = pose_res.boxes.xyxy.cpu().numpy()
        pose_kpts = pose_res.keypoints.data.cpu().numpy()

        pass_list = []

        # render each tracked child
        for box, tid in assignments:
            x1,y1,x2,y2 = map(int, box)
            # draw detection
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"ID{tid}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)

            # match pose
            kp = None
            for pb, pk in zip(pose_boxes, pose_kpts):
                if _iou(box, pb) > IOU_THRESHOLD:
                    kp = pk[:, :2]
                    break
            if kp is None:
                continue

            # draw skeleton
            for x,y in kp:
                cv2.circle(frame, (int(x),int(y)), 3, (0,255,0), -1)
            for i,j in _SKELETON:
                pt1 = tuple(kp[i].astype(int)); pt2 = tuple(kp[j].astype(int))
                cv2.line(frame, pt1, pt2, (0,255,255), 2)

            # compute features
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            memory[tid]["centers"].append([cx, cy])
            memory[tid]["foot_y"].append(float(kp[15,1]))
            centers_arr = np.array(memory[tid]["centers"])
            tilt = _bike_tilt(kp); posture = _posture_stability(kp)
            face = _face_direction(kp); amp = _foot_amplitude(memory[tid]["foot_y"])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            period = _kick_period(memory[tid]["foot_y"], fps)
            speed = float(np.mean(np.linalg.norm(np.diff(centers_arr,axis=0),axis=1))) if len(centers_arr)>1 else 0.0
            good = sum([abs(tilt)<=20, posture<=15, face<=10, amp>=5])
            pass_pct = int(100*good/4); pass_list.append(pass_pct)

            # feature text
            ft = f"tilt:{tilt:.1f} post:{posture:.1f} face:{face:.1f} amp:{amp:.1f}"
            ft += f" spd:{speed:.1f} per:{period:.2f}"
            cv2.putText(frame, ft, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        # global average
        if pass_list:
            avg = int(sum(pass_list)/len(pass_list))
            h,w = frame.shape[:2]; color=(0,255,0) if avg>=50 else (0,0,255)
            cv2.putText(frame, f"平均合格率:{avg}%", (w-200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)

        cv2.imshow("analysis", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()


if __name__=="__main__":
    show(Path(r"I:\chatgpt-main\chatgpt-main\data\test2.mp4"))
