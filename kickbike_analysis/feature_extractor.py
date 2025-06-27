"""Feature extraction using YOLOv8 and ByteTrack.

このモジュールは要件定義に沿って以下の処理を行う。

1. 動画フレームから人物と自転車を YOLOv8 で検出。
2. IoU によりバイクに乗っている人物を抽出。
3. ByteTrack で人物 ID を追跡し、時系列情報を保持。
4. YOLOv8-pose による骨格推定結果から各種特徴量を計算。
5. 1 人単位で平均特徴量を返す。

戻り値は ``(N, 6)`` 形状の ``numpy.ndarray`` で、各行は
[バイク傾き, 姿勢安定度, 顔向き変化, 足の振幅, 平均速度, 蹴り周期] を表す。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Dict, List, Tuple

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------

def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """IoU を計算する。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def _bike_tilt(kp: np.ndarray) -> float:
    """肩と腰の左右差から傾きを算出"""
    l_sh, r_sh = kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]
    sh_angle = np.arctan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0])
    hip_angle = np.arctan2(r_hip[1] - l_hip[1], r_hip[0] - l_hip[0])
    return float(np.degrees((sh_angle + hip_angle) / 2))


def _posture_stability(kp: np.ndarray) -> float:
    """肩-腰-膝の縦ラインのばらつきを評価"""
    line = kp[[5, 11, 13]]  # 左肩, 左腰, 左膝
    x_std = np.std(line[:, 0])
    return float(x_std)


def _face_direction(kp: np.ndarray) -> float:
    """顔の向きを数値化。鼻が目の中心からどれだけずれているか。"""
    nose = kp[0]
    eyes_center_x = (kp[1, 0] + kp[2, 0]) / 2
    return float(abs(nose[0] - eyes_center_x))


def _foot_amplitude(foot_seq: List[float]) -> float:
    if not foot_seq:
        return 0.0
    return float(np.max(foot_seq) - np.min(foot_seq))


def _kick_period(foot_seq: List[float], fps: float) -> float:
    if len(foot_seq) < 3:
        return 0.0
    idx, _ = find_peaks(np.array(foot_seq))
    if len(idx) < 2:
        return 0.0
    periods = np.diff(idx) / fps
    return float(np.mean(periods))


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def compute_frame_features(
    frames: Iterable[np.ndarray],
    fps: float = 30.0,
    return_crops: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Tuple[np.ndarray, List[int]]]]:
    """人物ごとの平均特徴量を計算する。

    Parameters
    ----------
    frames:
        解析対象のフレーム列。
    fps:
        フレームレート。蹴り周期計算に利用する。
    return_crops:
        True を指定すると人物ごとの切り出し画像も返す。

    Returns
    -------
    np.ndarray | Tuple[np.ndarray, List[Tuple[np.ndarray, List[int]]]]
        特徴量行列のみ、もしくは特徴量と切り出し画像の組。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")
    det_model.to(device)
    pose_model.to(device)

    tracker = BYTETracker()

    person_memory: Dict[int, Dict[str, List]] = defaultdict(
        lambda: {
            "centers": [],
            "foot_y": [],
            "keypoints": [],
            "crop": None,
        }
    )

    for frame in frames:
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
            tid = track.track_id
            box = track.tlbr  # x1,y1,x2,y2
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            person_memory[tid]["centers"].append([center_x, center_y])

            if person_memory[tid]["crop"] is None:
                # 対応するバイク領域を探索
                bike_box = None
                for bb in bike_boxes:
                    if _iou(box, bb) > 0.3:
                        bike_box = bb
                        break
                if bike_box is not None:
                    x1 = min(box[0], bike_box[0])
                    y1 = min(box[1], bike_box[1])
                    x2 = max(box[2], bike_box[2])
                    y2 = max(box[3], bike_box[3])
                else:
                    x1, y1, x2, y2 = box
                person_memory[tid]["crop"] = (frame.copy(), [int(x1), int(y1), int(x2), int(y2)])

            # 該当するポーズを取得
            kp = None
            for pose_box, kpt in zip(poses.boxes.xyxy.cpu().numpy(), poses.keypoints.cpu().numpy()):
                if _iou(box, pose_box) > 0.5:
                    kp = kpt[:, :2]
                    break
            if kp is not None:
                person_memory[tid]["keypoints"].append(kp)
                person_memory[tid]["foot_y"].append(float(kp[15, 1]))

    feature_vectors: List[np.ndarray] = []
    crops: List[Tuple[np.ndarray, List[int]]] = []
    for info in person_memory.values():
        centers = np.array(info["centers"])
        if len(centers) < 2:
            continue
        speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        avg_speed = float(np.mean(speeds))

        kps = [k for k in info["keypoints"] if k is not None]
        if not kps:
            continue
        tilts = [_bike_tilt(k) for k in kps]
        stability = [_posture_stability(k) for k in kps]
        face_var = [_face_direction(k) for k in kps]
        tilt_mean = float(np.mean(np.abs(tilts)))
        posture = float(np.mean(stability))
        face_dir = float(np.mean(face_var))

        foot_seq = info["foot_y"]
        amplitude = _foot_amplitude(foot_seq)
        period = _kick_period(foot_seq, fps)

        feature_vectors.append(
            np.array([tilt_mean, posture, face_dir, amplitude, avg_speed, period], dtype=np.float32)
        )
        if return_crops and info["crop"] is not None:
            crops.append(info["crop"])

    if not feature_vectors:
        if return_crops:
            return np.empty((0, 6), dtype=np.float32), []
        return np.empty((0, 6), dtype=np.float32)

    data = np.stack(feature_vectors)
    if return_crops:
        return data, crops
    return data

