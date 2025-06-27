from pathlib import Path
from typing import Tuple
import numpy as np

from .data_loader import load_video_frames
from .feature_extractor import compute_frame_features


def analyze(video_path: Path) -> Tuple[str, str]:
    """YOLOv8 から得た特徴量でキックバイク卒業可否を判断する。"""
    frames = list(load_video_frames(video_path))
    if not frames:
        return "不可", "動画を読み込めませんでした"

    features = compute_frame_features(frames)
    if features.size == 0:
        return "不可", "動きが検出できませんでした"

    vec = features[0]
    tilt, posture, face_dir, amp, speed, period = vec

    reasons = []
    if abs(tilt) > 20:
        reasons.append("バイクの傾きが大きい")
    if posture > 15:
        reasons.append("重心が安定しない")
    if face_dir > 10:
        reasons.append("顔の向きが定まらない")
    if amp < 5:
        reasons.append("蹴りが弱い")

    if not reasons:
        return "移行可能", "姿勢が安定しています"
    return "継続練習", "、".join(reasons)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m kickbike_analysis.analyze_video <video_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    result, detail = analyze(path)
    print(f"判定: {result}\n理由: {detail}")
