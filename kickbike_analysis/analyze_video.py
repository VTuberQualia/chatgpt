from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

from .data_loader import load_video_frames
from .feature_extractor import compute_frame_features



def analyze(video_path: Path) -> List[Tuple[str, str]]:
    """動画内の人物ごとの判定結果と画像を返す。"""
    frames = list(load_video_frames(video_path))
    if not frames:
        return [("不明", "動画を読み込めませんでした")]

    features, crops = compute_frame_features(frames, return_crops=True)

    if features.size == 0:
        return [("不明", "動きが検出できませんでした")]


    results: List[Tuple[str, str]] = []
    for idx, (vec, crop) in enumerate(zip(features, crops), start=1):
        tilt, posture, face_dir, amp, _speed, _period = vec
        reasons = []
        if abs(tilt) > 20:
            reasons.append("バイクの傾きが大きい")
        if posture > 15:
            reasons.append("重心が安定しない")
        if face_dir > 10:
            reasons.append("顔の向きが定まらない")
        if amp < 5:
            reasons.append("蹴りが弱い")

        label = "合格" if not reasons else "不合格"

        frame_img, bbox = crop
        x1, y1, x2, y2 = bbox
        img = frame_img[y1:y2, x1:x2].copy()
        color = (0, 255, 0) if label == "合格" else (0, 0, 255)
        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out_file = video_path.with_name(f"{video_path.stem}_person{idx}.jpg")
        cv2.imwrite(str(out_file), img)

        results.append((label, str(out_file)))

    return results



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m kickbike_analysis.analyze_video <video_path>")
        sys.exit(1)
    path = Path(sys.argv[1])

    results = analyze(path)
    for idx, (label, img_path) in enumerate(results, start=1):
        print(f"person {idx}: {label} -> {img_path}")

