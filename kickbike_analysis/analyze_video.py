from pathlib import Path
from typing import Tuple
import numpy as np

from .data_loader import load_video_frames

from .feature_extractor import compute_frame_features



def analyze(video_path: Path) -> Tuple[str, str]:
    """Heuristically analyze a kickbike video and return judgment and reason."""
    frames = list(load_video_frames(video_path))
    if not frames:
        return "不可", "動画を読み込めませんでした"


    features = compute_frame_features(frames)
    if features.size == 0:
        return "不可", "動きが検出できませんでした"

    center_var = float(np.std(features[:, :2]))
    speed_series = features[:, 2]
    smoothness = float(np.mean(np.abs(np.diff(speed_series))))
    angle_series = features[:, 7]
    angle_change = float(np.mean(np.abs(np.diff(angle_series))))

    reasons = []
    if center_var >= 15:
        reasons.append("揺れが大きい")
    if smoothness >= 2:
        reasons.append("速度変化が大きい")
    if angle_change >= 0.5:
        reasons.append("体の傾きが安定しない")

    if not reasons:
        return "可", "姿勢が安定しています"
    return "不可", "、".join(reasons)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m kickbike_analysis.analyze_video <video_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    result, detail = analyze(path)
    print(f"判定: {result}\n理由: {detail}")

