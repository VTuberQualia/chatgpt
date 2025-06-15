from pathlib import Path
from typing import Tuple
import numpy as np

from .data_loader import load_video_frames
from .feature_extractor import compute_motion_vectors


def analyze(video_path: Path) -> Tuple[str, str]:
    """Heuristically analyze a kickbike video and return judgment and reason."""
    frames = list(load_video_frames(video_path))
    if not frames:
        return "不可", "動画を読み込めませんでした"

    motion = compute_motion_vectors(frames)
    if motion.size == 0:
        return "不可", "動きが検出できませんでした"

    std_dev = float(np.std(motion))
    smoothness = float(np.mean(np.abs(np.diff(motion))))

    # Thresholds chosen empirically for placeholder logic
    if std_dev < 0.5 and smoothness < 0.3:
        return "可", "揺れが小さくスムーズに走行しています"
    reason = []
    if std_dev >= 0.5:
        reason.append("揺れが大きい")
    if smoothness >= 0.3:
        reason.append("速度変化が大きい")
    return "不可", "、".join(reason)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m kickbike_analysis.analyze_video <video_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    result, detail = analyze(path)
    print(f"判定: {result}\n理由: {detail}")

