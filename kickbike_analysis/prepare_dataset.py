from pathlib import Path
import numpy as np

from .data_loader import load_video_frames
from .feature_extractor import compute_frame_features


def prepare(dataset_dir: Path):
    """Extract per-frame features from videos under dataset_dir."""
    dataset_dir = Path(dataset_dir)
    for video in dataset_dir.glob("*.mp4"):
        frames = load_video_frames(video)
        feats = compute_frame_features(frames)
        out_file = video.with_suffix(".npy")
        np.save(out_file, feats)
        print(f"saved {out_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        user_input = input(
            "動画が保存されているフォルダを指定してください (空欄で 'data' フォルダを使用します): "
        ).strip()
        user_input = user_input.strip('"')
        path = Path(user_input) if user_input else Path("data")

    if path.suffix.lower() == ".mp4":
        path = path.parent

    prepare(path)
