"""動画から学習用の切り出し画像を保存する補助スクリプト。

指定したフォルダ内の ``.mp4`` 動画を順に処理し、
人物+バイク領域を切り出した画像を最大 ``limit`` 枚まで保存する。
将来の教師データ作成やアノテーション作業を想定している。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

try:  # pragma: no cover - スクリプト実行時のフォールバック
    from .data_loader import load_video_frames
    from .feature_extractor import compute_frame_features
except ImportError:  # run as standalone script
    from data_loader import load_video_frames
    from feature_extractor import compute_frame_features


def extract_images(video_dir: Path, out_dir: Path, limit: int = 1000) -> None:
    """動画から人物の切り出し画像を保存する。

    Parameters
    ----------
    video_dir: Path
        ``.mp4`` 動画が格納されたディレクトリ。
    out_dir: Path
        画像の保存先ディレクトリ。
    limit: int, default 1000
        保存する最大枚数。
    """

    video_dir = Path(video_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for video in video_dir.glob("*.mp4"):
        frames = load_video_frames(video)
        _, crops = compute_frame_features(frames, return_crops=True)
        for frame_img, bbox in crops:
            x1, y1, x2, y2 = bbox
            img = frame_img[y1:y2, x1:x2]
            out_file = out_dir / f"{video.stem}_{saved:04d}.jpg"
            cv2.imwrite(str(out_file), img)
            saved += 1
            if saved >= limit:
                print(f"{saved} 枚の画像を保存しました: {out_dir}")
                return
    print(f"{saved} 枚の画像を保存しました: {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="学習用の切り出し画像を作成する")
    parser.add_argument("video_dir", type=Path, help="動画があるディレクトリ")
    parser.add_argument("out_dir", type=Path, help="切り出し画像の保存先")
    parser.add_argument(
        "--limit", type=int, default=1000, help="保存する最大枚数 (デフォルト1000)"
    )
    args = parser.parse_args()
    extract_images(args.video_dir, args.out_dir, args.limit)
