from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2

# Allow running as a standalone script without package context
try:  # pragma: no cover - simple import fallback
    from .data_loader import load_video_frames
    from .feature_extractor import compute_frame_features
    from .quality_checker import check_quality
    from .scoring import score_features
except ImportError:  # run as script
    from data_loader import load_video_frames
    from feature_extractor import compute_frame_features
    from quality_checker import check_quality
    from scoring import score_features


def analyze(video_path: Path) -> List[Dict[str, object]]:
    """動画内の人物ごとのスコア・判定・アドバイスなどを返す。"""
    frames = list(load_video_frames(video_path))
    if not frames:
        return [
            {
                "score": 0,
                "category": "不明",
                "advices": ["動画を読み込めませんでした"],
                "image_path": "",
                "quality": [],
            }
        ]

    features, crops = compute_frame_features(frames, return_crops=True)

    if features.size == 0:
        return [
            {
                "score": 0,
                "category": "不明",
                "advices": ["動きが検出できませんでした"],
                "image_path": "",
                "quality": [],
            }
        ]

    results: List[Dict[str, object]] = []
    for idx, (vec, crop) in enumerate(zip(features, crops), start=1):
        score, category, advices = score_features(vec)

        frame_img, bbox = crop
        x1, y1, x2, y2 = bbox
        img = frame_img[y1:y2, x1:x2].copy()
        cv2.putText(
            img,
            f"{score} {category}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if category == "Ready" else (0, 0, 255),
            2,
        )
        out_file = video_path.with_name(f"{video_path.stem}_person{idx}.jpg")
        cv2.imwrite(str(out_file), img)

        quality = check_quality(frame_img, bbox)

        results.append(
            {
                "score": score,
                "category": category,
                "advices": advices,
                "image_path": str(out_file),
                "quality": quality,
            }
        )

    return results



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kickbike_analysis/analyze_video.py <video_path>")
        sys.exit(1)
    path = Path(sys.argv[1])

    results = analyze(path)
    for idx, r in enumerate(results, start=1):
        print(
            f"person {idx}: {r['score']}点 {r['category']} -> {r['image_path']}"
        )

