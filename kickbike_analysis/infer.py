"""学習済みモデルを用いた推論ユーティリティ。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import joblib

# パッケージ外からも実行できるようにする
try:  # pragma: no cover - import fallback
    from .data_loader import load_video_frames
    from .feature_extractor import compute_frame_features
    from .quality_checker import check_quality
    from .scoring import score_features
except ImportError:  # run as script
    from data_loader import load_video_frames
    from feature_extractor import compute_frame_features
    from quality_checker import check_quality
    from scoring import score_features


def predict(video_path: Path, model_path: Path) -> List[Dict[str, object]]:
    """動画から特徴量を抽出し学習済みモデルでカテゴリを推定する。"""

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

    model = joblib.load(model_path)

    results: List[Dict[str, object]] = []
    for idx, (vec, crop) in enumerate(zip(features, crops), start=1):
        pred = model.predict([vec])[0]
        prob = model.predict_proba([vec])[0]
        label_index = list(model.classes_).index(pred)
        score = int(prob[label_index] * 100)
        _, _, advices = score_features(vec)

        frame_img, bbox = crop
        x1, y1, x2, y2 = bbox
        img = frame_img[y1:y2, x1:x2].copy()
        cv2.putText(
            img,
            f"{score} {pred}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if pred == "Ready" else (0, 0, 255),
            2,
        )
        out_file = video_path.with_name(f"{video_path.stem}_person{idx}.jpg")
        cv2.imwrite(str(out_file), img)

        quality = check_quality(frame_img, bbox)

        results.append(
            {
                "score": score,
                "category": str(pred),
                "advices": advices,
                "image_path": str(out_file),
                "quality": quality,
            }
        )

    return results


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m kickbike_analysis.infer <video_path> <model.pkl>")
        sys.exit(1)

    res = predict(Path(sys.argv[1]), Path(sys.argv[2]))
    for r in res:
        print(f"{r['score']}点 {r['category']} -> {r['image_path']}")
        if r["advices"]:
            print("  アドバイス:", "、".join(r["advices"]))
        if r["quality"]:
            print("  品質:", "、".join(r["quality"]))
