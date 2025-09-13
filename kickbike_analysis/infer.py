"""推論用ユーティリティ。

学習済みモデルを必要とせず、`analyze_video` の結果をそのまま
返す簡易版である。将来的には学習済み分類器や強化学習モデルを
読み込んで使用する予定。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

# パッケージ外からも実行できるようにする
try:  # pragma: no cover - import fallback
    from .analyze_video import analyze
except ImportError:  # run as script
    from analyze_video import analyze


def predict(video_path: Path) -> List[Dict[str, object]]:
    """動画から人物ごとのスコアと判定を取得する。"""
    return analyze(video_path)


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m kickbike_analysis.infer <video_path>")
        sys.exit(1)

    res = predict(Path(sys.argv[1]))
    for r in res:
        print(f"{r['score']}点 {r['category']} -> {r['image_path']}")
        if r["advices"]:
            print("  アドバイス:", "、".join(r["advices"]))
        if r["quality"]:
            print("  品質:", "、".join(r["quality"]))
