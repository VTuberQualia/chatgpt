"""映像品質の簡易チェックを行うモジュール。"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def check_quality(frame: np.ndarray, bbox: List[int]) -> List[str]:
    """フレームと人物領域から品質上の問題を検出し説明を返す。

    チェック項目:
    - 被写体が小さすぎる
    - ブレが大きい
    - 暗すぎる/逆光
    - フレームから被写体がはみ出している
    """

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    issues: List[str] = []

    area_ratio = (x2 - x1) * (y2 - y1) / (w * h)
    if area_ratio < 0.05:
        issues.append("被写体が小さすぎます")

    if x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h:
        issues.append("被写体がフレームからはみ出しています")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 100:
        issues.append("映像がブレています")

    brightness = gray.mean()
    if brightness < 40:
        issues.append("映像が暗すぎます")

    return issues
