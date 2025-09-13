"""走行シーンを直線・カーブ・停止にセグメント化する簡易モジュール。"""

from __future__ import annotations

import numpy as np
from typing import List


def segment_motion(centers: np.ndarray) -> List[str]:
    """中心座標列からシーンラベル列を推定する。

    Parameters
    ----------
    centers: np.ndarray
        形状 ``(T, 2)`` の人物中心座標列。

    Returns
    -------
    List[str]
        各フレームに対応する ``"straight"``, ``"curve"``, ``"stop"`` の
        ラベル列。
    """

    if len(centers) < 2:
        return []

    diffs = np.diff(centers, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    dirs = np.arctan2(diffs[:, 1], diffs[:, 0])

    labels: List[str] = ["stop"]
    for i in range(len(speeds)):
        if speeds[i] < 1.0:
            labels.append("stop")
        else:
            if i > 0 and abs(dirs[i] - dirs[i - 1]) > np.deg2rad(20):
                labels.append("curve")
            else:
                labels.append("straight")
    return labels
