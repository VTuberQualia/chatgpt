"""特徴量から乗りこなしスコアを算出するモジュール。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def score_features(vec: np.ndarray) -> Tuple[int, str, List[str]]:
    """特徴量ベクトルからスコア・カテゴリ・アドバイスを返す。

    Parameters
    ----------
    vec: np.ndarray
        ``[傾き, 姿勢安定度, 顔向き変化, 蹴り振幅, 平均速度, 蹴り周期]``
        の6次元ベクトル。
    Returns
    -------
    score: int
        0-100 の乗りこなしスコア。
    category: str
        "Beginner" / "Almost" / "Ready" のいずれか。
    advices: List[str]
        次に行うべき練習や安全アドバイスを少なくとも3件返す。
    """

    tilt, posture, face_dir, amp, speed, period = vec

    score = 100
    advices: List[str] = []

    if abs(tilt) > 20:
        score -= 20
        advices.append("体の傾きを抑え、左右バランスを意識して走りましょう。")
    if posture > 15:
        score -= 20
        advices.append("背筋を伸ばし、重心を安定させる練習をしましょう。")
    if face_dir > 10:
        score -= 20
        advices.append("進行方向をしっかり見て、視線を前に保ちましょう。")
    if amp < 5:
        score -= 20
        advices.append("蹴り出しを強くし、勢いをつけて滑走する練習をしましょう。")
    if speed < 1.0:
        score -= 10
        advices.append("もう少しスピードを出して滑走の感覚をつかみましょう。")

    while len(advices) < 3:
        advices.append("ヘルメットと防具を着用し、安全に練習を続けましょう。")

    score = max(0, min(100, int(score)))

    if score >= 70:
        category = "Ready"
    elif score >= 40:
        category = "Almost"
    else:
        category = "Beginner"

    return score, category, advices
