"""Unsupervised clustering utilities using KMeans."""
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.cluster import KMeans

from .data_loader import load_video_frames
from .feature_extractor import compute_frame_features


def _video_features(video: Path) -> List[np.ndarray]:
    """Return person-wise feature vectors for a video."""
    feature_file = video.with_suffix(".npy")
    if feature_file.exists():
        motion = np.load(feature_file)
    else:
        frames = list(load_video_frames(video))
        motion = compute_frame_features(frames)
    if motion.size == 0:
        return []
    return [motion[i] for i in range(motion.shape[0])]


def train_clusters(dataset_dir: Path, n_clusters: int = 2) -> KMeans:
    """Cluster videos under ``dataset_dir`` and return fitted model."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"{dataset_dir} が見つかりません")

    videos = list(dataset_dir.glob("*.mp4"))
    features: List[np.ndarray] = []
    for video in videos:
        vecs = _video_features(video)
        if not vecs:
            continue
        features.extend(vecs)

    if not features:
        raise RuntimeError(
            f"{dataset_dir} に利用可能な動画が見つかりません。mp4ファイルが存在し、十分な動きがあるか確認してください。"
        )

    data = np.stack(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans


def save_model(model: KMeans, out_file: Path) -> None:
    joblib.dump(model, out_file)


def load_model(path: Path) -> KMeans:
    return joblib.load(path)


def predict_cluster(video: Path, model: KMeans) -> List[int]:
    """Return predicted cluster IDs for each detected person."""
    vecs = _video_features(video)
    if not vecs:
        return []
    data = np.stack(vecs)
    labels = model.predict(data)
    return [int(l) for l in labels]
