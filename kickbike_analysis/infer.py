"""Inference using a fitted clustering model."""
from pathlib import Path

from .cluster import load_model, predict_cluster
from .analyze_video import analyze


def predict(video_path: Path, model_path: Path) -> tuple[int, str]:
    """Return cluster id and reasoning for ``video_path``."""
    model = load_model(model_path)
    label = predict_cluster(video_path, model)
    _, reason = analyze(video_path)
    return label, reason


if __name__ == "__main__":
    result, detail = predict(Path("example.mp4"), Path("clusters.pkl"))
    print(f"Cluster: {result}\nReason: {detail}")

