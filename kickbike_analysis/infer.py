"""Inference using a fitted clustering model."""
from pathlib import Path

from .cluster import load_model, predict_cluster
from .analyze_video import analyze



def predict(video_path: Path, model_path: Path) -> list[tuple[int, str, str, str]]:
    """Return (cluster id, judge label, comment, image path) for each person."""
    model = load_model(model_path)
    labels = predict_cluster(video_path, model)
    analyses = analyze(video_path)
    results: list[tuple[int, str, str, str]] = []
    for idx, cid in enumerate(labels):
        if idx < len(analyses):
            label, comment, img = analyses[idx]
        else:
            label, comment, img = "不明", "", ""
        results.append((cid, label, comment, img))
    return results



if __name__ == "__main__":
    res = predict(Path("example.mp4"), Path("clusters.pkl"))
    for cid, label, comment, img in res:
        print(f"cluster {cid}: {label} ({comment}) -> {img}")

