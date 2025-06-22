"""Inference script for Kickbike Readiness Analyzer."""
from pathlib import Path
import torch

from .data_loader import load_video_frames
from .feature_extractor import compute_frame_features
from .analyze_video import analyze
from .model import ReadinessModel


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(video_path: Path, model_weights: Path) -> tuple[str, str]:
    """Return ``OK``/``NG`` and reasoning for the given video."""
    device = get_device()
    frames = list(load_video_frames(video_path))
    feats = compute_frame_features(frames)
    if feats.size == 0:
        return "NG", "動きが検出できませんでした"
    features = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    model = ReadinessModel(input_size=feats.shape[-1]).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    with torch.no_grad():
        prob = model(features).item()
    judge = "OK" if prob >= 0.5 else "NG"
    _, reason = analyze(video_path)
    return judge, reason


if __name__ == "__main__":
    result, detail = predict(Path("example.mp4"), Path("model.pt"))
    print(f"Prediction: {result}\nReason: {detail}")

