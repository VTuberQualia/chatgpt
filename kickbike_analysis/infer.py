"""Inference script for Kickbike Readiness Analyzer."""
from pathlib import Path
import argparse
import torch

from .data_loader import load_video_frames
from .feature_extractor import compute_motion_vectors
from .model import ReadinessModel


def predict(video_path: Path, model_weights: Path) -> str:
    """Return ``OK`` or ``NG`` for the given video."""
    frames = list(load_video_frames(video_path))
    motion = compute_motion_vectors(frames)
    features = torch.tensor(motion, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    model = ReadinessModel(input_size=1)
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    with torch.no_grad():
        prob = model(features).item()
    return "OK" if prob >= 0.5 else "NG"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kickbike readiness inference")
    parser.add_argument("video", type=Path, help="video file to analyze")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model.pt"),
        help="path to trained model weights",
    )
    args = parser.parse_args()

    result = predict(args.video, args.weights)
    print(f"Prediction: {result}")
