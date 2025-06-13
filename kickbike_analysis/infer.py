"""Inference script for Kickbike Readiness Analyzer."""
from pathlib import Path
import torch

from .data_loader import load_video_frames
from .feature_extractor import compute_motion_vectors
from .model import ReadinessModel



def predict(video_path: Path, model_weights: Path) -> str:
    """Return ``OK`` or ``NG`` for the given video."""
=======
def predict(video_path: Path, model_weights: Path) -> float:

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
    result = predict(Path('example.mp4'), Path('model.pt'))
    print(f"Prediction: {result}")
=======
        pred = model(features)
    return float(pred.item())

if __name__ == "__main__":
    readiness = predict(Path('example.mp4'), Path('model.pt'))
    print(f"Predicted readiness: {readiness:.2f}")

