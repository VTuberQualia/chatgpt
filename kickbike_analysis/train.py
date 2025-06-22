"""Training script for Kickbike Readiness Analyzer."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import load_video_frames
from .feature_extractor import compute_frame_features
from .model import ReadinessModel


def train(dataset_dir: Path, epochs: int = 10):
    """Train the readiness classifier with videos under ``dataset_dir``.

    A ``labels.csv`` file should map video file names to ``ok`` or ``ng``.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"{dataset_dir} が見つかりません")

    labels_file = dataset_dir / "labels.csv"
    label_map = {}
    if labels_file.exists():
        with labels_file.open() as f:
            for line in f:
                name, lab = line.strip().split(",")
                label_map[name] = 1.0 if lab.lower() == "ok" else 0.0

    videos = list(dataset_dir.glob("*.mp4"))
    features = []
    labels = []
    for video in videos:
        feature_file = video.with_suffix(".npy")
        if feature_file.exists():
            motion = np.load(feature_file)
        else:
            frames = list(load_video_frames(video))
            motion = compute_frame_features(frames)
        if motion.size == 0:
            continue
        features.append(torch.tensor(motion, dtype=torch.float32))
        label_value = label_map.get(video.name, 0.0)
        labels.append(torch.tensor([label_value]))

    if not features:
        raise RuntimeError(
            f"{dataset_dir} に利用可能な動画が見つかりません。mp4ファイルが存在し、十分な動きがあるか確認してください。"
        )

    # Pad sequences to the same length for this example
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels_tensor = torch.stack(labels)
    dataset = TensorDataset(padded, labels_tensor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = ReadinessModel(input_size=padded.shape[-1])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            preds = model(batch_x)
            loss = criterion(preds.squeeze(), batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


if __name__ == "__main__":
    import sys
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    trained_model = train(dataset_path)
    torch.save(trained_model.state_dict(), "model.pt")
