"""Training script for Kickbike Readiness Analyzer."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_loader import load_video_frames
from .feature_extractor import compute_motion_vectors
from .model import ReadinessModel


def train(dataset_dir: Path, epochs: int = 10):

    """Train the readiness classifier with videos under ``dataset_dir``.

    A ``labels.csv`` file should map video file names to ``ok`` or ``ng``.
    """
    labels_file = dataset_dir / "labels.csv"
    label_map = {}
    if labels_file.exists():
        with labels_file.open() as f:
            for line in f:
                name, lab = line.strip().split(',')
                label_map[name] = 1.0 if lab.lower() == 'ok' else 0.0

=======

    videos = list(dataset_dir.glob('*.mp4'))
    features = []
    labels = []
    for video in videos:
        frames = list(load_video_frames(video))
        motion = compute_motion_vectors(frames)
        features.append(torch.tensor(motion, dtype=torch.float32))

        label_value = label_map.get(video.name, 0.0)
        labels.append(torch.tensor([label_value]))
=======
        # Dummy label placeholder
        labels.append(torch.tensor([0.0]))


    # Pad sequences to the same length for this example
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels_tensor = torch.stack(labels)
    dataset = TensorDataset(padded, labels_tensor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = ReadinessModel(input_size=1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            preds = model(batch_x.unsqueeze(-1))
            loss = criterion(preds.squeeze(), batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    dataset_path = Path('data')
    trained_model = train(dataset_path)
    torch.save(trained_model.state_dict(), 'model.pt')
