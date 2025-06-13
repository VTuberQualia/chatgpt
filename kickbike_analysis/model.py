"""PyTorch model definition for readiness prediction.

Outputs the probability that the rider is ready (``OK`` class).
"""
import torch
from torch import nn


class ReadinessModel(nn.Module):
    """Simple LSTM-based classifier operating on motion features."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, sequence, features]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # use last time step
        logits = self.fc(out)
        return torch.sigmoid(logits)
