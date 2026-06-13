import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.attn = nn.Linear(channels, 1, bias=False)
        nn.init.normal_(self.attn.weight, std=0.01)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        scores = self.attn(x_t)
        weights = F.softmax(scores, dim=1)
        return (weights * x_t).sum(dim=1)


class Encoder(nn.Module):
    def __init__(self, in_channels=6, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.pool = nn.MaxPool1d(2)
        self.attn_pool = AttentionPooling(256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        h = self.attn_pool(x)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)

    @classmethod
    def load(cls, version="v1"):
        root = Path(__file__).resolve().parent
        weights_path = root / "weights" / f"encoder_{version}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Encoder weights not found: {weights_path}")
        device = torch.device("cpu")
        model = cls().to(device)
        model.load_state_dict(
            torch.load(str(weights_path), map_location=device, weights_only=True),
            strict=False,
        )
        model.eval()
        return model

    @classmethod
    def load_auto(cls):
        """Load best available encoder (v2 > v1)."""
        root = Path(__file__).resolve().parent
        for v in ["v2", "v1"]:
            p = root / "weights" / f"encoder_{v}.pth"
            if p.exists():
                return cls.load(v)
        raise FileNotFoundError("No encoder weights found")

    @torch.no_grad()
    def encode(self, array: "np.ndarray"):
        import numpy as np
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        tensor = torch.tensor(array, dtype=torch.float32, device=next(self.parameters()).device)
        mu, _ = self.forward(tensor)
        return mu.cpu().numpy()
