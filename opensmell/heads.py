from pathlib import Path

import torch
import torch.nn as nn


class ChemoprintHead(nn.Module):
    def __init__(self, latent_dim=256, chemo_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, chemo_dim)
        )

    def forward(self, z):
        return self.net(z)

    @classmethod
    def load(cls, version="v1"):
        root = Path(__file__).resolve().parent
        weights_path = root / "weights" / f"chemoprint_head_{version}.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"ChemoprintHead weights not found: {weights_path}")
        device = torch.device("cpu")
        model = cls().to(device)
        model.load_state_dict(
            torch.load(str(weights_path), map_location=device, weights_only=True)
        )
        model.eval()
        return model
