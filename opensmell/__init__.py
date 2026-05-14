import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from .encoder import Encoder
from .heads import ChemoprintHead
from .preprocessing import export_for_contribution as _export
from .preprocessing import load_csv, segment_and_normalize
from .result import SmellResult

_ROOT = Path(__file__).resolve().parent

_encoder: Optional[Encoder] = None
_chemoprint_head: Optional[ChemoprintHead] = None
_prototypes: Optional[np.ndarray] = None
_prototype_labels: Optional[list] = None


def _lazy_init():
    global _encoder, _chemoprint_head, _prototypes, _prototype_labels
    if _encoder is None:
        _encoder = Encoder.load("v1")
        _chemoprint_head = ChemoprintHead.load("v1")
        _prototypes = np.load(str(_ROOT / "data" / "prototypes.npy"))
        with open(str(_ROOT / "data" / "prototype_labels.json")) as f:
            _prototype_labels = json.load(f)


def process(filepath: str) -> SmellResult:
    _lazy_init()
    raw = load_csv(filepath)
    segments = segment_and_normalize(raw)
    latents = _encoder.encode(segments)
    latent = latents.mean(axis=0)
    latent_t = torch.tensor(latent[np.newaxis, :], dtype=torch.float32)
    chemo = _chemoprint_head(latent_t).detach().numpy().flatten()
    return _build_result(latent, chemo)


def process_array(array: np.ndarray) -> SmellResult:
    _lazy_init()
    if array.ndim == 2:
        segments = segment_and_normalize(array)
    else:
        segments = segment_and_normalize(array[0])
    latents = _encoder.encode(segments)
    latent = latents.mean(axis=0)
    latent_t = torch.tensor(latent[np.newaxis, :], dtype=torch.float32)
    chemo = _chemoprint_head(latent_t).detach().numpy().flatten()
    return _build_result(latent, chemo)


def _build_result(latent: np.ndarray, chemoprint: np.ndarray) -> SmellResult:
    sims = cosine_similarity(latent.reshape(1, -1), _prototypes)[0]
    best_idx = int(np.argmax(sims))
    confidence = float(sims[best_idx])
    substance = str(_prototype_labels[best_idx])
    warning = None
    should_contribute = False
    if confidence < 0.7:
        warning = (
            "This smell is not well-represented in the OpenSmell dataset. "
            "Your recording may be from a substance not yet catalogued. "
            "Consider contributing it at opensmell.org/contribute"
        )
        should_contribute = True
    return SmellResult(
        substance=substance,
        confidence=confidence,
        warning=warning,
        should_contribute=should_contribute,
        contribution_url="https://opensmell.onrender.com/contribute",
        chemoprint=chemoprint,
        latent=latent,
    )


export_for_contribution = _export

__all__ = [
    "process",
    "process_array",
    "export_for_contribution",
    "Encoder",
    "ChemoprintHead",
]
