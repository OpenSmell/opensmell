from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SmellResult:
    substance: str
    confidence: float
    warning: Optional[str]
    should_contribute: bool
    contribution_url: str
    chemoprint: np.ndarray
    latent: np.ndarray
