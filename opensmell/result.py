from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SmellResult:
    substance: Optional[str] = None
    confidence: Optional[float] = None
    warning: Optional[str] = None
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: list = field(default_factory=list)
    n_windows: int = 0

    @property
    def chemoprint(self) -> np.ndarray:
        if self.features.size == 0:
            return np.zeros(29, dtype=np.float32)
        n = min(29, len(self.features))
        return self.features[:n].copy()
