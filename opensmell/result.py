from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SmellResult:
    substance: Optional[str] = None
    confidence: Optional[float] = None
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: list = field(default_factory=list)
    n_windows: int = 0
