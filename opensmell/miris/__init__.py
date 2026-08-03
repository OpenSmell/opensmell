"""MIRIS sensor family (spectroscopic). Not yet implemented.

Framework contract: `opensmell.miris` will provide `normalize` (baseline
subtraction for IR), `features` (spectral peaks/intensities) and `smellability`
(IR absorption-band feasibility) once the hardware/algorithm lands.
"""

from __future__ import annotations

from typing import Any


def normalize(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("opensmell.miris is not implemented yet.")


def features(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("opensmell.miris is not implemented yet.")


def quality(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("opensmell.miris is not implemented yet.")
