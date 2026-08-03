"""Electrochemical sensor family. Not yet implemented.

Framework contract mirrors `opensmell.miris`: `normalize`, `features` and
`quality` will land here once electrochemical hardware support is added.
"""

from __future__ import annotations

from typing import Any


def normalize(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("opensmell.electrochemical is not implemented yet.")


def features(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("opensmell.electrochemical is not implemented yet.")


def quality(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("opensmell.electrochemical is not implemented yet.")
