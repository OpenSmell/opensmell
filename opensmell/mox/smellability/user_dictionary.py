"""User dictionary of live-resolved chemicals.

Mirrors `osmograph-web/lib/smellability/user-dictionary.ts` 1:1, replacing
localStorage with a JSON file on disk. Search and resolution consult it, but
entries are always visibly marked provisional (estimated) rather than curated —
the user's own growing lab notebook.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .types import Chemical

DICT_KEY = "osmell-user-dictionary"
DICT_MAX = 200

_dict_path: Optional[str] = None


def set_dictionary_path(path: Optional[str]) -> None:
    global _dict_path
    _dict_path = path


def _dictionary_file() -> str:
    if _dict_path:
        return _dict_path
    env = os.environ.get("OSMELL_DATA_DIR")
    if env:
        return os.path.join(env, "user-dictionary.json")
    return os.path.join(os.path.expanduser("~"), ".local", "share", "opensmell", "user-dictionary.json")


def read_user_dictionary() -> List[Chemical]:
    try:
        with open(_dictionary_file(), "r") as fh:
            parsed = json.load(fh)
        return [Chemical.from_dict(d) for d in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []


def save_to_user_dictionary(chemical: Chemical) -> bool:
    dictionary = read_user_dictionary()
    if any(c.id == chemical.id for c in dictionary):
        return False
    next_list = (dictionary + [chemical])[-DICT_MAX:]
    try:
        path = _dictionary_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump([c.to_dict() for c in next_list], fh)
        return True
    except Exception:
        return False


def remove_from_user_dictionary(chemical_id: str) -> None:
    dictionary = [c for c in read_user_dictionary() if c.id != chemical_id]
    try:
        path = _dictionary_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump([c.to_dict() for c in dictionary], fh)
    except Exception:
        # storage unavailable — ignore
        pass


def user_dictionary_by_id() -> Dict[str, Chemical]:
    return {c.id: c for c in read_user_dictionary()}
