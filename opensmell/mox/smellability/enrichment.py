"""Live PubChem enrichment (provisional chemicals).

Mirrors `osmograph-web/lib/smellability/enrichment.ts` 1:1, replacing
localStorage with a JSON file cache and `fetch` with `urllib`. Network calls are
synchronous (the desktop port has no browser event loop). Throttling, cache
sizing, and the boiling-point extraction logic are identical.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

PUB_PROPERTIES = "MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES"
CACHE_MAX = 200
MIN_INTERVAL_MS = 300
BP_TIMEOUT_MS = 8000

_cache_path: Optional[str] = None
_last_request_at = 0.0


@dataclass
class EnrichedChemical:
    name: str
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    iupac_name: Optional[str] = None
    smiles: Optional[str] = None
    source: str = "pubchem"
    fetched_at: str = ""

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "source": self.source, "fetchedAt": self.fetched_at}
        for key, attr in (
            ("molecularFormula", "molecular_formula"),
            ("molecularWeight", "molecular_weight"),
            ("iupacName", "iupac_name"),
            ("smiles", "smiles"),
        ):
            v = getattr(self, attr)
            if v is not None:
                d[key] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EnrichedChemical":
        return cls(
            name=d.get("name", ""),
            molecular_formula=d.get("molecularFormula"),
            molecular_weight=d.get("molecularWeight"),
            iupac_name=d.get("iupacName"),
            smiles=d.get("smiles"),
            source=d.get("source", "pubchem"),
            fetched_at=d.get("fetchedAt", ""),
        )


@dataclass
class EnrichedBoilingPoint:
    value_c: float
    source: str = "measured"
    note: str = "PubChem experimental property"


def set_cache_path(path: Optional[str]) -> None:
    global _cache_path
    _cache_path = path


def _data_dir() -> str:
    env = os.environ.get("OSMELL_DATA_DIR")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".local", "share", "opensmell")


def _cache_file() -> str:
    return _cache_path if _cache_path else os.path.join(_data_dir(), "pubchem-cache.json")


def _read_cache() -> Dict[str, EnrichedChemical]:
    try:
        with open(_cache_file(), "r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return {k: EnrichedChemical.from_dict(v) for k, v in data.items()}
    except Exception:
        return {}


def _write_cache(cache: Dict[str, EnrichedChemical]) -> None:
    try:
        path = _cache_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({k: v.to_dict() for k, v in cache.items()}, fh)
    except Exception:
        # storage full or unavailable — fail silently
        pass


def _throttle() -> None:
    global _last_request_at
    now = time.monotonic() * 1000
    wait = max(0.0, _last_request_at + MIN_INTERVAL_MS - now)
    if wait > 0:
        _last_request_at = now + wait
        time.sleep(wait / 1000)
    else:
        _last_request_at = now


def _fetch_json(url: str) -> Optional[dict]:
    with urllib.request.urlopen(url, timeout=BP_TIMEOUT_MS) as resp:
        data = json.load(resp)
    return data if isinstance(data, dict) else None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def lookup_pubchem(query: str) -> Optional[EnrichedChemical]:
    q = query.strip()
    if len(q) < 2:
        return None

    cache = _read_cache()
    key = q.lower()
    if key in cache:
        return cache[key]

    _throttle()

    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            + urllib.parse.quote(q)
            + "/property/"
            + PUB_PROPERTIES
            + "/JSON"
        )
        data = _fetch_json(url)
        props = ((data or {}).get("PropertyTable") or {}).get("Properties") or []
        p = props[0] if props else None
        if p is None:
            return None

        result = EnrichedChemical(
            name=p.get("IUPACName") or q,
            molecular_formula=p.get("MolecularFormula"),
            molecular_weight=_as_float(p.get("MolecularWeight")),
            iupac_name=p.get("IUPACName"),
            smiles=p.get("SMILES") or p.get("IsomericSMILES"),
            source="pubchem",
            fetched_at=_utc_now_iso(),
        )

        next_cache = dict(cache)
        next_cache[key] = result
        keys = list(next_cache.keys())
        if len(keys) > CACHE_MAX:
            for k in keys[: len(keys) - CACHE_MAX]:
                del next_cache[k]
        _write_cache(next_cache)
        return result
    except Exception:
        return None


def parse_boiling_point(raw: Any) -> Optional[float]:
    if not isinstance(raw, str):
        return None
    # pug_view often reports "281.6±35.0 °C" — the value before the ± range is
    # the reported boiling point, not the uncertainty.
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:±[^°\d]*\d+(?:\.\d+)?)?\s*°?\s*C", raw, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _extract_values(node: Any) -> List[float]:
    out: List[float] = []

    def walk(n: Any) -> None:
        if isinstance(n, list):
            for item in n:
                walk(item)
            return
        if isinstance(n, dict):
            obj = n
            value = obj.get("Value")
            if isinstance(value, dict):
                v = value
                swm = v.get("StringWithMarkup")
                if isinstance(swm, list):
                    for el in swm:
                        parsed = parse_boiling_point((el or {}).get("String"))
                        if parsed is not None:
                            out.append(parsed)
                    return
                num = v.get("Number")
                if isinstance(num, list):
                    for el in num:
                        parsed = el if isinstance(el, (int, float)) else parse_boiling_point(str(el))
                        if parsed is not None:
                            out.append(parsed)
                    return
                if isinstance(v.get("String"), str):
                    parsed = parse_boiling_point(v.get("String"))
                    if parsed is not None:
                        out.append(parsed)
                    return
            if isinstance(obj.get("StringWithMarkup"), str):
                parsed = parse_boiling_point(obj.get("StringWithMarkup"))
                if parsed is not None:
                    out.append(parsed)
                return
            for key in obj:
                walk(obj[key])

    walk(node)
    return out


def extract_boiling_point_c(sections: Union[List[Any], dict, None]) -> Optional[float]:
    hits: List[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            obj = node
            if obj.get("TOCHeading") == "Boiling Point":
                hits.extend(_extract_values(obj))
            for key in obj:
                if key == "TOCHeading":
                    continue
                walk(obj[key])

    walk(sections)

    # Prefer the first numeric °C value; several records list duplicates of the same number.
    for h in hits:
        if h is not None and h > -200 and h < 600:
            return h
    return None


def lookup_pubchem_boiling_point(query: str) -> Optional[EnrichedBoilingPoint]:
    q = query.strip()
    if len(q) < 2:
        return None

    # pug_view requires a CID, not a name — resolve it via the fast property endpoint first.
    prop_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + urllib.parse.quote(q)
        + "/property/CanonicalSMILES/JSON"
    )
    _throttle()
    cid: Optional[str] = None
    try:
        data = _fetch_json(prop_url)
        props = ((data or {}).get("PropertyTable") or {}).get("Properties") or []
        p = props[0] if props else None
        cid = str(p["CID"]) if p and p.get("CID") is not None else None
    except Exception:
        return None
    if not cid:
        return None

    _throttle()
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Boiling%20Point"
        data = _fetch_json(url)
        if not data or data.get("Fault") or not data.get("Record"):
            return None
        value_c = extract_boiling_point_c((data.get("Record") or {}).get("Section"))
        if value_c is None:
            return None
        return EnrichedBoilingPoint(value_c=value_c, source="measured", note="PubChem experimental property")
    except Exception:
        return None
