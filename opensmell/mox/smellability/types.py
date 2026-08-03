"""Smellability data model.

Mirrors `osmograph-web/lib/smellability/types.ts` 1:1. These records flow through
the 4-step feasibility chain (identity -> volatility -> headspace concentration ->
MOX redox check) and its ontology, search, and provisional layers. JSON
serialization uses camelCase keys so verdicts round-trip with the TypeScript
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, List, Literal, Optional, TypeVar

T = TypeVar("T")

DataSource = Literal["measured", "estimated", "unknown"]
CompositeKind = Literal["food", "beverage", "spice", "material", "product", "activity", "other"]
Verdict = Literal["green", "yellow", "red"]
VerdictConfidence = Literal["high", "medium", "low"]
SignalStrength = Literal["strong", "moderate", "weak", "none"]
ResponseSpeed = Literal["fast", "medium", "slow", "unknown"]
ResolvedEntityKind = Literal["chemical", "composite", "class"]
SignalBand = Literal["strong", "moderate", "weak", "marginal", "none"]


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


def _ser(v: Any) -> Any:
    if hasattr(v, "to_dict"):
        return v.to_dict()
    if isinstance(v, list):
        return [_ser(x) for x in v]
    return v


def _from_prop_or_none(d: Any, kind) -> Any:
    if d is None:
        return None
    return kind.from_dict(d)


@dataclass
class Property(Generic[T]):
    value: Optional[T]
    source: DataSource
    note: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"value": self.value, "source": self.source}
        if self.note is not None:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Property[Any]":
        return cls(value=d.get("value"), source=d.get("source", "unknown"), note=d.get("note"))


@dataclass
class AntoineCoeffs:
    a: float
    b: float
    c: float

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c}

    @classmethod
    def from_dict(cls, d: dict) -> "AntoineCoeffs":
        return cls(a=d["a"], b=d["b"], c=d["c"])


@dataclass
class ChemicalProperties:
    molecular_weight: Property[float]
    boiling_point: Property[float]
    vapor_pressure_25: Property[float]
    functional_groups: List[str] = field(default_factory=list)
    redox_active: bool = False
    non_redox: Optional[bool] = None
    gas: Optional[bool] = None
    odor_descriptor: Optional[str] = None
    antoine: Optional[AntoineCoeffs] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "molecularWeight": _ser(self.molecular_weight),
            "boilingPoint": _ser(self.boiling_point),
            "vaporPressure25": _ser(self.vapor_pressure_25),
            "functionalGroups": list(self.functional_groups),
            "redoxActive": self.redox_active,
        }
        if self.antoine is not None:
            d["antoine"] = self.antoine.to_dict()
        if self.non_redox is not None:
            d["nonRedox"] = self.non_redox
        if self.gas is not None:
            d["gas"] = self.gas
        if self.odor_descriptor is not None:
            d["odorDescriptor"] = self.odor_descriptor
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChemicalProperties":
        return cls(
            molecular_weight=Property.from_dict(d.get("molecularWeight") or {}),
            boiling_point=Property.from_dict(d.get("boilingPoint") or {}),
            vapor_pressure_25=Property.from_dict(d.get("vaporPressure25") or {}),
            functional_groups=list(d.get("functionalGroups") or []),
            redox_active=bool(d.get("redoxActive", False)),
            non_redox=d.get("nonRedox"),
            gas=d.get("gas"),
            odor_descriptor=d.get("odorDescriptor"),
            antoine=AntoineCoeffs.from_dict(d["antoine"]) if d.get("antoine") else None,
        )


@dataclass
class Chemical:
    id: str
    name: str
    synonyms: List[str]
    props: ChemicalProperties
    source_refs: List[str]
    cas: Optional[str] = None
    smiles: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "synonyms": list(self.synonyms),
            "props": self.props.to_dict(),
            "sourceRefs": list(self.source_refs),
        }
        if self.cas is not None:
            d["cas"] = self.cas
        if self.smiles is not None:
            d["smiles"] = self.smiles
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chemical":
        return cls(
            id=d["id"],
            name=d["name"],
            synonyms=list(d.get("synonyms") or []),
            props=ChemicalProperties.from_dict(d.get("props") or {}),
            source_refs=list(d.get("sourceRefs") or []),
            cas=d.get("cas"),
            smiles=d.get("smiles"),
        )


@dataclass
class CompositeConstituent:
    chemical_id: str
    weight_fraction: Property[float]

    def to_dict(self) -> dict:
        return {"chemicalId": self.chemical_id, "weightFraction": _ser(self.weight_fraction)}

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeConstituent":
        return cls(chemical_id=d.get("chemicalId", ""), weight_fraction=Property.from_dict(d.get("weightFraction") or {}))


@dataclass
class Composite:
    id: str
    name: str
    kind: CompositeKind
    synonyms: List[str]
    constituents: List[CompositeConstituent]
    source_refs: List[str]
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "synonyms": list(self.synonyms),
            "constituents": _ser(self.constituents),
            "sourceRefs": list(self.source_refs),
        }
        if self.notes is not None:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Composite":
        return cls(
            id=d["id"],
            name=d["name"],
            kind=d.get("kind", "other"),
            synonyms=list(d.get("synonyms") or []),
            constituents=[CompositeConstituent.from_dict(x) for x in (d.get("constituents") or [])],
            source_refs=list(d.get("sourceRefs") or []),
            notes=d.get("notes"),
        )


@dataclass
class ChainValue:
    label: str
    value: str
    source: DataSource

    def to_dict(self) -> dict:
        return {"label": self.label, "value": self.value, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "ChainValue":
        return cls(label=d["label"], value=d["value"], source=d.get("source", "unknown"))


@dataclass
class ChainStep:
    id: str
    label: str
    verdict: Verdict
    reason: str
    detail: str
    values: List[ChainValue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
            "values": _ser(self.values),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChainStep":
        return cls(
            id=d["id"],
            label=d["label"],
            verdict=d.get("verdict", "yellow"),
            reason=d["reason"],
            detail=d["detail"],
            values=[ChainValue.from_dict(x) for x in (d.get("values") or [])],
        )


@dataclass
class ConstituentVerdict:
    chemical_id: str
    name: str
    weight_fraction: float
    weight_source: DataSource
    steps: List[ChainStep]
    verdict: Verdict
    signal_strength: SignalStrength
    response_speed: ResponseSpeed
    signal_score: float

    def to_dict(self) -> dict:
        return {
            "chemicalId": self.chemical_id,
            "name": self.name,
            "weightFraction": self.weight_fraction,
            "weightSource": self.weight_source,
            "steps": _ser(self.steps),
            "verdict": self.verdict,
            "signalStrength": self.signal_strength,
            "responseSpeed": self.response_speed,
            "signalScore": self.signal_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConstituentVerdict":
        return cls(
            chemical_id=d.get("chemicalId", ""),
            name=d.get("name", ""),
            weight_fraction=d.get("weightFraction", 0.0),
            weight_source=d.get("weightSource", "unknown"),
            steps=[ChainStep.from_dict(x) for x in (d.get("steps") or [])],
            verdict=d.get("verdict", "yellow"),
            signal_strength=d.get("signalStrength", "none"),
            response_speed=d.get("responseSpeed", "unknown"),
            signal_score=d.get("signalScore", 0.0),
        )


@dataclass
class CrossCheck:
    sensor_count: int
    max_distinguishable: int
    library_substances: List[str]
    confusable: List[str]
    note: str

    def to_dict(self) -> dict:
        return {
            "sensorCount": self.sensor_count,
            "maxDistinguishable": self.max_distinguishable,
            "librarySubstances": list(self.library_substances),
            "confusable": list(self.confusable),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CrossCheck":
        return cls(
            sensor_count=d.get("sensorCount", 0),
            max_distinguishable=d.get("maxDistinguishable", 0),
            library_substances=list(d.get("librarySubstances") or []),
            confusable=list(d.get("confusable") or []),
            note=d.get("note", ""),
        )


@dataclass
class ResolvedEntity:
    kind: ResolvedEntityKind
    id: str
    name: str
    display_name: str
    match_hint: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "id": self.id,
            "name": self.name,
            "displayName": self.display_name,
            "matchHint": self.match_hint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResolvedEntity":
        return cls(
            kind=d.get("kind", "chemical"),
            id=d["id"],
            name=d["name"],
            display_name=d.get("displayName", d.get("name", "")),
            match_hint=d.get("matchHint", ""),
        )


@dataclass
class FeasibilityVerdict:
    entity_id: str
    entity_name: str
    kind: ResolvedEntityKind
    verdict: Verdict
    confidence: VerdictConfidence
    signal_strength: SignalStrength
    response_speed: ResponseSpeed
    constituents: List[ConstituentVerdict]
    steps: List[ChainStep]
    exposure_guidance: str
    dilution_guidance: str
    computed_at: str
    sensor_count: int
    notes: List[str]
    cross_check: Optional[CrossCheck] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "entityId": self.entity_id,
            "entityName": self.entity_name,
            "kind": self.kind,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "signalStrength": self.signal_strength,
            "responseSpeed": self.response_speed,
            "constituents": _ser(self.constituents),
            "steps": _ser(self.steps),
            "exposureGuidance": self.exposure_guidance,
            "dilutionGuidance": self.dilution_guidance,
            "computedAt": self.computed_at,
            "sensorCount": self.sensor_count,
            "notes": list(self.notes),
        }
        if self.cross_check is not None:
            d["crossCheck"] = self.cross_check.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeasibilityVerdict":
        return cls(
            entity_id=d.get("entityId", ""),
            entity_name=d.get("entityName", ""),
            kind=d.get("kind", "chemical"),
            verdict=d.get("verdict", "yellow"),
            confidence=d.get("confidence", "low"),
            signal_strength=d.get("signalStrength", "none"),
            response_speed=d.get("responseSpeed", "unknown"),
            constituents=[ConstituentVerdict.from_dict(x) for x in (d.get("constituents") or [])],
            steps=[ChainStep.from_dict(x) for x in (d.get("steps") or [])],
            exposure_guidance=d.get("exposureGuidance", ""),
            dilution_guidance=d.get("dilutionGuidance", ""),
            computed_at=d.get("computedAt", ""),
            sensor_count=d.get("sensorCount", 6),
            notes=list(d.get("notes") or []),
            cross_check=CrossCheck.from_dict(d["crossCheck"]) if d.get("crossCheck") else None,
        )


@dataclass
class SearchCandidate(ResolvedEntity):
    score: float = 0.0

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["score"] = self.score
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SearchCandidate":
        return cls(
            kind=d.get("kind", "chemical"),
            id=d["id"],
            name=d["name"],
            display_name=d.get("displayName", d.get("name", "")),
            match_hint=d.get("matchHint", ""),
            score=d.get("score", 0.0),
        )
