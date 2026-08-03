"""The 4-step feasibility chain.

Mirrors `osmograph-web/lib/smellability/chain.ts` 1:1:

    1. identity      — which substance, from the curated dictionary
    2. volatility    — vapor pressure at 25 °C (Antoine or Clausius-Clapeyron)
    3. signal        — saturated headspace vs the ~1 ppm MOX floor
    4. reactivity    — redox activity at the ~350 °C sensor surface

A constituent verdict is the worst of its steps; a composite verdict is a
weighted aggregation of its constituents; a class verdict is a coarse
class-level estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

from .compounds import COMPOUND_BY_ID, REFERENCE_COMPOUND
from .composites import COMPOSITE_BY_ID
from .constants import (
    AMBIENT_TEMP_C,
    AMBIENT_TEMP_K,
    CLASS_TERMS,
    DEFAULT_SENSOR_COUNT,
    MAX_SUBSTANCES,
    MOX_FLOOR_PPM,
    headspace_ppm_band,
    volatility_label,
)
from .transport import (
    IncidentFluxInput,
    delta_h_vap_trouton,
    diffusion_volume_from_mw,
    signal_ratio_vs_ref,
    vapor_pressure_antoine,
    vapor_pressure_clausius_clapeyron,
)
from .types import (
    ChainStep,
    ChainValue,
    Chemical,
    Composite,
    ConstituentVerdict,
    CrossCheck,
    DataSource,
    FeasibilityVerdict,
    ResponseSpeed,
    SignalStrength,
    Verdict,
    VerdictConfidence,
)
from .user_dictionary import user_dictionary_by_id

WORST = {"green": 0, "yellow": 1, "red": 2}


@dataclass
class ChainOptions:
    sensor_count: Optional[int] = None
    library_substances: Optional[List[str]] = None
    temp_c: Optional[float] = None


class EffectiveVaporPressure(NamedTuple):
    pa: float
    source: DataSource


class Guidance(NamedTuple):
    exposure: str
    dilution: str


def worst_verdict(a: Verdict, b: Verdict) -> Verdict:
    return a if WORST[a] >= WORST[b] else b


def signal_score(strength: SignalStrength) -> float:
    return {
        "strong": 1.0,
        "moderate": 0.6,
        "weak": 0.3,
        "none": 0.0,
    }[strength]


def speed_from_volatility(pa: Optional[float], gas: bool) -> ResponseSpeed:
    if gas or (pa is not None and pa >= 1000):
        return "fast"
    if pa is not None and pa >= 100:
        return "medium"
    if pa is not None and pa >= 1:
        return "slow"
    return "unknown"


def effective_vapor_pressure(c: Chemical) -> EffectiveVaporPressure:
    if c.props.vapor_pressure_25.value is not None:
        return EffectiveVaporPressure(c.props.vapor_pressure_25.value, c.props.vapor_pressure_25.source)
    if c.props.antoine is not None:
        a, b, cc = c.props.antoine.a, c.props.antoine.b, c.props.antoine.c
        return EffectiveVaporPressure(vapor_pressure_antoine(a, b, cc, AMBIENT_TEMP_C), "measured")
    if c.props.gas:
        return EffectiveVaporPressure(101325, "measured")
    if c.props.boiling_point.value is not None:
        t_boil_k = c.props.boiling_point.value + 273.15
        pa = vapor_pressure_clausius_clapeyron(AMBIENT_TEMP_K, t_boil_k, delta_h_vap_trouton(t_boil_k))
        return EffectiveVaporPressure(pa, "estimated")
    return EffectiveVaporPressure(0, "unknown")


def _signal_ratio(c: Chemical) -> tuple[float, DataSource]:
    vp = effective_vapor_pressure(c)
    if vp.source == "unknown" or vp.pa <= 0 or REFERENCE_COMPOUND is None:
        return (0.0, vp.source)
    rvp = effective_vapor_pressure(REFERENCE_COMPOUND)

    def _input(chem: Chemical, pv: float) -> IncidentFluxInput:
        mw = chem.props.molecular_weight.value
        return IncidentFluxInput(
            vapor_pressure_pa=pv,
            mol_weight_kg=(mw / 1000) if mw else 0.05,
            diffusion_volume_cm3=diffusion_volume_from_mw(mw) if mw else 55,
        )

    ratio = signal_ratio_vs_ref(_input(c, vp.pa), _input(REFERENCE_COMPOUND, rvp.pa))
    source: DataSource = (
        "measured" if vp.source == "measured" and rvp.source == "measured" else "estimated"
    )
    return (ratio, source)


def _headspace_ppm(c: Chemical) -> tuple[Optional[float], bool, DataSource]:
    vp = effective_vapor_pressure(c)
    if c.props.gas:
        return (None, True, "measured")
    if vp.source == "unknown" or vp.pa <= 0:
        return (None, False, "unknown")
    return ((vp.pa / 101325) * 1e6, False, vp.source)


def _fmt_pa(pa: Optional[float]) -> str:
    if pa is None:
        return "unknown"
    if pa >= 100000:
        return f"{pa / 1000:.0f} kPa"
    if pa >= 1000:
        return f"{pa / 1000:.2f} kPa"
    return f"{pa:.0f} Pa"


def _fmt_ratio(ratio: float) -> str:
    if ratio >= 10:
        return f"{ratio:.1f}× ethanol"
    if ratio >= 1:
        return f"{ratio:.2f}× ethanol"
    if ratio >= 0.1:
        return f"{ratio * 100:.0f}% of ethanol"
    return f"{ratio * 100:.1f}% of ethanol"


def _fmt_ppm(ppm: Optional[float]) -> str:
    if ppm is None:
        return "unknown"
    if ppm >= 10000:
        return f"{ppm / 1000:.0f}k"
    if ppm >= 100:
        return f"{round(ppm)}"
    return f"{ppm:.1f}"


def run_constituent_chain(c: Chemical) -> ConstituentVerdict:
    vp = effective_vapor_pressure(c)

    steps: List[ChainStep] = []

    mw = c.props.molecular_weight.value
    bp = c.props.boiling_point.value
    odour = f" Odour: {c.props.odor_descriptor}." if c.props.odor_descriptor else ""
    steps.append(
        ChainStep(
            id="identity",
            label="Identity & properties",
            verdict="green",
            reason=f"{c.name} resolved from the compound dictionary.",
            detail=(
                f"{c.name}{f' (CAS {c.cas})' if c.cas else ''}. "
                f"Molecular weight {f'{mw:.1f} g/mol' if mw is not None else 'unknown'}, "
                f"boiling point {f'{bp:.1f} °C' if bp is not None else 'unknown'}.{odour}"
            ),
            values=[
                ChainValue(
                    label="Molecular weight",
                    value=f"{mw:.1f} g/mol" if mw is not None else "unknown",
                    source=c.props.molecular_weight.source,
                ),
                ChainValue(
                    label="Boiling point",
                    value=f"{bp:.1f} °C" if bp is not None else "unknown",
                    source=c.props.boiling_point.source,
                ),
                ChainValue(label="Vapor pressure @ 25 °C", value=_fmt_pa(vp.pa), source=vp.source),
            ],
        )
    )

    vol_label = volatility_label(None if vp.source == "unknown" else vp.pa)
    vol_verdict: Verdict = "yellow"
    vol_reason = "Vapor pressure unknown — volatility cannot be assessed."
    if c.props.gas:
        vol_verdict = "green"
        vol_reason = f"{c.name} is a gas at room temperature — it is already in the vapor phase."
    elif vp.source != "unknown":
        if vol_label in ("very high", "high", "moderate"):
            vol_verdict = "green"
            vol_reason = f"{c.name} has {vol_label} volatility ({_fmt_pa(vp.pa)} at 25 °C) — it readily enters the headspace."
        elif vol_label == "low":
            vol_verdict = "yellow"
            vol_reason = f"{c.name} has low volatility ({_fmt_pa(vp.pa)} at 25 °C) — expect a slow, weak headspace unless the sample is warmed."
        else:
            vol_verdict = "red"
            vol_reason = f"{c.name} is effectively non-volatile at room temperature ({_fmt_pa(vp.pa)}) — it will not reach the sensor without heating."
    steps.append(
        ChainStep(
            id="volatility",
            label="Volatility",
            verdict=vol_verdict,
            reason=vol_reason,
            detail=(
                "Vapor pressure at 25 °C via Antoine equation where constants are curated, "
                "else Clausius-Clapeyron from the boiling point with Trouton's-rule enthalpy."
            ),
            values=[
                ChainValue(
                    label="Volatility class",
                    value="gas" if c.props.gas else vol_label,
                    source=vp.source,
                )
            ],
        )
    )

    head = _headspace_ppm(c)
    ratio_info = _signal_ratio(c)
    hs_band = "strong" if head[1] else (headspace_ppm_band(head[0]) if head[0] is not None else "unknown")
    sig_verdict: Verdict = "yellow"
    sig_reason = "Headspace concentration unknown — signal strength cannot be assessed."
    signal_strength: SignalStrength = "none"
    if hs_band != "unknown":
        if hs_band in ("strong", "moderate"):
            sig_verdict = "green"
            signal_strength = hs_band  # type: ignore[assignment]
            sig_reason = (
                f"{c.name} is a gas — the vapor phase is available at full concentration, "
                f"well above the ~{MOX_FLOOR_PPM} ppm MOX floor."
                if head[1]
                else f"Saturated headspace is ≈ {_fmt_ppm(head[0])} ppm — far above the ~{MOX_FLOOR_PPM} ppm MOX floor."
            )
        elif hs_band == "weak":
            sig_verdict = "yellow"
            signal_strength = "weak"
            sig_reason = (
                f"Saturated headspace is ≈ {_fmt_ppm(head[0])} ppm — detectable, but only "
                f"{max(1, round((head[0] or 0) / MOX_FLOOR_PPM))}× the MOX floor. Warm the sample and maximize surface area."
            )
        else:
            sig_verdict = "red"
            signal_strength = "weak" if hs_band == "marginal" else "none"
            sig_reason = (
                f"Saturated headspace is ≈ {_fmt_ppm(head[0])} ppm — within "
                f"{max(1, round((head[0] or 0) / MOX_FLOOR_PPM))}× of the ~{MOX_FLOOR_PPM} ppm floor "
                f"and unlikely to give a usable response."
            )
    steps.append(
        ChainStep(
            id="signal",
            label="Headspace concentration",
            verdict=sig_verdict,
            reason=sig_reason,
            detail=(
                "Saturated headspace is the mole fraction of the compound at its vapor pressure "
                "(p_vap / P_atm). It is the physical upper bound in an enclosed chamber and is "
                "compared against the practical MOX detection floor."
            ),
            values=[
                ChainValue(
                    label="Saturated headspace",
                    value=(
                        "full vapor phase (gas)"
                        if head[1]
                        else (f"{_fmt_ppm(head[0])} ppm" if head[0] is not None else "unknown")
                    ),
                    source=head[2],
                ),
                ChainValue(
                    label="Relative to ethanol",
                    value="unknown" if ratio_info[1] == "unknown" else _fmt_ratio(ratio_info[0]),
                    source=ratio_info[1],
                ),
            ],
        )
    )

    groups = ", ".join(c.props.functional_groups) if c.props.functional_groups else "no recognized functional groups"
    react_verdict: Verdict = "yellow"
    react_reason = f"Reactivity of {c.name} on MOX surfaces is not classified."
    if c.props.non_redox:
        react_verdict = "red"
        react_reason = f"{c.name} is not redox-active at MOX operating temperatures — it will not produce the surface reduction MOX sensors detect."
    elif c.props.redox_active:
        react_verdict = "green"
        react_reason = f"Contains {groups}; these are oxidized at the ~350 °C sensor surface, producing the resistance change MOX arrays detect."
    else:
        react_reason = f"{c.name} is not a reducing gas; any response is indirect (e.g. humidity/{'oxygen partial pressure' if c.name == 'oxygen' else 'matrix effects'})."
        if c.id == "water":
            react_reason = "Water is not a reducing VOC, but humidity strongly modulates MOX baseline resistance — expect a baseline shift rather than an analyte response."
    steps.append(
        ChainStep(
            id="reactivity",
            label="MOX reactivity",
            verdict=react_verdict,
            reason=react_reason,
            detail=(
                "MOX sensors respond to gases that undergo surface redox at operating temperature. "
                "Functional-group chemistry determines this; see the MOX boundaries in the science docs."
            ),
            values=[
                ChainValue(
                    label="Functional groups",
                    value=groups,
                    source="measured" if c.props.functional_groups else "estimated",
                )
            ],
        )
    )

    verdict: Verdict = "green"
    for s in steps:
        verdict = worst_verdict(verdict, s.verdict)

    speed = speed_from_volatility(None if vp.source == "unknown" else vp.pa, bool(c.props.gas))

    return ConstituentVerdict(
        chemical_id=c.id,
        name=c.name,
        weight_fraction=1,
        weight_source="measured",
        steps=steps,
        verdict=verdict,
        signal_strength=signal_strength,
        response_speed=speed,
        signal_score=signal_score(signal_strength),
    )


def _confidence_of(v: List[ConstituentVerdict]) -> VerdictConfidence:
    sources = [x.source for c in v for s in c.steps for x in s.values]
    if any(s == "unknown" for s in sources):
        return "low"
    if any(s == "estimated" for s in sources):
        return "medium"
    return "high"


def _build_cross_check(
    sensor_count: int,
    library: List[str],
    name: str,
    synonyms: List[str],
) -> CrossCheck:
    max_distinguishable = MAX_SUBSTANCES.get(sensor_count, 40)
    lower_name = name.lower()
    lower_syns = [s.lower() for s in synonyms]

    def _confusable(label: str) -> bool:
        l = label.lower()
        return l == lower_name or l in lower_syns or l in lower_name or lower_name in l

    confusable = [label for label in library if _confusable(label)]
    if not library:
        note = (
            f"At {sensor_count} sensors the array is rated to resolve roughly {max_distinguishable} "
            "distinct substances. Cross-sensitivity to your library is unknown until you add labeled sessions."
        )
    elif confusable:
        joined = '", "'.join(confusable)
        note = (
            f"At {sensor_count} sensors the array is rated to resolve roughly {max_distinguishable} "
            f'distinct substances. "{joined}" in your library may overlap with this substance\'s '
            "response — verify with a labeled exposure."
        )
    else:
        note = (
            f"At {sensor_count} sensors the array is rated to resolve roughly {max_distinguishable} "
            "distinct substances. No exact label overlap found in your library."
        )
    return CrossCheck(
        sensor_count=sensor_count,
        max_distinguishable=max_distinguishable,
        library_substances=library,
        confusable=confusable,
        note=note,
    )


def guidance(signal: SignalStrength, speed: ResponseSpeed) -> Guidance:
    base = "Capture a 30-60 s clean-air baseline first; record the exposure, then a recovery window."
    if signal == "strong" and speed == "fast":
        return Guidance(
            exposure=(
                f"{base} Signal is expected fast and strong — keep exposures short (10-30 s) and use "
                "an enclosed chamber or gentle airflow for repeatability."
            ),
            dilution="Start diluted (≈1:10 in clean air) and reduce dilution only if the response is small.",
        )
    if signal == "strong":
        return Guidance(
            exposure=(
                f"{base} Strong signal expected — an enclosed chamber and moderate exposure (20-40 s) "
                "will keep you out of saturation."
            ),
            dilution="A mild dilution (≈1:5) helps stay in the linear response region.",
        )
    if signal == "moderate":
        return Guidance(
            exposure=(
                f"{base} Moderately detectable — allow 30-60 s of exposure; a small chamber or gentle "
                "airflow improves repeatability."
            ),
            dilution="A mild dilution (≈1:3) may help stay in the linear region.",
        )
    if signal == "weak":
        return Guidance(
            exposure=(
                f"{base} Weak signal expected — maximize headspace (increase surface area, slightly "
                "warm the sample) and use a longer exposure window (60-120 s)."
            ),
            dilution="Avoid dilution — you need the maximum headspace concentration.",
        )
    return Guidance(
        exposure=f"{base} No usable signal is expected under normal conditions.",
        dilution="N/A — not expected to be detectable.",
    )


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def run_chemical_verdict(chemical: Chemical, opts: Optional[ChainOptions] = None) -> FeasibilityVerdict:
    opts = opts or ChainOptions()
    sensor_count = opts.sensor_count if opts.sensor_count is not None else DEFAULT_SENSOR_COUNT
    c = run_constituent_chain(chemical)
    library = opts.library_substances if opts.library_substances is not None else []
    cross_check = _build_cross_check(sensor_count, library, chemical.name, chemical.synonyms)
    confidence = _confidence_of([c])
    g = guidance(c.signal_strength, c.response_speed)
    return FeasibilityVerdict(
        entity_id=chemical.id,
        entity_name=chemical.name,
        kind="chemical",
        verdict=c.verdict,
        confidence=confidence,
        signal_strength=c.signal_strength,
        response_speed=c.response_speed,
        constituents=[c],
        steps=c.steps,
        exposure_guidance=g.exposure,
        dilution_guidance=g.dilution,
        cross_check=cross_check,
        computed_at=_utc_now_iso(),
        sensor_count=sensor_count,
        notes=[],
    )


def run_composite_verdict(composite: Composite, opts: Optional[ChainOptions] = None) -> FeasibilityVerdict:
    opts = opts or ChainOptions()
    sensor_count = opts.sensor_count if opts.sensor_count is not None else DEFAULT_SENSOR_COUNT
    constituents: List[ConstituentVerdict] = []
    for c in composite.constituents:
        chemical = COMPOUND_BY_ID.get(c.chemical_id)
        if chemical is None:
            continue
        v = run_constituent_chain(chemical)
        v.weight_fraction = c.weight_fraction.value or 0
        v.weight_source = c.weight_fraction.source
        constituents.append(v)

    total_weight = sum(v.weight_fraction for v in constituents)
    if total_weight > 0:
        for v in constituents:
            v.weight_fraction /= total_weight

    red_weight = 0.0
    non_green_weight = 0.0
    for v in constituents:
        if v.verdict == "red":
            red_weight += v.weight_fraction
        if v.verdict != "green":
            non_green_weight += v.weight_fraction

    verdict: Verdict = "green"
    if red_weight > 0.5:
        verdict = "red"
    elif non_green_weight > 0.4:
        verdict = "yellow"

    # The dominant constituent defines the character of the headspace: the grade
    # follows the constituent that contributes the most expected signal.
    dominant: Optional[ConstituentVerdict] = None
    best_contribution = -1.0
    for v in constituents:
        contribution = v.weight_fraction * v.signal_score
        if contribution > best_contribution:
            best_contribution = contribution
            dominant = v
    signal_strength: SignalStrength = dominant.signal_strength if dominant else "none"
    response_speed: ResponseSpeed = dominant.response_speed if dominant else "unknown"

    confidence = _confidence_of(constituents)
    library = opts.library_substances if opts.library_substances is not None else []
    cross_check = _build_cross_check(sensor_count, library, composite.name, composite.synonyms)
    g = guidance(signal_strength, response_speed)

    notes: List[str] = []
    if any(v.weight_source == "estimated" for v in constituents):
        notes.append(
            "Constituent abundances are literature estimates (GC-MS studies) and vary with ripeness, cultivar, and preparation."
        )
    if composite.notes:
        notes.append(composite.notes)

    return FeasibilityVerdict(
        entity_id=composite.id,
        entity_name=composite.name,
        kind="composite",
        verdict=verdict,
        confidence=confidence,
        signal_strength=signal_strength,
        response_speed=response_speed,
        constituents=constituents,
        steps=[],
        exposure_guidance=g.exposure,
        dilution_guidance=g.dilution,
        cross_check=cross_check,
        computed_at=_utc_now_iso(),
        sensor_count=sensor_count,
        notes=notes,
    )


def run_class_verdict(class_key: str, opts: Optional[ChainOptions] = None) -> FeasibilityVerdict:
    term = CLASS_TERMS[class_key]
    opts = opts or ChainOptions()
    sensor_count = opts.sensor_count if opts.sensor_count is not None else DEFAULT_SENSOR_COUNT
    label = term.label
    steps: List[ChainStep] = [
        ChainStep(
            id="identity",
            label="Identity",
            verdict="green",
            reason=f"You asked about the {label} class of compounds.",
            detail="Many individual compounds fall in this class; resolve to a specific compound for a precise verdict.",
            values=[ChainValue(label="Class", value=label, source="measured")],
        ),
        ChainStep(
            id="volatility",
            label="Volatility",
            verdict="yellow",
            reason=f"Volatility varies across the {label.lower()} — small members are volatile, larger ones much less so.",
            detail="Grade depends on molecular weight and functional groups.",
            values=[ChainValue(label="Vapor pressure @ 25 °C", value="varies by compound", source="unknown")],
        ),
        ChainStep(
            id="signal",
            label="Headspace signal",
            verdict="yellow",
            reason="Expected to be detectable if a sufficiently volatile member is exposed.",
            detail="Use the specific-compound path for a numeric signal grade.",
            values=[ChainValue(label="Signal vs ethanol", value="varies by compound", source="unknown")],
        ),
        ChainStep(
            id="reactivity",
            label="MOX reactivity",
            verdict="green",
            reason=f"{label} are oxidized at the ~350 °C MOX surface — class-level chemistry is redox-active.",
            detail="MOX sensors respond to these functional groups (see science docs).",
            values=[ChainValue(label="Redox active", value="yes", source="measured")],
        ),
    ]
    library = opts.library_substances if opts.library_substances is not None else []
    return FeasibilityVerdict(
        entity_id=f"class:{class_key}",
        entity_name=label,
        kind="class",
        verdict="yellow",
        confidence="low",
        signal_strength="moderate",
        response_speed="medium",
        constituents=[],
        steps=steps,
        exposure_guidance=guidance("moderate", "medium").exposure,
        dilution_guidance=guidance("moderate", "medium").dilution,
        cross_check=_build_cross_check(sensor_count, library, label, []),
        computed_at=_utc_now_iso(),
        sensor_count=sensor_count,
        notes=["Class-level verdict only — resolve to a specific compound for a precise, actionable result."],
    )


def resolve_and_run(
    entity_id: str,
    kind: str,
    opts: Optional[ChainOptions] = None,
) -> Optional[FeasibilityVerdict]:
    if kind == "chemical":
        c = COMPOUND_BY_ID.get(entity_id) or user_dictionary_by_id().get(entity_id)
        return run_chemical_verdict(c, opts) if c else None
    if kind == "composite":
        comp = COMPOSITE_BY_ID.get(entity_id)
        return run_composite_verdict(comp, opts) if comp else None
    if kind == "class":
        key = entity_id.replace("^class:", "", 1) if entity_id.startswith("class:") else entity_id
        return run_class_verdict(key, opts) if key in CLASS_TERMS else None
    return None
