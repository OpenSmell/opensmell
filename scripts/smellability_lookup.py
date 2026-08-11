"""Smellability lookup — grade a substance or mixture for MOX feasibility.

Answers the question "will my e-nose actually smell it?" with the 4-step
physics chain (identity -> volatility -> signal -> reactivity). Accepts:

  - a substance name       python scripts/smellability_lookup.py ethanol
  - a SMILES string        python scripts/smellability_lookup.py --smiles O=C=O
  - a mixture              python scripts/smellability_lookup.py --mix "isoamyl acetate:0.6,ethanol:0.4"
                           (constituents may be names or SMILES strings)

Mixture constituents are graded as a weighted composite; constituents given as
SMILES are inferred via group contributions (Joback) and never pretend to be
curated measurements — confidence drops to low/medium accordingly.

Exit codes: 0 = verdict produced, 2 = query unresolved, 3 = usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opensmell import smellability  # noqa: E402
from opensmell.smellability import (  # noqa: E402
    COMPOUND_BY_ID,
    ChainOptions,
    Composite,
    CompositeConstituent,
    Property,
    exact_resolve,
    search_substances,
)


SMILES_HINT = re.compile(r"[=#()\[\]/0-9]")
MAX_CHOICES = 8


def looks_like_smiles(query: str) -> bool:
    q = query.strip()
    return bool(re.search(SMILES_HINT, q))


def resolve_name(query: str):
    """Exact-resolve a name; on failure, surface the best search choices."""
    entity = exact_resolve(query)
    if entity is not None:
        return entity
    if looks_like_smiles(query):
        try:
            chem = smellability.chemical_from_smiles(query)
        except ValueError:
            return None
        return chem
    return None


def _resolve_member(token: str):
    """Resolve a mixture member (name or SMILES) to a Chemical."""
    entity = resolve_name(token)
    if entity is None:
        return None, f"cannot resolve constituent {token!r}"
    if hasattr(entity, "kind"):
        if entity.kind != "chemical":
            return None, f"{token!r} resolved to a {entity.kind}, not a chemical — expand it into compounds first"
        chemical = COMPOUND_BY_ID.get(entity.id)
        if chemical is None:
            return None, f"{token!r} resolved but has no curated chemical record"
        return chemical, None
    if not hasattr(entity, "props"):
        return None, f"{token!r} resolved to an unsupported entity"
    return entity, None


def parse_mix(spec: str):
    """Parse 'name-or-smiles:weight,name2:weight2'. Weight is optional (default equal)."""
    members = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            token, _, weight = chunk.rpartition(":")
            weight = float(weight)
        else:
            token, weight = chunk, None
        if not token.strip():
            raise ValueError(f"empty constituent in mixture spec: {chunk!r}")
        members.append((token.strip(), weight))
    if not members:
        raise ValueError("mixture spec is empty")
    return members


def build_mix_verdict(members, opts):
    chemicals = {}
    constituents = []
    problems = []
    total_weight = sum((w if w is not None else 0.0) for _, w in members)
    explicit_weights = any(w is not None for _, w in members)
    auto_weight = 1.0 / len(members)
    for i, (token, weight) in enumerate(members):
        chemical, err = _resolve_member(token)
        if err:
            problems.append(err)
            continue
        if explicit_weights:
            w = weight if weight is not None else 0.0
        else:
            w = auto_weight
        chemicals[chemical.id] = chemical
        constituents.append(
            CompositeConstituent(
                chemical_id=chemical.id,
                weight_fraction=Property(value=w, source="measured" if weight is not None else "estimated"),
            )
        )
    if not constituents:
        raise ValueError("no constituents could be resolved")
    if problems:
        sys.stderr.write("WARNING\n")
        for p in problems:
            sys.stderr.write(f"  - {p}\n")
    composite = Composite(
        id="mix-cli",
        name=", ".join(
            f"{chemicals[c.chemical_id].name} {c.weight_fraction.value:.2f}".strip() for c in constituents
        ),
        kind="other",
        synonyms=[],
        constituents=constituents,
        source_refs=["user-supplied mixture (CLI)"],
        notes="User-supplied mixture; weights are as given and not literature-verified." if explicit_weights else None,
    )
    verdict = smellability.run_composite_verdict(composite, opts, constituent_chemicals=chemicals)
    if explicit_weights and total_weight > 0:
        verdict.notes = list(verdict.notes) + [f"Total given weight: {total_weight:.2f} (not normalized)."]
    return verdict


def format_verdict(v, as_json: bool) -> str:
    if as_json:
        return json.dumps(v.to_dict(), indent=2, sort_keys=True)
    lines = []
    lines.append("=" * 60)
    lines.append(f"{v.entity_name}  [{v.kind}]")
    lines.append(f"  VERDICT      {v.verdict.upper():<6}  signal={v.signal_strength}  speed={v.response_speed}")
    lines.append(f"  CONFIDENCE   {v.confidence}")
    lines.append(f"  SENSOR COUNT {v.sensor_count}")
    if v.constituents:
        lines.append("  CONSTITUENTS")
        for c in v.constituents:
            lines.append(
                f"    - {c.name:<28} w={c.weight_fraction:.2f}  {c.verdict.upper():<6} "
                f"signal={c.signal_strength}"
            )
    for step in v.steps:
        lines.append(f"  [{step.id:<10}] {step.verdict.upper():<6} {step.reason}")
        for val in step.values:
            lines.append(f"      {val.label}: {val.value}  ({val.source})")
    if v.cross_check is not None:
        cc = v.cross_check
        lines.append(f"  CROSS-CHECK  max distinguishable = {cc.max_distinguishable}")
        if cc.confusable:
            lines.append(f"    confusable with: {', '.join(cc.confusable)}")
        lines.append(f"    {cc.note}")
    lines.append(f"  EXPOSURE     {v.exposure_guidance}")
    lines.append(f"  DILUTION     {v.dilution_guidance}")
    for n in v.notes or []:
        lines.append(f"  NOTE         {n}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _choices_text(query: str) -> str:
    results = search_substances(query, MAX_CHOICES)
    if not results:
        return f"no matches for {query!r}"
    lines = [f"no exact match for {query!r}; closest:"]
    for r in results:
        lines.append(f"  - {r.name:<28} ({r.match_hint})  score={r.score}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="smellability_lookup", description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="?", help="substance name (or SMILES if it contains SMILES syntax)")
    parser.add_argument("--smiles", help="grade a SMILES string directly (e.g. O=C=O)")
    parser.add_argument("--mix", help='mixture as "a:0.6,b:0.4" — members may be names or SMILES')
    parser.add_argument("--sensor-count", type=int, default=6, help="sensor count for the capacity cross-check (default 6)")
    parser.add_argument("--library", help="comma-separated substances already in your labeled library")
    parser.add_argument("--json", action="store_true", help="machine-readable output (FeasibilityVerdict.to_dict)")
    args = parser.parse_args(argv)

    opts = ChainOptions(sensor_count=args.sensor_count)
    if args.library:
        opts.library_substances = [s.strip() for s in args.library.split(",") if s.strip()]

    try:
        if args.mix:
            members = parse_mix(args.mix)
            verdict = build_mix_verdict(members, opts)
        elif args.smiles:
            verdict = smellability.run_chemical_verdict(
                smellability.chemical_from_smiles(args.smiles), opts
            )
        elif args.query:
            try:
                resolved = resolve_name(args.query)
            except ValueError as e:
                print(f"error: {e}")
                return 3
            if resolved is None:
                if looks_like_smiles(args.query):
                    print(f"error: could not parse {args.query!r} as a name or SMILES")
                    return 3
                print(_choices_text(args.query))
                return 2
            if hasattr(resolved, "kind"):
                verdict = smellability.resolve_and_run(resolved.id, resolved.kind, opts)
                if verdict is None:
                    print(f"could not run verdict for {resolved.display_name!r}")
                    return 3
            else:
                verdict = smellability.run_chemical_verdict(resolved, opts)
        else:
            parser.error("provide a query, --smiles, or --mix")
            return 3
    except ValueError as e:
        print(f"error: {e}")
        return 3

    print(format_verdict(verdict, args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
