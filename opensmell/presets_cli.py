"""`opensmell-presets` — inspect the shipped configuration presets.

A preset is a configuration file, so the questions about it are questions about
configuration: what is in the file, does it hold together, what does it resolve
to once the derived numbers are computed. This command answers those and nothing
else. It deliberately does not emit JSON schema, does not generate code, and does
not upload anything — a preset is small, readable and meant to be edited by hand,
and a tool that makes it opaque works against that.

Subcommands:

    list                 ids of every shipped preset
    show <id|path>       one preset as text, or as JSON with --json
    validate [<id>]      load and cross-check every preset, or one of them
    field <id> <path>    one resolved field, e.g. `field food_freshness
                         baseline.duration_s`

Exit codes: 0 success, 1 a preset failed validation or the field could not be
resolved, 2 a bad command line.

Every subcommand reports the same advisory list for a preset (`advisories()`),
because those are the observations a user must read before deploying and they
would otherwise be invisible in every other view.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from .presets import (
    Preset,
    PresetError,
    PresetNotFoundError,
    list_presets,
    load_all_presets,
    load_preset,
    presets_dir,
    resolve_field,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _print(preset: Preset, stream=None) -> None:
    """Render one preset as text, with its advisories attached.

    Advisory placement is deliberate: they follow the block they refer to rather
    than being collected at the end, because "the cadence is below 1 Hz" is a
    statement about the cadence line and reads as a footnote three screens later.
    """
    out = stream if stream is not None else sys.stdout
    write = out.write

    write(f"{preset.id}\n{'=' * len(preset.id)}\n")
    write(f"{preset.name}\n\n")
    write(f"  {preset.description.strip()}\n\n")

    write(
        f"  cadence         {preset.sampling_rate_hz:g} Hz"
        f"  (1 sample per {preset.cadence_period_s:g} s,"
        f" {preset.n_channels} channels)\n"
    )
    if preset.cadence_rationale.strip():
        write(f"                   {preset.cadence_rationale.strip()}\n")
    for flag in preset.timing.flags:
        write(f"  ! timing: {flag}\n")
    floor = preset.min_detectable_duration_s
    write(
        f"  resolvable      {floor:g} s minimum event"
        f"  (2 samples)\n"
        f"  R0 baseline     {preset.r0_samples} samples"
        f" = {preset.baseline.duration_s:g} s"
        f" ({preset.baseline.method})\n"
    )
    if preset.r0_samples < 2:
        write("  ! baseline: fewer than 2 samples is not a usable R0\n")
    write(
        f"  settling        {preset.baseline.settling_s:g} s"
        f"   warm-up {preset.baseline.warmup_s:g} s"
        f" ({preset.baseline.warmup_s / 3600.0:g} h)\n"
    )

    write(f"\n  sensors ({preset.n_channels})\n")
    for sensor in preset.sensors:
        targets = ", ".join(sensor.targets) if sensor.targets else "not tabulated"
        write(f"    {sensor.id:<10} {sensor.model:<8} {sensor.role}\n")
        write(f"               targets: {targets}\n")
        if sensor.clean_air_ratio is not None:
            write(f"               clean air ratio: {sensor.clean_air_ratio:g}\n")
        if sensor.note.strip():
            write(f"               {sensor.note.strip()}\n")

    if preset.context_columns:
        write(
            "\n  context columns "
            + ", ".join(preset.context_columns)
            + "\n    preserved as metadata; never scored as sensor channels\n"
        )

    write("\n  features\n    " + ", ".join(preset.feature_groups) + "\n")
    missing = [g for g in preset.feature_groups if not preset.requests(g)]
    if missing:
        write(f"    ! not selected: {missing}\n")

    thresholds = preset.thresholds
    write(
        "\n  thresholds"
        f"      dead noise {thresholds.dead_noise_ceiling:g}"
        f"   drift floor {thresholds.drift_abs_floor:g}"
        f"   drift ceiling {thresholds.drift_rel_ceiling:g}\n"
    )
    if thresholds.basis.strip():
        write(f"    {thresholds.basis.strip()}\n")

    budget = preset.false_alarm_budget
    if budget is None:
        write("\n  false alarms   none set: not a continuously paged monitor\n")
    else:
        write(
            f"\n  false alarms   {budget.alarms_per_day:g} per day"
            f" (~1 per {_period(budget.alarms_per_day)})\n"
        )
        if budget.rationale.strip():
            write(f"    {budget.rationale.strip()}\n")

    if preset.provisional:
        write("\n  needs empirical validation\n    " + "\n    ".join(preset.provisional) + "\n")

    if preset.notes.strip():
        write(f"\n  notes\n    {preset.notes.strip()}\n")

    for note in preset.advisories():
        write(f"  ! {note}\n")

    if preset.source_path:
        write(f"\n  source: {preset.source_path}\n")


def _period(alarms_per_day: float) -> str:
    """Human interval for a per-day alarm rate."""
    if alarms_per_day <= 0:
        return "never"
    days = 1.0 / alarms_per_day
    if days >= 1.0:
        return f"{days:.0f} day(s)"
    hours = days * 24.0
    return f"{hours:.0f} hour(s)"


def _cmd_list(args: argparse.Namespace) -> int:
    directory = presets_dir()
    if args.json:
        payload = []
        for preset in load_all_presets().values():
            payload.append(
                {
                    "id": preset.id,
                    "name": preset.name,
                    "cadence_hz": preset.sampling_rate_hz,
                    "channels": preset.n_channels,
                    "feature_groups": preset.feature_groups,
                    "advisories": preset.advisories(),
                    "source_path": preset.source_path,
                }
            )
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    ids = list_presets()
    width = max(len(i) for i in ids) if ids else 0
    print(f"{len(ids)} presets in {directory}\n")
    for preset_id in ids:
        preset = load_preset(preset_id)
        budget = preset.false_alarm_budget
        rate = f"{budget.alarms_per_day:g}/day" if budget else "no FA budget"
        print(
            f"  {preset_id:<{width}}  {preset.sampling_rate_hz:>6g} Hz"
            f"  {preset.n_channels} ch  {rate}"
        )
        print(f"  {'':<{width}}  {preset.name}")
    return EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    preset = load_preset(args.preset)
    if args.json:
        print(json.dumps(preset.to_dict(), indent=2))
    else:
        _print(preset)
    return EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.preset:
        presets = [load_preset(args.preset)]
    else:
        presets = list(load_all_presets().values())

    failures: List[str] = []
    for preset in presets:
        advisories = preset.advisories()
        if args.strict and advisories:
            failures.append(f"{preset.id}: {advisories[0]}")
            print(f"FAIL  {preset.id}: {advisories[0]}")
            continue
        print(f"ok    {preset.id}")
        for note in advisories:
            print(f"      ! {note}")

    checked = len(presets)
    print(f"\n{checked - len(failures)}/{checked} presets valid, from {presets_dir()}")
    if failures:
        print(f"{len(failures)} failed")
        return EXIT_FAILED
    return EXIT_OK


def _cmd_field(args: argparse.Namespace) -> int:
    preset = load_preset(args.preset)
    try:
        value = resolve_field(preset.to_dict(), args.path)
    except (PresetError, KeyError) as exc:
        print(f"opensmell-presets: no field {args.path!r} in {preset.id}", file=sys.stderr)
        print(f"opensmell-presets: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2))
    else:
        print(value)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opensmell-presets",
        description="Inspect the shipped OpenSmell configuration presets.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_list = sub.add_parser("list", help="list every shipped preset")
    p_list.add_argument("--json", action="store_true", help="emit JSON")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="show one preset")
    p_show.add_argument("preset", metavar="<id|path>", help="preset id or file path")
    p_show.add_argument("--json", action="store_true", help="emit JSON")
    p_show.set_defaults(func=_cmd_show)

    p_validate = sub.add_parser("validate", help="load and cross-check presets")
    p_validate.add_argument(
        "preset",
        nargs="?",
        metavar="<id>",
        help="one preset; default is every shipped preset",
    )
    p_validate.add_argument(
        "--strict",
        action="store_true",
        help="treat advisories as failures",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_field = sub.add_parser("field", help="print one resolved field")
    p_field.add_argument("preset", metavar="<id|path>", help="preset id or file path")
    p_field.add_argument("path", metavar="<path>", help="dotted field path")
    p_field.set_defaults(func=_cmd_field)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        return args.func(args)
    except PresetNotFoundError as exc:
        print(f"opensmell-presets: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except PresetError as exc:
        print(f"opensmell-presets: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except BrokenPipeError:  # pragma: no cover - shell pipelines
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())