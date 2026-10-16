"""``opensmell-mcp`` — a read-only MCP server over the OpenSmell SDK.

This server exposes a small, honest slice of the SDK to an AI agent over the
Model Context Protocol: the shipped presets, recording validation, a feature and
quality summary, and per-channel sensor-health features. It exists so an agent
can answer questions about a recording without re-deriving the format, and so it
can discover what it *cannot* do (``capabilities()``) instead of inventing an
answer.

Design rules, enforced by construction rather than by documentation:

* **Read-only.** No tool writes a file, mutates the SDK, or produces side
  effects. Every failure is returned as a JSON object with an actionable
  ``error`` string — never a raised exception, never a traceback.
* **Offline.** No network calls, no telemetry. Files on the local disk only.
* **No hardware.** Nothing here opens a serial port or talks to a device. A
  tool that would need the rig says so and points at the desktop app.

The four-way anomaly verdict (Normal / GradualDrift / Poisoning / Dead) lives in
the Rust SDK (``opensmell-rs/src/anomaly/verdict.rs``) and is deliberately *not*
reimplemented here. ``compute_health_features`` returns the health features the
verdict is built from and states that the verdict itself is out of reach. A
second implementation of a safety-critical policy would drift from the first;
the boundary is a feature, not a gap to paper over.

Transport: stdio. Registered as the ``opensmell-mcp`` console script.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .features import run_processor
from .ingest import ingest_file
from .io import parse_osmell_file
from .mox.features import compute_channel_health
from .presets import (
    PresetError,
    PresetNotFoundError,
    load_preset,
)
from .presets import list_presets as _list_preset_ids
from .quality import compute_quality
from .types import DEFAULT_SYNTHETIC_RATE_HZ, OsmellFile, QualityReport, r0_window_samples

SERVER_NAME = "opensmell"
SERVER_VERSION = "1.0.0"

SUPPORTED_SUFFIXES = (".osmell", ".csv", ".txt")
R0_DEFAULT_RULE = "clamp(floor(0.15 * N), 5, 30) when no window is declared"

READ_ONLY_NOTE = (
    "This server is read-only and offline: it never writes files, calls the "
    "network, or touches hardware. To record or export, use the OpenSmell "
    "desktop app."
)

# The Rust verdict is the only correct implementation of the four-way policy.
# Kept as a named constant so both capabilities() and compute_health_features()
# say the same thing in the same words.
VERDICT_LIMITATION = (
    "The four-way anomaly verdict (Normal / GradualDrift / Poisoning / Dead) is "
    "implemented in Rust at opensmell-rs/src/anomaly/verdict.rs and is not "
    "callable from this Python server. This tool returns the raw sensor-health "
    "features the verdict consumes (drift_rate, sensitivity_decay, noise_floor, "
    "hysteresis); it does not classify a channel. Do not present these numbers "
    "as a verdict."
)

# --------------------------------------------------------------------------- #
# Small result / sanitization helpers
# --------------------------------------------------------------------------- #


def _ok(**payload: Any) -> Dict[str, Any]:
    return {"ok": True, **payload}


def _error(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


def _sanitize(value: Any) -> Any:
    """Return a JSON-safe copy: numpy scalars/arrays become Python natives.

    The SDK returns a few ``numpy.float64`` values; ``json.dumps`` rejects them.
    Every tool funnels its result through here so the MCP layer only ever sees
    plain dicts, lists, str, int, float, bool and None.
    """
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    return value


def _guarded(fn):
    """Wrap a tool so it can never raise into the MCP transport.

    Input-schema validation still happens at the SDK boundary (an unknown or
    missing *required* argument is a protocol error and should be); everything
    inside the tool — file I/O, parsing, feature extraction — is caught here and
    converted to an actionable ``{"ok": false, "error": ...}`` result.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the contract is "never raise"
            return _error(
                f"{fn.__name__} failed unexpectedly: {exc}. "
                "Check that the path points at a readable recording (see the "
                "capabilities tool for supported formats); if it is a valid "
                "file this is a bug, not a data problem."
            )
        return _sanitize(result)

    return wrapper


def _resolve_existing_file(path: Any):
    """Return ``(Path, None)`` for a readable file, else ``(None, error_dict)``."""
    if not isinstance(path, str) or not path.strip():
        return None, _error(
            "path must be a non-empty string pointing at a .osmell, .csv, or "
            ".txt recording, e.g. 'recording.osmell' or '/data/run1.csv'."
        )
    p = Path(path).expanduser()
    if not p.exists():
        return None, _error(
            f"file not found at {p}. This server only reads files that already "
            f"exist on disk, and relative paths resolve against the server "
            f"process working directory ({Path.cwd()}). Check the path or copy "
            f"the recording next to the server."
        )
    if p.is_dir():
        return None, _error(
            f"{p} is a directory, not a recording. Point at a single .osmell, "
            f".csv, or .txt file."
        )
    return p, None


def _quality_summary(report: Optional[QualityReport]) -> Optional[Dict[str, Any]]:
    if report is None:
        return None
    flags = report.flags
    return {
        "total": report.total,
        "badge": report.badge,
        "subscores": {
            name: {"value": sub.value, "reason": sub.reason}
            for name, sub in report.subscores.items()
        },
        "flags": {
            "dead_sensors": list(flags.dead_sensors),
            "unsorted_rows": flags.unsorted_rows,
            "non_finite_samples": flags.non_finite_samples,
            "used_default_adc_max": flags.used_default_adc_max,
            "used_median_sampling_rate": flags.used_median_sampling_rate,
            "no_baseline": flags.no_baseline,
            "empty_recording": flags.empty_recording,
            "time_unit_mismatch": flags.time_unit_mismatch,
        },
        "notes": list(report.notes),
        "reasons": dict(report.reasons),
    }


def _load_recording(path: Any):
    """Load a recording into an ``OsmellFile``.

    Returns ``(info, None)`` on success with ``info`` a dict of
    ``path``, ``format``, ``file``, ``sample_count``, ``sampling_rate_hz``,
    ``sampling_rate_source``, ``report`` and ``warnings``; otherwise
    ``(None, error_dict)`` with an actionable message.
    """
    p, err = _resolve_existing_file(path)
    if err is not None:
        return None, err

    suffix = p.suffix.lower()
    if suffix == ".osmell":
        try:
            file = parse_osmell_file(str(p))
        except Exception as exc:  # noqa: BLE001
            return None, _error(
                f"could not read the .osmell bundle at {p}: {exc}. A .osmell "
                f"file is a zip containing manifest.json and data.csv; if it "
                f"was truncated or hand-edited, re-export it from the desktop app."
            )
        sr = file.manifest.sensor.sampling_rate_hz
        report = None
        warnings: List[str] = []
        try:
            report = compute_quality(
                file,
                sample_count=len(file.time),
                guess_sampling_rate_hz=float(sr) if sr else 0.0,
            )
        except NotImplementedError as exc:
            warnings.append(f"quality scorer unavailable: {exc}")
        return (
            {
                "path": str(p),
                "format": "osmell",
                "file": file,
                "sample_count": len(file.time),
                "sampling_rate_hz": float(sr) if sr else None,
                "sampling_rate_source": "manifest" if sr else "undeclared",
                "report": report,
                "warnings": warnings,
            },
            None,
        )

    if suffix in (".csv", ".txt"):
        session = ingest_file(str(p))
        if not session.ok:
            return None, _error(
                f"could not load {p} as a recording: {session.error}. "
                f"This server expects a CSV with a header row and numeric "
                f"sensor columns (an optional timestamp_ms/elapsed_ms column is "
                f"used for timing)."
            )
        rate = session.file.manifest.sensor.sampling_rate_hz
        if session.time_source == "synthetic":
            # The manifest carries a rate, but it was synthesized from the
            # default because the CSV had no time column; say so rather than
            # letting it pass as measured.
            rate_source = "synthetic"
        elif rate:
            rate_source = "manifest"
        else:
            rate_source = "unknown"
        return (
            {
                "path": str(p),
                "format": "csv",
                "file": session.file,
                "sample_count": len(session.file.time),
                "sampling_rate_hz": rate,
                "sampling_rate_source": rate_source,
                "report": session.report,
                "warnings": list(session.warnings),
            },
            None,
        )

    return None, _error(
        f"unsupported recording type '{suffix or p.name}' at {p}. Supported "
        f"extensions: .osmell, .csv, .txt."
    )


# --------------------------------------------------------------------------- #
# Tool implementations (pure logic; registered with descriptions in build_server)
# --------------------------------------------------------------------------- #


def list_presets() -> Dict[str, Any]:  # type: ignore[no-redef]
    """List the 12 shipped use-case presets (ids and names)."""
    try:
        ids = _list_preset_ids()
    except Exception as exc:  # noqa: BLE001
        return _error(
            f"could not enumerate presets: {exc}. The preset directory may be "
            f"missing; set OPENSMELL_PRESETS_DIR to a folder of *.yaml presets."
        )
    presets: List[Dict[str, Any]] = []
    for preset_id in ids:
        entry: Dict[str, Any] = {"id": preset_id, "name": None}
        try:
            entry["name"] = load_preset(preset_id).name
        except PresetError as exc:
            entry["error"] = f"preset failed to load: {exc}"
        presets.append(entry)
    return _ok(count=len(presets), presets=presets)


def get_preset(preset_id: str) -> Dict[str, Any]:
    """Return one full preset as JSON, with advisories attached."""
    if not isinstance(preset_id, str) or not preset_id.strip():
        return _error(
            "preset_id must be a non-empty string; call list_presets for the "
            "valid ids."
        )
    preset_id = preset_id.strip()
    try:
        preset = load_preset(preset_id)
    except PresetNotFoundError:
        try:
            valid = ", ".join(_list_preset_ids())
        except Exception:  # noqa: BLE001
            valid = "(could not enumerate; see list_presets)"
        return _error(
            f"unknown preset '{preset_id}'. Valid ids: {valid}."
        )
    except PresetError as exc:
        return _error(f"preset '{preset_id}' failed to load: {exc}")
    data = preset.to_dict()
    data["advisories"] = list(preset.advisories())
    data["source_path"] = preset.source_path
    return _ok(preset=data)


def validate_recording(path: str) -> Dict[str, Any]:
    """Validate a recording and return structured, actionable issues."""
    p, err = _resolve_existing_file(path)
    if err is not None:
        return err

    suffix = p.suffix.lower()
    if suffix == ".osmell":
        return _validate_osmell(p)
    if suffix in (".csv", ".txt"):
        return _validate_csv(p)
    return _error(
        f"unsupported recording type '{suffix or p.name}' at {p}. Supported "
        f"extensions: .osmell, .csv, .txt."
    )


def _validate_osmell(p: Path) -> Dict[str, Any]:
    try:
        file = parse_osmell_file(str(p))
    except Exception as exc:  # noqa: BLE001
        return _ok(
            path=str(p),
            format="osmell",
            valid=False,
            errors=[
                f"could not read the .osmell bundle: {exc}. A .osmell file is "
                f"a zip containing manifest.json and data.csv; re-export it if "
                f"it was truncated or hand-edited."
            ],
            warnings=[],
            next_steps=READ_ONLY_NOTE,
        )

    errors: List[str] = []
    warnings: List[str] = []
    manifest = file.manifest
    channels = [c.id for c in manifest.sensor.channels]
    if not channels:
        errors.append(
            "manifest declares no sensor channels (sensor.channels is empty); "
            "nothing can be scored."
        )
    if not file.time:
        errors.append("data.csv contains no rows.")
    for cid in channels:
        if cid not in file.data:
            errors.append(
                f"channel '{cid}' is declared in the manifest but missing from "
                f"data.csv."
            )
    if not manifest.sensor.sampling_rate_hz:
        warnings.append(
            "manifest does not declare a sampling rate; per-second features "
            "will use a default and should be treated as approximate."
        )
    return _ok(
        path=str(p),
        format="osmell",
        valid=not errors,
        sensor_type=manifest.sensor.sensor_type,
        channels=channels,
        sample_count=len(file.time),
        errors=errors,
        warnings=warnings,
        next_steps=READ_ONLY_NOTE,
    )


def _validate_csv(p: Path) -> Dict[str, Any]:
    session = ingest_file(str(p))
    warnings = list(session.warnings or [])
    if not session.ok:
        return _ok(
            path=str(p),
            format="csv",
            valid=False,
            errors=[session.error or "unknown ingest error"],
            warnings=warnings,
            next_steps=(
                "Fix the CSV: keep a header row, at least one numeric sensor "
                "column, and at least one data row. A timestamp_ms/elapsed_ms "
                "column is optional. Nothing was written; re-run validate_recording "
                "after editing."
            ),
        )
    file = session.file
    channels = [c.id for c in file.manifest.sensor.channels]
    return _ok(
        path=str(p),
        format="csv",
        valid=True,
        sensor_type=session.sensor_type,
        channels=channels,
        sample_count=len(file.time),
        time_source=session.time_source,
        errors=[],
        warnings=warnings,
        quality=_quality_summary(session.report),
        next_steps=READ_ONLY_NOTE,
    )


def analyze_recording(path: str) -> Dict[str, Any]:
    """Validate a recording, extract features, and summarize quality."""
    info, err = _load_recording(path)
    if err is not None:
        return err

    file: OsmellFile = info["file"]
    sensor_type = file.manifest.sensor.sensor_type
    try:
        processed = run_processor(file)
    except Exception as exc:  # noqa: BLE001
        return _error(
            f"feature extraction failed for {info['path']} (sensor type "
            f"'{sensor_type}'): {exc}. The file loaded but its features could "
            f"not be computed; this usually means a malformed channel."
        )

    feature_blocks = processed.get("features") or []
    features = [
        {
            "channel": block.channel,
            "r0": block.r0,
            "relative_amplitude": block.relative_amplitude,
            "direction": block.direction,
            "dead": block.dead,
            "saturation_index": block.saturation_index,
            "rise_time_ms": block.rise_time_ms,
            "decay_time_ms": block.decay_time_ms,
            "auc": block.auc,
            "endpoint_delta": block.endpoint_delta,
        }
        for block in feature_blocks
    ]

    warnings = list(info["warnings"])
    if not features:
        warnings.append(
            f"no framework features were produced for sensor type "
            f"'{sensor_type}'; only quality/validation information is available."
        )

    return _ok(
        path=info["path"],
        format=info["format"],
        sensor_type=sensor_type,
        channels=[c.id for c in file.manifest.sensor.channels],
        sample_count=info["sample_count"],
        sampling_rate_hz=info["sampling_rate_hz"],
        sampling_rate_source=info["sampling_rate_source"],
        quality=_quality_summary(info["report"]),
        features=features,
        warnings=warnings,
        note=(
            "Features are deterministic summaries of this recording. This tool "
            "does not train, does not predict a substance, and writes nothing. "
            + READ_ONLY_NOTE
        ),
    )


def compute_health_features(
    path: str, r0_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Compute per-channel drift/noise/hysteresis under the R0 contract."""
    if r0_samples is not None and (
        isinstance(r0_samples, bool) or not isinstance(r0_samples, int)
    ):
        return _error(
            f"r0_samples must be an integer sample count or omitted; got "
            f"{r0_samples!r}. Omit it to use the contract default "
            f"({R0_DEFAULT_RULE})."
        )
    if r0_samples is not None and r0_samples <= 0:
        return _error(
            f"r0_samples must be a positive integer or omitted; got "
            f"{r0_samples}. A non-positive window is not meaningful, so the "
            f"contract default ({R0_DEFAULT_RULE}) applies only when the "
            f"argument is absent."
        )

    info, err = _load_recording(path)
    if err is not None:
        return err

    file: OsmellFile = info["file"]
    sr = info["sampling_rate_hz"]
    if not sr or sr <= 0:
        sr = float(DEFAULT_SYNTHETIC_RATE_HZ)
        sr_source = "default_synthetic"
    else:
        sr = float(sr)
        sr_source = info["sampling_rate_source"]

    channels: List[Dict[str, Any]] = []
    missing: List[str] = []
    for channel in file.manifest.sensor.channels:
        cid = channel.id
        series = file.data.get(cid)
        if series is None:
            missing.append(cid)
            continue
        arr = np.asarray(series, dtype=np.float64)
        window = r0_window_samples(len(arr), r0_samples)
        health = compute_channel_health(arr, r0_samples=r0_samples, sr=sr)
        channels.append(
            {
                "channel": cid,
                "n_samples": int(len(arr)),
                "r0_window_samples": int(window),
                "drift_rate": health["drift_rate"],
                "sensitivity_decay": health["sensitivity_decay"],
                "noise_floor": health["noise_floor"],
                "hysteresis": health["hysteresis"],
            }
        )

    warnings: List[str] = list(info["warnings"])
    if missing:
        warnings.append(
            f"channels {missing} are declared in the manifest but absent from "
            f"the data; they were skipped."
        )
    if sr_source in ("default_synthetic", "synthetic", "unknown"):
        warnings.append(
            "no measured sampling rate was available; drift_rate is scaled by "
            f"{sr:g} Hz (source: {sr_source}) and is approximate."
        )

    return _ok(
        path=info["path"],
        format=info["format"],
        sensor_type=file.manifest.sensor.sensor_type,
        sampling_rate_hz=sr,
        sampling_rate_source=sr_source,
        r0_contract={
            "declared_r0_samples": r0_samples,
            "default_rule": R0_DEFAULT_RULE,
            "note": (
                "A declared r0_samples window wins verbatim; otherwise the "
                "window is a floored, capped 15% of the recording, which is "
                "cadence-independent in the fraction region."
            ),
        },
        channels=channels,
        warnings=warnings,
        verdict=None,
        verdict_available=False,
        verdict_limitation=VERDICT_LIMITATION,
    )


def capabilities() -> Dict[str, Any]:
    """Report what this server can and cannot do."""
    return _ok(
        server=SERVER_NAME,
        version=SERVER_VERSION,
        transport="stdio",
        tools=[
            {
                "name": "list_presets",
                "purpose": "ids and names of the 12 shipped use-case presets",
            },
            {
                "name": "get_preset",
                "purpose": "one full preset (sensors, cadence, features, thresholds)",
            },
            {
                "name": "validate_recording",
                "purpose": "structured validation of an existing recording file",
            },
            {
                "name": "analyze_recording",
                "purpose": "features + quality summary for a recording",
            },
            {
                "name": "compute_health_features",
                "purpose": "per-channel drift/noise/hysteresis under the R0 contract",
            },
            {
                "name": "capabilities",
                "purpose": "this description",
            },
        ],
        can={
            "read_files": list(SUPPORTED_SUFFIXES),
            "presets": "list and return the shipped presets",
            "validate": "structural validation with actionable error strings",
            "features": "MOX framework features and quality scoring",
            "health": (
                "per-channel drift_rate, sensitivity_decay, noise_floor, "
                "hysteresis with declared-window-or-clamp(floor(0.15*N),5,30) R0"
            ),
        },
        cannot={
            "write": "no tool writes, deletes, or moves files",
            "network": "no network calls and no telemetry of any kind",
            "hardware": (
                "no device access; the server never opens a serial port. Use the "
                "OpenSmell desktop app to record from a rig."
            ),
            "train_or_predict": (
                "no model training or substance prediction; use the SDK's "
                "train()/predict() in a script"
            ),
            "verdict": VERDICT_LIMITATION,
            "miris_electrochemical_features": (
                "feature/quality extraction is only implemented for MOX (and "
                "unknown arrays scored device-agnostically); MIRIS and "
                "electrochemical recordings validate but do not get features"
            ),
        },
        errors={
            "shape": '{"ok": false, "error": "<actionable message>"}',
            "policy": "no stack traces; failures name the path and the next step",
        },
        constraints=READ_ONLY_NOTE,
    )


# --------------------------------------------------------------------------- #
# MCP wiring
# --------------------------------------------------------------------------- #

_DESCRIPTIONS = {
    "list_presets": (
        "List the 12 shipped OpenSmell use-case presets. Returns each preset's "
        "stable id and human-readable name. Call this first to discover valid "
        "preset_id values for get_preset. Read-only; takes no arguments; never "
        "touches the network or hardware."
    ),
    "get_preset": (
        "Return one full OpenSmell preset as JSON: sensor list and targets, "
        "sampling cadence (Hz), baseline/R0 window, requested feature groups, "
        "anomaly thresholds, false-alarm budget, and non-fatal advisories. "
        "Input: preset_id, one of the ids from list_presets. Failure: returns "
        '{"ok": false, "error": ...} naming the valid ids when preset_id is '
        "unknown or the preset file fails to load."
    ),
    "validate_recording": (
        "Validate a single existing .osmell, .csv, or .txt recording without "
        "modifying it. Returns valid (bool), errors[] (actionable, structural), "
        "warnings[] (non-fatal adopt-don't-reject decisions), the detected "
        "sensor type, channel ids, sample count and timing source. Never raises, "
        "never writes. Failure: a missing path, a directory, or an unsupported "
        'extension returns {"ok": false, "error": ...} telling you what to do '
        "instead. Success may still report valid=false with reasons in errors[]."
    ),
    "analyze_recording": (
        "Validate a recording and compute deterministic features plus quality: "
        "per-channel R0, relative amplitude, direction, dead flag, saturation "
        "index, rise/decay time, AUC and endpoint delta, together with the "
        "quality score, badge and flags. Input: path to an existing .osmell, "
        ".csv, or .txt file. Read-only and offline. Returns "
        '{"ok": false, "error": ...} if the file cannot be loaded or its sensor '
        "family has no feature extractor. Does not train, predict a substance, "
        "or write anything."
    ),
    "compute_health_features": (
        "Compute per-channel sensor-health features for an existing recording: "
        "drift_rate (least-squares dR/dt in units/second), sensitivity_decay "
        "(last-third minus first-third mean, over R0), noise_floor (baseline "
        "std / R0) and hysteresis (adsorption vs desorption path ratio). "
        "r0_samples optionally declares the baseline window in samples; when "
        "omitted the R0 contract default applies: clamp(floor(0.15*N), 5, 30). "
        "Read-only and offline. IMPORTANT: this returns raw health features, NOT "
        "the four-way Normal/GradualDrift/Poisoning/Dead verdict, which lives in "
        "the Rust SDK and is not callable here; the response states this "
        'explicitly. Failure: {"ok": false, "error": ...} for a bad path, an '
        "unloadable file, or a non-positive/non-integer r0_samples."
    ),
    "capabilities": (
        "Describe what this MCP server can and cannot do, so an agent can plan "
        "instead of guessing: available tools, supported file formats and sensor "
        "families, the read-only / offline / no-hardware constraints, where the "
        "four-way anomaly verdict actually lives (Rust, not exposed here), and "
        "how failures are reported. Call this when unsure whether a task is in "
        "scope. Takes no arguments."
    ),
}


def _register(server: Any, fn) -> None:
    kwargs: Dict[str, Any] = {
        "name": fn.__name__,
        "description": _DESCRIPTIONS[fn.__name__],
    }
    try:
        server.add_tool(fn, structured_output=False, **kwargs)
    except TypeError:
        # Older SDKs without structured_output; plain text output is the default.
        server.add_tool(fn, **kwargs)


def build_server() -> Any:
    """Construct the MCP server with all tools registered.

    Imported lazily so ``import opensmell.mcp_server`` works even when the
    optional ``mcp`` dependency is not installed; only running the server needs
    it.
    """
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2
    except ImportError:  # pragma: no cover - exercised only on mcp 1.x
        from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore

    server = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "Read-only, offline tools over the OpenSmell digital-olfaction SDK. "
            "Call capabilities() for scope, list_presets() to discover preset "
            "ids, and validate_recording()/analyze_recording() on local files. "
            "No tool writes or uses hardware; the four-way anomaly verdict is "
            "not available here."
        ),
    )
    for fn in (
        list_presets,
        get_preset,
        validate_recording,
        analyze_recording,
        compute_health_features,
        capabilities,
    ):
        _register(server, _guarded(fn))
    return server


def main() -> None:
    """Console entry point: run the server over stdio."""
    build_server().run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
