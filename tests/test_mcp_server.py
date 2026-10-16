"""The ``opensmell-mcp`` read-only MCP server.

Covers the contract an agent depends on: a well-formed tool list, each tool's
happy path on a real fixture, actionable (never-raised) error paths, and a
``capabilities()`` report that matches what is actually implemented.
"""

import asyncio
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import opensmell as o
from opensmell import mcp_server as mcp

pytest.importorskip("mcp", reason="optional MCP server dependency not installed")

FIXTURE = os.path.join(os.path.dirname(__file__), "cinnamon_1.csv")

EXPECTED_TOOLS = {
    "list_presets",
    "get_preset",
    "validate_recording",
    "analyze_recording",
    "compute_health_features",
    "capabilities",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _write_tiny_csv(path, n=40, channels=("VOC", "Alcohol")):
    rows = ["timestamp_ms," + ",".join(channels)]
    for i in range(n):
        value = 100 + 10 * math.sin(i / 5.0) + 0.5 * i
        rows.append(f"{i * 100}," + ",".join(str(int(value + 3 * c)) for c, _ in enumerate(channels)))
    path.write_text("\n".join(rows) + "\n")
    return path


def _list_tools():
    async def _inner():
        return await mcp.build_server().list_tools()

    return asyncio.run(_inner())


def _call(name, arguments):
    async def _inner():
        result = await mcp.build_server().call_tool(name, arguments)
        return json.loads(result.content[0].text)

    return asyncio.run(_inner())


# --------------------------------------------------------------------------- #
# tool list is well-formed
# --------------------------------------------------------------------------- #


def test_tool_list_is_well_formed():
    tools = _list_tools()
    names = {tool.name for tool in tools}
    assert names == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description and tool.description.strip()
        schema = tool.input_schema
        assert schema is not None
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)


def test_only_path_and_preset_tools_require_arguments():
    schemas = {tool.name: tool.input_schema for tool in _list_tools()}
    assert schemas["list_presets"].get("required", []) == []
    assert schemas["capabilities"].get("required", []) == []
    assert schemas["get_preset"]["required"] == ["preset_id"]
    assert schemas["validate_recording"]["required"] == ["path"]
    assert schemas["analyze_recording"]["required"] == ["path"]
    assert schemas["compute_health_features"]["required"] == ["path"]
    assert set(schemas["compute_health_features"]["properties"]) == {"path", "r0_samples"}


# --------------------------------------------------------------------------- #
# list_presets
# --------------------------------------------------------------------------- #


def test_list_presets_happy_path():
    result = mcp.list_presets()
    assert result["ok"] is True
    assert result["count"] == 12
    ids = [entry["id"] for entry in result["presets"]]
    assert ids == o.list_presets()
    assert all(entry["name"] for entry in result["presets"])


def test_list_presets_via_sdk():
    result = _call("list_presets", {})
    assert result["ok"] is True and result["count"] == 12


# --------------------------------------------------------------------------- #
# get_preset
# --------------------------------------------------------------------------- #


def test_get_preset_happy_path():
    result = mcp.get_preset("food_freshness")
    assert result["ok"] is True
    preset = result["preset"]
    assert preset["id"] == "food_freshness"
    for key in ("sensors", "cadence_hz", "feature_groups", "thresholds", "r0_samples"):
        assert key in preset
    assert isinstance(preset["advisories"], list)


def test_get_preset_unknown_lists_valid_ids():
    result = mcp.get_preset("does_not_exist")
    assert result["ok"] is False
    assert "does_not_exist" in result["error"]
    assert "food_freshness" in result["error"]


def test_get_preset_rejects_empty_id():
    result = mcp.get_preset("   ")
    assert result["ok"] is False
    assert "list_presets" in result["error"]


# --------------------------------------------------------------------------- #
# validate_recording
# --------------------------------------------------------------------------- #


def test_validate_recording_csv_happy_path(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    result = mcp.validate_recording(str(path))
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["format"] == "csv"
    assert result["channels"] == ["VOC", "Alcohol"]
    assert result["sample_count"] == 40
    assert result["errors"] == []


def test_validate_recording_existing_fixture():
    result = mcp.validate_recording(FIXTURE)
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["sensor_type"] == "mox"


def test_validate_recording_missing_file_is_actionable():
    result = mcp.validate_recording("/no/such/recording.osmell")
    assert result["ok"] is False
    assert "file not found" in result["error"]
    assert "/no/such/recording.osmell" in result["error"]
    assert "Traceback" not in json.dumps(result)


def test_validate_recording_directory_is_actionable(tmp_path):
    result = mcp.validate_recording(str(tmp_path))
    assert result["ok"] is False
    assert "directory" in result["error"]


def test_validate_recording_unsupported_extension(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("hello")
    result = mcp.validate_recording(str(p))
    assert result["ok"] is False
    assert "unsupported" in result["error"]


def test_validate_recording_header_only_csv_is_invalid_not_raised(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("VOC,Alcohol\n")
    result = mcp.validate_recording(str(p))
    assert result["ok"] is True
    assert result["valid"] is False
    assert result["errors"]
    assert all(isinstance(e, str) and e for e in result["errors"])


def test_validate_recording_osmell_roundtrip(tmp_path):
    session = o.ingest_file(FIXTURE)
    osmell = tmp_path / "run.osmell"
    osmell.write_bytes(o.build_osmell(session.file))
    result = mcp.validate_recording(str(osmell))
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["format"] == "osmell"
    assert result["sample_count"] == len(session.file.time)


# --------------------------------------------------------------------------- #
# analyze_recording
# --------------------------------------------------------------------------- #


def test_analyze_recording_happy_path(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    result = mcp.analyze_recording(str(path))
    assert result["ok"] is True
    assert result["sensor_type"] == "mox"
    assert len(result["features"]) == 2
    for block in result["features"]:
        assert block["channel"] in ("VOC", "Alcohol")
        assert "r0" in block and "dead" in block
    assert result["quality"]["badge"] in ("Excellent", "Good", "Fair", "Poor", "Unknown")


def test_analyze_recording_returns_quality_for_fixture():
    result = mcp.analyze_recording(FIXTURE)
    assert result["ok"] is True
    assert len(result["features"]) == 6
    assert result["quality"]["total"] is not None


def test_analyze_recording_missing_file_is_actionable():
    result = mcp.analyze_recording("/nope/x.csv")
    assert result["ok"] is False
    assert isinstance(result["error"], str) and result["error"]


# --------------------------------------------------------------------------- #
# compute_health_features
# --------------------------------------------------------------------------- #


def test_compute_health_features_happy_path(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    result = mcp.compute_health_features(str(path))
    assert result["ok"] is True
    assert result["sampling_rate_hz"] == 10.0
    assert len(result["channels"]) == 2
    for block in result["channels"]:
        for key in ("drift_rate", "sensitivity_decay", "noise_floor", "hysteresis"):
            assert isinstance(block[key], float)
        assert isinstance(block["r0_window_samples"], int)


def test_compute_health_features_r0_contract_default(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv", n=40)
    result = mcp.compute_health_features(str(path))
    # clamp(floor(0.15 * 40), 5, 30) == 6
    assert result["channels"][0]["r0_window_samples"] == 6
    assert result["r0_contract"]["declared_r0_samples"] is None


def test_compute_health_features_declared_window_wins(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    result = mcp.compute_health_features(str(path), r0_samples=12)
    assert result["ok"] is True
    assert all(block["r0_window_samples"] == 12 for block in result["channels"])


def test_compute_health_features_rejects_bad_window(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    for bad in (0, -3, "ten"):
        result = mcp.compute_health_features(str(path), r0_samples=bad)
        assert result["ok"] is False
        assert isinstance(result["error"], str) and result["error"]


def test_compute_health_features_states_verdict_limitation(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    result = mcp.compute_health_features(str(path))
    assert result["verdict"] is None
    assert result["verdict_available"] is False
    assert "Rust" in result["verdict_limitation"]
    assert "opensmell-rs" in result["verdict_limitation"]


def test_compute_health_features_reports_declared_sampling_rate():
    result = mcp.compute_health_features(FIXTURE)
    assert result["ok"] is True
    # The cinnamon fixture carries no time column, so the honest source is
    # "synthetic", not a measured "manifest" rate.
    assert result["sampling_rate_source"] == "synthetic"
    assert any("sampling rate" in w for w in result["warnings"])


# --------------------------------------------------------------------------- #
# capabilities reflects what is implemented
# --------------------------------------------------------------------------- #


def test_capabilities_happy_path():
    result = mcp.capabilities()
    assert result["ok"] is True
    assert result["server"] == "opensmell"
    assert result["transport"] == "stdio"
    assert {t["name"] for t in result["tools"]} == EXPECTED_TOOLS


def test_capabilities_declares_limits():
    result = mcp.capabilities()
    cannot = result["cannot"]
    for key in ("write", "network", "hardware", "verdict"):
        assert key in cannot and cannot[key]
    assert "Rust" in cannot["verdict"]
    assert "opensmell-rs" in cannot["verdict"]
    assert result["can"]["read_files"] == [".osmell", ".csv", ".txt"]


def test_capabilities_matches_registered_tools():
    declared = {t["name"] for t in mcp.capabilities()["tools"]}
    registered = {tool.name for tool in _list_tools()}
    assert declared == registered


# --------------------------------------------------------------------------- #
# contract: errors are strings, never exceptions, and nothing is written
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("get_preset", {"preset_id": ""}),
        ("validate_recording", {"path": "/no/such/file.csv"}),
        ("analyze_recording", {"path": "/no/such/file.csv"}),
        ("compute_health_features", {"path": "/no/such/file.csv"}),
        ("compute_health_features", {"path": "/no/such/file.csv", "r0_samples": -1}),
    ],
)
def test_error_paths_are_actionable_json(name, arguments):
    result = _call(name, arguments)
    assert result["ok"] is False
    assert isinstance(result["error"], str) and result["error"]
    assert "Traceback" not in json.dumps(result)


def test_valid_tools_are_json_serializable(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    for name, arguments in [
        ("list_presets", {}),
        ("get_preset", {"preset_id": "food_freshness"}),
        ("validate_recording", {"path": str(path)}),
        ("analyze_recording", {"path": str(path)}),
        ("compute_health_features", {"path": str(path)}),
        ("capabilities", {}),
    ]:
        json.dumps(_call(name, arguments))


def test_tools_do_not_write_to_disk(tmp_path):
    path = _write_tiny_csv(tmp_path / "tiny.csv")
    before = sorted(os.listdir(tmp_path))
    mcp.validate_recording(str(path))
    mcp.analyze_recording(str(path))
    mcp.compute_health_features(str(path))
    mcp.list_presets()
    mcp.capabilities()
    assert sorted(os.listdir(tmp_path)) == before
