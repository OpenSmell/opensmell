"""Tests for the configuration presets and their loader.

Two kinds of test live here and the distinction matters:

* tests about the twelve shipped files — that they load, that their numbers agree
  with the constants table and with the SDK's own timing and feature machinery,
  and that the honesty markers (`provisional:`, `basis:`) still cover what they
  claim to cover;
* tests about the loader — that a malformed file produces an error naming the
  field, because a configuration loader that silently accepts a broken file is
  worse than no loader.

The malformed-file tests write real YAML to `tmp_path` and load it by path, so
they exercise the same code path a user hits with their own file.
"""

import json
import re

import pytest

from opensmell import presets as P
from opensmell.constants import clean_air_ratio, is_power_law_sensor, sensor_gases, sensor_models
from opensmell.mox.timing import min_detectable_duration, timing_report
from opensmell.presets_cli import main

EXPECTED_IDS = [
    "agriculture_storage",
    "breath_markers",
    "cleaning_verification",
    "food_freshness",
    "gas_leak_fire_smoke",
    "gas_purity_contamination",
    "industrial_process_fault",
    "mould_damp",
    "perfume_authenticity",
    "pest_rodent_detection",
    "ripeness_storage_aging",
    "voc_air_quality",
]

MINIMAL = """\
id: {id}
name: Test preset
description: A minimal valid preset.
sensor_type: mox
{cadence}cadence_rationale: >
    A probe preset; the cadence is chosen to make the loader tests readable.
sensors:
  - id: mq135
    model: MQ-135
    role: primary
    targets: [NH4]
    clean_air_ratio: 3.6
feature_groups: [global]
baseline:
  duration_s: 1800
thresholds:
  dead_noise_ceiling: 0.02
  drift_abs_floor: 0.001
  drift_rel_ceiling: 0.01
  basis: >
    SDK defaults, kept unchanged; a probe preset has no reason to move them.
notes: >
    PROBE_NOTES
"""


def write(tmp_path, body, name="probe.yaml"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def shipped():
    return P.load_all_presets()


@pytest.fixture(scope="module")
def gas_leak():
    """One multi-channel preset, as a fixture for the field-path tests."""
    return P.load_preset("gas_leak_fire_smoke").to_dict()


# ---------------------------------------------------------------------------
# The shipped files.
# ---------------------------------------------------------------------------


class TestShippedPresets:
    def test_twelve_presets_ship(self):
        assert P.list_presets() == EXPECTED_IDS

    def test_every_use_case_is_covered_once(self):
        # One file per requested use case, and the id is the file stem, so this
        # also proves no duplicate configuration under two names.
        assert len(set(P.list_presets())) == 12

    def test_every_preset_loads(self, shipped):
        assert sorted(shipped) == EXPECTED_IDS
        for preset in shipped.values():
            assert preset.id
            assert preset.name
            assert preset.description.strip()
            assert preset.source_path.endswith(f"{preset.id}.yaml")

    def test_cadence_is_mandatory_and_positive(self, shipped):
        for preset in shipped.values():
            assert preset.sampling_rate_hz > 0
            assert preset.cadence_period_s > 0
            assert preset.sampling_rate_hz * preset.cadence_period_s == pytest.approx(1.0)

    def test_every_preset_explains_its_cadence(self, shipped):
        # The user asked for the reasoning behind each cadence to be written
        # down; a preset with an empty rationale is a regression of that request.
        for preset in shipped.values():
            assert len(preset.cadence_rationale.strip()) > 40, preset.id

    def test_every_sensor_is_justified(self, shipped):
        for preset in shipped.values():
            for sensor in preset.sensors:
                assert len(sensor.note.strip()) > 40, f"{preset.id}/{sensor.id}"

    def test_models_exist_in_the_constants_table(self, shipped):
        for preset in shipped.values():
            for sensor in preset.sensors:
                assert sensor.model in sensor_models(), f"{preset.id}/{sensor.id}"

    def test_target_gases_are_tabulated_for_their_model(self, shipped):
        for preset in shipped.values():
            for sensor in preset.sensors:
                for gas in sensor.targets:
                    assert gas in sensor_gases(sensor.model), f"{preset.id}/{gas}"

    def test_declared_clean_air_ratios_equal_the_constants_table(self, shipped):
        # The loader enforces this too; asserting it here means a future edit to
        # the validator cannot quietly let a wrong ratio through.
        checked = 0
        for preset in shipped.values():
            for sensor in preset.sensors:
                if is_power_law_sensor(sensor.model):
                    assert sensor.clean_air_ratio == pytest.approx(clean_air_ratio(sensor.model))
                    checked += 1
        assert checked > 40

    def test_power_law_channels_declare_a_ratio(self, shipped):
        # A channel the constants table can check must be checked; omitting the
        # ratio where one exists would let a wrong element into a preset.
        for preset in shipped.values():
            for sensor in preset.sensors:
                if is_power_law_sensor(sensor.model):
                    assert sensor.clean_air_ratio is not None, f"{preset.id}/{sensor.id}"

    def test_untabulated_channels_declare_no_ratio(self, shipped):
        for preset in shipped.values():
            for sensor in preset.sensors:
                if not is_power_law_sensor(sensor.model):
                    assert sensor.clean_air_ratio is None, f"{preset.id}/{sensor.id}"

    def test_channel_ids_are_unique_within_a_preset(self, shipped):
        for preset in shipped.values():
            ids = [s.id for s in preset.sensors]
            assert len(ids) == len(set(ids)), preset.id

    def test_every_preset_has_a_primary_channel(self, shipped):
        for preset in shipped.values():
            roles = [s.role for s in preset.sensors]
            assert "primary" in roles, preset.id

    def test_roles_are_known(self, shipped):
        for preset in shipped.values():
            for sensor in preset.sensors:
                assert sensor.role in P.SENSOR_ROLES

    def test_feature_groups_are_real_groups_that_produce_features(self, shipped):
        for preset in shipped.values():
            assert preset.feature_groups, preset.id
            names = preset.feature_names()
            for group in preset.feature_groups:
                assert group in P.FEATURE_GROUPS
                marker = P.FEATURE_GROUP_MARKERS[group]
                assert any(marker in n for n in names), f"{preset.id}/{group}"

    def test_feature_groups_are_unique_and_ordered_as_the_extractor_orders_them(self, shipped):
        for preset in shipped.values():
            assert len(set(preset.feature_groups)) == len(preset.feature_groups), preset.id
            order = [P.FEATURE_GROUPS.index(g) for g in preset.feature_groups]
            assert order == sorted(order), preset.id

    def test_context_columns_are_known_hints(self, shipped):
        from opensmell.types import CONTEXT_COLUMN_HINTS

        for preset in shipped.values():
            for column in preset.context_columns:
                assert column in CONTEXT_COLUMN_HINTS, f"{preset.id}/{column}"

    def test_r0_samples_follows_duration_and_cadence(self, shipped):
        for preset in shipped.values():
            expected = round(preset.baseline.duration_s * preset.sampling_rate_hz)
            assert preset.r0_samples == expected, preset.id
            assert preset.r0_samples >= 2, preset.id

    def test_baseline_durations_are_the_documented_protocol(self, shipped):
        # 30-minute baseline, 5-minute settling, 48-hour new-sensor warm-up,
        # all from HARDWARE.md. A preset may lengthen them with a stated reason
        # but must not shorten them silently.
        for preset in shipped.values():
            assert preset.baseline.duration_s >= P.DEFAULT_BASELINE_SECONDS, preset.id
            assert preset.baseline.settling_s >= P.OPERATIONAL_SETTLING_SECONDS, preset.id
            assert preset.baseline.warmup_s >= P.NEW_SENSOR_WARMUP_SECONDS, preset.id
            assert preset.baseline.method in ("median-of-leading-window",)

    def test_thresholds_are_positive_and_bounded(self, shipped):
        for preset in shipped.values():
            t = preset.thresholds
            assert 0 < t.dead_noise_ceiling
            assert 0 < t.drift_abs_floor
            assert 0 < t.drift_rel_ceiling

    def test_threshold_overrides_carry_a_basis(self, shipped):
        for preset in shipped.values():
            t = preset.thresholds
            deviates = {
                k: getattr(t, k) for k, v in P.VERDICT_THRESHOLD_DEFAULTS.items()
                if getattr(t, k) != v
            }
            if deviates:
                assert t.basis.strip(), f"{preset.id} deviates on {deviates}"
            else:
                assert t.basis.strip(), f"{preset.id} should say why it kept the defaults"

    def test_low_rate_presets_do_not_request_kinetic_features(self, shipped):
        # A preset at 1 sample/minute asking for decay times would be reporting
        # the sample spacing back as chemistry.
        for preset in shipped.values():
            if preset.sampling_rate_hz < 1.0:
                assert not preset.requests_cadence_sensitive(), preset.id

    def test_min_detectable_duration_agrees_with_the_timing_module(self, shipped):
        for preset in shipped.values():
            assert preset.min_detectable_duration_s == pytest.approx(
                min_detectable_duration(preset.sampling_rate_hz)
            )

    def test_timing_report_agrees_with_the_timing_module(self, shipped):
        for preset in shipped.values():
            direct = timing_report(
                sampling_rate_hz=preset.sampling_rate_hz, n_channels=preset.n_channels
            )
            assert preset.timing.min_detectable_duration_s == pytest.approx(
                direct.min_detectable_duration_s
            )
            assert preset.timing.flags == direct.flags

    def test_cadence_matches_a_documented_rate(self, shipped):
        # Every cadence is one of the rates this workspace has grounds for:
        # 10 Hz kinetics, 2 Hz production, 1 Hz corpora, 1/60 Hz spoilage.
        assert {round(p.sampling_rate_hz, 4) for p in shipped.values()} <= {
            10.0, 2.0, 1.0, 0.5, round(1 / 60, 4),
        }

    def test_safety_preset_has_the_loosest_budget_of_the_monitored_ones(self, shipped):
        # Encodes the design intent rather than a measurement: a life-safety
        # alarm may be called weekly, an IAQ unit may be called twice a day.
        gas = shipped["gas_leak_fire_smoke"].false_alarm_budget
        iaq = shipped["voc_air_quality"].false_alarm_budget
        assert gas.alarms_per_day < iaq.alarms_per_day

    def test_bench_presets_have_no_alarm_budget(self, shipped):
        # A per-day alarm rate is meaningless for a measurement made once by a
        # person; setting one would imply a monitoring duty that does not exist.
        for preset_id in ("gas_purity_contamination", "breath_markers", "cleaning_verification"):
            assert shipped[preset_id].false_alarm_budget is None, preset_id

    def test_every_budget_is_positive_and_justified(self, shipped):
        for preset in shipped.values():
            budget = preset.false_alarm_budget
            if budget is None:
                continue
            assert budget.alarms_per_day > 0, preset.id
            assert len(budget.rationale.strip()) > 40, preset.id

    def test_provisional_entries_resolve_to_real_fields(self, shipped):
        for preset in shipped.values():
            data = preset.to_dict()
            for path in preset.provisional:
                assert P.resolve_field(data, path) is not None, f"{preset.id}:{path}"

    def test_files_that_flag_unvalidated_numbers_flag_them_in_provisional(self):
        # The user's requirement: a figure the author was not confident about
        # must be marked. The prose marker and the machine-readable
        # `provisional:` list have to agree, because the list is what a tool
        # reads and the prose is what a reader sees.
        markers = ("NEEDS EMPIRICAL VALIDATION", "not a measured operating point")
        for path in P.preset_files():
            text = path.read_text(encoding="utf-8")
            flagged = any(marker in text for marker in markers)
            preset = P.load_preset(str(path))
            assert flagged == bool(preset.provisional), path.name

    def test_ephylene_gap_is_stated_where_it_applies(self, shipped):
        # No ethylene-selective channel exists in the constants table, so the two
        # ripening/smell presets that would want one have to say so out loud.
        for preset_id in ("food_freshness", "ripeness_storage_aging"):
            assert "ethylene" in shipped[preset_id].notes.lower(), preset_id
            for sensor in shipped[preset_id].sensors:
                assert "ethylene" not in [g.lower() for g in sensor.targets]

    def test_notes_are_present_everywhere(self, shipped):
        for preset in shipped.values():
            assert len(preset.notes.strip()) > 80, preset.id

    def test_sensor_descriptor_round_trips_into_the_data_model(self, shipped):
        # A preset is only useful if its array can be written into the .osmell
        # data model, so this goes all the way to a serialized bundle.
        from opensmell.io import build_osmell, parse_osmell
        from opensmell.types import OsmellFile, OsmellManifest

        for preset in shipped.values():
            descriptor = preset.to_sensor_descriptor()
            assert descriptor.sensor_type == preset.sensor_type
            assert descriptor.sampling_rate_hz == pytest.approx(preset.sampling_rate_hz)
            assert [c.id for c in descriptor.channels] == [s.id for s in preset.sensors]
            file = OsmellFile(
                manifest=OsmellManifest(sensor=descriptor),
                time=[0, 1, 2],
                data={s.id: [1.0, 2.0, 3.0] for s in preset.sensors},
            )
            parsed = parse_osmell(build_osmell(file))
            assert [c.id for c in parsed.manifest.sensor.channels] == [
                s.id for s in preset.sensors
            ]
            assert parsed.manifest.sensor.sampling_rate_hz == pytest.approx(
                preset.sampling_rate_hz
            )

    def test_to_dict_is_json_serialisable(self, shipped):
        for preset in shipped.values():
            json.dumps(preset.to_dict())


# ---------------------------------------------------------------------------
# The loader.
# ---------------------------------------------------------------------------


class TestLoaderHappyPath:
    def test_load_by_id(self):
        assert P.load_preset("food_freshness").id == "food_freshness"

    def test_load_by_path(self, tmp_path):
        path = write(tmp_path, MINIMAL.format(id="probe", cadence="cadence_hz: 1\n"))
        assert P.load_preset(path).id == "probe"

    def test_load_all_returns_id_keyed_mapping(self, shipped):
        assert all(isinstance(k, str) and v.id == k for k, v in shipped.items())

    def test_period_and_rate_forms_agree(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 2\n")
        body += "cadence_period_s: 0.5\n"
        assert P.load_preset(write(tmp_path, body)).sampling_rate_hz == 2.0

    def test_yml_extension_is_accepted(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n")
        assert P.load_preset(write(tmp_path, body, "probe.yml")).id == "probe"

    def test_folded_block_scalar_joins_lines(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "PROBE_NOTES", "folded line one\n    folded line two"
        )
        assert "line one folded line two" in P.load_preset(write(tmp_path, body)).notes

    def test_literal_block_scalar_keeps_newlines(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "notes: >\n    PROBE_NOTES", "notes: |\n    first\n    second"
        )
        assert P.load_preset(write(tmp_path, body)).notes.strip() == "first\nsecond"

    def test_quoted_hash_is_not_a_comment(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "name: Test preset", 'name: "Test # preset"'
        )
        assert P.load_preset(write(tmp_path, body)).name == "Test # preset"

    def test_scientific_notation_is_accepted(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1e0\n")
        assert P.load_preset(write(tmp_path, body)).sampling_rate_hz == 1.0


class TestLoaderErrors:
    def test_missing_cadence_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="")
        with pytest.raises(P.PresetError, match="cadence"):
            P.load_preset(write(tmp_path, body))

    def test_non_positive_cadence_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_period_s: 0\n")
        with pytest.raises(P.PresetError, match="cadence"):
            P.load_preset(write(tmp_path, body))

    def test_negative_cadence_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: -1\n")
        with pytest.raises(P.PresetError, match="cadence"):
            P.load_preset(write(tmp_path, body))

    def test_disagreeing_cadence_forms_are_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n")
        body += "cadence_period_s: 60\n"
        with pytest.raises(P.PresetError, match="cadence"):
            P.load_preset(write(tmp_path, body))

    def test_unknown_feature_group_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "feature_groups: [global]", "feature_groups: [global, wizardry]"
        )
        with pytest.raises(P.PresetError, match="wizardry"):
            P.load_preset(write(tmp_path, body))

    def test_duplicate_feature_group_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "feature_groups: [global]", "feature_groups: [global, global]"
        )
        with pytest.raises(P.PresetError, match="duplicate"):
            P.load_preset(write(tmp_path, body))

    def test_no_feature_groups_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "feature_groups: [global]", "feature_groups: []"
        )
        with pytest.raises(P.PresetError, match="feature_groups"):
            P.load_preset(write(tmp_path, body))

    def test_unknown_sensor_model_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "model: MQ-135", "model: MQ-999"
        )
        with pytest.raises(P.PresetError, match="MQ-999"):
            P.load_preset(write(tmp_path, body))

    def test_untabulated_target_gas_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "targets: [NH4]", "targets: [Unobtainium]"
        )
        with pytest.raises(P.PresetError, match="Unobtainium"):
            P.load_preset(write(tmp_path, body))

    def test_wrong_clean_air_ratio_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "clean_air_ratio: 3.6", "clean_air_ratio: 4.0"
        )
        with pytest.raises(P.PresetError, match="clean_air_ratio"):
            P.load_preset(write(tmp_path, body))

    def test_missing_clean_air_ratio_for_a_tabulated_model_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "    clean_air_ratio: 3.6\n", ""
        )
        with pytest.raises(P.PresetError, match="clean_air_ratio"):
            P.load_preset(write(tmp_path, body))

    def test_duplicate_sensor_id_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "feature_groups: [global]",
            "  - id: mq135\n    model: MQ-135\n    role: primary\n"
            "feature_groups: [global]",
        )
        with pytest.raises(P.PresetError, match="mq135"):
            P.load_preset(write(tmp_path, body))

    def test_unknown_role_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "role: primary", "role: oracle"
        )
        with pytest.raises(P.PresetError, match="oracle"):
            P.load_preset(write(tmp_path, body))

    def test_unknown_top_level_key_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n") + "colour: blue\n"
        with pytest.raises(P.PresetError, match="colour"):
            P.load_preset(write(tmp_path, body))

    def test_non_positive_baseline_duration_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "duration_s: 1800", "duration_s: 0"
        )
        with pytest.raises(P.PresetError, match="duration_s"):
            P.load_preset(write(tmp_path, body))

    def test_conflicting_duration_and_sample_count_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "  duration_s: 1800", "  duration_s: 1800\n  r0_samples: 5000"
        )
        with pytest.raises(P.PresetError, match="r0_samples"):
            P.load_preset(write(tmp_path, body))

    def test_tuned_threshold_without_a_basis_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "  dead_noise_ceiling: 0.02", "  dead_noise_ceiling: 0.01"
        ).replace("  basis: >\n", "  x_basis: >\n")
        with pytest.raises(P.PresetError, match="basis"):
            P.load_preset(write(tmp_path, body))

    def test_provisional_entry_that_resolves_nothing_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n")
        body += "provisional:\n  - cadence_hz\n  - not_a_field\n"
        with pytest.raises(P.PresetError, match="not_a_field"):
            P.load_preset(write(tmp_path, body))

    def test_id_must_match_the_file_name(self, tmp_path):
        body = MINIMAL.format(id="mismatched", cadence="cadence_hz: 1\n")
        with pytest.raises(P.PresetError, match="does not match the file name"):
            P.load_preset(write(tmp_path, body))

    def test_upper_case_id_is_an_error(self, tmp_path):
        body = MINIMAL.format(id="Not_Snake", cadence="cadence_hz: 1\n")
        with pytest.raises(P.PresetError, match="snake_case"):
            P.load_preset(write(tmp_path, body, "Not_Snake.yaml"))

    def test_unknown_preset_name_lists_what_exists(self):
        with pytest.raises(P.PresetNotFoundError, match="Available"):
            P.load_preset("no_such_preset")

    def test_missing_file_path_is_reported(self, tmp_path):
        with pytest.raises(P.PresetNotFoundError):
            P.load_preset(str(tmp_path / "absent.yaml"))

    def test_tab_indentation_is_rejected(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "  duration_s: 1800", "\tduration_s: 1800"
        )
        with pytest.raises(P.PresetError):
            P.load_preset(write(tmp_path, body))

    def test_flow_mapping_is_rejected(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n").replace(
            "  duration_s: 1800", "  duration_s: {n: 1}"
        )
        with pytest.raises(P.PresetError):
            P.load_preset(write(tmp_path, body))

    def test_empty_file_is_rejected(self, tmp_path):
        with pytest.raises(P.PresetError):
            P.load_preset(write(tmp_path, "# nothing here\n"))

    def test_anchor_is_rejected(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n") + "notes: &a hello\n"
        with pytest.raises(P.PresetError):
            P.load_preset(write(tmp_path, body))

    def test_stray_line_without_a_key_is_rejected(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="cadence_hz: 1\n") + "loose text\n"
        with pytest.raises(P.PresetError):
            P.load_preset(write(tmp_path, body))

    def test_error_message_names_the_file_and_line(self, tmp_path):
        body = MINIMAL.format(id="probe", cadence="")
        body += "sensors: notalist\n"
        with pytest.raises(P.PresetError) as excinfo:
            P.load_preset(write(tmp_path, body))
        message = str(excinfo.value)
        assert "probe.yaml" in message
        assert re.search(r"probe\.yaml:\d+", message)


class TestResolveField:
    def test_nested_key(self, gas_leak):
        assert P.resolve_field(gas_leak, "baseline.duration_s") == 1800.0

    def test_list_index(self, gas_leak):
        assert P.resolve_field(gas_leak, "sensors[0].model") == "MQ-2"

    def test_bare_list_index(self, gas_leak):
        assert P.resolve_field(gas_leak, "feature_groups[0]") == "device_agnostic"

    def test_missing_key(self, gas_leak):
        with pytest.raises(KeyError, match="absent"):
            P.resolve_field(gas_leak, "absent")

    def test_index_out_of_range(self, gas_leak):
        with pytest.raises(KeyError, match="out of range"):
            P.resolve_field(gas_leak, "sensors[99].model")

    def test_malformed_index(self, gas_leak):
        with pytest.raises(KeyError, match="not a list index"):
            P.resolve_field(gas_leak, "sensors[x].model")

    def test_index_into_a_mapping(self, gas_leak):
        with pytest.raises(KeyError, match="not a list index"):
            P.resolve_field(gas_leak, "baseline[0]")

    def test_unclosed_bracket(self, gas_leak):
        with pytest.raises(KeyError, match="unclosed"):
            P.resolve_field(gas_leak, "sensors[0")


class TestPresetsDirectory:
    def test_ships_in_the_source_tree(self):
        assert P.presets_dir().is_dir()
        assert len(list(P.presets_dir().glob("*.yaml"))) == 12

    def test_environment_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(P.PRESET_DIR_ENV, str(tmp_path))
        with pytest.raises(P.PresetNotFoundError, match="food_freshness"):
            P.load_preset("food_freshness")

        body = MINIMAL.format(id="elsewhere", cadence="cadence_hz: 1\n")
        write(tmp_path, body, "elsewhere.yaml")
        assert P.load_preset("elsewhere").id == "elsewhere"

    def test_bogus_override_falls_back_to_the_shipped_presets(self, tmp_path, monkeypatch):
        monkeypatch.setenv(P.PRESET_DIR_ENV, str(tmp_path / "absent"))
        assert P.list_presets() == EXPECTED_IDS

    def test_empty_override_directory_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setenv(P.PRESET_DIR_ENV, str(tmp_path))
        with pytest.raises(P.PresetNotFoundError, match=str(tmp_path)):
            P.list_presets()


# ---------------------------------------------------------------------------
# The command line.
# ---------------------------------------------------------------------------


class TestCli:
    def test_list_succeeds_and_names_every_preset(self, capsys):
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        for preset_id in EXPECTED_IDS:
            assert preset_id in out
        assert "12 presets" in out

    def test_list_json(self, capsys):
        assert main(["list", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert [p["id"] for p in payload] == EXPECTED_IDS

    def test_show_renders_the_preset(self, capsys):
        assert main(["show", "food_freshness"]) == 0
        out = capsys.readouterr().out
        assert "food_freshness" in out
        assert "cadence" in out
        assert "clean air ratio" in out

    def test_show_json_round_trips(self, capsys):
        assert main(["show", "voc_air_quality", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["id"] == "voc_air_quality"
        assert len(payload["sensors"]) >= 4
        assert payload["cadence_hz"] == 1.0

    def test_validate_all_succeeds(self, capsys):
        assert main(["validate"]) == 0
        out = capsys.readouterr().out
        assert "12/12 presets valid" in out
        assert "FAIL" not in out

    def test_validate_reports_advisories_without_failing(self, capsys):
        assert main(["validate"]) == 0
        out = capsys.readouterr().out
        assert "cadence_below_1hz" in out

    def test_validate_strict_turns_advisories_into_failures(self, capsys):
        # The four per-minute presets carry a timing advisory by design, so
        # --strict has to fail: that is what makes the flag worth having.
        assert main(["validate", "--strict"]) == 1
        assert "failed" in capsys.readouterr().out

    def test_validate_one_preset(self, capsys):
        assert main(["validate", "gas_leak_fire_smoke"]) == 0
        out = capsys.readouterr().out
        assert "ok    gas_leak_fire_smoke" in out

    def test_field_prints_a_value(self, capsys):
        assert main(["field", "food_freshness", "baseline.duration_s"]) == 0
        assert capsys.readouterr().out.strip() == "1800.0"

    def test_field_handles_an_index(self, capsys):
        assert main(["field", "gas_leak_fire_smoke", "sensors[1].model"]) == 0
        assert capsys.readouterr().out.strip() == "MQ-4"

    def test_field_reports_a_missing_field(self, capsys):
        assert main(["field", "food_freshness", "nope"]) == 1
        assert "nope" in capsys.readouterr().err

    def test_unknown_preset_is_a_clean_error(self, capsys):
        assert main(["show", "nope"]) == 1
        err = capsys.readouterr().err
        assert "opensmell-presets:" in err
        assert "Available" in err

    def test_malformed_file_is_a_clean_error(self, tmp_path, capsys):
        path = write(tmp_path, MINIMAL.format(id="probe", cadence=""))
        assert main(["validate", path]) == 1
        assert "cadence" in capsys.readouterr().err

    def test_no_command_is_a_usage_error(self, capsys):
        assert main([]) == 2
        assert "usage" in capsys.readouterr().err.lower()

    def test_argv_accepts_a_sequence(self):
        assert main(("list",)) == 0