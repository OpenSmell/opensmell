"""Smellability parity tests.

Mirrors the web suite `osmograph-web/lib/smellability/__tests__/{chain,groups,
ontology,transport,provisional}.test.ts` 1:1. Every assertion here exists on the
TypeScript side; keep them in lockstep when the chain's behavior changes.
"""

from __future__ import annotations

import pytest

from opensmell.mox.smellability import (
    COMPOSITE_BY_ID,
    COMPOUND_BY_ID,
    MAX_SUBSTANCES,
    MOX_BOUNDARIES,
    EnrichedBoilingPoint,
    EnrichedChemical,
    IncidentFluxInput,
    build_provisional_chemical,
    delta_h_vap_trouton,
    diffusion_volume_from_mw,
    dominant_percept,
    estimate_vapor_pressure_from_boiling_point,
    headspace_ppm_band,
    infer_functional_groups,
    kekule_to_aromatic,
    percepts_for,
    perceptual_summary,
    relevant_boundaries,
    run_chemical_verdict,
    run_class_verdict,
    run_composite_verdict,
    signal_ratio_vs_ref,
    vapor_pressure_antoine,
    vapor_pressure_clausius_clapeyron,
    volatility_label,
)


def chem(id_: str):
    return COMPOUND_BY_ID[id_]


def _same(a: list, b: list) -> bool:
    return sorted(a) == sorted(b)


def _groups(smiles: str) -> list:
    return sorted(infer_functional_groups(smiles))


# ---------------------------------------------------------------- band tables

class TestBandTables:
    def test_volatility_bands_match_the_science_doc(self):
        assert volatility_label(10000) == "very high"
        assert volatility_label(9999.9) == "high"
        assert volatility_label(100) == "moderate"
        assert volatility_label(1) == "low"
        assert volatility_label(0.5) == "negligible"
        assert volatility_label(None) == "unknown"

    def test_headspace_ppm_bands_match_the_1_ppm_mox_floor(self):
        assert headspace_ppm_band(1000) == "strong"
        assert headspace_ppm_band(999.9) == "moderate"
        assert headspace_ppm_band(100) == "moderate"
        assert headspace_ppm_band(10) == "weak"
        assert headspace_ppm_band(1) == "marginal"
        assert headspace_ppm_band(0.9) == "none"

    def test_array_capacity_reconciles_with_canonical_table_2(self):
        assert MAX_SUBSTANCES[3] == 6
        assert MAX_SUBSTANCES[6] == 40
        assert MAX_SUBSTANCES[12] == 200
        assert MAX_SUBSTANCES[24] == 10000


# ----------------------------------------------------------- chemical verdicts

class TestChemicalVerdicts:
    def test_ethanol_green_strong_fast_high_confidence(self):
        v = run_chemical_verdict(chem("ethanol"))
        assert v.verdict == "green"
        assert v.signal_strength == "strong"
        assert v.response_speed == "fast"
        assert v.confidence == "high"

    def test_acetone_green_strong_fast(self):
        v = run_chemical_verdict(chem("acetone"))
        assert v.verdict == "green"
        assert v.signal_strength == "strong"
        assert v.response_speed == "fast"

    def test_hydrogen_sulfide_gas_green_strong_fast(self):
        v = run_chemical_verdict(chem("hydrogen-sulfide"))
        assert v.verdict == "green"
        assert v.signal_strength == "strong"
        assert v.response_speed == "fast"

    def test_co2_is_a_hard_stop_reactivity_step_red(self):
        v = run_chemical_verdict(chem("carbon-dioxide"))
        assert v.verdict == "red"
        reactivity = next(s for s in v.steps if s.id == "reactivity")
        assert reactivity.verdict == "red"

    def test_n2_is_a_hard_stop(self):
        assert run_chemical_verdict(chem("nitrogen")).verdict == "red"

    def test_isoamyl_acetate_green_strong_medium(self):
        v = run_chemical_verdict(chem("isoamyl-acetate"))
        assert v.verdict == "green"
        assert v.signal_strength == "strong"
        assert v.confidence == "medium"

    def test_cinnamaldehyde_yellow_weak_slow(self):
        v = run_chemical_verdict(chem("cinnamaldehyde"))
        assert v.verdict == "yellow"
        assert v.signal_strength == "weak"
        assert v.response_speed == "slow"

    def test_eugenol_yellow_weak_slow(self):
        v = run_chemical_verdict(chem("eugenol"))
        assert v.verdict == "yellow"
        assert v.signal_strength == "weak"
        assert v.response_speed == "slow"

    def test_water_yellow_baseline_shift_not_analyte(self):
        v = run_chemical_verdict(chem("water"))
        assert v.verdict == "yellow"
        assert v.signal_strength == "strong"
        reactivity = next(s for s in v.steps if s.id == "reactivity")
        assert "baseline shift" in reactivity.reason

    def test_estimated_properties_downgrade_confidence_to_medium(self):
        assert run_chemical_verdict(chem("cinnamaldehyde")).confidence == "medium"
        assert run_chemical_verdict(chem("isoamyl-acetate")).confidence == "medium"


# -------------------------------------------------------- composite verdicts

class TestCompositeVerdicts:
    def test_banana_green_strong(self):
        v = run_composite_verdict(COMPOSITE_BY_ID["banana"])
        assert v.verdict == "green"
        assert v.signal_strength == "strong"

    def test_cinnamon_yellow_weak(self):
        v = run_composite_verdict(COMPOSITE_BY_ID["cinnamon"])
        assert v.verdict == "yellow"
        assert v.signal_strength == "weak"

    def test_sewer_green_h2s_dominates(self):
        v = run_composite_verdict(COMPOSITE_BY_ID["sewer"])
        assert v.verdict == "green"
        assert v.signal_strength == "strong"

    def test_gasoline_green(self):
        assert run_composite_verdict(COMPOSITE_BY_ID["gasoline"]).verdict == "green"

    def test_rotten_egg_green(self):
        assert run_composite_verdict(COMPOSITE_BY_ID["rotten-egg"]).verdict == "green"

    def test_car_exhaust_green(self):
        assert run_composite_verdict(COMPOSITE_BY_ID["car-exhaust"]).verdict == "green"

    def test_composite_weights_are_normalized_to_sum_to_1(self):
        v = run_composite_verdict(COMPOSITE_BY_ID["banana"])
        assert abs(sum(c.weight_fraction for c in v.constituents) - 1) < 1e-6


class TestClassVerdicts:
    def test_alcohol_class_yellow_low_confidence_with_resolve_note(self):
        v = run_class_verdict("alcohol")
        assert v.verdict == "yellow"
        assert v.confidence == "low"
        assert "specific compound" in " ".join(v.notes)


class TestCrossCheckCapacity:
    def test_reports_canonical_distinguishable_count_for_6_sensors(self):
        from opensmell.mox.smellability import ChainOptions

        v = run_chemical_verdict(chem("ethanol"), ChainOptions(sensor_count=6))
        assert v.cross_check.max_distinguishable == 40

    def test_flags_confusable_labels_from_user_library(self):
        from opensmell.mox.smellability import ChainOptions

        v = run_chemical_verdict(
            chem("ethanol"),
            ChainOptions(sensor_count=6, library_substances=["ethanol", "hand sanitizer"]),
        )
        assert "ethanol" in v.cross_check.confusable


# ------------------------------------------------------ transport (thermo)

class TestVaporPressureAntoine:
    def _within(self, actual: float, expected: float, pct: float):
        lo = expected * (1 - pct / 100)
        hi = expected * (1 + pct / 100)
        assert lo < actual < hi, f"{actual} vs {expected} (±{pct}%)"

    def test_ethanol_about_7_87_kpa(self):
        self._within(vapor_pressure_antoine(8.20417, 1642.89, 230.3, 25), 7870, 2)

    def test_acetone_about_30_6_kpa(self):
        self._within(vapor_pressure_antoine(7.11714, 1210.595, 229.664, 25), 30600, 2)

    def test_methanol_about_16_9_kpa(self):
        self._within(vapor_pressure_antoine(8.08097, 1582.271, 239.726, 25), 16900, 2)

    def test_benzene_about_12_7_kpa(self):
        self._within(vapor_pressure_antoine(6.90565, 1211.033, 220.79, 25), 12700, 2)

    def test_isopropanol_about_6_0_kpa_looser(self):
        self._within(vapor_pressure_antoine(8.87829, 2010.33, 252.636, 25), 6020, 5)

    def test_rises_monotonically_with_temperature(self):
        a = vapor_pressure_antoine(8.20417, 1642.89, 230.3, 20)
        b = vapor_pressure_antoine(8.20417, 1642.89, 230.3, 30)
        assert b > a


class TestClausiusClapeyronAndTrouton:
    def test_water_from_boiling_point_within_order_of_magnitude_of_3_2_kpa(self):
        t_boil_k = 100 + 273.15
        p = vapor_pressure_clausius_clapeyron(298.15, t_boil_k, delta_h_vap_trouton(t_boil_k))
        assert p > 1000
        assert p < 10000

    def test_delta_h_vap_trouton_is_88x_boiling_temperature(self):
        assert abs(delta_h_vap_trouton(373.15) - 32837.2) < 0.1


class TestDiffusionAndFluxRatio:
    def test_diffusion_volume_is_1_1x_molecular_weight(self):
        assert abs(diffusion_volume_from_mw(100) - 110) < 1e-9

    def test_h2s_flux_is_about_14_5x_ethanol(self):
        ethanol = IncidentFluxInput(vapor_pressure_pa=7870, mol_weight_kg=0.04607, diffusion_volume_cm3=50.68)
        h2s = IncidentFluxInput(vapor_pressure_pa=101325, mol_weight_kg=0.03408, diffusion_volume_cm3=37.49)
        ratio = signal_ratio_vs_ref(h2s, ethanol)
        assert ratio > 13
        assert ratio < 16

    def test_zero_vapor_pressure_compound_contributes_no_flux(self):
        ethanol = IncidentFluxInput(vapor_pressure_pa=7870, mol_weight_kg=0.04607, diffusion_volume_cm3=50.68)
        inert = IncidentFluxInput(vapor_pressure_pa=0, mol_weight_kg=0.1, diffusion_volume_cm3=110)
        assert signal_ratio_vs_ref(inert, ethanol) == 0


# ------------------------------------------------------------- groups (SMILES)

class TestKekuleAromaticRingDetection:
    def test_normalises_a_linear_kekule_benzene_ring(self):
        assert kekule_to_aromatic("C1=CC=CC=C1") == "c1ccccc1"
        assert _same(_groups("C1=CC=CC=C1"), ["aromatic"])

    def test_detects_phenol_with_oh_straight_on_the_ring(self):
        assert _same(_groups("OC1=CC=CC=C1"), ["aromatic", "phenol"])
        assert _same(_groups("C1=CC=C(C=C1)O"), ["aromatic", "phenol"])

    def test_detects_guaiacol_as_phenol_plus_ether(self):
        assert _same(_groups("COC1=CC=CC=C1O"), ["aromatic", "phenol", "ether"])

    def test_does_not_mistake_anisole_methoxy_for_a_phenol(self):
        assert _same(_groups("COC1=CC=CC=C1"), ["aromatic", "ether"])

    def test_vanillin_reads_as_aromatic_aldehyde_phenol_ether(self):
        assert _same(_groups("COC1=C(C=CC(=C1)C=O)O"), ["aromatic", "aldehyde", "phenol", "ether"])

    def test_cinnamaldehyde_keeps_its_real_alkene_and_terminal_aldehyde(self):
        assert _same(_groups("C1=CC=C(C=C1)/C=C/C=O"), ["aromatic", "aldehyde", "alkene"])

    def test_benzaldehyde_reads_as_aromatic_aldehyde(self):
        assert _same(_groups("C1=CC=C(C=C1)C=O"), ["aromatic", "aldehyde"])

    def test_eugenol_kekule_form_reads_as_phenol_ether_alkene(self):
        assert _same(_groups("C=CCC1=CC(=C(C=C1)O)OC"), ["aromatic", "phenol", "ether", "alkene"])

    def test_furfural_reads_as_furan(self):
        g = infer_functional_groups("O=CC1=COC=C1")
        assert "furan" in g
        assert "aromatic" in g
        assert "aldehyde" in g
        assert _groups("O=CC1=COC=C1") != sorted(["aromatic", "aldehyde", "furan"])

    def test_styrene_keeps_its_vinyl_alkene(self):
        assert _same(_groups("C=CC1=CC=CC=C1"), ["aromatic", "alkene"])


class TestNonAromaticRings:
    def test_limonene_stays_a_terpene_alkene_not_aromatic(self):
        g = infer_functional_groups("CC1=CCC(CC1)C(C)=C")
        assert "aromatic" not in g
        assert "alkene" in g

    def test_menthol_saturated_ring_is_not_aromatic(self):
        assert "aromatic" not in infer_functional_groups("CC1CCC(C(C1)O)C(C)C")

    def test_pinene_bicyclic_ring_is_not_aromatic(self):
        assert "aromatic" not in infer_functional_groups("CC1=CCC2CC1C2(C)C")

    def test_cyclohexene_and_cyclohexane_are_not_aromatic(self):
        assert "aromatic" not in infer_functional_groups("C1CCC=CC1")
        assert "aromatic" not in infer_functional_groups("C1CCCCC1")


class TestCuratedAromaticForm:
    def test_phenol_guaiacol_furfural_in_lowercase_form(self):
        assert _same(_groups("Oc1ccccc1"), ["aromatic", "phenol"])
        assert _same(_groups("COc1ccccc1O"), ["aromatic", "phenol", "ether"])
        assert "furan" in infer_functional_groups("O=Cc1ccco1")

    def test_eugenol_in_curated_form_keeps_phenol_ether_alkene(self):
        assert _same(_groups("COc1cc(CC=C)ccc1O"), ["aromatic", "phenol", "ether", "alkene"])

    def test_cinnamaldehyde_in_curated_form_reads_the_same(self):
        assert _same(_groups("O=C/C=C/c1ccccc1"), ["aromatic", "aldehyde", "alkene"])


class TestCoreFunctionalGroups:
    def test_alcohols_ketones_acids_esters(self):
        assert "alcohol" in infer_functional_groups("CCO")
        assert "ketone" in infer_functional_groups("CC(=O)C")
        assert "carboxylic acid" in infer_functional_groups("CC(=O)O")
        assert "ester" in infer_functional_groups("CCOC(=O)C")
        assert "diketone" in infer_functional_groups("CC(=O)C(C)=O")

    def test_sulfur_chemistry(self):
        assert "thiol" in infer_functional_groups("CS")
        assert "thioether" in infer_functional_groups("CSC")
        assert "sulfur" in infer_functional_groups("CSC")

    def test_alkanes_and_hetero_atom_only_molecules(self):
        assert "alkane" in infer_functional_groups("CCCCCC")
        assert infer_functional_groups("N") == ["amine"]
        assert infer_functional_groups("S") == ["sulfur"]

    def test_empty_or_missing_smiles_stays_silent(self):
        assert infer_functional_groups(None) == []
        assert infer_functional_groups("") == []


# --------------------------------------------------------- ontology (percepts)

class TestPerceptualOntology:
    def test_isoamyl_acetate_reads_as_fruity_sweet_esters(self):
        ps = percepts_for(chem("isoamyl-acetate"))
        assert "fruity-ester" in [p.id for p in ps]

    def test_cinnamaldehyde_reads_as_spicy_balsamic(self):
        assert dominant_percept(chem("cinnamaldehyde")).id == "spicy-balsamic"

    def test_limonene_reads_as_citrus_terpenic(self):
        assert dominant_percept(chem("limonene")).id == "citrus-terpenic"

    def test_hydrogen_sulfide_reads_as_sulfurous(self):
        ps = percepts_for(chem("hydrogen-sulfide"))
        assert "sulfurous" in [p.id for p in ps]

    def test_ethanol_reads_as_alcoholic(self):
        ps = percepts_for(chem("ethanol"))
        assert "alcoholic" in [p.id for p in ps]

    def test_low_volatility_percepts_are_flagged(self):
        ps = percepts_for(chem("cinnamaldehyde"))
        assert any(p.id == "spicy-balsamic" for p in ps)


class TestMoxBoundaries:
    def test_has_canonical_boundaries_4_can_5_cannot(self):
        can = [b for b in MOX_BOUNDARIES if b.capability]
        cannot = [b for b in MOX_BOUNDARIES if not b.capability]
        assert len(can) == 4
        assert len(cannot) == 5
        assert {b.id for b in cannot} >= {"structure", "concentration", "non-redox", "trace", "mixture"}

    def test_co2_verdict_highlights_the_non_redox_boundary(self):
        v = run_chemical_verdict(chem("carbon-dioxide"))
        assert "non-redox" in relevant_boundaries(v)

    def test_composite_verdict_highlights_the_mixture_boundary(self):
        v = run_composite_verdict(COMPOSITE_BY_ID["banana"])
        assert "mixture" in relevant_boundaries(v)

    def test_class_verdict_highlights_the_structure_boundary(self):
        v = run_class_verdict("alcohol")
        assert "structure" in relevant_boundaries(v)


class TestPerceptualSummaryWording:
    def test_red_non_redox_verdict_says_chemistry_is_not_redox_active(self):
        v = run_chemical_verdict(chem("carbon-dioxide"))
        assert "not redox-active" in perceptual_summary(v, percepts_for(chem("carbon-dioxide")))

    def test_strong_green_verdict_names_the_reducing_response(self):
        v = run_chemical_verdict(chem("ethanol"))
        assert "clear reducing response" in perceptual_summary(v, percepts_for(chem("ethanol")))

    def test_low_volatility_spice_warns_about_weak_slow_signal(self):
        v = run_chemical_verdict(chem("cinnamaldehyde"))
        assert "weak, slow" in perceptual_summary(v, percepts_for(chem("cinnamaldehyde")))


# --------------------------------------------------- provisional (PubChem path)

def _enriched(**partial) -> EnrichedChemical:
    base = dict(
        name="Vanillin",
        smiles="COC1=C(C=CC(=C1)C=O)O",
        molecular_weight=152.15,
        source="pubchem",
        fetched_at="2026-01-01T00:00:00.000Z",
    )
    base.update(partial)
    return EnrichedChemical(**base)


def _bp(value_c: float) -> EnrichedBoilingPoint:
    return EnrichedBoilingPoint(value_c=value_c, source="measured", note="PubChem experimental property")


class TestBuildProvisionalChemical:
    def test_flags_everything_derived_from_pubchem_as_estimated(self):
        c = build_provisional_chemical(_enriched(), _bp(285))
        assert c.props.vapor_pressure_25.source == "estimated"
        assert c.props.vapor_pressure_25.value > 0
        assert c.props.molecular_weight.source == "measured"
        assert "phenol" in c.props.functional_groups

    def test_leaves_vapor_pressure_unknown_when_no_boiling_point(self):
        c = build_provisional_chemical(_enriched(), None)
        assert c.props.vapor_pressure_25.value is None
        assert c.props.vapor_pressure_25.source == "unknown"
        assert c.props.boiling_point.value is None

    @pytest.mark.parametrize(
        "name",
        ["N2", "O2", "CO2", "Ar", "He", "Ne", "nitrogen", "oxygen", "argon", "carbon dioxide"],
    )
    def test_inorganic_inerts_are_non_redox(self, name):
        c = build_provisional_chemical(_enriched(name=name, smiles=None), None)
        assert c.props.functional_groups == []
        assert c.props.non_redox is True
        assert c.props.redox_active is False

    @pytest.mark.parametrize("name", ["H2", "CO", "H2S", "NH3", "ammonia", "hydrogen sulfide"])
    def test_reducing_gases_and_organics_are_redox_active(self, name):
        c = build_provisional_chemical(_enriched(name=name), None)
        assert c.props.non_redox is None
        assert c.props.redox_active is True

    def test_inorganic_molecules_without_groups_are_redox_active_only_if_reducing(self):
        water = build_provisional_chemical(_enriched(name="H2O", smiles="O"), None)
        assert water.props.functional_groups == []
        assert water.props.redox_active is False
        assert water.props.non_redox is None

    def test_cites_pubchem_as_its_source(self):
        c = build_provisional_chemical(_enriched(), _bp(285))
        assert "PubChem (live lookup)" in c.source_refs


class TestEstimateVaporPressureFromBoilingPoint:
    def test_high_bp_leads_to_low_vp(self):
        ethanol = estimate_vapor_pressure_from_boiling_point(78.2)
        vanillin = estimate_vapor_pressure_from_boiling_point(285)
        assert ethanol > vanillin

    def test_matches_the_chains_estimated_branch_for_known_substance(self):
        p = estimate_vapor_pressure_from_boiling_point(78.2)
        assert p > 5000
        assert p < 50000
