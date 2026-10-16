"""Sensor technology ontology (Part 17 of the brief).

Maps sensing technology -> transduction mechanism -> observables -> latent
physical state -> response phenotype.  Deliberately NOT MOX-only: the goal is
an abstraction that can host MOX, MEMS chemiresistors, capacitive,
electrochemical, photoionisation and optical modalities so that an OpenSmell
protocol can talk about *what the sensor measures* regardless of vendor.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransductionChain:
    """One link in the sensing chain.

    technology: sensor technology name (e.g. "mox", "electrochemical")
    mechanism: physico-chemical transduction mechanism (e.g. "ionosorption
        + barrier-limited transport")
    observables: list of raw electrical observables (resistance, current,
        capacitance, voltage, impedance, photon counts...)
    latent_state: list of latent physical quantities that drive observables
        (surface coverage, barrier height, carrier density, electrode
        potential, ...)
    environment_sensitivity: which environmental variables enter directly
    model_hint: name of the model family in physics/ that applies (nullable)
    """

    technology: str
    mechanism: str
    observables: list = field(default_factory=list)
    latent_state: list = field(default_factory=list)
    environment_sensitivity: list = field(default_factory=list)
    model_hint: str = ""


SENSOR_TECHNOLOGIES: dict[str, TransductionChain] = {
    "mox": TransductionChain(
        technology="mox",
        mechanism="O2 ionosorption -> surface charge -> depletion -> barrier-limited transport",
        observables=["resistance", "conductance"],
        latent_state=["O- coverage", "band bending psi_s", "depletion width W",
                      "surface electron density n_s"],
        environment_sensitivity=["temperature", "humidity", "flow", "pO2"],
        model_hint="physics.depletion + physics.transport + physics.power_law_analysis",
    ),
    "mems_chemiresistor": TransductionChain(
        technology="mems_cmos",
        mechanism="CMOS microhotplate-based chemiresistive film, temperature-modulated",
        observables=["resistance", "temperature-modulated resistance", "impedance"],
        latent_state=["film coverage", "temperature setpoint", "modulation phase state"],
        environment_sensitivity=["temperature", "humidity"],
        model_hint="physics.temperature + physics.power_law_analysis (per-setpoint)",
    ),
    "electrochemical": TransductionChain(
        technology="electrochemical",
        mechanism="redox reaction at sensing electrode -> Faradaic current (amperometric)",
        observables=["current", "voltage", "charge"],
        latent_state=["analyte flux", "electrode potential", "diffusion layer",
                      "reaction product accumulation"],
        environment_sensitivity=["temperature", "pressure", "humidity"],
        model_hint="Faraday / diffusion-controlled current; NOT MOX depletion physics",
    ),
    "capacitive": TransductionChain(
        technology="capacitive",
        mechanism="analyte changes film dielectric constant / swelling -> capacitance",
        observables=["capacitance", "dielectric loss"],
        latent_state=["film permittivity", "swelling ratio", "absorbed mass"],
        environment_sensitivity=["temperature", "humidity"],
        model_hint="Clausius-Mossotti / swelling models (not implemented here)",
    ),
    "photoionization": TransductionChain(
        technology="photoionization",
        mechanism="UV photoionisation of VOC -> ion current proportional to concentration",
        observables=["ion current"],
        latent_state=["ion flux", "ionisation potential response: molecules with IP > lamp eV invisible"],
        environment_sensitivity=["temperature", "humidity"],
        model_hint="ionisation cross-section; sharp IP cutoff (information filter)",
    ),
    "optical_ir": TransductionChain(
        technology="optical",
        mechanism="IR absorption (non-dispersive or FTIR) -> Lambert-Beer attenuation",
        observables=["absorbance spectrum", "photocurrent"],
        latent_state=["absorber column density per band", "path length"],
        environment_sensitivity=["temperature", "pressure"],
        model_hint="Beer-Lambert (fundamental), absorption cross-sections",
    ),
    "gravimetric": TransductionChain(
        technology="gravimetric",
        mechanism="analyte mass loading shifts resonant frequency (QCM/SAW)",
        observables=["resonant frequency", "phase"],
        latent_state=["adsorbed mass", "viscoelastic loading"],
        environment_sensitivity=["temperature", "humidity"],
        model_hint="Sauerbrey / Kanazawa-Gordon",
    ),
}


def sensor_technology_catalog() -> dict:
    """Expose the catalog for reports."""
    return {k: v for k, v in SENSOR_TECHNOLOGIES.items()}


def transduction_chain(technology: str) -> TransductionChain:
    """Lookup a technology chain; raises KeyError for unknown names."""
    return SENSOR_TECHNOLOGIES[technology]


__all__ = ["TransductionChain", "sensor_technology_catalog", "transduction_chain",
           "SENSOR_TECHNOLOGIES"]