"""
Chemical World Model — Simulation Engine

This module represents a chemical system as a state and performs
deterministic calculations on that state.

It does not pretend to discover arbitrary chemical reactions.
Reaction-specific calculations are performed when a reaction is
explicitly supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite, log10
from typing import Any, Mapping

from chemistry import (
    load_elements,
    molar_mass,
    parse_formula,
    parse_charge,
    ideal_gas_pressure,
    ideal_gas_volume,
    ideal_gas_moles,
    arrhenius_rate_constant,
    delta_gibbs,
    equilibrium_constant_from_delta_g,
    balance_reaction,
    stoichiometric_extent_limiting_reagent,
    reaction_quotient,
    mixture_phase_summary,
    charge_balance,
    ionic_strength,
    ph_from_h_concentration,
    poh_from_oh_concentration,
    kw_at_temperature,
    oxidation_states,
    evaluate_solubility,
    SOLUBILITY_DATA,
    cell_potential,
    STANDARD_REDUCTION_POTENTIALS,
)

ELEMENTS = load_elements()


class SimulationError(ValueError):
    """Raised when a simulation input is invalid."""


@dataclass
class ChemicalState:
    composition: dict[str, float]
    temperature_k: float
    pressure_atm: float
    volume_l: float | None = None
    pH: float | None = None
    solvent: str | None = None
    time_s: float = 0.0
    energy_j: float = 0.0
    electric_field_v_m: float = 0.0
    magnetic_field_t: float = 0.0


@dataclass
class ReactionSpec:
    reactants: dict[str, float]
    products: dict[str, float]

    activation_energy_j_mol: float | None = None
    pre_exponential_factor_s_inv: float | None = None

    delta_h_j_mol: float | None = None
    delta_s_j_mol_k: float | None = None
    standard_delta_g_j_mol: float | None = None


def _number(name: str, value: Any, minimum: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise SimulationError(f"{name} must be numeric.")

    if not isfinite(value) or value < minimum:
        raise SimulationError(
            f"{name} must be finite and >= {minimum}."
        )

    return value


def _normalize_composition(
    composition: Mapping[str, float]
) -> dict[str, float]:

    if not composition:
        raise SimulationError("composition cannot be empty.")

    result = {}

    for formula, amount in composition.items():

        if not isinstance(formula, str) or not formula.strip():
            raise SimulationError(
                "Every chemical formula must be a non-empty string."
            )

        amount = _number(
            f"amount for {formula}",
            amount,
            minimum=0.0,
        )

        # Validate formula
        try:
            parse_formula(formula, elements=ELEMENTS)
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

        result[formula.strip()] = amount

    if sum(result.values()) <= 0:
        raise SimulationError(
            "At least one chemical component must contain more than zero moles."
        )

    return result


def create_state(
    composition: Mapping[str, float],
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    volume_l: float | None = None,
    pH: float | None = None,
    solvent: str | None = None,
    time_s: float = 0.0,
    energy_j: float = 0.0,
    electric_field_v_m: float = 0.0,
    magnetic_field_t: float = 0.0,
) -> ChemicalState:

    temperature_k = _number(
        "temperature_k",
        temperature_k,
        minimum=0.000001,
    )

    pressure_atm = _number(
        "pressure_atm",
        pressure_atm,
        minimum=0.000001,
    )

    if volume_l is not None:
        volume_l = _number(
            "volume_l",
            volume_l,
            minimum=0.000001,
        )

    if pH is not None:
        pH = _number("pH", pH)

        if pH > 14:
            raise SimulationError(
                "pH must be between 0 and 14."
            )

    time_s = _number(
        "time_s",
        time_s,
        minimum=0.0,
    )

    return ChemicalState(
        composition=_normalize_composition(composition),
        temperature_k=temperature_k,
        pressure_atm=pressure_atm,
        volume_l=volume_l,
        pH=pH,
        solvent=solvent,
        time_s=time_s,
        energy_j=float(energy_j),
        electric_field_v_m=float(electric_field_v_m),
        magnetic_field_t=float(magnetic_field_t),
    )


def elemental_composition(
    composition: Mapping[str, float]
) -> dict[str, float]:

    totals = {}

    for formula, moles in composition.items():

        atoms = parse_formula(formula)

        for element, count in atoms.items():

            totals[element] = (
                totals.get(element, 0.0)
                + count * moles
            )

    return totals


def _ionic_summary(state: ChemicalState) -> dict[str, Any] | None:
    """Only present when the composition actually contains charged species
    (parsed via each formula's charge notation, e.g. 'Na+', 'SO4^2-')."""

    charged_species = {
        formula: moles
        for formula, moles in state.composition.items()
        if parse_charge(formula) != 0
    }

    if not charged_species:
        return None

    summary: dict[str, Any] = {
        "charge_balance": charge_balance(state.composition),
    }

    if state.volume_l is not None and state.volume_l > 0:

        concentrations = {
            formula: moles / state.volume_l
            for formula, moles in charged_species.items()
        }

        summary["ion_concentrations_mol_l"] = concentrations
        summary["ionic_strength_mol_l"] = ionic_strength(concentrations)

        h_conc = concentrations.get("H+")
        oh_conc = concentrations.get("OH-")

        if h_conc is not None and h_conc > 0:
            summary["pH"] = ph_from_h_concentration(h_conc)

        if oh_conc is not None and oh_conc > 0:
            summary["pOH"] = poh_from_oh_concentration(oh_conc)

        if "pH" in summary or "pOH" in summary:
            p_kw = -log10(kw_at_temperature(state.temperature_k))
            if "pH" in summary and "pOH" not in summary:
                summary["pOH"] = p_kw - summary["pH"]
            elif "pOH" in summary and "pH" not in summary:
                summary["pH"] = p_kw - summary["pOH"]
            summary["kw_at_temperature"] = kw_at_temperature(state.temperature_k)

    else:
        summary["note"] = (
            "Supply volume_l to compute ionic strength and pH from these ion amounts."
        )

    return summary


def calculate_state_properties(
    state: ChemicalState
) -> dict[str, Any]:

    masses = {}

    total_mass_g = 0.0
    total_moles = sum(state.composition.values())

    for formula, amount in state.composition.items():

        mass = molar_mass(
            formula,
            ELEMENTS,
        )

        masses[formula] = mass

        total_mass_g += mass * amount

    result = {
        "total_moles": total_moles,
        "total_mass_g": total_mass_g,
        "molar_masses_g_mol": masses,
        "elemental_moles": elemental_composition(
            state.composition
        ),
    }

    # Ideal-gas calculations are provided as reference calculations.
    result["ideal_gas_volume_l"] = ideal_gas_volume(
        moles=total_moles,
        temperature_k=state.temperature_k,
        pressure_atm=state.pressure_atm,
    )

    if state.volume_l is not None:

        result["ideal_gas_pressure_atm"] = ideal_gas_pressure(
            moles=total_moles,
            temperature_k=state.temperature_k,
            volume_l=state.volume_l,
        )

        result["ideal_gas_moles"] = ideal_gas_moles(
            temperature_k=state.temperature_k,
            pressure_atm=state.pressure_atm,
            volume_l=state.volume_l,
        )

    result["phase_summary"] = mixture_phase_summary(
        state.composition,
        state.temperature_k,
        state.pressure_atm,
        elements=ELEMENTS,
    )

    result["ionic_summary"] = _ionic_summary(state)

    return result


def _oxidation_and_redox(balanced: dict[str, Any]) -> dict[str, Any]:
    reactant_ox = {f: oxidation_states(f, elements=ELEMENTS) for f in balanced["reactants"]}
    product_ox = {f: oxidation_states(f, elements=ELEMENTS) for f in balanced["products"]}

    reactant_states_by_element: dict[str, set] = {}
    for info in reactant_ox.values():
        for el, val in info["oxidation_states"].items():
            reactant_states_by_element.setdefault(el, set()).add(val)

    product_states_by_element: dict[str, set] = {}
    for info in product_ox.values():
        for el, val in info["oxidation_states"].items():
            product_states_by_element.setdefault(el, set()).add(val)

    changed_elements = []
    indeterminate_elements = []

    for el in set(reactant_states_by_element) & set(product_states_by_element):
        r_states = reactant_states_by_element[el]
        p_states = product_states_by_element[el]
        if "indeterminate" in r_states or "indeterminate" in p_states:
            indeterminate_elements.append(el)
        elif r_states != p_states:
            changed_elements.append({
                "element": el,
                "reactant_states": sorted(r_states),
                "product_states": sorted(p_states),
            })

    if changed_elements:
        is_redox: Any = True
    elif indeterminate_elements:
        is_redox = "indeterminate"
    else:
        is_redox = False

    result = {
        "reactant_oxidation_states": reactant_ox,
        "product_oxidation_states": product_ox,
        "is_redox": is_redox,
        "changed_elements": changed_elements,
    }
    if indeterminate_elements:
        result["indeterminate_elements"] = indeterminate_elements
    return result


def _detect_electrochemistry(balanced: dict[str, Any]) -> dict[str, Any] | None:
    """Only succeeds when the reaction's species match a curated half-
    reaction couple's exact formula spelling on both the oxidized and
    reduced side — otherwise returns None (no fabricated E-cell)."""
    reduction_couple = None
    oxidation_couple = None

    for label, data in STANDARD_REDUCTION_POTENTIALS.items():
        ox, red = data["oxidized_form"], data["reduced_form"]
        if ox in balanced["reactants"] and red in balanced["products"]:
            reduction_couple = label
        if red in balanced["reactants"] and ox in balanced["products"]:
            oxidation_couple = label

    if reduction_couple and oxidation_couple and reduction_couple != oxidation_couple:
        return cell_potential(reduction_couple, oxidation_couple)
    return None


def _equilibrium_status(
    state: ChemicalState,
    balanced: dict[str, Any],
    k: float,
) -> dict[str, Any]:
    all_species = list(balanced["reactants"]) + list(balanced["products"])

    if state.volume_l is not None and state.volume_l > 0:
        basis = "concentration_mol_l"
        activities = {f: state.composition.get(f, 0.0) / state.volume_l for f in all_species}
    else:
        basis = "moles (no volume_l supplied — this is not a true concentration)"
        activities = {f: state.composition.get(f, 0.0) for f in all_species}

    if any(activities[f] <= 0 for f in all_species):
        return {
            "reaction_quotient": None,
            "basis": basis,
            "reason": (
                "One or more species has zero or unspecified amount in the "
                "current composition — Q is not defined."
            ),
        }

    q = reaction_quotient(activities, balanced["products"]) / reaction_quotient(
        activities, balanced["reactants"]
    )

    ratio = (q / k) if k != 0 else float("inf")
    if 0.99 <= ratio <= 1.01:
        direction = "at_equilibrium"
    elif q < k:
        direction = "forward (toward products)"
    else:
        direction = "reverse (toward reactants)"

    return {
        "reaction_quotient": q,
        "equilibrium_constant": k,
        "basis": basis,
        "direction": direction,
    }


def _precipitation_check(
    state: ChemicalState,
    balanced: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = [f for f in balanced["products"] if f in SOLUBILITY_DATA]
    if not candidates or not state.volume_l:
        return None

    results = {}
    for compound in candidates:
        dissociation = SOLUBILITY_DATA[compound]["dissociation"]
        concentrations = {}
        for ion in dissociation:
            moles = state.composition.get(ion)
            if moles is None:
                concentrations = None
                break
            concentrations[ion] = moles / state.volume_l
        if concentrations is not None:
            results[compound] = evaluate_solubility(compound, concentrations)

    return results or None


def calculate_reaction(
    state: ChemicalState,
    reaction: ReactionSpec,
) -> dict[str, Any]:

    try:
        balanced = balance_reaction(
            reaction.reactants,
            reaction.products,
            elements=ELEMENTS,
        )
    except ValueError as exc:
        raise SimulationError(str(exc)) from exc

    result = {
        "balanced_reactants": balanced["reactants"],
        "balanced_products": balanced["products"],
        "balanced_equation": balanced["equation"],
    }

    # If the state's composition includes every balanced reactant species,
    # work out the limiting reagent and the resulting product yield. This
    # is skipped (not guessed) when the state doesn't specify moles for
    # every reactant.
    reactant_coeffs = balanced["reactants"]
    available_moles = {
        formula: state.composition.get(formula)
        for formula in reactant_coeffs
    }

    if all(amount is not None for amount in available_moles.values()):

        try:
            limiting_species, extent_mol = stoichiometric_extent_limiting_reagent(
                available_moles,
                reactant_coeffs,
            )
        except ValueError as exc:
            raise SimulationError(str(exc)) from exc

        result["limiting_reagent"] = limiting_species
        result["reaction_extent_mol"] = extent_mol
        result["reactant_moles_consumed"] = {
            formula: coeff * extent_mol
            for formula, coeff in reactant_coeffs.items()
        }
        result["product_moles_formed"] = {
            formula: coeff * extent_mol
            for formula, coeff in balanced["products"].items()
        }
    else:
        result["limiting_reagent"] = None
        result["reaction_extent_mol"] = None

    if (
        reaction.activation_energy_j_mol is not None
        and reaction.pre_exponential_factor_s_inv is not None
    ):

        result["rate_constant_s_inv"] = arrhenius_rate_constant(
            reaction.pre_exponential_factor_s_inv,
            reaction.activation_energy_j_mol,
            state.temperature_k,
        )

    elif (
        reaction.activation_energy_j_mol is not None
        or reaction.pre_exponential_factor_s_inv is not None
    ):

        raise SimulationError(
            "Both activation energy and pre-exponential factor "
            "are required for Arrhenius calculations."
        )

    if (
        reaction.delta_h_j_mol is not None
        and reaction.delta_s_j_mol_k is not None
    ):

        dg = delta_gibbs(
            reaction.delta_h_j_mol,
            reaction.delta_s_j_mol_k,
            state.temperature_k,
        )

        result["delta_g_j_mol"] = dg

        result["equilibrium_constant"] = (
            equilibrium_constant_from_delta_g(
                dg,
                state.temperature_k,
            )
        )

    elif reaction.standard_delta_g_j_mol is not None:

        dg = reaction.standard_delta_g_j_mol

        result["delta_g_j_mol"] = dg

        result["equilibrium_constant"] = (
            equilibrium_constant_from_delta_g(
                dg,
                state.temperature_k,
            )
        )

    redox_info = _oxidation_and_redox(balanced)
    result.update(redox_info)

    result["electrochemistry"] = _detect_electrochemistry(balanced)

    if "equilibrium_constant" in result:
        result["equilibrium"] = _equilibrium_status(
            state, balanced, result["equilibrium_constant"]
        )
    else:
        result["equilibrium"] = {
            "reaction_quotient": None,
            "reason": (
                "No thermodynamic data (delta_H+delta_S or standard delta_G) "
                "was supplied, so no K is available to compare against."
            ),
        }

    result["precipitation"] = _precipitation_check(state, balanced)

    return result


def simulate(
    state: ChemicalState,
    reaction: ReactionSpec | None = None,
) -> dict[str, Any]:

    result = {
        "state": asdict(state),
        "properties": calculate_state_properties(state),
        "reaction": None,
        "warnings": [],
    }

    if reaction is not None:

        result["reaction"] = calculate_reaction(
            state,
            reaction,
        )

    result["warnings"].append(
        "Ideal-gas calculations are reference calculations and "
        "do not replace real-fluid or molecular simulation models."
    )

    if reaction is None:

        result["warnings"].append(
            "No reaction pathway was supplied. The engine does not "
            "invent reaction products from element names alone."
        )

    return result


def simulate_payload(
    payload: Mapping[str, Any]
) -> dict[str, Any]:

    state = create_state(
        composition=payload.get("composition", {}),
        temperature_k=payload.get(
            "temperature_k",
            298.15,
        ),
        pressure_atm=payload.get(
            "pressure_atm",
            1.0,
        ),
        volume_l=payload.get("volume_l"),
        pH=payload.get("pH"),
        solvent=payload.get("solvent"),
        time_s=payload.get("time_s", 0.0),
        energy_j=payload.get("energy_j", 0.0),
        electric_field_v_m=payload.get(
            "electric_field_v_m",
            0.0,
        ),
        magnetic_field_t=payload.get(
            "magnetic_field_t",
            0.0,
        ),
    )

    reaction_payload = payload.get("reaction")

    reaction = None

    if reaction_payload is not None:

        reaction = ReactionSpec(
            reactants=reaction_payload.get(
                "reactants",
                {},
            ),
            products=reaction_payload.get(
                "products",
                {},
            ),
            activation_energy_j_mol=reaction_payload.get(
                "activation_energy_j_mol"
            ),
            pre_exponential_factor_s_inv=reaction_payload.get(
                "pre_exponential_factor_s_inv"
            ),
            delta_h_j_mol=reaction_payload.get(
                "delta_h_j_mol"
            ),
            delta_s_j_mol_k=reaction_payload.get(
                "delta_s_j_mol_k"
            ),
            standard_delta_g_j_mol=reaction_payload.get(
                "standard_delta_g_j_mol"
            ),
        )

    return simulate(
        state,
        reaction=reaction,
    )
