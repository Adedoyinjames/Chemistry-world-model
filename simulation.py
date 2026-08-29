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
from math import isfinite
from typing import Any, Mapping

from chemistry import (
    load_elements,
    molar_mass,
    parse_formula,
    ideal_gas_pressure,
    ideal_gas_volume,
    ideal_gas_moles,
    arrhenius_rate_constant,
    delta_gibbs,
    equilibrium_constant_from_delta_g,
    balance_reaction,
    stoichiometric_extent_limiting_reagent,
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

    return result


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
