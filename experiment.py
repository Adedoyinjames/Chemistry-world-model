"""Multi-step experiment engine.

An "experiment" is a fully-specified, replayable script: an initial
ChemicalState + composition, an optional registered reaction, and an
ordered list of steps. This module is deliberately STATELESS at the API
layer — run_experiment() takes the complete step list every time and
replays it from scratch, returning a full recorded timeline. That means
every experiment is reproducible from its own parameters alone (no
server-side session to lose or desync), at the cost of resending the step
list on each call — a deliberate tradeoff that keeps the deploy
dependency-light (no database).

Supported step types (each produces one recorded timeline entry):
  - add_material:    {"type": "add_material", "formula": str, "moles": float}
  - remove_material: {"type": "remove_material", "formula": str, "moles": float}
  - heat:            {"type": "heat", "target_temperature_k": float}
  - cool:            {"type": "cool", "target_temperature_k": float}
  - change_pressure: {"type": "change_pressure", "target_pressure_atm": float}
  - change_ph:       {"type": "change_ph", "target_ph": float}
  - change_solvent:  {"type": "change_solvent", "solvent": str}
  - apply_energy:    {"type": "apply_energy", "energy_j": float, "degrees_of_freedom": int (optional, default 3)}
  - apply_field:     {"type": "apply_field", "electric_field_v_m": float (optional), "magnetic_field_t": float (optional)}
  - wait:            {"type": "wait", "duration_s": float}
"""
from __future__ import annotations

from typing import Any, Mapping

from simulation import (
    ChemicalState,
    ReactionSpec,
    SimulationError,
    create_state,
    calculate_state_properties,
    calculate_reaction,
    ELEMENTS,
)
from chemistry import molar_mass
from kinetics import integrate_elementary_kinetics, KineticsError

R = 8.31446261815324  # J mol^-1 K^-1


class ExperimentError(ValueError):
    """Raised when an experiment step or definition is invalid."""


def _apply_add_material(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    formula = step.get("formula")
    moles = step.get("moles")
    if not formula or moles is None:
        raise ExperimentError("add_material requires 'formula' and 'moles'.")
    try:
        molar_mass(formula, elements=ELEMENTS)  # validates the formula
    except ValueError as exc:
        raise ExperimentError(str(exc)) from exc
    if moles < 0:
        raise ExperimentError("add_material 'moles' must be non-negative.")
    state.composition[formula] = state.composition.get(formula, 0.0) + float(moles)
    return []


def _apply_remove_material(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    formula = step.get("formula")
    moles = step.get("moles")
    if not formula or moles is None:
        raise ExperimentError("remove_material requires 'formula' and 'moles'.")
    if moles < 0:
        raise ExperimentError("remove_material 'moles' must be non-negative.")

    available = state.composition.get(formula, 0.0)
    warnings = []
    if moles > available:
        warnings.append(
            f"Requested removal of {moles} mol {formula} but only {available} mol "
            f"was present — removed all of it instead."
        )
    state.composition[formula] = max(available - moles, 0.0)
    if state.composition[formula] == 0.0:
        del state.composition[formula]
    if not state.composition:
        raise ExperimentError("remove_material would leave the composition empty.")
    return warnings


def _apply_heat(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    target = step.get("target_temperature_k")
    if target is None:
        raise ExperimentError("heat requires 'target_temperature_k'.")
    if target < state.temperature_k:
        raise ExperimentError(
            f"heat step's target ({target} K) is below the current temperature "
            f"({state.temperature_k} K) — use a 'cool' step instead."
        )
    state.temperature_k = float(target)
    return []


def _apply_cool(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    target = step.get("target_temperature_k")
    if target is None:
        raise ExperimentError("cool requires 'target_temperature_k'.")
    if target <= 0:
        raise ExperimentError("cool 'target_temperature_k' must be positive.")
    if target > state.temperature_k:
        raise ExperimentError(
            f"cool step's target ({target} K) is above the current temperature "
            f"({state.temperature_k} K) — use a 'heat' step instead."
        )
    state.temperature_k = float(target)
    return []


def _apply_change_pressure(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    target = step.get("target_pressure_atm")
    if target is None:
        raise ExperimentError("change_pressure requires 'target_pressure_atm'.")
    if target <= 0:
        raise ExperimentError("target_pressure_atm must be positive.")
    state.pressure_atm = float(target)
    return []


def _apply_change_ph(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    target = step.get("target_ph")
    if target is None:
        raise ExperimentError("change_ph requires 'target_ph'.")
    if not (0 <= target <= 14):
        raise ExperimentError("target_ph must be between 0 and 14.")
    state.pH = float(target)
    return [
        "change_ph sets pH directly as an idealized control input — it "
        "represents adding an unspecified strong acid/base to reach that "
        "pH, not a specific modeled reagent or ion concentration change."
    ]


def _apply_change_solvent(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    solvent = step.get("solvent")
    if not solvent:
        raise ExperimentError("change_solvent requires 'solvent'.")
    state.solvent = str(solvent)
    return [
        "Solvent is recorded for reference; no solvation/activity-"
        "coefficient model is applied based on it."
    ]


def _apply_energy(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    energy_j = step.get("energy_j")
    if energy_j is None:
        raise ExperimentError("apply_energy requires 'energy_j'.")
    dof = step.get("degrees_of_freedom", 3)

    state.energy_j += float(energy_j)

    total_moles = sum(state.composition.values())
    warnings = []
    if total_moles > 0:
        delta_t = (2.0 * float(energy_j)) / (dof * total_moles * R)
        state.temperature_k = max(state.temperature_k + delta_t, 1e-6)
        warnings.append(
            f"Temperature change from applied energy uses the ideal-gas "
            f"equipartition approximation (delta_T = 2E/(f n R), f={dof}) — "
            "not a real heat-capacity model for the actual substances present."
        )
    return warnings


def _apply_field(state: ChemicalState, step: Mapping[str, Any]) -> list[str]:
    if "electric_field_v_m" in step:
        state.electric_field_v_m = float(step["electric_field_v_m"])
    if "magnetic_field_t" in step:
        state.magnetic_field_t = float(step["magnetic_field_t"])
    return [
        "Electric/magnetic field values are recorded for reference; no "
        "electrochemical or magnetic effect is computed from them."
    ]


def _apply_wait(
    state: ChemicalState,
    step: Mapping[str, Any],
    reaction: ReactionSpec | None,
    rate_constant_s_inv: float | None,
) -> tuple[list[str], dict[str, Any] | None]:
    duration = step.get("duration_s")
    if duration is None:
        raise ExperimentError("wait requires 'duration_s'.")
    if duration < 0:
        raise ExperimentError("duration_s must be non-negative.")

    state.time_s += float(duration)

    if reaction is None or rate_constant_s_inv is None:
        return (
            ["No reaction with a rate constant is registered — waiting advances "
             "time_s only, with no composition change."],
            None,
        )

    from chemistry import balance_reaction
    try:
        balanced = balance_reaction(reaction.reactants, reaction.products, elements=ELEMENTS)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from exc

    amount_kind = "concentration_mol_l" if state.volume_l else "moles"
    species = set(balanced["reactants"]) | set(balanced["products"])
    if state.volume_l:
        initial_amounts = {f: state.composition.get(f, 0.0) / state.volume_l for f in species}
    else:
        initial_amounts = {f: state.composition.get(f, 0.0) for f in species}

    try:
        kinetics_result = integrate_elementary_kinetics(
            initial_amounts=initial_amounts,
            reactant_coeffs=balanced["reactants"],
            product_coeffs=balanced["products"],
            rate_constant_s_inv=rate_constant_s_inv,
            duration_s=duration,
            amount_kind=amount_kind,
        )
    except KineticsError as exc:
        raise ExperimentError(str(exc)) from exc

    for formula, final_amount in kinetics_result["final_amounts"].items():
        if state.volume_l:
            state.composition[formula] = final_amount * state.volume_l
        else:
            state.composition[formula] = final_amount
    state.composition = {f: m for f, m in state.composition.items() if m > 1e-15}

    return (
        [kinetics_result["assumption"]],
        kinetics_result,
    )


_STEP_HANDLERS = {
    "add_material": _apply_add_material,
    "remove_material": _apply_remove_material,
    "heat": _apply_heat,
    "cool": _apply_cool,
    "change_pressure": _apply_change_pressure,
    "change_ph": _apply_change_ph,
    "change_solvent": _apply_change_solvent,
    "apply_energy": _apply_energy,
    "apply_field": _apply_field,
}


def run_experiment(
    initial_state_payload: Mapping[str, Any],
    steps: list[Mapping[str, Any]],
    reaction_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a full experiment script from scratch and return the
    recorded timeline. Every entry is reproducible: re-running this
    function with the same arguments produces the same timeline."""

    state = create_state(
        composition=initial_state_payload.get("composition", {}),
        temperature_k=initial_state_payload.get("temperature_k", 298.15),
        pressure_atm=initial_state_payload.get("pressure_atm", 1.0),
        volume_l=initial_state_payload.get("volume_l"),
        pH=initial_state_payload.get("pH"),
        solvent=initial_state_payload.get("solvent"),
        time_s=initial_state_payload.get("time_s", 0.0),
        energy_j=initial_state_payload.get("energy_j", 0.0),
        electric_field_v_m=initial_state_payload.get("electric_field_v_m", 0.0),
        magnetic_field_t=initial_state_payload.get("magnetic_field_t", 0.0),
    )

    reaction = None
    rate_constant_s_inv = None
    if reaction_payload is not None:
        reaction = ReactionSpec(
            reactants=reaction_payload.get("reactants", {}),
            products=reaction_payload.get("products", {}),
            activation_energy_j_mol=reaction_payload.get("activation_energy_j_mol"),
            pre_exponential_factor_s_inv=reaction_payload.get("pre_exponential_factor_s_inv"),
            delta_h_j_mol=reaction_payload.get("delta_h_j_mol"),
            delta_s_j_mol_k=reaction_payload.get("delta_s_j_mol_k"),
            standard_delta_g_j_mol=reaction_payload.get("standard_delta_g_j_mol"),
        )

    from dataclasses import asdict

    timeline = [{
        "step_index": 0,
        "step": {"type": "initial_state"},
        "state": asdict(state),
        "properties": calculate_state_properties(state),
        "reaction_analysis": (
            calculate_reaction(state, reaction) if reaction is not None else None
        ),
        "kinetics": None,
        "warnings": [],
    }]

    if reaction is not None:
        r0 = timeline[0]["reaction_analysis"]
        if r0 and "rate_constant_s_inv" in r0:
            rate_constant_s_inv = r0["rate_constant_s_inv"]

    all_assumptions: set[str] = set()

    for i, step in enumerate(steps, start=1):
        step_type = step.get("type")
        if step_type not in _STEP_HANDLERS and step_type != "wait":
            raise ExperimentError(
                f"Unknown step type {step_type!r} at index {i}. Valid types: "
                f"{sorted(list(_STEP_HANDLERS) + ['wait'])}"
            )

        kinetics_entry = None
        if step_type == "wait":
            warnings, kinetics_entry = _apply_wait(state, step, reaction, rate_constant_s_inv)
        else:
            warnings = _STEP_HANDLERS[step_type](state, step)

        for w in warnings:
            all_assumptions.add(w)

        reaction_analysis = None
        if reaction is not None:
            try:
                reaction_analysis = calculate_reaction(state, reaction)
                if reaction_analysis and "rate_constant_s_inv" in reaction_analysis:
                    rate_constant_s_inv = reaction_analysis["rate_constant_s_inv"]
            except SimulationError as exc:
                reaction_analysis = {"error": str(exc)}

        timeline.append({
            "step_index": i,
            "step": dict(step),
            "state": asdict(state),
            "properties": calculate_state_properties(state),
            "reaction_analysis": reaction_analysis,
            "kinetics": kinetics_entry,
            "warnings": warnings,
        })

    history = {
        "time_s": [entry["state"]["time_s"] for entry in timeline],
        "temperature_k": [entry["state"]["temperature_k"] for entry in timeline],
        "pressure_atm": [entry["state"]["pressure_atm"] for entry in timeline],
        "total_moles": [entry["properties"]["total_moles"] for entry in timeline],
    }

    return {
        "timeline": timeline,
        "history": history,
        "assumptions": sorted(all_assumptions),
        "reproducible_from": {
            "initial_state": initial_state_payload,
            "reaction": reaction_payload,
            "steps": steps,
        },
    }
