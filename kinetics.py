"""Time-evolution kinetics for the reaction engine.

This integrates concentration (or, lacking a volume, raw moles) forward in
time using an assumed elementary / mass-action rate law:

    rate = k * Π( [reactant_i] ** balanced_coefficient_i )

This is the standard simplifying assumption for reactions with a known
Arrhenius rate constant and no separately-measured rate law — it is exact
for genuinely elementary reactions and only approximate for multi-step
mechanisms. Every result generated here says so explicitly; the frontend
should surface that label, not just the numbers.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping


class KineticsError(ValueError):
    """Raised for invalid kinetics-integration input."""


def _rate(
    concentrations: dict[str, float],
    reactant_coeffs: Mapping[str, int],
    rate_constant: float,
) -> float:
    rate = rate_constant
    for species, coeff in reactant_coeffs.items():
        c = max(concentrations[species], 0.0)
        rate *= c ** coeff
    return rate


def _derivatives(
    concentrations: dict[str, float],
    reactant_coeffs: Mapping[str, int],
    product_coeffs: Mapping[str, int],
    rate_constant: float,
) -> dict[str, float]:
    rate = _rate(concentrations, reactant_coeffs, rate_constant)
    d: dict[str, float] = {species: 0.0 for species in concentrations}
    for species, coeff in reactant_coeffs.items():
        d[species] -= coeff * rate
    for species, coeff in product_coeffs.items():
        d[species] = d.get(species, 0.0) + coeff * rate
    return d


def integrate_elementary_kinetics(
    initial_amounts: Mapping[str, float],
    reactant_coeffs: Mapping[str, int],
    product_coeffs: Mapping[str, int],
    rate_constant_s_inv: float,
    duration_s: float,
    n_steps: int = 60,
    amount_kind: str = "concentration_mol_l",
) -> dict[str, Any]:
    """Integrate an assumed elementary rate law forward over `duration_s`
    using 4th-order Runge-Kutta, clamping any species to zero once
    depleted (a reaction can't run past its limiting reagent).

    `amount_kind` should be 'concentration_mol_l' when a volume was
    available to convert moles to molarity, or 'moles' when it wasn't
    (in which case the shape of the curve is still meaningful but the
    numbers are not true concentrations — this is reported back so the
    frontend can label the axis honestly).
    """
    if duration_s < 0 or not isfinite(duration_s):
        raise KineticsError("duration_s must be non-negative and finite")
    if rate_constant_s_inv < 0 or not isfinite(rate_constant_s_inv):
        raise KineticsError("rate_constant_s_inv must be non-negative and finite")
    if n_steps < 1:
        raise KineticsError("n_steps must be at least 1")

    all_species = set(initial_amounts) | set(reactant_coeffs) | set(product_coeffs)
    state = {s: float(initial_amounts.get(s, 0.0)) for s in all_species}
    for s, v in state.items():
        if v < 0 or not isfinite(v):
            raise KineticsError(f"Initial amount for {s!r} must be non-negative and finite")

    # Auto-refine step count for fast/stiff kinetics: cap k*dt at ~0.1 so
    # RK4 stays accurate, up to a performance ceiling. Without this, a
    # large rate constant + coarse n_steps silently produces garbage
    # (an under-resolved step can overshoot so far that naive clamping
    # would fabricate product mass out of nowhere).
    max_total_steps = 4000
    if rate_constant_s_inv > 0 and duration_s > 0:
        target_dt = 0.1 / rate_constant_s_inv
        needed_steps = int(duration_s / target_dt) + 1
        n_steps = min(max(n_steps, needed_steps), max_total_steps)

    dt = duration_s / n_steps
    times = [0.0]
    trajectory = {s: [state[s]] for s in all_species}

    for step in range(n_steps):
        k1 = _derivatives(state, reactant_coeffs, product_coeffs, rate_constant_s_inv)
        s2 = {s: state[s] + 0.5 * dt * k1[s] for s in state}
        k2 = _derivatives(s2, reactant_coeffs, product_coeffs, rate_constant_s_inv)
        s3 = {s: state[s] + 0.5 * dt * k2[s] for s in state}
        k3 = _derivatives(s3, reactant_coeffs, product_coeffs, rate_constant_s_inv)
        s4 = {s: state[s] + dt * k3[s] for s in state}
        k4 = _derivatives(s4, reactant_coeffs, product_coeffs, rate_constant_s_inv)

        deltas = {
            s: (dt / 6.0) * (k1[s] + 2 * k2[s] + 2 * k3[s] + k4[s])
            for s in state
        }

        # If this step would deplete any species below zero, scale the
        # WHOLE step proportionally (not just clamp the offender) so every
        # species' change stays in the correct stoichiometric ratio —
        # equivalent to hitting a limiting reagent exactly at zero.
        scale = 1.0
        for s in state:
            if deltas[s] < 0 and state[s] + deltas[s] < 0:
                scale = min(scale, state[s] / (-deltas[s]))
        scale = max(scale, 0.0)

        for s in state:
            state[s] = max(state[s] + scale * deltas[s], 0.0)

        times.append((step + 1) * dt)
        for s in all_species:
            trajectory[s].append(state[s])

    return {
        "times_s": times,
        "trajectories": trajectory,
        "amount_kind": amount_kind,
        "final_amounts": {s: state[s] for s in all_species},
        "assumption": (
            "Assumes an elementary (mass-action) rate law derived from the "
            "balanced stoichiometry: rate = k * product of [reactant]^coeff. "
            "This is exact only for genuinely elementary reactions; "
            "multi-step mechanisms may follow a different rate law entirely."
        ),
    }
