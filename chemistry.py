"""Core deterministic chemistry and physical-chemistry calculations.

The module is deliberately dependency-light and keeps scientific assumptions explicit.
It does not invent reaction data: reaction thermochemistry must be supplied by a
validated source when equilibrium/energy calculations are requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import exp, isfinite, log
from pathlib import Path
import json
import re
from typing import Any, Mapping, Sequence

R = 8.31446261815324  # J mol^-1 K^-1
R_L_ATM = 0.082057366080960  # L atm mol^-1 K^-1

_DATA_PATH = Path(__file__).with_name("elements.json")


def load_elements(path: str | Path = _DATA_PATH) -> dict[str, dict]:
    """Load the elemental registry indexed by chemical symbol."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    elements = payload.get("elements")
    if not isinstance(elements, list) or len(elements) != 118:
        raise ValueError("elements.json must contain exactly 118 elements")
    registry = {e["symbol"]: e for e in elements}
    if len(registry) != 118:
        raise ValueError("Element symbols must be unique")
    return registry


ELEMENTS = load_elements()


def element(symbol: str, elements: Mapping[str, dict] | None = None) -> dict:
    """Return one element record, validating its symbol.

    `elements` optionally overrides the module-level registry (this is what
    lets callers like simulation.py pass their own loaded registry through).
    """
    registry = elements if elements is not None else ELEMENTS
    try:
        return registry[symbol]
    except KeyError as exc:
        raise ValueError(f"Unknown element symbol: {symbol}") from exc


def _parse_group(
    text: str,
    index: int = 0,
    stop: str | None = None,
    elements: Mapping[str, dict] | None = None,
) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    while index < len(text):
        ch = text[index]
        if stop and ch == stop:
            return counts, index + 1
        if ch == '(' or ch == '[':
            closing = ')' if ch == '(' else ']'
            inner, index = _parse_group(text, index + 1, closing, elements=elements)
            m = re.match(r'(\d+)', text[index:])
            multiplier = int(m.group(1)) if m else 1
            index += len(m.group(1)) if m else 0
            for sym, n in inner.items():
                counts[sym] = counts.get(sym, 0) + n * multiplier
            continue
        if ch.isupper():
            j = index + 1
            if j < len(text) and text[j].islower():
                j += 1
            sym = text[index:j]
            element(sym, elements=elements)
            m = re.match(r'(\d+)', text[j:])
            number = int(m.group(1)) if m else 1
            index = j + (len(m.group(1)) if m else 0)
            counts[sym] = counts.get(sym, 0) + number
            continue
        raise ValueError(f"Invalid formula at position {index}: {text!r}")
    if stop:
        raise ValueError(f"Unclosed group in formula: {text!r}")
    return counts, index


def parse_formula(formula: str, elements: Mapping[str, dict] | None = None) -> dict[str, int]:
    """Parse formulas such as H2O, Ca(OH)2, Al2(SO4)3 into element counts."""
    formula = formula.strip()
    if not formula:
        raise ValueError("Formula cannot be empty")
    counts, end = _parse_group(formula, elements=elements)
    if end != len(formula):
        raise ValueError(f"Invalid formula: {formula!r}")
    return dict(sorted(counts.items()))


def molar_mass(formula: str, elements: Mapping[str, dict] | None = None) -> float:
    """Return molar mass in g/mol using the elemental atomic weights."""
    composition = parse_formula(formula, elements=elements)
    return sum(
        n * element(sym, elements=elements)["atomic_weight_u"]
        for sym, n in composition.items()
    )


def mass_to_moles(mass_g: float, formula: str, elements: Mapping[str, dict] | None = None) -> float:
    if mass_g < 0 or not isfinite(mass_g):
        raise ValueError("mass_g must be a finite non-negative number")
    return mass_g / molar_mass(formula, elements=elements)


def moles_to_mass(moles: float, formula: str, elements: Mapping[str, dict] | None = None) -> float:
    if moles < 0 or not isfinite(moles):
        raise ValueError("moles must be a finite non-negative number")
    return moles * molar_mass(formula, elements=elements)


def ideal_gas_pressure(moles: float, temperature_k: float, volume_l: float) -> float:
    """Ideal-gas pressure in atm."""
    _validate_positive("moles", moles)
    _validate_positive("temperature_k", temperature_k)
    _validate_positive("volume_l", volume_l)
    return moles * R_L_ATM * temperature_k / volume_l


def ideal_gas_volume(moles: float, temperature_k: float, pressure_atm: float) -> float:
    """Ideal-gas volume in liters."""
    _validate_positive("moles", moles)
    _validate_positive("temperature_k", temperature_k)
    _validate_positive("pressure_atm", pressure_atm)
    return moles * R_L_ATM * temperature_k / pressure_atm


def ideal_gas_moles(temperature_k: float, pressure_atm: float, volume_l: float) -> float:
    """Ideal-gas amount in mol."""
    _validate_positive("temperature_k", temperature_k)
    _validate_positive("pressure_atm", pressure_atm)
    _validate_positive("volume_l", volume_l)
    return pressure_atm * volume_l / (R_L_ATM * temperature_k)


def arrhenius_rate_constant(a: float, activation_energy_j_mol: float, temperature_k: float) -> float:
    """Arrhenius rate constant k = A exp(-Ea/RT)."""
    _validate_positive("a", a)
    _validate_positive("temperature_k", temperature_k)
    if activation_energy_j_mol < 0 or not isfinite(activation_energy_j_mol):
        raise ValueError("activation_energy_j_mol must be finite and non-negative")
    exponent = -activation_energy_j_mol / (R * temperature_k)
    return a * exp(exponent)


def delta_gibbs(delta_h_j_mol: float, delta_s_j_mol_k: float, temperature_k: float) -> float:
    """Compute ΔG = ΔH - TΔS in J/mol."""
    _validate_positive("temperature_k", temperature_k)
    if not all(isfinite(x) for x in (delta_h_j_mol, delta_s_j_mol_k)):
        raise ValueError("Thermodynamic inputs must be finite")
    return delta_h_j_mol - temperature_k * delta_s_j_mol_k


def equilibrium_constant_from_delta_g(delta_g_j_mol: float, temperature_k: float) -> float:
    """Compute K from ΔG°: K = exp(-ΔG°/RT).

    Extremely large/small values are clipped to avoid floating-point overflow;
    the clipping range is reported by the returned finite value rather than a
    fabricated physical bound.
    """
    _validate_positive("temperature_k", temperature_k)
    if not isfinite(delta_g_j_mol):
        raise ValueError("delta_g_j_mol must be finite")
    exponent = -delta_g_j_mol / (R * temperature_k)
    exponent = max(min(exponent, 700.0), -700.0)
    return exp(exponent)


def reaction_quotient(activities: Mapping[str, float], stoichiometric_coefficients: Mapping[str, int]) -> float:
    """Compute Q = Π a_i^ν_i from dimensionless activities."""
    q = 1.0
    for species, nu in stoichiometric_coefficients.items():
        if species not in activities:
            raise ValueError(f"Missing activity for species {species!r}")
        activity = activities[species]
        if activity <= 0 or not isfinite(activity):
            raise ValueError(f"Activity for {species!r} must be positive and finite")
        q *= activity ** nu
    return q


def balance_reaction(
    reactants: Sequence[str] | Mapping[str, float],
    products: Sequence[str] | Mapping[str, float],
    elements: Mapping[str, dict] | None = None,
) -> dict[str, Any]:
    """Balance a chemical equation using exact rational linear algebra.

    `reactants`/`products` may be a list of formulas, or a dict keyed by
    formula (any values in the dict are ignored — only the species present
    matter, since the correct integer coefficients are solved for here
    rather than trusted from caller input).

    Returns a dict: {"reactants": {formula: coeff}, "products": {formula: coeff},
    "equation": "<human-readable balanced equation>"}.
    """
    import sympy as sp

    reactant_formulas = list(reactants)
    product_formulas = list(products)

    if not reactant_formulas or not product_formulas:
        raise ValueError("At least one reactant and one product formula are required")

    species = reactant_formulas + product_formulas
    compositions = [parse_formula(f, elements=elements) for f in species]
    element_symbols = sorted({e for c in compositions for e in c})

    matrix = []
    for e in element_symbols:
        row = []
        for i, comp in enumerate(compositions):
            sign = 1 if i < len(reactant_formulas) else -1
            row.append(sign * comp.get(e, 0))
        matrix.append(row)

    ns = sp.Matrix(matrix).nullspace()
    if not ns:
        raise ValueError("Reaction cannot be balanced with the supplied species")

    vector = ns[0]
    lcm = sp.ilcm(*[term.q for term in vector])
    ints = [int(term * lcm) for term in vector]

    from math import gcd as _gcd
    common = 0
    for x in ints:
        common = _gcd(common, abs(x))
    ints = [x // common for x in ints]

    if any(x < 0 for x in ints):
        ints = [-x for x in ints]

    left, right = ints[:len(reactant_formulas)], ints[len(reactant_formulas):]
    if not all(left) or not all(right):
        raise ValueError("Degenerate balance returned a zero coefficient")

    reactant_coeffs = dict(zip(reactant_formulas, left))
    product_coeffs = dict(zip(product_formulas, right))

    def _side(coeffs: dict[str, int]) -> str:
        return " + ".join(
            f"{coeff} {formula}" if coeff != 1 else formula
            for formula, coeff in coeffs.items()
        )

    equation = f"{_side(reactant_coeffs)} -> {_side(product_coeffs)}"

    return {
        "reactants": reactant_coeffs,
        "products": product_coeffs,
        "equation": equation,
    }


def stoichiometric_extent_limiting_reagent(
    reactants: Mapping[str, float],
    stoichiometry: Mapping[str, float],
) -> tuple[str, float]:
    """Return limiting reactant and maximum reaction extent in mol.

    `reactants` contains available moles; `stoichiometry` contains positive
    reactant coefficients. Product coefficients should not be supplied here.
    """
    if not reactants or not stoichiometry:
        raise ValueError("Reactants and stoichiometry cannot be empty")
    extents = []
    for species, coeff in stoichiometry.items():
        _validate_positive(f"stoichiometry[{species}]", coeff)
        if species not in reactants:
            raise ValueError(f"Missing available moles for {species!r}")
        amount = reactants[species]
        if amount < 0 or not isfinite(amount):
            raise ValueError(f"Invalid available moles for {species!r}")
        extents.append((species, amount / coeff))
    return min(extents, key=lambda x: x[1])


def _validate_positive(name: str, value: float) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be positive and finite")
