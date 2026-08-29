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


_CARET_CHARGE_RE = re.compile(r'\^(\d*)([+-])$')
_BARE_TRAILING_SIGN_RE = re.compile(r'(\d*)([+-])$')
_SINGLE_ELEMENT_RE = re.compile(r'^[A-Z][a-z]?$')


def split_charge(formula: str) -> tuple[str, int]:
    """Split a trailing ionic charge notation off a formula.

    Caret notation is unambiguous and always wins: 'SO4^2-' -> ('SO4', -2).

    Without a caret, a digit immediately before the sign is genuinely
    ambiguous — it could be the charge magnitude ('Fe3+' = Fe, charge +3)
    or a formula subscript with an implied magnitude-1 charge ('MnO4-' =
    MnO4, charge -1; the 4 is oxygen's subscript, not the charge). This is
    resolved with one rule: the digit is treated as a charge magnitude
    only when what remains after removing it is a single bare element
    symbol (e.g. 'Fe', 'Ca', 'Al'); otherwise it's left as part of the
    formula and the bare sign implies magnitude 1.

    Known limitation: this rule cannot disambiguate a diatomic radical ion
    of a single element from a charged monatomic ion when both are
    plausible (e.g. superoxide should be written 'O2^-', not 'O2-', since
    'O2-' would otherwise read as oxide with the 2 misparsed as charge).
    Always use caret notation for any species where this matters.
    """
    stripped = formula.strip()

    m = _CARET_CHARGE_RE.search(stripped)
    if m:
        magnitude_str, sign = m.groups()
        magnitude = int(magnitude_str) if magnitude_str else 1
        charge = magnitude if sign == '+' else -magnitude
        neutral = stripped[:m.start()].rstrip()
        if not neutral:
            raise ValueError(f"Invalid formula: {formula!r} has a charge but no atoms")
        return neutral, charge

    m2 = _BARE_TRAILING_SIGN_RE.search(stripped)
    if not m2:
        return stripped, 0

    digit_str, sign = m2.groups()

    if digit_str:
        candidate_remainder = stripped[:m2.start()]
        if _SINGLE_ELEMENT_RE.fullmatch(candidate_remainder):
            magnitude = int(digit_str)
            charge = magnitude if sign == '+' else -magnitude
            return candidate_remainder, charge
        # Digit belongs to the formula (a subscript) — bare sign, magnitude 1.
        neutral = stripped[:-1]
        if not neutral:
            raise ValueError(f"Invalid formula: {formula!r} has a charge but no atoms")
        return neutral, (1 if sign == '+' else -1)

    neutral = stripped[:m2.start()]
    if not neutral:
        raise ValueError(f"Invalid formula: {formula!r} has a charge but no atoms")
    return neutral, (1 if sign == '+' else -1)


def parse_charge(formula: str) -> int:
    """Return the net ionic charge encoded in a formula (0 if none)."""
    _, charge = split_charge(formula)
    return charge


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
    """Parse formulas such as H2O, Ca(OH)2, Al2(SO4)3, or ions like Fe3+/SO4^2-
    into element counts. A trailing charge notation is stripped first (use
    parse_charge to recover it) — it does not affect atom counts or mass."""
    formula = formula.strip()
    if not formula:
        raise ValueError("Formula cannot be empty")
    neutral, _charge = split_charge(formula)
    counts, end = _parse_group(neutral, elements=elements)
    if end != len(neutral):
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
    charges = [parse_charge(f) for f in species]
    element_symbols = sorted({e for c in compositions for e in c})

    matrix = []
    for e in element_symbols:
        row = []
        for i, comp in enumerate(compositions):
            sign = 1 if i < len(reactant_formulas) else -1
            row.append(sign * comp.get(e, 0))
        matrix.append(row)

    # Charge is a conserved quantity exactly like an atom count: if any
    # species carries a nonzero charge, require net charge to balance too
    # (this is what makes ionic/redox equations solvable, not just atom-balanced).
    if any(c != 0 for c in charges):
        charge_row = [
            (1 if i < len(reactant_formulas) else -1) * charges[i]
            for i in range(len(species))
        ]
        matrix.append(charge_row)

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


def kw_at_temperature(temperature_k: float) -> float:
    """Water's ion-product constant Kw at a given temperature, via the
    van't Hoff equation anchored at Kw(298.15 K) = 1.0e-14.

    Uses the standard enthalpy of water autoionization, ΔH ≈ +55.8 kJ/mol.
    This is a well-established approximation (constant ΔH over the range),
    not exact far from 298 K — real Kw(T) deviates somewhat at extremes.
    """
    _validate_positive("temperature_k", temperature_k)
    kw_ref = 1.0e-14
    t_ref = 298.15
    delta_h_ionization = 55800.0  # J/mol, standard reference value
    ln_ratio = -delta_h_ionization / R * (1.0 / temperature_k - 1.0 / t_ref)
    return kw_ref * exp(ln_ratio)


def ph_from_h_concentration(h_conc_mol_l: float) -> float:
    """pH = -log10([H+]). Requires molarity, i.e. moles/volume already computed."""
    if h_conc_mol_l <= 0 or not isfinite(h_conc_mol_l):
        raise ValueError("h_conc_mol_l must be positive and finite")
    return -log(h_conc_mol_l, 10)


def poh_from_oh_concentration(oh_conc_mol_l: float) -> float:
    """pOH = -log10([OH-])."""
    if oh_conc_mol_l <= 0 or not isfinite(oh_conc_mol_l):
        raise ValueError("oh_conc_mol_l must be positive and finite")
    return -log(oh_conc_mol_l, 10)


def h_concentration_from_ph(pH: float) -> float:
    if not isfinite(pH):
        raise ValueError("pH must be finite")
    return 10 ** (-pH)


def ionic_strength(concentrations_mol_l: Mapping[str, float]) -> float:
    """Ionic strength I = 1/2 * Σ(c_i * z_i^2), using charge parsed from each
    species' formula (e.g. 'Na+', 'SO4^2-'). Species with zero charge don't
    contribute. Concentrations must be in mol/L."""
    total = 0.0
    for species, conc in concentrations_mol_l.items():
        if conc < 0 or not isfinite(conc):
            raise ValueError(f"Concentration for {species!r} must be non-negative and finite")
        z = parse_charge(species)
        total += conc * (z ** 2)
    return 0.5 * total


def charge_balance(charged_moles: Mapping[str, float], tolerance_fraction: float = 0.02) -> dict[str, Any]:
    """Check whether a set of species (with ionic formulas) and their moles
    represents a charge-neutral solution — a real solution cannot have a net
    charge, so an imbalance usually means a counterion is missing from the
    composition, not a chemistry result in itself.

    Returns net_charge_mol (signed), total_ionic_moles (for scaling the
    tolerance), and is_balanced (within tolerance_fraction of total ionic
    moles, or exactly zero when there are no charged species at all).
    """
    net_charge = 0.0
    total_ionic = 0.0
    for species, moles in charged_moles.items():
        if moles < 0 or not isfinite(moles):
            raise ValueError(f"Moles for {species!r} must be non-negative and finite")
        z = parse_charge(species)
        if z != 0:
            net_charge += z * moles
            total_ionic += abs(z) * moles

    if total_ionic == 0:
        return {"net_charge_mol": 0.0, "total_ionic_moles": 0.0, "is_balanced": True}

    is_balanced = abs(net_charge) <= tolerance_fraction * total_ionic
    return {
        "net_charge_mol": net_charge,
        "total_ionic_moles": total_ionic,
        "is_balanced": is_balanced,
    }


_FIXED_OXIDATION_STATES = {
    "F": -1,
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Fr": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Ra": 2,
    "O": -2,
    "H": 1,
}


def oxidation_states(formula: str, elements: Mapping[str, dict] | None = None) -> dict[str, Any]:
    """Best-effort oxidation-state assignment from standard fixed rules
    (Group 1 = +1, Group 2 = +2, F = -1, O = -2 default, H = +1 default)
    plus charge-balance algebra for at most one remaining unknown element.

    This is a bounded heuristic, not a general solver. Known exceptions
    (peroxides, superoxides, metal hydrides, mixed-oxidation-state
    compounds like Fe3O4) are NOT special-cased. When the fixed rules
    produce an inconsistent or non-integer result, every affected element
    is reported as "indeterminate" with a reason, instead of guessing —
    a deliberate accuracy-over-coverage tradeoff.
    """
    neutral, charge = split_charge(formula)
    atom_counts = parse_formula(formula, elements=elements)

    if len(atom_counts) == 1:
        (only_element,) = atom_counts.keys()
        count = atom_counts[only_element]
        if charge == 0:
            return {
                "formula": formula,
                "oxidation_states": {only_element: 0},
                "indeterminate": False,
                "note": "Elemental (homonuclear) form: oxidation state is 0 by definition.",
            }
        if charge % count == 0:
            return {
                "formula": formula,
                "oxidation_states": {only_element: charge // count},
                "indeterminate": False,
                "note": "Monatomic ion: oxidation state equals charge per atom.",
            }
        return {
            "formula": formula,
            "oxidation_states": {only_element: "indeterminate"},
            "indeterminate": True,
            "note": "Charge does not divide evenly across identical atoms.",
        }

    known: dict[str, int] = {}
    unknown_elements: list[str] = []
    for el in atom_counts:
        if el in _FIXED_OXIDATION_STATES:
            known[el] = _FIXED_OXIDATION_STATES[el]
        else:
            unknown_elements.append(el)

    known_sum = sum(atom_counts[el] * known[el] for el in known)

    if not unknown_elements:
        if known_sum == charge:
            return {
                "formula": formula,
                "oxidation_states": known,
                "indeterminate": False,
                "note": "Assigned entirely from standard fixed rules.",
            }
        return {
            "formula": formula,
            "oxidation_states": {el: "indeterminate" for el in atom_counts},
            "indeterminate": True,
            "note": (
                "Standard oxidation-state rules are inconsistent for this "
                "formula (commonly a peroxide/superoxide, metal hydride, or "
                "other exception this solver does not model)."
            ),
        }

    if len(unknown_elements) > 1:
        return {
            "formula": formula,
            "oxidation_states": {
                **known,
                **{el: "indeterminate" for el in unknown_elements},
            },
            "indeterminate": True,
            "note": (
                "More than one element's oxidation state cannot be derived "
                "from fixed rules alone — insufficient information."
            ),
        }

    (unknown_el,) = unknown_elements
    remaining_charge = charge - known_sum
    count = atom_counts[unknown_el]
    if remaining_charge % count != 0:
        return {
            "formula": formula,
            "oxidation_states": {**known, unknown_el: "indeterminate"},
            "indeterminate": True,
            "note": (
                f"Non-integer oxidation state solved for {unknown_el} — likely "
                "a mixed-oxidation-state compound, which this solver cannot resolve."
            ),
        }

    known[unknown_el] = remaining_charge // count
    return {
        "formula": formula,
        "oxidation_states": known,
        "indeterminate": False,
        "note": f"{unknown_el} solved via charge balance; other elements from fixed rules.",
    }


SOLUBILITY_DATA_SOURCE = (
    "Standard 25 C reference Ksp values, cross-checked against the CRC "
    "Handbook of Chemistry and Physics where available; treat as textbook-"
    "grade reference data, not a primary source. CaSO4 in particular has "
    "notable variance across literature sources (roughly 2e-5 to 9e-5) "
    "depending on hydrate form and method — verify independently if precision matters."
)

SOLUBILITY_DATA: dict[str, dict[str, Any]] = {
    "AgCl": {"ksp": 1.77e-10, "dissociation": {"Ag+": 1, "Cl-": 1}},
    "AgBr": {"ksp": 5.35e-13, "dissociation": {"Ag+": 1, "Br-": 1}},
    "AgI": {"ksp": 8.52e-17, "dissociation": {"Ag+": 1, "I-": 1}},
    "Ag2CrO4": {"ksp": 1.12e-12, "dissociation": {"Ag+": 2, "CrO4^2-": 1}},
    "BaSO4": {"ksp": 1.08e-10, "dissociation": {"Ba2+": 1, "SO4^2-": 1}},
    "CaCO3": {"ksp": 3.36e-9, "dissociation": {"Ca2+": 1, "CO3^2-": 1}},
    "CaF2": {"ksp": 3.45e-11, "dissociation": {"Ca2+": 1, "F-": 2}},
    "CaSO4": {"ksp": 4.93e-5, "dissociation": {"Ca2+": 1, "SO4^2-": 1}},
    "Ca(OH)2": {"ksp": 5.02e-6, "dissociation": {"Ca2+": 1, "OH-": 2}},
    "Mg(OH)2": {"ksp": 5.61e-12, "dissociation": {"Mg2+": 1, "OH-": 2}},
    "Fe(OH)3": {"ksp": 2.79e-39, "dissociation": {"Fe3+": 1, "OH-": 3}},
    "Fe(OH)2": {"ksp": 4.87e-17, "dissociation": {"Fe2+": 1, "OH-": 2}},
    "PbCl2": {"ksp": 1.70e-5, "dissociation": {"Pb2+": 1, "Cl-": 2}},
    "PbSO4": {"ksp": 2.53e-8, "dissociation": {"Pb2+": 1, "SO4^2-": 1}},
    "PbI2": {"ksp": 9.80e-9, "dissociation": {"Pb2+": 1, "I-": 2}},
    "ZnS": {"ksp": 2.93e-25, "dissociation": {"Zn2+": 1, "S^2-": 1}},
    "CuS": {"ksp": 6.30e-36, "dissociation": {"Cu2+": 1, "S^2-": 1}},
}


def evaluate_solubility(
    compound: str,
    ion_concentrations_mol_l: Mapping[str, float],
) -> dict[str, Any]:
    """Compare the reaction quotient Qsp for a compound's dissolution to its
    Ksp, given current ion concentrations in mol/L. Returns saturation
    state and, if supersaturated, the extent (mol/L) that would precipitate
    out to bring Qsp back down to Ksp — found by bisection on the
    generalized solubility-equilibrium equation for the compound's actual
    dissociation stoichiometry (works for 1:1, 1:2, 2:1, etc.).

    Compounds not in SOLUBILITY_DATA return an explicit "unsupported"
    result rather than a fabricated Ksp.
    """
    if compound not in SOLUBILITY_DATA:
        return {
            "compound": compound,
            "supported": False,
            "reason": f"No reference Ksp available for {compound!r}.",
        }

    entry = SOLUBILITY_DATA[compound]
    ksp = entry["ksp"]
    dissociation = entry["dissociation"]

    missing = [ion for ion in dissociation if ion not in ion_concentrations_mol_l]
    if missing:
        return {
            "compound": compound,
            "supported": True,
            "ksp": ksp,
            "reason": f"Missing concentration(s) for: {', '.join(missing)}",
        }

    def qsp_at_extent(extent: float) -> float:
        q = 1.0
        for ion, coeff in dissociation.items():
            c = ion_concentrations_mol_l[ion] - coeff * extent
            if c <= 0:
                return -1.0  # signals "ran out of an ion" to the caller
            q *= c ** coeff
        return q

    q_initial = qsp_at_extent(0.0)

    if q_initial < ksp:
        state = "unsaturated"
    elif q_initial == ksp:
        state = "saturated"
    else:
        state = "supersaturated"

    result = {
        "compound": compound,
        "supported": True,
        "ksp": ksp,
        "qsp_initial": q_initial,
        "saturation_state": state,
    }

    if state == "supersaturated":
        # Bisect for the extent x in (0, x_max) where Qsp(x) = Ksp.
        # x_max is bounded by the smallest stoichiometric limit of any ion.
        x_max = min(
            ion_concentrations_mol_l[ion] / coeff
            for ion, coeff in dissociation.items()
        )
        lo, hi = 0.0, x_max
        for _ in range(100):
            mid = (lo + hi) / 2
            q_mid = qsp_at_extent(mid)
            if q_mid < 0 or q_mid < ksp:
                hi = mid
            else:
                lo = mid
        result["precipitate_extent_mol_l"] = lo
        result["note"] = (
            "Estimated extent assumes ideal (dilute) solution behavior — "
            "no activity-coefficient correction for ionic strength."
        )

    return result


FARADAY_CONSTANT = 96485.33212  # C/mol

STANDARD_REDUCTION_POTENTIALS_SOURCE = (
    "Standard reduction potentials E° (V vs SHE, 25 C), standard textbook "
    "reference values. Half-reactions are labeled by the oxidized/reduced "
    "couple only (not fully written out); electron counts below are for "
    "the couple as commonly tabulated."
)

# {couple_label: {"e_volts": E°, "electrons": n, "oxidized_form": ..., "reduced_form": ...}}
STANDARD_REDUCTION_POTENTIALS: dict[str, dict[str, Any]] = {
    "F2/F-": {"e_volts": 2.87, "electrons": 2, "oxidized_form": "F2", "reduced_form": "F-"},
    "MnO4-/Mn2+": {"e_volts": 1.51, "electrons": 5, "oxidized_form": "MnO4-", "reduced_form": "Mn2+"},
    "Cl2/Cl-": {"e_volts": 1.36, "electrons": 2, "oxidized_form": "Cl2", "reduced_form": "Cl-"},
    "O2/H2O": {"e_volts": 1.23, "electrons": 4, "oxidized_form": "O2", "reduced_form": "H2O"},
    "Ag+/Ag": {"e_volts": 0.80, "electrons": 1, "oxidized_form": "Ag+", "reduced_form": "Ag"},
    "Fe3+/Fe2+": {"e_volts": 0.77, "electrons": 1, "oxidized_form": "Fe3+", "reduced_form": "Fe2+"},
    "Cu2+/Cu": {"e_volts": 0.34, "electrons": 2, "oxidized_form": "Cu2+", "reduced_form": "Cu"},
    "H+/H2": {"e_volts": 0.00, "electrons": 2, "oxidized_form": "H+", "reduced_form": "H2"},
    "Pb2+/Pb": {"e_volts": -0.13, "electrons": 2, "oxidized_form": "Pb2+", "reduced_form": "Pb"},
    "Ni2+/Ni": {"e_volts": -0.26, "electrons": 2, "oxidized_form": "Ni2+", "reduced_form": "Ni"},
    "Fe2+/Fe": {"e_volts": -0.44, "electrons": 2, "oxidized_form": "Fe2+", "reduced_form": "Fe"},
    "Zn2+/Zn": {"e_volts": -0.76, "electrons": 2, "oxidized_form": "Zn2+", "reduced_form": "Zn"},
    "Al3+/Al": {"e_volts": -1.66, "electrons": 3, "oxidized_form": "Al3+", "reduced_form": "Al"},
    "Mg2+/Mg": {"e_volts": -2.37, "electrons": 2, "oxidized_form": "Mg2+", "reduced_form": "Mg"},
    "Na+/Na": {"e_volts": -2.71, "electrons": 1, "oxidized_form": "Na+", "reduced_form": "Na"},
    "Ca2+/Ca": {"e_volts": -2.87, "electrons": 2, "oxidized_form": "Ca2+", "reduced_form": "Ca"},
    "K+/K": {"e_volts": -2.93, "electrons": 1, "oxidized_form": "K+", "reduced_form": "K"},
    "Li+/Li": {"e_volts": -3.04, "electrons": 1, "oxidized_form": "Li+", "reduced_form": "Li"},
}


def find_reduction_couple(species: str) -> str | None:
    """Find a curated half-reaction couple where `species` is either the
    oxidized or reduced form. Returns the couple label, or None if not found."""
    for label, data in STANDARD_REDUCTION_POTENTIALS.items():
        if species in (data["oxidized_form"], data["reduced_form"]):
            return label
    return None


def cell_potential(cathode_couple: str, anode_couple: str) -> dict[str, Any]:
    """Standard cell potential for a galvanic cell: E°cell = E°cathode - E°anode
    (cathode = reduction happens; anode = oxidation happens), plus the
    derived ΔG° and equilibrium constant K at 298.15 K.

    Both couple labels must exist in STANDARD_REDUCTION_POTENTIALS (e.g.
    'Cu2+/Cu', 'Zn2+/Zn'). n (electrons transferred) is taken as the least
    common multiple of the two half-reactions' electron counts — the
    standard convention for balancing a full redox reaction from two
    half-reactions with different electron counts.
    """
    if cathode_couple not in STANDARD_REDUCTION_POTENTIALS:
        raise ValueError(f"No reference reduction potential for {cathode_couple!r}")
    if anode_couple not in STANDARD_REDUCTION_POTENTIALS:
        raise ValueError(f"No reference reduction potential for {anode_couple!r}")

    e_cathode = STANDARD_REDUCTION_POTENTIALS[cathode_couple]["e_volts"]
    e_anode = STANDARD_REDUCTION_POTENTIALS[anode_couple]["e_volts"]
    e_cell = e_cathode - e_anode

    from math import lcm as _lcm
    n = _lcm(
        STANDARD_REDUCTION_POTENTIALS[cathode_couple]["electrons"],
        STANDARD_REDUCTION_POTENTIALS[anode_couple]["electrons"],
    )

    delta_g = -n * FARADAY_CONSTANT * e_cell
    k = equilibrium_constant_from_delta_g(delta_g, 298.15)

    return {
        "cathode_couple": cathode_couple,
        "anode_couple": anode_couple,
        "e_cathode_v": e_cathode,
        "e_anode_v": e_anode,
        "e_cell_v": e_cell,
        "electrons_transferred": n,
        "delta_g_j_mol": delta_g,
        "equilibrium_constant": k,
        "is_spontaneous": e_cell > 0,
    }


COMPOUND_PHASE_DATA: dict[str, dict[str, Any]] = {
    "H2O": {"melting_point_k": 273.15, "boiling_point_k": 373.15},
    "NaCl": {"melting_point_k": 1074.0, "boiling_point_k": 1686.0},
    "NH3": {"melting_point_k": 195.4, "boiling_point_k": 239.8},
    "CH4": {"melting_point_k": 90.7, "boiling_point_k": 111.7},
    "C2H5OH": {"melting_point_k": 159.0, "boiling_point_k": 351.5},
    "C6H12O6": {"melting_point_k": 423.0, "note": "Decomposes near this temperature rather than melting cleanly."},
    "H2SO4": {"melting_point_k": 283.5, "boiling_point_k": 610.0, "note": "Decomposes on boiling."},
    "NaOH": {"melting_point_k": 596.0, "boiling_point_k": 1661.0},
    "CO2": {"note": "Sublimes directly solid->gas at 1 atm (triple point is ~5.1 atm); no liquid phase at 1 atm."},
    "CaCO3": {"note": "Decomposes (to CaO + CO2) around 1098 K at 1 atm rather than melting cleanly."},
}


def determine_phase(
    formula: str,
    temperature_k: float,
    pressure_atm: float = 1.0,
    elements: Mapping[str, dict] | None = None,
) -> dict[str, Any]:
    """Determine solid/liquid/gas for one species from reference melting/
    boiling point data. Elements use elements.json data directly; a small
    set of common compounds use a curated reference table; anything else
    is honestly reported as 'unknown' rather than guessed. Only strictly
    valid near 1 atm — no pressure-dependence model is applied beyond a
    caveat when pressure is far from 1 atm."""
    neutral, _charge = split_charge(formula)
    atom_counts = parse_formula(formula, elements=elements)

    melt = boil = None
    source = None
    special_note = None

    if neutral in COMPOUND_PHASE_DATA:
        data = COMPOUND_PHASE_DATA[neutral]
        melt = data.get("melting_point_k")
        boil = data.get("boiling_point_k")
        special_note = data.get("note")
        source = "curated_compound_reference"
    elif len(atom_counts) == 1:
        (el,) = atom_counts.keys()
        record = element(el, elements=elements)
        melt = record.get("melting_point_k")
        boil = record.get("boiling_point_k")
        source = "element_reference"

    result: dict[str, Any] = {
        "formula": formula,
        "source": source,
        "melting_point_k": melt,
        "boiling_point_k": boil,
    }
    if special_note:
        result["note"] = special_note
    if abs(pressure_atm - 1.0) > 0.05:
        result["pressure_caveat"] = (
            "Phase estimate assumes ~1 atm reference pressure; not corrected "
            "for the supplied pressure."
        )

    if melt is None and boil is None:
        result["phase"] = "unknown"
        result["reason"] = f"No reference melting/boiling data available for {formula!r}."
        return result

    if melt is not None and boil is not None:
        if temperature_k < melt:
            result["phase"] = "solid"
        elif temperature_k < boil:
            result["phase"] = "liquid"
        else:
            result["phase"] = "gas"
    elif melt is not None:
        result["phase"] = "solid" if temperature_k < melt else "unknown_above_melting_point"
    else:  # only boil known
        result["phase"] = "gas" if temperature_k >= boil else "unknown_below_boiling_point"

    return result


def mixture_phase_summary(
    composition: Mapping[str, float],
    temperature_k: float,
    pressure_atm: float = 1.0,
    elements: Mapping[str, dict] | None = None,
) -> dict[str, Any]:
    """Per-species phase for every component of a mixture, plus mole
    fractions grouped by phase. Species with no reference data land in an
    honest 'unknown' bucket rather than being assumed to be anything."""
    total_moles = sum(composition.values())
    per_species = {}
    phase_mole_fractions: dict[str, float] = {}

    for formula, moles in composition.items():
        info = determine_phase(formula, temperature_k, pressure_atm, elements=elements)
        per_species[formula] = info
        phase = info["phase"]
        fraction = (moles / total_moles) if total_moles > 0 else 0.0
        phase_mole_fractions[phase] = phase_mole_fractions.get(phase, 0.0) + fraction

    return {
        "per_species": per_species,
        "phase_mole_fractions": phase_mole_fractions,
    }


def _validate_positive(name: str, value: float) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be positive and finite")
