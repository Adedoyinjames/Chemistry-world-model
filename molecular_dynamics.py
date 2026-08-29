"""Molecular motion visualization backend.

This is a genuine (if simplified) physics simulation, not scripted
animation: initial particle speeds are sampled from the real 2D
Maxwell-Boltzmann distribution for each species' actual molar mass and
the system's temperature, and particles then evolve under real elastic
collisions (momentum + kinetic energy conserving) with each other and
with the box walls.

Two honest simplifications, both documented and surfaced in the output:

1. Reduced (non-SI) units. Real molecular speeds at everyday temperatures
   are hundreds of m/s — far too fast to animate meaningfully in a small
   on-screen box. Positions/velocities here use a single fixed
   calibration constant (K_DISPLAY) instead of true SI units, chosen so
   speeds render nicely at ordinary room-temperature-ish scales. This
   preserves every *relative* physical relationship exactly (heavier
   species move slower at the same T, by exactly sqrt(m1/m2); a species
   moves faster at higher T, by exactly sqrt(T1/T2)) — it only rescales
   the absolute display speed, which is a visualization choice, not a
   physical claim.
2. 2D hard-disk collisions detected by discrete-time overlap (not
   continuous time-of-impact). This is the standard simplification used
   in teaching-oriented molecular dynamics visualizations; at high speed
   or low frame rate a very fast pair can occasionally pass through each
   other in one step. Each individual collision that IS detected is
   resolved with an exact elastic-collision formula (momentum and
   kinetic energy conserving).
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping

from chemistry import element, molar_mass, parse_formula, split_charge

K_DISPLAY = 40.0  # calibration constant, reduced units — see module docstring
DEFAULT_BOX_SIZE = 100.0
MAX_PARTICLES = 150


def _species_mass_amu(formula: str, elements: Mapping[str, dict] | None = None) -> float:
    return molar_mass(formula, elements=elements)


def _species_radius(formula: str, elements: Mapping[str, dict] | None = None) -> float:
    neutral, _charge = split_charge(formula)
    atoms = parse_formula(formula, elements=elements)
    radii = []
    for sym, count in atoms.items():
        r = element(sym, elements=elements)["van_der_waals_radius_angstrom"]
        radii.extend([r] * count)
    avg_radius = sum(radii) / len(radii) if radii else 1.5
    # Slightly grow multi-atom species so they read as visually larger than
    # a single atom of similar element — a display heuristic, not a real
    # molecular-volume calculation.
    size_bonus = 1.0 + 0.15 * (sum(atoms.values()) - 1)
    return avg_radius * 1.5 * size_bonus


def initialize_particles(
    composition: Mapping[str, float],
    temperature_k: float,
    elements: Mapping[str, dict] | None = None,
    max_particles: int = MAX_PARTICLES,
    box_size: float = DEFAULT_BOX_SIZE,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build an initial particle ensemble for a composition at a given
    temperature. Particle *counts* are proportional to mole fraction
    (capped at max_particles total); particle *speeds* are drawn from the
    real 2D Maxwell-Boltzmann distribution for each species' molar mass at
    the given temperature, in the reduced unit system described above."""
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if not composition:
        raise ValueError("composition cannot be empty")

    rng = random.Random(seed)
    total_moles = sum(composition.values())

    counts: dict[str, int] = {}
    for formula, moles in composition.items():
        fraction = moles / total_moles if total_moles > 0 else 0.0
        counts[formula] = max(1, round(fraction * max_particles))

    # Re-scale down to max_particles if rounding pushed us over.
    total_count = sum(counts.values())
    if total_count > max_particles:
        scale = max_particles / total_count
        counts = {f: max(1, round(c * scale)) for f, c in counts.items()}

    particles = []
    pid = 0
    for formula, count in counts.items():
        mass = _species_mass_amu(formula, elements=elements)
        radius = _species_radius(formula, elements=elements)
        sigma = math.sqrt(K_DISPLAY * temperature_k / mass)  # per-axis std dev

        for _ in range(count):
            # Maxwell-Boltzmann speed via two independent Gaussian velocity
            # components (this IS the 2D Maxwell-Boltzmann speed
            # distribution — a Rayleigh distribution — by construction).
            vx = rng.gauss(0.0, sigma)
            vy = rng.gauss(0.0, sigma)
            x = rng.uniform(radius, box_size - radius)
            y = rng.uniform(radius, box_size - radius)
            particles.append({
                "id": pid,
                "species": formula,
                "x": x, "y": y,
                "vx": vx, "vy": vy,
                "radius": radius,
                "mass": mass,
            })
            pid += 1

    return {
        "particles": particles,
        "box_size": box_size,
        "instantaneous_temperature_k": _instantaneous_temperature(particles),
        "target_temperature_k": temperature_k,
        "unit_system_note": (
            "Positions/velocities use reduced display units, not SI — see "
            "molecular_dynamics module docs. Relative speed relationships "
            "between species and temperatures are physically exact; "
            "absolute on-screen speed is a visualization choice."
        ),
    }


def _instantaneous_temperature(particles: list[dict[str, Any]]) -> float:
    """Kinetic-theory diagnostic: back out the temperature implied by the
    ensemble's actual current kinetic energy (2 translational DOF in 2D:
    <KE> per particle = K_DISPLAY * T)."""
    if not particles:
        return 0.0
    total_ke = sum(0.5 * p["mass"] * (p["vx"] ** 2 + p["vy"] ** 2) for p in particles)
    return total_ke / (len(particles) * K_DISPLAY)


def step_particles(
    particles: list[dict[str, Any]],
    dt: float,
    box_size: float = DEFAULT_BOX_SIZE,
    target_temperature_k: float | None = None,
) -> dict[str, Any]:
    """Advance the ensemble by one timestep: integrate positions, resolve
    wall collisions (elastic reflection) and pairwise particle collisions
    (exact elastic-collision impulse, conserving momentum and kinetic
    energy), then optionally apply a velocity-rescaling thermostat toward
    `target_temperature_k` (a standard, real MD technique — used here so
    that changing the temperature control mid-experiment visibly and
    correctly speeds up or slows down the ensemble)."""
    particles = [dict(p) for p in particles]  # don't mutate caller's list

    for p in particles:
        p["x"] += p["vx"] * dt
        p["y"] += p["vy"] * dt

    for p in particles:
        r = p["radius"]
        if p["x"] - r < 0:
            p["x"] = r
            p["vx"] = abs(p["vx"])
        elif p["x"] + r > box_size:
            p["x"] = box_size - r
            p["vx"] = -abs(p["vx"])
        if p["y"] - r < 0:
            p["y"] = r
            p["vy"] = abs(p["vy"])
        elif p["y"] + r > box_size:
            p["y"] = box_size - r
            p["vy"] = -abs(p["vy"])

    n = len(particles)
    for i in range(n):
        a = particles[i]
        for j in range(i + 1, n):
            b = particles[j]
            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]
            dist = math.hypot(dx, dy)
            min_dist = a["radius"] + b["radius"]
            if dist >= min_dist or dist < 1e-9:
                continue

            nx, ny = dx / dist, dy / dist
            rvx, rvy = a["vx"] - b["vx"], a["vy"] - b["vy"]
            vel_along_normal = rvx * nx + rvy * ny

            if vel_along_normal > 0:  # approaching along the line of centers
                m1, m2 = a["mass"], b["mass"]
                impulse = (2 * vel_along_normal) / (1 / m1 + 1 / m2)
                a["vx"] -= (impulse / m1) * nx
                a["vy"] -= (impulse / m1) * ny
                b["vx"] += (impulse / m2) * nx
                b["vy"] += (impulse / m2) * ny

            overlap = min_dist - dist
            correction = overlap / 2
            a["x"] -= nx * correction
            a["y"] -= ny * correction
            b["x"] += nx * correction
            b["y"] += ny * correction

    if target_temperature_k is not None and target_temperature_k > 0:
        current_t = _instantaneous_temperature(particles)
        if current_t > 1e-9:
            scale = math.sqrt(target_temperature_k / current_t)
            for p in particles:
                p["vx"] *= scale
                p["vy"] *= scale

    return {
        "particles": particles,
        "instantaneous_temperature_k": _instantaneous_temperature(particles),
    }
