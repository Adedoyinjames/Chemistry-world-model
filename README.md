<div align="center">

<img src="https://i.ibb.co/TqYHPb8P/Chat-GPT-Image-Aug-29-2026-08-56-46-PM.png" alt="Chemical World Model banner" width="100%" />

# Chemical World Model

**A deterministic chemistry simulation API — matter state, reactions, kinetics, equilibrium, ionic chemistry, redox, and molecular motion, all computed server-side.**

[![X](https://img.shields.io/badge/X-%40Adedoyinjames__-000000?style=flat-square&logo=x&logoColor=white)](https://x.com/Adedoyinjames_)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ifeoluwa%20Adedoyin-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ifeoluwa-adedoyin/)
[![Email](https://img.shields.io/badge/Email-ifeoluwaadedoyin19%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:ifeoluwaadedoyin19@gmail.com)

</div>

---

## Quick Links

| | |
|---|---|
| 🌐 Web App | [chemistry-world-model-frontend.vercel.app](https://chemistry-world-model-frontend.vercel.app/) |
| 🚀 Live API | [chemistry-world-model.onrender.com](https://chemistry-world-model.onrender.com) |
| 📖 Interactive Docs | [chemistry-world-model.onrender.com/docs](https://chemistry-world-model.onrender.com/docs) |
| 💓 Health Check | [chemistry-world-model.onrender.com/health](https://chemistry-world-model.onrender.com/health) |
| ✉️ Contact | [ifeoluwaadedoyin19@gmail.com](mailto:ifeoluwaadedoyin19@gmail.com) |

---

## Overview

Chemical World Model is a FastAPI backend that represents a chemical system as an explicit state — composition, temperature, pressure, volume, pH, solvent — and performs deterministic, reference-data-backed calculations on it. It does not use an LLM to "guess" chemistry: every number either comes from a real physical formula, a curated reference constant (atomic weights, melting/boiling points, Ksp, standard reduction potentials), or is explicitly reported as `unknown` / `indeterminate` / `unsupported` when the data or method genuinely doesn't cover a case.

Built for a computational-laboratory workflow: define a system, optionally register a reaction, then either take a single snapshot (`/simulate`) or script a full multi-step experiment over time (`/experiment/run`) — heat it, cool it, add reagents, wait for kinetics to play out — and get back a fully reproducible, timestamped record of everything that happened.

## Features

**Core composition & properties**
- Formula parsing with nested groups (`Ca(OH)2`, `Al2(SO4)3`) and ionic notation (`Fe3+`, `SO4^2-`)
- Molar mass, elemental composition, ideal-gas law calculations
- Automatic reaction balancing via exact rational linear algebra (handles charge conservation too, so ionic/redox equations balance correctly)

**Matter state**
- Solid/liquid/gas determination from real melting/boiling point data (all 118 elements, plus common compounds) — reports `unknown` honestly rather than guessing

**Ionic & acid-base chemistry**
- Charge balance validation, ionic strength, pH/pOH with temperature-dependent Kw (van't Hoff)

**Reaction direction & equilibrium**
- Reaction quotient Q vs equilibrium constant K, with a clear forward/reverse/at-equilibrium verdict

**Redox & electrochemistry**
- Self-validating oxidation-state solver (flags peroxides, hydrides, and mixed-oxidation compounds as indeterminate instead of guessing wrong)
- Standard reduction potentials, cell potential, ΔG, and K for recognized half-reaction couples

**Precipitation & solubility**
- Ksp vs Qsp comparison against a curated reference table, with a general bisection solver for precipitate yield at any stoichiometry

**Time-evolving kinetics**
- RK4 integration of elementary (mass-action) rate laws, with automatic step refinement so fast kinetics stay numerically stable

**Multi-step experiments**
- A stateless, fully reproducible experiment timeline: add/remove material, heat, cool, change pressure/pH/solvent, apply energy or fields, wait — each step recorded with its resulting state and analysis

**Molecular motion visualization**
- A real (if simplified) 2D physics engine: particle velocities are sampled from the actual Maxwell-Boltzmann distribution for each species' molar mass and the system's temperature, then evolved via true elastic collisions — verified to conserve kinetic energy to machine precision

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status |
| `POST /formulas/validate` | Validate a single formula, get element counts + molar mass |
| `POST /simulate` | Full single-snapshot analysis of a composition (+ optional reaction) |
| `POST /experiment/run` | Run a scripted multi-step experiment, get the full recorded timeline |
| `POST /particles/init` | Seed a molecular-motion particle ensemble |
| `POST /particles/step` | Advance the particle ensemble by one physics timestep |

Full request/response schemas are in the interactive docs at [`/docs`](https://chemistry-world-model.onrender.com/docs).

## Example

```bash
curl -X POST https://chemistry-world-model.onrender.com/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "composition": {"H2": 2, "O2": 1},
    "temperature_k": 298.15,
    "volume_l": 10
  }'
```

## Tech Stack

FastAPI · Pydantic · SymPy (exact rational equation balancing) · pure-Python numerics (no numpy/scipy dependency) · deployed on Render

## Scientific Integrity

Every simplification in this project is documented in the code and surfaced in the API's own output (as a `note`, `assumption`, or explicit `"unknown"`/`"indeterminate"` value) rather than hidden. If something isn't backed by real reference data or a real physical formula, the API says so instead of fabricating a number.

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Requires Python 3.12 (see `.python-version`).

## Author

Built by **Adedoyin Ifeoluwa James** ([@Adedoyinjames_](https://x.com/Adedoyinjames_) · [LinkedIn](https://www.linkedin.com/in/ifeoluwa-adedoyin/) · [email](mailto:ifeoluwaadedoyin19@gmail.com))
