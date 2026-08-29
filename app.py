"""
Chemical World Model API

Start with:

    pip install fastapi uvicorn
    uvicorn app:app --reload

API documentation:

    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from simulation import (
    ELEMENTS,
    SimulationError,
    molar_mass,
    parse_formula,
    simulate_payload,
)
from experiment import run_experiment, ExperimentError
from molecular_dynamics import initialize_particles, step_particles


app = FastAPI(
    title="Chemical World Model API",
    description=(
        "API for representing chemical systems and calculating "
        "their physical and chemical state."
    ),
    version="1.0.0",
)

# CORS: comma-separated list of allowed origins, e.g.
#   ALLOWED_ORIGINS="https://your-frontend.vercel.app,http://localhost:5173"
# Defaults to "*" (any origin) so it works out of the box; lock this down
# to your real frontend domain(s) once you deploy it.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = (
    ["*"]
    if _allowed_origins.strip() == "*"
    else [o.strip() for o in _allowed_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    composition: dict[str, float] = Field(
        ...,
        description=(
            "Chemical formulas mapped to amounts in moles."
        ),
        examples=[
            {
                "H2": 2,
                "O2": 1
            }
        ],
    )

    temperature_k: float = Field(
        default=298.15,
        gt=0,
    )

    pressure_atm: float = Field(
        default=1.0,
        gt=0,
    )

    volume_l: float | None = Field(
        default=None,
        gt=0,
    )

    pH: float | None = Field(
        default=None,
        ge=0,
        le=14,
    )

    solvent: str | None = None

    time_s: float = Field(
        default=0.0,
        ge=0,
    )

    energy_j: float = 0.0

    electric_field_v_m: float = 0.0

    magnetic_field_t: float = 0.0

    reaction: dict[str, Any] | None = None


class ExperimentRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    initial_state: dict[str, Any] = Field(
        ...,
        description="Same shape as a /simulate body's state fields (composition required).",
        examples=[{"composition": {"H2": 4, "O2": 1}, "temperature_k": 298.15, "volume_l": 10}],
    )

    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ordered list of step objects. Each needs a 'type' field: "
            "add_material, remove_material, heat, cool, change_pressure, "
            "change_ph, change_solvent, apply_energy, apply_field, wait — "
            "plus that type's specific fields (see documentation)."
        ),
    )

    reaction: dict[str, Any] | None = Field(
        default=None,
        description="Same shape as /simulate's reaction field. Enables kinetics during 'wait' steps.",
    )


class ParticlesInitRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    composition: dict[str, float] = Field(...)
    temperature_k: float = Field(..., gt=0)
    max_particles: int = Field(default=120, ge=1, le=150)
    box_size: float = Field(default=100.0, gt=0)
    seed: int | None = None


class ParticlesStepRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    particles: list[dict[str, Any]] = Field(
        ...,
        description="The 'particles' array from a prior /particles/init or /particles/step response.",
    )
    dt: float = Field(..., gt=0)
    box_size: float = Field(default=100.0, gt=0)
    target_temperature_k: float | None = Field(default=None, gt=0)


class FormulaValidationRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    formula: str = Field(
        ...,
        examples=["Ca(OH)2"],
    )


@app.post("/formulas/validate")
def validate_formula(
    request: FormulaValidationRequest,
):
    """Cheap, single-formula check for live frontend input validation —
    no state or reaction required. Returns 422 with a readable error for
    an invalid formula instead of a full simulation error."""

    try:

        element_counts = parse_formula(
            request.formula,
            elements=ELEMENTS,
        )

        mass = molar_mass(
            request.formula,
            elements=ELEMENTS,
        )

        return {
            "valid": True,
            "formula": request.formula.strip(),
            "element_counts": element_counts,
            "molar_mass_g_mol": mass,
        }

    except ValueError as error:

        return JSONResponse(
            status_code=422,
            content={
                "valid": False,
                "formula": request.formula,
                "error": str(error),
            },
        )


@app.post("/experiment/run")
def run_experiment_endpoint(
    request: ExperimentRequest,
):
    """Replay a full multi-step experiment script and return the recorded
    timeline (one entry per step, each reproducible from the returned
    'reproducible_from' block). Stateless: resend the full step list to
    replay or extend an experiment — nothing is stored server-side."""

    try:

        result = run_experiment(
            initial_state_payload=request.initial_state,
            steps=request.steps,
            reaction_payload=request.reaction,
        )

        return result

    except (SimulationError, ExperimentError) as error:

        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_experiment",
                "detail": str(error),
            },
        )

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "error": "experiment_failure",
                "detail": str(error),
            },
        )


@app.post("/particles/init")
def particles_init(
    request: ParticlesInitRequest,
):
    """Build an initial molecular-motion particle ensemble for a
    composition at a given temperature (real Maxwell-Boltzmann-sampled
    velocities — see molecular_dynamics module docs for the reduced-unit
    system used). Feed the returned 'particles' array into /particles/step
    on each animation tick."""

    try:

        result = initialize_particles(
            composition=request.composition,
            temperature_k=request.temperature_k,
            elements=ELEMENTS,
            max_particles=request.max_particles,
            box_size=request.box_size,
            seed=request.seed,
        )

        return result

    except ValueError as error:

        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_particles_request",
                "detail": str(error),
            },
        )

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "error": "particles_failure",
                "detail": str(error),
            },
        )


@app.post("/particles/step")
def particles_step(
    request: ParticlesStepRequest,
):
    """Advance a particle ensemble by one physics timestep (real elastic
    collisions with walls and other particles; optional thermostat toward
    target_temperature_k when the experiment's temperature changes)."""

    try:

        result = step_particles(
            particles=request.particles,
            dt=request.dt,
            box_size=request.box_size,
            target_temperature_k=request.target_temperature_k,
        )

        return result

    except (ValueError, KeyError, TypeError) as error:

        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_particles_step",
                "detail": str(error),
            },
        )

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "error": "particles_failure",
                "detail": str(error),
            },
        )


@app.get("/")
def root():

    return {
        "name": "Chemical World Model",
        "status": "online",
        "description": (
            "Computational environment for modelling "
            "chemical states under controlled conditions."
        ),
        "documentation": "/docs",
        "simulation_endpoint": "/simulate",
        "formula_validation_endpoint": "/formulas/validate",
        "experiment_endpoint": "/experiment/run",
        "particles_init_endpoint": "/particles/init",
        "particles_step_endpoint": "/particles/step",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/simulate")
def simulate_chemical_system(
    request: SimulationRequest,
):

    try:

        result = simulate_payload(
            request.model_dump(
                exclude_none=True
            )
        )

        return result

    except SimulationError as error:

        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_simulation",
                "detail": str(error),
            },
        )

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "error": "simulation_failure",
                "detail": str(error),
            },
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
