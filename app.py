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
