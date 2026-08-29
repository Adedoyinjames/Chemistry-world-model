"""
Chemical World Model API

Start with:

    pip install fastapi uvicorn
    uvicorn app:app --reload

API documentation:

    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from simulation import (
    SimulationError,
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
        port=8000,
        reload=False,
    )
