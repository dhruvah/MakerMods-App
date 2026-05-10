"""ACT orchestrator phase monitoring and control API."""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ── In-memory ACT state ──────────────────────────────────────────────────────

class _ACTState:
    def __init__(self) -> None:
        self.phase: str = "init"
        self.phase_start: float = time.monotonic()
        self.paused: bool = False
        self.available_phases: list[str] = [
            "pick_and_drop",
            "press_lever",
            "wait_for_toast",
            "pick_out_of_toaster",
        ]

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_start = time.monotonic()

    def elapsed_s(self) -> float:
        return round(time.monotonic() - self.phase_start, 1)


_state = _ACTState()


# ── Response / request models ────────────────────────────────────────────────

class PhaseStatusResponse(BaseModel):
    phase: str
    elapsed_s: float
    paused: bool
    available_phases: list[str]


class AdvancePhaseRequest(BaseModel):
    phase_name: str


class SetPhasesRequest(BaseModel):
    phases: list[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/status", response_model=PhaseStatusResponse)
async def get_status():
    """Return current phase, elapsed time, and available phases."""
    return PhaseStatusResponse(
        phase=_state.phase,
        elapsed_s=_state.elapsed_s(),
        paused=_state.paused,
        available_phases=_state.available_phases,
    )


@router.post("/advance-phase")
async def advance_phase(request: AdvancePhaseRequest):
    """Manually set the active phase and reset the elapsed timer."""
    if request.phase_name not in _state.available_phases and request.phase_name != "init":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown phase '{request.phase_name}'. Available: {_state.available_phases}",
        )
    _state.set_phase(request.phase_name)
    return {"phase": request.phase_name, "elapsed_s": 0.0}


@router.post("/set-phases")
async def set_phases(request: SetPhasesRequest):
    """Replace the available phase list."""
    if not request.phases:
        raise HTTPException(status_code=400, detail="At least one phase is required")
    _state.available_phases = request.phases
    # Reset to init if current phase is no longer valid
    if _state.phase not in request.phases:
        _state.set_phase("init")
    return {"phases": request.phases}


@router.post("/reset")
async def reset_state():
    """Reset phase tracking to initial state."""
    _state.phase = "init"
    _state.phase_start = time.monotonic()
    _state.paused = False
    return {"message": "ACT state reset"}
