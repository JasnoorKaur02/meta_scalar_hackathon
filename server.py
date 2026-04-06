"""
server.py — FastAPI server for the Supply Chain Disruption environment.

Endpoints:
  POST /reset   body: {task, seed}               → {session_id, observation, done}
  POST /step    body: {session_id, command}       → {session_id, observation, reward, done, info}
  GET  /state   query: ?session_id=<id>           → SCState dict
  GET  /health                                    → {status: "ok"}
  GET  /                                          → env metadata
"""

import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from supply_chain_env import SupplyChainEnv, SCAction

# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Supply Chain Disruption Manager",
    description="OpenEnv-compliant RL environment for supply chain crisis management.",
    version="1.0.0",
)

# In-memory session store: session_id → SupplyChainEnv instance
_sessions: Dict[str, SupplyChainEnv] = {}


# ── Request/Response Schemas ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "assess_disruption"
    seed: int = 42


class StepRequest(BaseModel):
    session_id: str
    command: str


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> SupplyChainEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env


def _safe_info(info: dict) -> dict:
    """Strip non-JSON-serialisable values (e.g. sets) from info dict."""
    out = {}
    for k, v in info.items():
        if isinstance(v, set):
            out[k] = sorted(v)
        elif isinstance(v, dict):
            out[k] = _safe_info(v)
        elif isinstance(v, (str, int, float, bool, list, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/")
def info():
    """Environment metadata."""
    return {
        "name": "supply-chain-disruption-env",
        "version": "1.0.0",
        "description": (
            "OpenEnv-compliant RL environment for supply chain disruption management. "
            "An AI agent learns to handle crises across a multi-node supply chain via "
            "natural language actions."
        ),
        "tasks": {
            "assess_disruption": {
                "difficulty": "easy",
                "max_steps": 1,
                "description": "Single-step structured assessment of a disruption event.",
            },
            "resolve_disruption": {
                "difficulty": "medium",
                "max_steps": 5,
                "budget": 200000,
                "description": "Take up to 5 recovery actions within a $200k budget.",
            },
            "cascade_management": {
                "difficulty": "hard",
                "max_steps": 10,
                "budget": 200000,
                "description": "10-day crisis with cascading disruptions and daily simulation.",
            },
        },
        "action_types": [
            "reroute_supplier", "expedite_shipping", "reallocate_stock",
            "pause_factory", "activate_emergency_supplier", "notify_client",
            "assess_situation",
        ],
        "endpoints": {
            "POST /reset": "Start a new episode",
            "POST /step": "Take an action",
            "GET  /state": "Get current state summary (?session_id=...)",
            "GET  /health": "Liveness probe",
        },
        "active_sessions": len(_sessions),
    }


@app.post("/reset")
def reset(req: ResetRequest):
    """
    Start a new episode.

    Returns session_id that must be passed to /step and /state.
    """
    try:
        env = SupplyChainEnv()
        obs = env.reset(task=req.task, seed=req.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    session_id = str(uuid.uuid4())
    _sessions[session_id] = env

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "done": False,
    }


@app.post("/step")
def step(req: StepRequest):
    """
    Take one action in an existing episode.

    Body: {session_id: str, command: str}
    """
    env = _get_session(req.session_id)

    try:
        obs, reward, done, info = env.step(SCAction(command=req.command))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "session_id": req.session_id,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": _safe_info(info),
    }


@app.get("/state")
def state(session_id: str):
    """
    Return a lightweight SCState summary for the given session.

    Query param: ?session_id=<uuid>
    """
    env = _get_session(session_id)
    return env.state().model_dump()
