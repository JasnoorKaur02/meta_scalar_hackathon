"""
server.py — FastAPI server for the Supply Chain Disruption environment.

Endpoints:
  POST /reset   body: {task, seed, difficulty}       -> {session_id, observation, done}
  POST /step    body: {session_id, command}          -> {session_id, observation, reward, done, info}
  GET  /state   query: ?session_id=<id>              -> SCState dict
  GET  /tasks                                        -> task metadata with difficulty info
  GET  /health                                       -> {status: "ok"}
  GET  /                                             -> env metadata
"""

import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supply_chain_env import SupplyChainEnv, SCAction

# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Supply Chain Disruption Manager",
    description="OpenEnv-compliant RL environment for supply chain crisis management.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> SupplyChainEnv instance
_sessions: Dict[str, SupplyChainEnv] = {}


# ── Request/Response Schemas ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "assess_disruption"
    seed: int = 42
    difficulty: int = 2   # 1 (easiest) to 5 (hardest)


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
    """Strip non-JSON-serialisable values (sets) from info dict."""
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


@app.get("/tasks")
def tasks():
    """Return all available tasks with difficulty metadata."""
    return {
        "tasks": {
            "assess_disruption": {
                "difficulty_level": "easy",
                "recommended_difficulty": 2,
                "max_steps": 1,
                "description": "Single-step structured assessment of a disruption event.",
            },
            "resolve_disruption": {
                "difficulty_level": "medium",
                "recommended_difficulty": 2,
                "max_steps": 5,
                "budget": 200000,
                "description": "Take up to 5 recovery actions within a $200k budget.",
            },
            "cascade_management": {
                "difficulty_level": "hard",
                "recommended_difficulty": 2,
                "max_steps": 10,
                "budget": 200000,
                "description": "10-day crisis with cascading disruptions and daily simulation.",
            },
            "budget_optimization": {
                "difficulty_level": "hard",
                "recommended_difficulty": 2,
                "max_steps": 30,
                "budget": 500000,
                "description": "30-day long-horizon cost optimization with 3 disruption waves.",
            },
            "supplier_negotiation": {
                "difficulty_level": "expert",
                "recommended_difficulty": 2,
                "max_steps": 8,
                "budget": 300000,
                "description": "Pre-negotiate supply contracts before an announced disruption hits.",
            },
        },
        "difficulty_scale": {
            "1": "Easiest — 2x starting stock, 2x budget",
            "2": "Normal — default stock and budget",
            "3": "Hard — 75% stock, 90% budget",
            "4": "Very Hard — 50% stock, 75% budget + extra disruption",
            "5": "Expert — 25% stock, 50% budget + two extra disruptions",
        },
        "progression_suggestion": [
            "assess_disruption (difficulty 1)",
            "assess_disruption (difficulty 2-3)",
            "resolve_disruption (difficulty 1-2)",
            "cascade_management (difficulty 1-2)",
            "supplier_negotiation (difficulty 2)",
            "budget_optimization (difficulty 2)",
            "cascade_management (difficulty 3-5)",
            "budget_optimization (difficulty 4-5)",
        ],
    }


@app.get("/")
def info():
    """Environment metadata."""
    return {
        "name": "supply-chain-disruption-env",
        "version": "2.0.0",
        "description": (
            "OpenEnv-compliant RL environment for supply chain disruption management. "
            "An AI agent learns to handle crises across a multi-node supply chain via "
            "natural language actions. Now with 5 tasks, 16 disruption scenarios, "
            "dynamic difficulty (1-5), and a Gradio visual dashboard."
        ),
        "tasks": {
            "assess_disruption":   {"difficulty": "easy",   "max_steps": 1},
            "resolve_disruption":  {"difficulty": "medium", "max_steps": 5,  "budget": 200000},
            "cascade_management":  {"difficulty": "hard",   "max_steps": 10, "budget": 200000},
            "budget_optimization": {"difficulty": "hard",   "max_steps": 30, "budget": 500000},
            "supplier_negotiation":{"difficulty": "expert", "max_steps": 8,  "budget": 300000},
        },
        "disruption_count": 16,
        "dynamic_difficulty": "1 (easiest) to 5 (hardest)",
        "action_types": [
            "reroute_supplier", "expedite_shipping", "reallocate_stock",
            "pause_factory", "activate_emergency_supplier", "notify_client",
            "negotiate_contract", "assess_situation",
        ],
        "endpoints": {
            "POST /reset":  "Start a new episode (params: task, seed, difficulty)",
            "POST /step":   "Take an action",
            "GET  /state":  "Get current state summary (?session_id=...)",
            "GET  /tasks":  "List all tasks with difficulty info",
            "GET  /health": "Liveness probe",
        },
        "active_sessions": len(_sessions),
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    """
    Start a new episode.

    Returns session_id that must be passed to /step and /state.
    Difficulty 1 = easiest (2x stock/budget), 5 = hardest (0.25x stock, 0.5x budget).
    """
    if req is None:
        req = ResetRequest()
    try:
        env = SupplyChainEnv()
        obs = env.reset(task=req.task, seed=req.seed, difficulty=req.difficulty)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)
