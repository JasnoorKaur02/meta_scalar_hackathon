"""
inference.py — Baseline inference script for the Supply Chain Disruption environment.

Runs all five tasks sequentially using an LLM via the OpenAI-compatible API.

Mandatory stdout format:
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<json_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL  — LLM API base URL  (default: https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — API key (checked first)
  API_KEY       — API key fallback
  ENV_BASE_URL  — environment server URL (default: http://localhost:7860)
  TASK          — single task to run (optional; runs all 5 if unset)
  DIFFICULTY    — difficulty level 1-5 (default: 2)
"""

import json
import os
import sys

import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

HF_TOKEN: str = os.getenv("HF_TOKEN")
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")
DIFFICULTY: int = int(os.getenv("DIFFICULTY", "2"))

ENV_NAME = "supply-chain-disruption-env"

# ── System Prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "assess_disruption": """\
You are a supply chain risk analyst. You will be shown a disruption event with current \
warehouse stock, supplier status, and factory configurations.

Your job is to assess the situation by outputting EXACTLY this single line:
  assess situation: affected_components=[comp1,comp2,...] severity=<low|medium|high|critical> days_of_stock=<integer> factories_at_risk=[FAC_ID1,FAC_ID2,...]

Rules:
- affected_components: components whose supply is directly disrupted (chip, sensor, casing, motor, battery)
- severity: the disruption's severity level
- days_of_stock: minimum integer days any at-risk factory can keep running before a \
  disrupted component runs out (calculate from warehouse stock ÷ daily consumption rate)
- factories_at_risk: factories that use any affected component

Factory daily consumption = production_rate × recipe_quantity_per_component.
Example: FAC_ALPHA (80 units/day, chip×2) uses 160 chips/day. WH_NORTH has 400 chips → 2 days.

Output ONLY the single assessment line. No other text.""",

    "resolve_disruption": """\
You are a supply chain crisis manager. A disruption is active and you have a $200,000 budget \
and 5 action slots to restore supply chain integrity.

Available actions (issue ONE per turn):
  reroute supplier from <SUP_X> to <SUP_Y> for <N> <component>
    Cost: N × dest_supplier_cost × 1.3  |  shipment after supplier lead-time
  expedite <N> <component> from <WH_X> to <FAC_Y>
    Cost: N × $15 (airfreight)  |  immediate stock transfer
  reallocate <N> <component> from <WH_X> to <WH_Y>
    Cost: N × $5 (ground)  |  immediate stock transfer
  pause <FAC_X> for <N> days
    Cost: N × $5,000 (idle)  |  stops consumption & production
  activate emergency supplier for <N> <component>
    Cost: N × $120 (spot market)  |  immediate stock injection
  notify client <ORD_X> of <N> day delay
    Cost: $0  |  preserves client trust
  assess situation
    Cost: $0  |  shows full status report

Entity IDs:
  Suppliers : SUP_A (chip,sensor $50 7d), SUP_B (motor,sensor $80 5d),
              SUP_C (casing $30 4d), SUP_D (battery,motor $60 3d), SUP_E (chip,battery $55 6d)
  Warehouses: WH_NORTH (Chicago), WH_SOUTH (Dallas), WH_WEST (Los Angeles)
  Factories : FAC_ALPHA (Detroit→WH_NORTH), FAC_BETA (Houston→WH_SOUTH), FAC_GAMMA (Seattle→WH_WEST)
  Orders    : ORD_001 (CRITICAL), ORD_002 (HIGH), ORD_003 (MEDIUM), ORD_004 (LOW)

Strategy tips:
- Protect CRITICAL and HIGH orders first.
- Expedite or reallocate stock quickly — factories starve if any ONE component runs out.
- Diversify your actions (use multiple action types) for a better score.
- Stay under $200,000 total budget.

Reply with ONE action command per turn. No explanation needed.""",

    "cascade_management": """\
You are managing a supply chain through a 10-day cascading crisis. Each turn you take ONE \
action, then one day passes and factories either produce or starve.

This is a hard scenario:
- Multiple disruptions are active from day 1.
- A new disruption will be injected at step 4 — adapt when you see it.
- Every day an unfulfilled order is past its deadline incurs a penalty.
- You have a $200,000 budget across all 10 steps.

Available actions (one per step):
  reroute supplier from <SUP_X> to <SUP_Y> for <N> <component>
  expedite <N> <component> from <WH_X> to <FAC_Y>
  reallocate <N> <component> from <WH_X> to <WH_Y>
  pause <FAC_X> for <N> days
  activate emergency supplier for <N> <component>
  notify client <ORD_X> of <N> day delay
  assess situation

Key facts:
- Factories produce at FULL rate or ZERO (binary — one short component = factory starved).
- FAC_ALPHA 80/day, FAC_BETA 60/day, FAC_GAMMA 50/day.
- ORD_001 $12k/day late, ORD_002 $8k/day, ORD_003 $5k/day, ORD_004 $3k/day.
- Graded on: fulfillment rate, priority-weighted on-time, financial score, factory uptime.
- Emergency supplier ($120/unit) is the fastest fix but expensive.
- Rerouted shipments take lead_time days to arrive.
- Reallocate/expedite are immediate and cheap.

Prioritise: fill CRITICAL orders on time > reduce daily penalties > keep factories running > stay in budget.
Reply with ONE action command per turn.""",

    "budget_optimization": """\
You are optimizing a supply chain over a 30-day horizon with a $500,000 budget. \
Your goal is to minimize cost per unit produced while avoiding late penalties.

This is a long-horizon task:
- Three disruption waves will hit at steps 1, 10, and 20 — anticipate them.
- You have 30 action slots total across 30 simulated days.
- Graded primarily on cost-per-unit efficiency and order fulfillment.

Available actions (one per step):
  reroute supplier from <SUP_X> to <SUP_Y> for <N> <component>
  expedite <N> <component> from <WH_X> to <FAC_Y>
  reallocate <N> <component> from <WH_X> to <WH_Y>
  pause <FAC_X> for <N> days
  activate emergency supplier for <N> <component>
  notify client <ORD_X> of <N> day delay
  negotiate contract with <SUP_X> for <N> <component>
  assess situation

Orders (T4 set):
  ORD_A01 (CRITICAL, 600 units, due day 8,  $15k/day)
  ORD_A02 (HIGH,     500 units, due day 12, $10k/day)
  ORD_A03 (HIGH,     400 units, due day 16, $8k/day)
  ORD_A04 (MEDIUM,   350 units, due day 20, $5k/day)
  ORD_A05 (MEDIUM,   250 units, due day 25, $3.5k/day)
  ORD_A06 (LOW,      200 units, due day 28, $2k/day)

Cost tips:
- Reallocate ($5/unit) and expedite ($15/unit) are cheapest.
- Pre-negotiate contracts to lock in discounts before disruptions inflate prices.
- Emergency supplier ($120/unit) is last resort — avoid it for cost efficiency.
- Benchmark: aim for <$100 cost per unit produced.

Reply with ONE action command per turn.""",

    "supplier_negotiation": """\
You are a procurement strategist. Intelligence indicates that a major disruption \
is coming in 5 steps. Your job is to negotiate supply contracts NOW, before prices spike.

Key mechanic:
  negotiate contract with <SUP_X> for <N> <component>
    - Locks in units at a supplier discount (8–18% off normal price)
    - Delivery is GUARANTEED even if the supplier gets disrupted
    - Much cheaper than emergency spot market ($120/unit) after disruption

Discount rates:
  SUP_A (chip, sensor): 12% off   SUP_B (motor, sensor): 10% off
  SUP_C (casing):       18% off   SUP_D (battery, motor): 15% off
  SUP_E (chip, battery): 8% off

You also have access to:
  reallocate <N> <component> from <WH_X> to <WH_Y>  ($5/unit)
  expedite <N> <component> from <WH_X> to <FAC_Y>   ($15/unit)
  activate emergency supplier for <N> <component>    ($120/unit — expensive!)
  notify client <ORD_X> of <N> day delay
  assess situation

Watch the UPCOMING DISRUPTION field in each observation to track what's coming.

Graded on:
  - Supply security: do your contracts cover the disruption period needs?
  - Cost efficiency: savings vs. spot market alternative
  - Diversification: contract with multiple suppliers

Strategy: negotiate 2–4 contracts covering the disrupted components, then \
reallocate stock for immediate gaps. Don't overspend — $300k budget.

Reply with ONE action command per turn.""",
}

# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_user_prompt(obs: dict) -> str:
    lines = [
        f"Day {obs['day']} / {obs['max_days']}  |  "
        f"Budget: ${obs['budget_remaining']:,.0f}  |  "
        f"Penalties so far: ${obs['total_penalties_so_far']:,.0f}  |  "
        f"Difficulty: {obs.get('difficulty', 2)}/5",
        "",
    ]

    # Upcoming disruption intel (task 5)
    upcoming = obs.get("upcoming_disruption", "")
    if upcoming:
        lines.append(f"!! INTEL: {upcoming}")
        lines.append("")

    # Active disruptions
    lines.append("── Active Disruptions ──────────────────────────────")
    if obs["active_disruptions"]:
        for d in obs["active_disruptions"]:
            lines.append(
                f"  [{d['severity'].upper()}] {d['name']}  "
                f"{d['days_remaining']}d remaining  "
                f"suppliers:{d['affected_suppliers']}  "
                f"components:{d['affected_components']}"
            )
    else:
        lines.append("  None")

    # Orders
    lines.append("")
    lines.append("── Orders ──────────────────────────────────────────")
    for o in obs["orders"]:
        pct = int(100 * o["units_fulfilled"] / max(o["units_required"], 1))
        lines.append(
            f"  {o['id']} [{o['priority']:8s}]  "
            f"{o['units_fulfilled']:4d}/{o['units_required']:4d} ({pct:3d}%)  "
            f"due day {o['due_day']}  ${o['late_penalty_per_day']:,.0f}/day  "
            f"status:{o['status']}"
        )

    # Warehouses
    lines.append("")
    lines.append("── Warehouse Stock ─────────────────────────────────")
    for wh in obs["warehouses"]:
        stock = "  ".join(
            f"{k}:{wh['stock'].get(k, 0)}" for k in ["chip", "sensor", "casing", "motor", "battery"]
        )
        lines.append(f"  {wh['id']} ({wh['location']}): {stock}")

    # Suppliers
    lines.append("")
    lines.append("── Suppliers ───────────────────────────────────────")
    for s in obs["suppliers"]:
        status = f"DISRUPTED ({s['disruption_days_remaining']}d)" if s["disrupted"] else "OK"
        lines.append(
            f"  {s['id']} {s['name']:22s}  {status:18s}  "
            f"{str(s['components']):20s}  ${s['cost_per_unit']}/u  "
            f"lead {s['lead_time_days']}d"
        )

    # Factories
    lines.append("")
    lines.append("── Factories ───────────────────────────────────────")
    for f in obs["factories"]:
        lines.append(
            f"  {f['id']} ({f['location']}) -> {f['served_by']}  "
            f"status:{f['status']:7s}  "
            f"today:{f['units_produced_today']:3d}  "
            f"total:{f['total_units_produced']:4d}"
        )

    # Last action result
    if obs.get("last_action_result"):
        lines.append("")
        lines.append("── Last Action Result ──────────────────────────────")
        result = obs["last_action_result"]
        if len(result) > 800:
            result = result[:800] + "..."
        lines.append(result)

    return "\n".join(lines)


# ── Task Runner ────────────────────────────────────────────────────────────────

def run_task(task: str, seed: int = 42, difficulty: int = 2) -> dict:
    """
    Run one complete task episode and print mandatory log lines.
    Returns {task, score, rewards, success}.
    """
    try:
        resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task": task, "seed": seed, "difficulty": difficulty},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)
        print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
        return {"task": task, "score": 0.0, "rewards": [], "success": False}

    data = resp.json()
    session_id = data["session_id"]
    obs = data["observation"]
    max_steps = obs.get("max_days", 10)

    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards = []
    last_error = None

    for step_n in range(1, max_steps + 1):
        user_prompt = _build_user_prompt(obs)
        action_str = "assess situation"
        error = None

        try:
            completion = _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[task]},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                temperature=0.4,
            )
            action_str = completion.choices[0].message.content.strip()
        except Exception as exc:
            error = str(exc)
            action_str = "assess situation"
            last_error = error

        reward = 0.0
        done = False

        try:
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"session_id": session_id, "command": action_str},
                timeout=30,
            )
            if step_resp.status_code == 200:
                step_data = step_resp.json()
                obs = step_data["observation"]
                reward = step_data["reward"]
                done = step_data["done"]
            else:
                error = f"HTTP {step_resp.status_code}: {step_resp.text[:200]}"
                last_error = error
                done = True
        except Exception as exc:
            error = str(exc)
            last_error = error
            done = True

        rewards.append(reward)

        print(
            f"[STEP] step={step_n} "
            f"action={json.dumps(action_str)} "
            f"reward={reward:.2f} "
            f"done={str(done).lower()} "
            f"error={json.dumps(error)}",
            flush=True,
        )

        if done:
            break

    score = rewards[-1] if rewards else 0.0
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    success = last_error is None

    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} "
        f"score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return {"task": task, "score": score, "rewards": rewards, "success": success}


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    single_task = os.getenv("TASK")
    if single_task:
        tasks = [single_task]
    else:
        tasks = [
            "assess_disruption",
            "resolve_disruption",
            "cascade_management",
            "budget_optimization",
            "supplier_negotiation",
        ]

    results = []
    for task in tasks:
        result = run_task(task, difficulty=DIFFICULTY)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)
    print(
        f"\nOverall average score: {avg_score:.3f}  "
        "(" + ", ".join(f"{r['task']}={r['score']:.3f}" for r in results) + ")",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
