"""
supply_chain_env.py — OpenEnv-compliant SupplyChainEnv.

Implements:
  reset(task, seed) → SCObservation
  step(SCAction)    → (SCObservation, float, bool, dict)
  state()           → SCState

Three tasks:
  assess_disruption   — 1 step,  grade structured NL assessment
  resolve_disruption  — 5 steps, $200k budget, recovery planning
  cascade_management  — 10 steps, cascading disruptions, daily simulation
"""

import copy
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from world import (
    SUPPLIERS, WAREHOUSES, FACTORIES, ORDERS, DISRUPTIONS,
    COMPONENTS, PRIORITY_ORDER, MAX_DAILY_PENALTY, MAX_TOTAL_PENALTY,
)
from parser import parse_action, parse_assessment

# ════════════════════════════════════════════════════════════════════════════
#  Pydantic Models (OpenEnv spec)
# ════════════════════════════════════════════════════════════════════════════

class SupplierStatus(BaseModel):
    id: str
    name: str
    location: str
    components: List[str]
    reliability: float
    lead_time_days: int
    cost_per_unit: float
    disrupted: bool
    disruption_days_remaining: int


class WarehouseStatus(BaseModel):
    id: str
    name: str
    location: str
    stock: Dict[str, int]


class FactoryStatus(BaseModel):
    id: str
    name: str
    location: str
    served_by: str
    production_rate: int
    recipe: Dict[str, int]
    status: str                  # "running" | "starved" | "paused"
    units_produced_today: int
    total_units_produced: int
    pause_days_remaining: int


class OrderStatus(BaseModel):
    id: str
    customer: str
    units_required: int
    units_fulfilled: int
    due_day: int
    late_penalty_per_day: float
    priority: str
    status: str                  # "pending" | "fulfilled" | "late"
    days_late: int
    total_penalties_accrued: float


class DisruptionInfo(BaseModel):
    id: str
    name: str
    severity: str
    affected_suppliers: List[str]
    affected_components: List[str]
    days_remaining: int


class SCAction(BaseModel):
    command: str


class SCObservation(BaseModel):
    task: str
    day: int
    max_days: int
    budget_remaining: float
    total_penalties_so_far: float
    active_disruptions: List[DisruptionInfo]
    suppliers: List[SupplierStatus]
    warehouses: List[WarehouseStatus]
    factories: List[FactoryStatus]
    orders: List[OrderStatus]
    last_action_result: str
    score_so_far: float
    done: bool
    available_actions: List[str]


class SCState(BaseModel):
    task: str
    day: int
    budget_remaining: float
    total_penalties: float
    orders_fulfilled: int
    orders_total: int
    factories_running: int
    done: bool
    cumulative_reward: float


# ════════════════════════════════════════════════════════════════════════════
#  World State Initialisation & Mutation
# ════════════════════════════════════════════════════════════════════════════

_TASK_MAX_STEPS = {
    "assess_disruption": 1,
    "resolve_disruption": 5,
    "cascade_management": 10,
}

_TASK_BUDGET = {
    "assess_disruption": 200_000.0,
    "resolve_disruption": 200_000.0,
    "cascade_management": 200_000.0,
}


def _init_world(task: str) -> Dict:
    """Return a fresh mutable world state (deep copy of seed data)."""
    suppliers = {
        k: {**copy.deepcopy(v), "disrupted": False, "disruption_days_remaining": 0}
        for k, v in SUPPLIERS.items()
    }
    warehouses = {
        k: {**copy.deepcopy(v), "stock": copy.deepcopy(v["stock"])}
        for k, v in WAREHOUSES.items()
    }
    factories = {
        k: {
            **copy.deepcopy(v),
            "status": "running",
            "units_produced_today": 0,
            "total_units_produced": 0,
            "pause_days_remaining": 0,
        }
        for k, v in FACTORIES.items()
    }
    orders = {
        k: {
            **copy.deepcopy(v),
            "units_required": v["units"],  # mutable (order_impact can change it)
            "units_fulfilled": 0,
            "status": "pending",
            "days_late": 0,
            "total_penalties_accrued": 0.0,
            "notified": False,
        }
        for k, v in ORDERS.items()
    }

    return {
        "task": task,
        "day": 0,
        "budget_remaining": _TASK_BUDGET[task],
        "total_cost": 0.0,
        "total_penalties": 0.0,
        "step_count": 0,
        "cumulative_reward": 0.0,
        "score_so_far": 0.0,
        "done": False,
        "last_action_result": "Episode started.",
        "action_types_used": set(),  # tracks diversity
        "pending_shipments": [],     # {arrives_day, warehouse, component, units}
        "active_disruptions": {},    # disruption_id → disruption dict + days_remaining
        "suppliers": suppliers,
        "warehouses": warehouses,
        "factories": factories,
        "orders": orders,
    }


def _apply_disruption(world: Dict, disruption_id: str, override_days: Optional[int] = None) -> None:
    """Apply a disruption event to the live world state."""
    d = copy.deepcopy(DISRUPTIONS[disruption_id])
    days = override_days if override_days is not None else d["delay_days"]

    # Mark affected suppliers as disrupted (only when delay_days > 0)
    if days > 0:
        for sup_id in d["affected_suppliers"]:
            if sup_id in world["suppliers"]:
                world["suppliers"][sup_id]["disrupted"] = True
                world["suppliers"][sup_id]["disruption_days_remaining"] = days

    # Apply immediate stock removals
    for wh_id, deltas in d.get("stock_impact", {}).items():
        if wh_id in world["warehouses"]:
            for comp, delta in deltas.items():
                cur = world["warehouses"][wh_id]["stock"].get(comp, 0)
                world["warehouses"][wh_id]["stock"][comp] = max(0, cur + delta)

    # Apply order demand multipliers
    for order_id, mult in d.get("order_impact", {}).items():
        if order_id in world["orders"]:
            world["orders"][order_id]["units_required"] = int(
                world["orders"][order_id]["units_required"] * mult
            )

    # Register in active disruptions (keep at least 1 day so it appears in obs)
    world["active_disruptions"][disruption_id] = {
        **d,
        "days_remaining": max(days, 1),
    }


def _simulate_day(world: Dict) -> float:
    """
    Advance the world by one day.

    Order of operations:
      1. Increment day counter
      2. Arrive pending shipments
      3. Accrue late penalties (BEFORE production so even partial-day lateness is charged)
      4. Factory production (binary: full-rate or starved)
      5. Allocate production to orders by priority
      6. Tick down disruption counters; resolve expired ones

    Returns the total penalties accrued THIS day.
    """
    world["day"] += 1
    day = world["day"]

    # ── 1. Pending shipments ─────────────────────────────────────────────────
    for shipment in world["pending_shipments"][:]:
        if shipment["arrives_day"] <= day:
            wh = shipment["warehouse"]
            comp = shipment["component"]
            if wh in world["warehouses"]:
                world["warehouses"][wh]["stock"][comp] = (
                    world["warehouses"][wh]["stock"].get(comp, 0) + shipment["units"]
                )
            world["pending_shipments"].remove(shipment)

    # ── 2. Accrue late penalties (start-of-production snapshot) ─────────────
    daily_penalty = 0.0
    for order in world["orders"].values():
        if order["status"] != "fulfilled" and day > order["due_day"]:
            order["days_late"] = day - order["due_day"]
            p = order["late_penalty_per_day"]
            order["total_penalties_accrued"] += p
            world["total_penalties"] += p
            daily_penalty += p

    # ── 3. Factory production ────────────────────────────────────────────────
    total_produced = 0
    for fac in world["factories"].values():
        fac["units_produced_today"] = 0

        if fac["pause_days_remaining"] > 0:
            fac["status"] = "paused"
            fac["pause_days_remaining"] -= 1
            continue

        wh_id = fac["served_by"]
        wh_stock = world["warehouses"][wh_id]["stock"]
        recipe = fac["recipe"]
        rate = fac["production_rate"]

        # Binary check: can we produce at full rate?
        can_produce = all(
            wh_stock.get(comp, 0) >= qty * rate
            for comp, qty in recipe.items()
        )

        if can_produce:
            for comp, qty in recipe.items():
                wh_stock[comp] -= qty * rate
            fac["status"] = "running"
            fac["units_produced_today"] = rate
            fac["total_units_produced"] += rate
            total_produced += rate
        else:
            fac["status"] = "starved"

    # ── 4. Allocate production to orders (CRITICAL first) ───────────────────
    available = total_produced
    for priority in PRIORITY_ORDER:
        for order in world["orders"].values():
            if order["priority"] == priority and order["status"] != "fulfilled":
                need = order["units_required"] - order["units_fulfilled"]
                give = min(need, available)
                order["units_fulfilled"] += give
                available -= give
                if order["units_fulfilled"] >= order["units_required"]:
                    order["status"] = "fulfilled"
                if available <= 0:
                    break
        if available <= 0:
            break

    # ── 5. Tick down active disruptions ─────────────────────────────────────
    for dis_id in list(world["active_disruptions"].keys()):
        dis = world["active_disruptions"][dis_id]
        dis["days_remaining"] -= 1
        if dis["days_remaining"] <= 0:
            # Un-disrupt suppliers
            for sup_id in dis["affected_suppliers"]:
                if sup_id in world["suppliers"]:
                    world["suppliers"][sup_id]["disrupted"] = False
                    world["suppliers"][sup_id]["disruption_days_remaining"] = 0
            del world["active_disruptions"][dis_id]

    return daily_penalty


# ════════════════════════════════════════════════════════════════════════════
#  Action Execution
# ════════════════════════════════════════════════════════════════════════════

def _action_cost(world: Dict, parsed: Dict) -> float:
    """Calculate cost of a parsed action without applying it."""
    t = parsed.get("type")
    if t == "reroute_supplier":
        sup_id = parsed.get("to_supplier")
        units = parsed.get("units") or 100
        cpu = world["suppliers"].get(sup_id, {}).get("cost_per_unit", 100.0) if sup_id else 100.0
        return units * cpu * 1.3
    if t == "expedite_shipping":
        return (parsed.get("units") or 100) * 15.0
    if t == "reallocate_stock":
        return (parsed.get("units") or 100) * 5.0
    if t == "pause_factory":
        return (parsed.get("days") or 1) * 5_000.0
    if t == "activate_emergency_supplier":
        return (parsed.get("units") or 100) * 120.0
    if t in ("notify_client", "assess_situation"):
        return 0.0
    return 0.0


def _execute_action(world: Dict, parsed: Dict) -> Tuple[float, str, bool]:
    """
    Apply a parsed action to world state.
    Returns (step_reward, message, success).
    """
    t = parsed.get("type", "unknown")

    if t == "unknown":
        return 0.0, f"Unrecognised command: '{parsed.get('raw', '')}'. Try 'assess situation' to see options.", False

    if t == "assess_situation":
        world["action_types_used"].add(t)
        report = _situation_report(world)
        return 0.05, report, True

    cost = _action_cost(world, parsed)
    if cost > world["budget_remaining"]:
        return 0.0, (
            f"Insufficient budget — need ${cost:,.0f}, "
            f"have ${world['budget_remaining']:,.0f}."
        ), False

    world["action_types_used"].add(t)

    if t == "reroute_supplier":
        return _do_reroute(world, parsed, cost)
    if t == "expedite_shipping":
        return _do_expedite(world, parsed, cost)
    if t == "reallocate_stock":
        return _do_reallocate(world, parsed, cost)
    if t == "pause_factory":
        return _do_pause(world, parsed, cost)
    if t == "activate_emergency_supplier":
        return _do_emergency(world, parsed, cost)
    if t == "notify_client":
        return _do_notify(world, parsed, cost)

    return 0.0, f"Unknown action type: {t}", False


def _deduct_cost(world: Dict, cost: float) -> None:
    world["budget_remaining"] -= cost
    world["total_cost"] += cost


def _most_depleted_warehouse(world: Dict, component: str) -> str:
    """Return the warehouse ID with the lowest stock of component."""
    return min(
        world["warehouses"].keys(),
        key=lambda w: world["warehouses"][w]["stock"].get(component, 0),
    )


def _do_reroute(world: Dict, p: Dict, cost: float) -> Tuple[float, str, bool]:
    to_sup = p.get("to_supplier")
    from_sup = p.get("from_supplier")
    comp = p.get("component")
    units = p.get("units") or 100

    if not to_sup or to_sup not in world["suppliers"]:
        return 0.0, f"Unknown destination supplier '{to_sup}'.", False
    if not comp:
        return 0.0, "No component specified for reroute.", False
    if comp not in world["suppliers"][to_sup]["components"]:
        return 0.0, f"{to_sup} does not supply '{comp}'.", False

    _deduct_cost(world, cost)
    lead = world["suppliers"][to_sup]["lead_time_days"]
    arrives = world["day"] + lead
    target_wh = _most_depleted_warehouse(world, comp)

    world["pending_shipments"].append({
        "arrives_day": arrives,
        "warehouse": target_wh,
        "component": comp,
        "units": units,
    })
    return 0.15, (
        f"Rerouted {units}× {comp} from {from_sup} → {to_sup}. "
        f"Shipment of {units} arrives at {target_wh} on day {arrives}. "
        f"Cost: ${cost:,.0f}."
    ), True


def _do_expedite(world: Dict, p: Dict, cost: float) -> Tuple[float, str, bool]:
    from_wh = p.get("from_warehouse")
    to_fac = p.get("to_factory")
    comp = p.get("component")
    units = p.get("units") or 100

    if not from_wh or from_wh not in world["warehouses"]:
        return 0.0, f"Unknown source warehouse '{from_wh}'.", False
    if not to_fac or to_fac not in world["factories"]:
        return 0.0, f"Unknown target factory '{to_fac}'.", False
    if not comp:
        return 0.0, "No component specified for expedite.", False

    to_wh = world["factories"][to_fac]["served_by"]
    if from_wh == to_wh:
        return 0.0, f"{from_wh} already serves {to_fac} — expedite within same warehouse is a no-op.", False

    available = world["warehouses"][from_wh]["stock"].get(comp, 0)
    actual = min(units, available)
    if actual <= 0:
        return 0.0, f"No '{comp}' available in {from_wh}.", False

    actual_cost = actual * 15.0
    _deduct_cost(world, actual_cost)
    world["warehouses"][from_wh]["stock"][comp] -= actual
    world["warehouses"][to_wh]["stock"][comp] = (
        world["warehouses"][to_wh]["stock"].get(comp, 0) + actual
    )
    return 0.20, (
        f"Expedited {actual}× {comp} from {from_wh} → {to_fac} ({to_wh}). "
        f"Cost: ${actual_cost:,.0f}."
    ), True


def _do_reallocate(world: Dict, p: Dict, cost: float) -> Tuple[float, str, bool]:
    from_wh = p.get("from_warehouse")
    to_wh = p.get("to_warehouse")
    comp = p.get("component")
    units = p.get("units") or 100

    if not from_wh or from_wh not in world["warehouses"]:
        return 0.0, f"Unknown source warehouse '{from_wh}'.", False
    if not to_wh or to_wh not in world["warehouses"]:
        return 0.0, f"Unknown destination warehouse '{to_wh}'.", False
    if from_wh == to_wh:
        return 0.0, "Source and destination warehouses are the same.", False
    if not comp:
        return 0.0, "No component specified for reallocate.", False

    available = world["warehouses"][from_wh]["stock"].get(comp, 0)
    actual = min(units, available)
    if actual <= 0:
        return 0.0, f"No '{comp}' available in {from_wh}.", False

    actual_cost = actual * 5.0
    _deduct_cost(world, actual_cost)
    world["warehouses"][from_wh]["stock"][comp] -= actual
    world["warehouses"][to_wh]["stock"][comp] = (
        world["warehouses"][to_wh]["stock"].get(comp, 0) + actual
    )
    return 0.10, (
        f"Reallocated {actual}× {comp} from {from_wh} → {to_wh}. "
        f"Cost: ${actual_cost:,.0f}."
    ), True


def _do_pause(world: Dict, p: Dict, cost: float) -> Tuple[float, str, bool]:
    fac_id = p.get("factory")
    days = p.get("days") or 1

    if not fac_id or fac_id not in world["factories"]:
        return 0.0, f"Unknown factory '{fac_id}'.", False

    _deduct_cost(world, cost)
    world["factories"][fac_id]["pause_days_remaining"] = days
    return 0.05, (
        f"Paused {fac_id} for {days} day(s). "
        f"Factory will resume after day {world['day'] + days}. "
        f"Cost: ${cost:,.0f}."
    ), True


def _do_emergency(world: Dict, p: Dict, cost: float) -> Tuple[float, str, bool]:
    comp = p.get("component")
    units = p.get("units") or 100
    target = p.get("target_warehouse")

    if not comp:
        return 0.0, "No component specified for emergency order.", False

    _deduct_cost(world, cost)

    if not target or target not in world["warehouses"]:
        target = _most_depleted_warehouse(world, comp)

    world["warehouses"][target]["stock"][comp] = (
        world["warehouses"][target]["stock"].get(comp, 0) + units
    )
    return 0.20, (
        f"Emergency order fulfilled: {units}× {comp} added to {target}. "
        f"Cost: ${cost:,.0f} (spot-market premium)."
    ), True


def _do_notify(world: Dict, p: Dict, _cost: float) -> Tuple[float, str, bool]:
    order_id = p.get("order_id")
    delay = p.get("expected_delay_days") or 1

    if not order_id or order_id not in world["orders"]:
        return 0.0, f"Unknown order '{order_id}'.", False

    world["orders"][order_id]["notified"] = True
    world["orders"][order_id]["expected_delay"] = delay
    customer = world["orders"][order_id]["customer"]
    return 0.05, (
        f"Notification sent to {customer} ({order_id}): "
        f"expected delay of {delay} day(s). No cost incurred."
    ), True


# ════════════════════════════════════════════════════════════════════════════
#  Observation / State Builders
# ════════════════════════════════════════════════════════════════════════════

_AVAILABLE_ACTIONS = [
    "reroute supplier from <SUP_X> to <SUP_Y> for <N> <component>",
    "expedite <N> <component> from <WH_X> to <FAC_Y>",
    "reallocate <N> <component> from <WH_X> to <WH_Y>",
    "pause <FAC_X> for <N> days",
    "activate emergency supplier for <N> <component>",
    "notify client <ORD_X> of <N> day delay",
    "assess situation",
]


def _build_observation(world: Dict) -> SCObservation:
    task = world["task"]
    max_days = _TASK_MAX_STEPS[task]

    suppliers = [
        SupplierStatus(
            id=s["id"],
            name=s["name"],
            location=s["location"],
            components=s["components"],
            reliability=s["reliability"],
            lead_time_days=s["lead_time_days"],
            cost_per_unit=s["cost_per_unit"],
            disrupted=s["disrupted"],
            disruption_days_remaining=s["disruption_days_remaining"],
        )
        for s in world["suppliers"].values()
    ]

    warehouses = [
        WarehouseStatus(
            id=w["id"],
            name=w["name"],
            location=w["location"],
            stock=dict(w["stock"]),
        )
        for w in world["warehouses"].values()
    ]

    factories = [
        FactoryStatus(
            id=f["id"],
            name=f["name"],
            location=f["location"],
            served_by=f["served_by"],
            production_rate=f["production_rate"],
            recipe=dict(f["recipe"]),
            status=f["status"],
            units_produced_today=f["units_produced_today"],
            total_units_produced=f["total_units_produced"],
            pause_days_remaining=f["pause_days_remaining"],
        )
        for f in world["factories"].values()
    ]

    orders = [
        OrderStatus(
            id=o["id"],
            customer=o["customer"],
            units_required=o["units_required"],
            units_fulfilled=o["units_fulfilled"],
            due_day=o["due_day"],
            late_penalty_per_day=o["late_penalty_per_day"],
            priority=o["priority"],
            status=o["status"],
            days_late=o["days_late"],
            total_penalties_accrued=o["total_penalties_accrued"],
        )
        for o in world["orders"].values()
    ]

    disruptions = [
        DisruptionInfo(
            id=d["id"],
            name=d["name"],
            severity=d["severity"],
            affected_suppliers=d["affected_suppliers"],
            affected_components=d.get("affected_components", []),
            days_remaining=d["days_remaining"],
        )
        for d in world["active_disruptions"].values()
    ]

    return SCObservation(
        task=task,
        day=world["day"],
        max_days=max_days,
        budget_remaining=world["budget_remaining"],
        total_penalties_so_far=world["total_penalties"],
        active_disruptions=disruptions,
        suppliers=suppliers,
        warehouses=warehouses,
        factories=factories,
        orders=orders,
        last_action_result=world["last_action_result"],
        score_so_far=world["score_so_far"],
        done=world["done"],
        available_actions=_AVAILABLE_ACTIONS,
    )


def _build_state(world: Dict) -> SCState:
    fac_running = sum(
        1 for f in world["factories"].values() if f["status"] == "running"
    )
    orders_fulfilled = sum(
        1 for o in world["orders"].values() if o["status"] == "fulfilled"
    )
    return SCState(
        task=world["task"],
        day=world["day"],
        budget_remaining=world["budget_remaining"],
        total_penalties=world["total_penalties"],
        orders_fulfilled=orders_fulfilled,
        orders_total=len(world["orders"]),
        factories_running=fac_running,
        done=world["done"],
        cumulative_reward=world["cumulative_reward"],
    )


# ════════════════════════════════════════════════════════════════════════════
#  Situation Report (assess_situation action)
# ════════════════════════════════════════════════════════════════════════════

def _situation_report(world: Dict) -> str:
    lines = [
        f"=== Situation Report — Day {world['day']} ===",
        f"Budget remaining : ${world['budget_remaining']:>10,.0f}",
        f"Total penalties  : ${world['total_penalties']:>10,.0f}",
        "",
        "Active disruptions:",
    ]
    if world["active_disruptions"]:
        for d in world["active_disruptions"].values():
            lines.append(
                f"  [{d['severity'].upper()}] {d['name']} — "
                f"{d['days_remaining']}d remaining — "
                f"affects {d['affected_suppliers']} — "
                f"components: {d.get('affected_components', [])}"
            )
    else:
        lines.append("  None")

    lines += ["", "Warehouse stock:"]
    for wh in world["warehouses"].values():
        stock_str = "  ".join(f"{c}:{world['warehouses'][wh['id']]['stock'].get(c,0)}" for c in COMPONENTS)
        lines.append(f"  {wh['id']} ({wh['location']}): {stock_str}")

    lines += ["", "Factory status:"]
    for fac in world["factories"].values():
        lines.append(
            f"  {fac['id']}: {fac['status'].upper():8s} | "
            f"produced today {fac['units_produced_today']:3d} | "
            f"total {fac['total_units_produced']:4d}"
        )

    lines += ["", "Order status:"]
    for o in world["orders"].values():
        pct = int(100 * o["units_fulfilled"] / max(o["units_required"], 1))
        lines.append(
            f"  {o['id']} [{o['priority']:8s}] "
            f"{o['units_fulfilled']:4d}/{o['units_required']:4d} ({pct:3d}%) "
            f"due day {o['due_day']} — {o['status']}"
        )

    pending = world["pending_shipments"]
    if pending:
        lines += ["", "Pending shipments:"]
        for s in pending:
            lines.append(
                f"  {s['units']}× {s['component']} → {s['warehouse']} "
                f"arrives day {s['arrives_day']}"
            )

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Task 1 Grader
# ════════════════════════════════════════════════════════════════════════════

def _compute_days_of_stock(disruption: Dict, world: Dict) -> int:
    """
    Minimum integer days any at-risk factory can keep running before it runs
    out of any affected component, given current warehouse stock.
    Returns 30 if no factory uses affected components.
    """
    affected = set(disruption.get("affected_components", []))
    if not affected:
        return 30

    min_days = float("inf")
    for fac in world["factories"].values():
        recipe = fac["recipe"]
        rate = fac["production_rate"]
        wh_stock = world["warehouses"][fac["served_by"]]["stock"]

        uses_affected = any(c in recipe for c in affected)
        if not uses_affected:
            continue

        for comp in affected:
            if comp in recipe:
                per_day = recipe[comp] * rate
                if per_day > 0:
                    days = wh_stock.get(comp, 0) // per_day
                    min_days = min(min_days, days)

    return int(min_days) if min_days != float("inf") else 30


def _compute_factories_at_risk(disruption: Dict, world: Dict) -> List[str]:
    """Return list of factory IDs that use any affected component."""
    affected = set(disruption.get("affected_components", []))
    return [
        fac_id
        for fac_id, fac in world["factories"].items()
        if any(c in fac["recipe"] for c in affected)
    ]


def _grade_task1(response: str, disruption: Dict, world: Dict) -> Tuple[float, Dict]:
    """
    Grade a Task-1 assessment response.

    Scoring:
      affected_components  30%  F1 vs ground truth
      severity             25%  exact=1.0, ±1 level=0.6, ±2=0.2, else=0
      days_of_stock        25%  exact=1.0, ±1=0.7, ±3=0.3, else=0
      factories_at_risk    20%  recall (correct / total expected)
    """
    parsed = parse_assessment(response)

    # Ground truth
    true_comps = set(disruption.get("affected_components", []))
    true_sev = disruption["severity"].lower()
    true_days = _compute_days_of_stock(disruption, world)
    true_facs = set(_compute_factories_at_risk(disruption, world))

    # ── affected_components (F1) ─────────────────────────────────────────────
    agent_comps = set(parsed["affected_components"])
    if true_comps or agent_comps:
        tp = len(agent_comps & true_comps)
        prec = tp / len(agent_comps) if agent_comps else 0.0
        rec = tp / len(true_comps) if true_comps else 1.0
        comp_score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    else:
        comp_score = 1.0

    # ── severity ─────────────────────────────────────────────────────────────
    sev_levels = ["low", "medium", "high", "critical"]
    agent_sev = (parsed["severity"] or "").lower()
    if agent_sev == true_sev:
        sev_score = 1.0
    elif agent_sev in sev_levels and true_sev in sev_levels:
        diff = abs(sev_levels.index(agent_sev) - sev_levels.index(true_sev))
        sev_score = max(0.0, 1.0 - diff * 0.4)
    else:
        sev_score = 0.0

    # ── days_of_stock ────────────────────────────────────────────────────────
    agent_days = parsed["days_of_stock"]
    if agent_days is not None:
        diff = abs(agent_days - true_days)
        if diff == 0:
            days_score = 1.0
        elif diff <= 1:
            days_score = 0.7
        elif diff <= 3:
            days_score = 0.3
        else:
            days_score = 0.0
    else:
        days_score = 0.0

    # ── factories_at_risk (recall) ───────────────────────────────────────────
    agent_facs = set(parsed["factories_at_risk"])
    if true_facs:
        fac_score = len(agent_facs & true_facs) / len(true_facs)
    else:
        fac_score = 1.0 if not agent_facs else 0.5

    total = (
        comp_score * 0.30
        + sev_score * 0.25
        + days_score * 0.25
        + fac_score * 0.20
    )
    total = float(max(0.0, min(1.0, total)))

    info = {
        "component_score": comp_score,
        "severity_score": sev_score,
        "days_score": days_score,
        "factory_score": fac_score,
        "true_components": sorted(true_comps),
        "agent_components": sorted(agent_comps),
        "true_severity": true_sev,
        "agent_severity": agent_sev,
        "true_days": true_days,
        "agent_days": agent_days,
        "true_factories": sorted(true_facs),
        "agent_factories": sorted(agent_facs),
    }
    return total, info


# ════════════════════════════════════════════════════════════════════════════
#  Task 2 Grader
# ════════════════════════════════════════════════════════════════════════════

def _grade_task2(world: Dict) -> Tuple[float, Dict]:
    """
    Composite grader for resolve_disruption.

    budget_compliance        20%  1.0 if under budget
    priority_orders_protected 35%  estimated coverage of CRITICAL+HIGH demand
    action_diversity         25%  unique useful action types / 5
    efficiency               20%  balanced spend (too little or too much = lower)
    """
    budget = _TASK_BUDGET["resolve_disruption"]
    spent = world["total_cost"]

    # Budget compliance
    if spent <= budget:
        budget_score = 1.0
    else:
        budget_score = max(0.0, 1.0 - (spent - budget) / budget)

    # Priority orders protected — estimate producible units
    total_stock: Dict[str, int] = {}
    for wh in world["warehouses"].values():
        for comp, qty in wh["stock"].items():
            total_stock[comp] = total_stock.get(comp, 0) + qty

    # Min days any factory can keep running
    min_run_days = float("inf")
    for fac in world["factories"].values():
        wh_stock = world["warehouses"][fac["served_by"]]["stock"]
        rate = fac["production_rate"]
        recipe = fac["recipe"]
        for comp, per_unit in recipe.items():
            per_day = per_unit * rate
            if per_day > 0:
                d = wh_stock.get(comp, 0) / per_day
                min_run_days = min(min_run_days, d)
    if min_run_days == float("inf"):
        min_run_days = 0.0

    # Units producible in 6 days (resolve window), plus pending shipments
    producible = min(min_run_days, 6.0) * (80 + 60 + 50)
    pending = sum(s["units"] for s in world["pending_shipments"])
    total_supply = producible + pending

    # CRITICAL + HIGH unfulfilled demand
    hi_demand = sum(
        o["units_required"] - o["units_fulfilled"]
        for o in world["orders"].values()
        if o["priority"] in ("CRITICAL", "HIGH") and o["status"] != "fulfilled"
    )
    priority_score = min(1.0, total_supply / max(hi_demand, 1.0)) if hi_demand > 0 else 1.0

    # Action diversity
    useful = {
        "reroute_supplier", "expedite_shipping", "reallocate_stock",
        "activate_emergency_supplier", "notify_client",
    }
    diversity = len(world["action_types_used"] & useful) / 5.0

    # Efficiency: reward spending a meaningful fraction of budget
    if spent <= 0:
        eff = 0.1
    elif spent < budget * 0.05:
        eff = 0.3
    elif spent <= budget:
        # Linearly ramp from 0.5 (low spend) to 1.0 (near budget)
        eff = 0.5 + 0.5 * (spent / budget)
    else:
        eff = max(0.0, 1.0 - (spent - budget) / budget)

    total = (
        budget_score * 0.20
        + priority_score * 0.35
        + diversity * 0.25
        + eff * 0.20
    )
    total = float(max(0.0, min(1.0, total)))

    info = {
        "budget_score": budget_score,
        "priority_score": priority_score,
        "diversity_score": diversity,
        "efficiency_score": eff,
        "spent": spent,
        "action_types_used": list(world["action_types_used"]),
    }
    return total, info


# ════════════════════════════════════════════════════════════════════════════
#  Task 3 Grader
# ════════════════════════════════════════════════════════════════════════════

def _grade_task3(world: Dict) -> Tuple[float, Dict]:
    """
    Composite grader for cascade_management.

    order_fulfillment_rate   35%  total fulfilled / total required
    on_time_delivery_rate    25%  on-time fulfilled / total orders
    financial_score          25%  1 - actual_penalties / max_possible_penalties
    budget_efficiency        15%  spend within budget, reward active use
    """
    orders = list(world["orders"].values())
    total_required = sum(o["units_required"] for o in orders)
    total_fulfilled = sum(o["units_fulfilled"] for o in orders)
    fulfillment_rate = total_fulfilled / max(total_required, 1)

    on_time = sum(
        1 for o in orders
        if o["status"] == "fulfilled" and o["days_late"] == 0
    )
    on_time_rate = on_time / len(orders)

    # Maximum possible penalty: sum(penalty × max_late_days) for each order
    day = world["day"]
    max_poss = sum(
        o["late_penalty_per_day"] * max(0, day - o["due_day"])
        for o in orders
    )
    actual_pen = world["total_penalties"]
    financial_score = max(0.0, 1.0 - actual_pen / max(max_poss, 1.0))

    # Budget efficiency
    budget = _TASK_BUDGET["cascade_management"]
    spent = world["total_cost"]
    if spent <= 0:
        budget_eff = 0.1
    elif spent <= budget:
        budget_eff = min(1.0, 0.4 + 0.6 * (spent / budget))
    else:
        budget_eff = max(0.0, 1.0 - (spent - budget) / budget)

    total = (
        fulfillment_rate * 0.35
        + on_time_rate * 0.25
        + financial_score * 0.25
        + budget_eff * 0.15
    )
    total = float(max(0.0, min(1.0, total)))

    info = {
        "fulfillment_rate": fulfillment_rate,
        "on_time_rate": on_time_rate,
        "financial_score": financial_score,
        "budget_efficiency": budget_eff,
        "total_penalties": actual_pen,
        "max_possible_penalties": max_poss,
    }
    return total, info


# ════════════════════════════════════════════════════════════════════════════
#  Main Environment Class
# ════════════════════════════════════════════════════════════════════════════

class SupplyChainEnv:
    """
    OpenEnv-compliant supply chain disruption management environment.

    Usage:
        env = SupplyChainEnv()
        obs = env.reset(task="cascade_management", seed=42)
        obs, reward, done, info = env.step(SCAction(command="assess situation"))
        state = env.state()
    """

    def __init__(self) -> None:
        self._world: Optional[Dict] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task: str = "assess_disruption", seed: int = 42) -> SCObservation:
        """
        Initialise a new episode.
        Same (task, seed) always produces the same starting world state.
        """
        if task not in _TASK_MAX_STEPS:
            raise ValueError(f"Unknown task '{task}'. Choose from {list(_TASK_MAX_STEPS)}")

        self._world = _init_world(task)

        # Apply initial disruptions per task
        if task == "assess_disruption":
            _apply_disruption(self._world, "port_strike_asia")
            self._world["_assessment_disruption_id"] = "port_strike_asia"

        elif task == "resolve_disruption":
            _apply_disruption(self._world, "factory_fire_sup_b")
            _apply_disruption(self._world, "logistics_collapse_americas")

        elif task == "cascade_management":
            _apply_disruption(self._world, "dual_disruption")

        self._world["last_action_result"] = (
            f"[{task.upper()}] Episode reset. Assess the situation and take action."
        )
        return _build_observation(self._world)

    def step(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        """Take one step. Returns (observation, reward, done, info)."""
        if self._world is None:
            raise RuntimeError("Call reset() before step().")
        if self._world["done"]:
            raise RuntimeError("Episode is done — call reset() to start a new episode.")

        task = self._world["task"]
        if task == "assess_disruption":
            return self._step_task1(action)
        elif task == "resolve_disruption":
            return self._step_task2(action)
        elif task == "cascade_management":
            return self._step_task3(action)
        else:
            raise RuntimeError(f"Unknown task '{task}'.")

    def state(self) -> SCState:
        """Return a lightweight summary of the current episode state."""
        if self._world is None:
            raise RuntimeError("Call reset() before state().")
        return _build_state(self._world)

    # ── Task 1: assess_disruption ─────────────────────────────────────────────

    def _step_task1(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        w = self._world
        w["step_count"] += 1

        dis_id = w.get("_assessment_disruption_id", "port_strike_asia")
        disruption = DISRUPTIONS[dis_id]
        reward, grade_info = _grade_task1(action.command, disruption, w)

        reward = float(max(0.0, min(1.0, reward)))
        w["cumulative_reward"] += reward
        w["score_so_far"] = reward
        w["done"] = True

        w["last_action_result"] = (
            f"Assessment graded: {reward:.3f}\n"
            f"  components   {grade_info['component_score']:.2f} "
            f"  (got {grade_info['agent_components']}, expected {grade_info['true_components']})\n"
            f"  severity     {grade_info['severity_score']:.2f} "
            f"  (got '{grade_info['agent_severity']}', expected '{grade_info['true_severity']}')\n"
            f"  days_of_stock {grade_info['days_score']:.2f} "
            f"  (got {grade_info['agent_days']}, expected {grade_info['true_days']})\n"
            f"  factories    {grade_info['factory_score']:.2f} "
            f"  (got {grade_info['agent_factories']}, expected {grade_info['true_factories']})"
        )

        return _build_observation(w), reward, True, {"grade": grade_info}

    # ── Task 2: resolve_disruption ────────────────────────────────────────────

    def _step_task2(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        w = self._world
        w["step_count"] += 1
        max_steps = _TASK_MAX_STEPS["resolve_disruption"]

        parsed = parse_action(action.command)
        step_reward, msg, success = _execute_action(w, parsed)
        step_reward = float(max(0.0, min(1.0, step_reward)))

        done = w["step_count"] >= max_steps

        if done:
            final_reward, grade_info = _grade_task2(w)
            final_reward = float(max(0.0, min(1.0, final_reward)))
            w["score_so_far"] = final_reward
            w["cumulative_reward"] += final_reward
            w["done"] = True
            w["last_action_result"] = (
                msg + f"\n\n[FINAL SCORE: {final_reward:.3f}] "
                f"Budget:{grade_info['budget_score']:.2f} "
                f"Orders:{grade_info['priority_score']:.2f} "
                f"Diversity:{grade_info['diversity_score']:.2f} "
                f"Efficiency:{grade_info['efficiency_score']:.2f}"
            )
            return _build_observation(w), final_reward, True, {
                "parsed_action": {k: v for k, v in parsed.items() if k != "raw"},
                "success": success,
                "grade": grade_info,
            }

        w["cumulative_reward"] += step_reward
        w["score_so_far"] = w["cumulative_reward"] / w["step_count"]
        w["last_action_result"] = msg

        return _build_observation(w), step_reward, False, {
            "parsed_action": {k: v for k, v in parsed.items() if k != "raw"},
            "success": success,
        }

    # ── Task 3: cascade_management ────────────────────────────────────────────

    def _step_task3(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        w = self._world
        w["step_count"] += 1
        max_steps = _TASK_MAX_STEPS["cascade_management"]

        # ── Inject new disruption at step 4 (before action/simulation) ────────
        new_disruption_msg = ""
        if w["step_count"] == 4 and "factory_fire_sup_b" not in w["active_disruptions"]:
            _apply_disruption(w, "factory_fire_sup_b")
            new_disruption_msg = (
                "\n\n⚠️  NEW DISRUPTION DETECTED: Factory Fire at BetaCraft Germany (SUP_B)! "
                "Motors and sensors production halted for 12 days!"
            )

        # ── Execute agent action ──────────────────────────────────────────────
        parsed = parse_action(action.command)
        action_reward, msg, success = _execute_action(w, parsed)
        action_reward = float(max(0.0, min(1.0, action_reward)))

        # ── Simulate one day ──────────────────────────────────────────────────
        daily_penalty = _simulate_day(w)

        # Per-step reward: weighted combination of action quality + penalty avoidance
        penalty_reward = float(max(0.0, 1.0 - daily_penalty / MAX_DAILY_PENALTY))
        step_reward = float(max(0.0, min(1.0, action_reward * 0.3 + penalty_reward * 0.7)))

        w["last_action_result"] = msg + new_disruption_msg
        done = w["step_count"] >= max_steps

        if done:
            final_reward, grade_info = _grade_task3(w)
            final_reward = float(max(0.0, min(1.0, final_reward)))
            w["score_so_far"] = final_reward
            w["cumulative_reward"] += final_reward
            w["done"] = True
            w["last_action_result"] += (
                f"\n\n[FINAL SCORE: {final_reward:.3f}] "
                f"Fulfillment:{grade_info['fulfillment_rate']:.2f} "
                f"OnTime:{grade_info['on_time_rate']:.2f} "
                f"Financial:{grade_info['financial_score']:.2f} "
                f"BudgetEff:{grade_info['budget_efficiency']:.2f}"
            )
            return _build_observation(w), final_reward, True, {
                "parsed_action": {k: v for k, v in parsed.items() if k != "raw"},
                "success": success,
                "daily_penalty": daily_penalty,
                "grade": grade_info,
            }

        w["cumulative_reward"] += step_reward
        w["score_so_far"] = w["cumulative_reward"] / w["step_count"]

        return _build_observation(w), step_reward, False, {
            "parsed_action": {k: v for k, v in parsed.items() if k != "raw"},
            "success": success,
            "daily_penalty": daily_penalty,
        }
