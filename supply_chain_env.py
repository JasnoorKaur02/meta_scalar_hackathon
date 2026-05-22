"""
supply_chain_env.py — OpenEnv-compliant SupplyChainEnv.

Implements:
  reset(task, seed, difficulty) -> SCObservation
  step(SCAction)                -> (SCObservation, float, bool, dict)
  state()                       -> SCState

Five tasks:
  assess_disruption    — 1 step,   grade structured NL assessment
  resolve_disruption   — 5 steps,  $200k budget, recovery planning
  cascade_management   — 10 steps, cascading disruptions, daily simulation
  budget_optimization  — 30 steps, $500k budget, long-horizon cost efficiency
  supplier_negotiation — 8 steps,  $300k budget, pre-disrupt contract strategy

Dynamic difficulty (1–5) scales starting stock and budget.
"""

import copy
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from world import (
    SUPPLIERS, WAREHOUSES, FACTORIES, ORDERS, ORDERS_T4, DISRUPTIONS,
    COMPONENTS, PRIORITY_ORDER,
    MAX_DAILY_PENALTY, MAX_TOTAL_PENALTY,
    MAX_DAILY_PENALTY_T4,
    SUPPLIER_NEGOTIATION_DISCOUNTS,
    BUDGET_OPT_DISRUPTION_SCHEDULE,
)
from parser import parse_action, parse_assessment

# ── Pydantic Models (OpenEnv spec) ────────────────────────────────────────────

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
    # Extra context fields (empty string / 0.0 when not applicable)
    upcoming_disruption: str = ""
    difficulty: int = 2


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
    difficulty: int = 2


# ── World Config ──────────────────────────────────────────────────────────────

_TASK_MAX_STEPS = {
    "assess_disruption":   1,
    "resolve_disruption":  5,
    "cascade_management":  10,
    "budget_optimization": 30,
    "supplier_negotiation": 8,
}

_TASK_BUDGET = {
    "assess_disruption":   200_000.0,
    "resolve_disruption":  200_000.0,
    "cascade_management":  200_000.0,
    "budget_optimization": 500_000.0,
    "supplier_negotiation": 300_000.0,
}

# Difficulty multipliers: stock_mult scales all starting inventory,
# budget_mult scales the task budget.
_DIFFICULTY_SCALE = {
    1: {"stock_mult": 2.00, "budget_mult": 2.00},
    2: {"stock_mult": 1.00, "budget_mult": 1.00},
    3: {"stock_mult": 0.75, "budget_mult": 0.90},
    4: {"stock_mult": 0.50, "budget_mult": 0.75},
    5: {"stock_mult": 0.25, "budget_mult": 0.50},
}

# Extra disruptions injected at task start for difficulty 4/5
_DIFFICULTY_EXTRA_DISRUPTIONS = {
    4: ["demand_spike"],
    5: ["demand_spike", "severe_weather_midwest"],
}

# Upcoming disruption announced to agent for supplier_negotiation task
_T5_ANNOUNCED_DISRUPTION = "triple_disruption"
_T5_DISRUPTION_STEP = 5


# ── World State Initialisation ────────────────────────────────────────────────

def _init_world(task: str, difficulty: int = 2) -> Dict:
    """Return a fresh mutable world state (deep copy of seed data)."""
    scale = _DIFFICULTY_SCALE.get(difficulty, _DIFFICULTY_SCALE[2])
    stock_mult = scale["stock_mult"]
    budget_mult = scale["budget_mult"]

    suppliers = {
        k: {**copy.deepcopy(v), "disrupted": False, "disruption_days_remaining": 0}
        for k, v in SUPPLIERS.items()
    }

    warehouses = {
        k: {
            **copy.deepcopy(v),
            "stock": {
                comp: max(0, int(qty * stock_mult))
                for comp, qty in v["stock"].items()
            },
        }
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

    # Select order set based on task
    order_template = ORDERS_T4 if task == "budget_optimization" else ORDERS
    orders = {
        k: {
            **copy.deepcopy(v),
            "units_required": v["units"],
            "units_fulfilled": 0,
            "status": "pending",
            "days_late": 0,
            "total_penalties_accrued": 0.0,
            "notified": False,
        }
        for k, v in order_template.items()
    }

    base_budget = _TASK_BUDGET[task]
    return {
        "task": task,
        "difficulty": difficulty,
        "day": 0,
        "budget_remaining": base_budget * budget_mult,
        "total_cost": 0.0,
        "total_penalties": 0.0,
        "step_count": 0,
        "cumulative_reward": 0.0,
        "score_so_far": 0.0,
        "done": False,
        "last_action_result": "Episode started.",
        "action_types_used": set(),
        "pending_shipments": [],       # {arrives_day, warehouse, component, units}
        "active_disruptions": {},      # disruption_id -> disruption dict + days_remaining
        "factory_uptime": {            # tracks running/starved/paused days per factory
            fid: {"days_running": 0, "days_starved": 0, "days_paused": 0}
            for fid in FACTORIES
        },
        "contracts_secured": [],       # {supplier, component, units, locked_cost, delivery_day}
        "suppliers": suppliers,
        "warehouses": warehouses,
        "factories": factories,
        "orders": orders,
    }


def _apply_disruption(world: Dict, disruption_id: str, override_days: Optional[int] = None) -> None:
    """Apply a disruption event to the live world state."""
    d = copy.deepcopy(DISRUPTIONS[disruption_id])
    days = override_days if override_days is not None else d["delay_days"]

    if days > 0:
        for sup_id in d["affected_suppliers"]:
            if sup_id in world["suppliers"]:
                world["suppliers"][sup_id]["disrupted"] = True
                world["suppliers"][sup_id]["disruption_days_remaining"] = days

    for wh_id, deltas in d.get("stock_impact", {}).items():
        if wh_id in world["warehouses"]:
            for comp, delta in deltas.items():
                cur = world["warehouses"][wh_id]["stock"].get(comp, 0)
                world["warehouses"][wh_id]["stock"][comp] = max(0, cur + delta)

    for order_id, mult in d.get("order_impact", {}).items():
        if order_id in world["orders"]:
            world["orders"][order_id]["units_required"] = int(
                world["orders"][order_id]["units_required"] * mult
            )

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
      3. Accrue late penalties
      4. Factory production (binary: full-rate or starved)
      5. Allocate production to orders by priority
      6. Track factory uptime
      7. Tick down disruption counters

    Returns the total penalties accrued THIS day.
    """
    world["day"] += 1
    day = world["day"]

    # 1. Pending shipments
    for shipment in world["pending_shipments"][:]:
        if shipment["arrives_day"] <= day:
            wh = shipment["warehouse"]
            comp = shipment["component"]
            if wh in world["warehouses"]:
                world["warehouses"][wh]["stock"][comp] = (
                    world["warehouses"][wh]["stock"].get(comp, 0) + shipment["units"]
                )
            world["pending_shipments"].remove(shipment)

    # 2. Accrue late penalties
    daily_penalty = 0.0
    for order in world["orders"].values():
        if order["status"] != "fulfilled" and day > order["due_day"]:
            order["days_late"] = day - order["due_day"]
            p = order["late_penalty_per_day"]
            order["total_penalties_accrued"] += p
            world["total_penalties"] += p
            daily_penalty += p

    # 3. Factory production
    total_produced = 0
    for fac in world["factories"].values():
        fac["units_produced_today"] = 0

        if fac["pause_days_remaining"] > 0:
            fac["status"] = "paused"
            fac["pause_days_remaining"] -= 1
            total_produced += 0
            continue

        wh_id = fac["served_by"]
        wh_stock = world["warehouses"][wh_id]["stock"]
        recipe = fac["recipe"]
        rate = fac["production_rate"]

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

    # 4. Allocate production to orders (CRITICAL first)
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

    # 5. Track factory uptime
    uptime = world.get("factory_uptime", {})
    for fid, fac in world["factories"].items():
        if fid in uptime:
            if fac["status"] == "running":
                uptime[fid]["days_running"] += 1
            elif fac["status"] == "starved":
                uptime[fid]["days_starved"] += 1
            elif fac["status"] == "paused":
                uptime[fid]["days_paused"] += 1

    # 6. Tick down active disruptions
    for dis_id in list(world["active_disruptions"].keys()):
        dis = world["active_disruptions"][dis_id]
        dis["days_remaining"] -= 1
        if dis["days_remaining"] <= 0:
            for sup_id in dis["affected_suppliers"]:
                if sup_id in world["suppliers"]:
                    world["suppliers"][sup_id]["disrupted"] = False
                    world["suppliers"][sup_id]["disruption_days_remaining"] = 0
            del world["active_disruptions"][dis_id]

    return daily_penalty


# ── Action Execution ──────────────────────────────────────────────────────────

def _action_cost(world: Dict, parsed: Dict) -> float:
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
    if t == "negotiate_contract":
        sup_id = parsed.get("supplier")
        units = parsed.get("units") or 100
        if sup_id and sup_id in world["suppliers"]:
            cpu = world["suppliers"][sup_id]["cost_per_unit"]
            discount = SUPPLIER_NEGOTIATION_DISCOUNTS.get(sup_id, 0.10)
            return units * cpu * (1.0 - discount)
        return units * 80.0
    if t in ("notify_client", "assess_situation"):
        return 0.0
    return 0.0


def _execute_action(world: Dict, parsed: Dict) -> Tuple[float, str, bool]:
    """Apply a parsed action. Returns (step_reward, message, success)."""
    t = parsed.get("type", "unknown")

    if t == "unknown":
        return 0.0, (
            f"Unrecognised command: '{parsed.get('raw', '')}'. "
            "Try 'assess situation' to see options."
        ), False

    if t == "assess_situation":
        world["action_types_used"].add(t)
        return 0.05, _situation_report(world), True

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
    if t == "negotiate_contract":
        return _do_negotiate_contract(world, parsed, cost)

    return 0.0, f"Unknown action type: {t}", False


def _deduct_cost(world: Dict, cost: float) -> None:
    world["budget_remaining"] -= cost
    world["total_cost"] += cost


def _most_depleted_warehouse(world: Dict, component: str) -> str:
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
        f"Rerouted {units}x {comp} from {from_sup} -> {to_sup}. "
        f"Shipment arrives at {target_wh} on day {arrives}. "
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
        return 0.0, f"{from_wh} already serves {to_fac} — expedite is a no-op.", False

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
        f"Expedited {actual}x {comp} from {from_wh} -> {to_fac} ({to_wh}). "
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
        f"Reallocated {actual}x {comp} from {from_wh} -> {to_wh}. "
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
        f"Resumes after day {world['day'] + days}. "
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
        f"Emergency order: {units}x {comp} added to {target}. "
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


def _do_negotiate_contract(world: Dict, p: Dict, cost: float) -> Tuple[float, str, bool]:
    """
    Lock in a forward supply contract with a supplier at a negotiated discount.
    The shipment is guaranteed to arrive even if the supplier later gets disrupted.
    """
    sup_id = p.get("supplier")
    comp = p.get("component")
    units = p.get("units") or 100

    if not sup_id or sup_id not in world["suppliers"]:
        return 0.0, f"Unknown supplier '{sup_id}'.", False
    if not comp:
        return 0.0, "No component specified for contract.", False
    if comp not in world["suppliers"][sup_id]["components"]:
        return 0.0, f"{sup_id} does not supply '{comp}'.", False

    _deduct_cost(world, cost)

    discount = SUPPLIER_NEGOTIATION_DISCOUNTS.get(sup_id, 0.10)
    lead = world["suppliers"][sup_id]["lead_time_days"]
    arrives = world["day"] + lead
    target_wh = _most_depleted_warehouse(world, comp)

    # Guaranteed contract shipment (separate tracking list)
    contract = {
        "supplier": sup_id,
        "component": comp,
        "units": units,
        "locked_cost": cost,
        "delivery_day": arrives,
        "target_warehouse": target_wh,
    }
    world["contracts_secured"].append(contract)

    # Also queue as a guaranteed pending shipment (always arrives)
    world["pending_shipments"].append({
        "arrives_day": arrives,
        "warehouse": target_wh,
        "component": comp,
        "units": units,
        "guaranteed": True,
    })

    return 0.25, (
        f"Contract secured with {sup_id}: {units}x {comp} at "
        f"{int(discount * 100)}% discount. "
        f"Delivery to {target_wh} on day {arrives}. "
        f"Locked cost: ${cost:,.0f} (vs. spot market ~${units * world['suppliers'][sup_id]['cost_per_unit'] * 1.3:,.0f})."
    ), True


# ── Observation / State Builders ──────────────────────────────────────────────

_AVAILABLE_ACTIONS = [
    "reroute supplier from <SUP_X> to <SUP_Y> for <N> <component>",
    "expedite <N> <component> from <WH_X> to <FAC_Y>",
    "reallocate <N> <component> from <WH_X> to <WH_Y>",
    "pause <FAC_X> for <N> days",
    "activate emergency supplier for <N> <component>",
    "notify client <ORD_X> of <N> day delay",
    "negotiate contract with <SUP_X> for <N> <component>",
    "assess situation",
]


def _build_observation(world: Dict, upcoming_disruption: str = "") -> SCObservation:
    task = world["task"]
    max_days = _TASK_MAX_STEPS[task]

    suppliers = [
        SupplierStatus(
            id=s["id"], name=s["name"], location=s["location"],
            components=s["components"], reliability=s["reliability"],
            lead_time_days=s["lead_time_days"], cost_per_unit=s["cost_per_unit"],
            disrupted=s["disrupted"],
            disruption_days_remaining=s["disruption_days_remaining"],
        )
        for s in world["suppliers"].values()
    ]

    warehouses = [
        WarehouseStatus(
            id=w["id"], name=w["name"], location=w["location"],
            stock=dict(w["stock"]),
        )
        for w in world["warehouses"].values()
    ]

    factories = [
        FactoryStatus(
            id=f["id"], name=f["name"], location=f["location"],
            served_by=f["served_by"], production_rate=f["production_rate"],
            recipe=dict(f["recipe"]), status=f["status"],
            units_produced_today=f["units_produced_today"],
            total_units_produced=f["total_units_produced"],
            pause_days_remaining=f["pause_days_remaining"],
        )
        for f in world["factories"].values()
    ]

    orders = [
        OrderStatus(
            id=o["id"], customer=o["customer"],
            units_required=o["units_required"], units_fulfilled=o["units_fulfilled"],
            due_day=o["due_day"], late_penalty_per_day=o["late_penalty_per_day"],
            priority=o["priority"], status=o["status"],
            days_late=o["days_late"],
            total_penalties_accrued=o["total_penalties_accrued"],
        )
        for o in world["orders"].values()
    ]

    disruptions = [
        DisruptionInfo(
            id=d["id"], name=d["name"], severity=d["severity"],
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
        upcoming_disruption=upcoming_disruption,
        difficulty=world.get("difficulty", 2),
    )


def _build_state(world: Dict) -> SCState:
    fac_running = sum(1 for f in world["factories"].values() if f["status"] == "running")
    orders_fulfilled = sum(1 for o in world["orders"].values() if o["status"] == "fulfilled")
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
        difficulty=world.get("difficulty", 2),
    )


# ── Situation Report ──────────────────────────────────────────────────────────

def _situation_report(world: Dict) -> str:
    lines = [
        f"=== Situation Report — Day {world['day']} ===",
        f"Budget remaining : ${world['budget_remaining']:>10,.0f}",
        f"Total penalties  : ${world['total_penalties']:>10,.0f}",
        f"Difficulty       : {world.get('difficulty', 2)}/5",
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
        stock_str = "  ".join(
            f"{c}:{world['warehouses'][wh['id']]['stock'].get(c, 0)}"
            for c in COMPONENTS
        )
        lines.append(f"  {wh['id']} ({wh['location']}): {stock_str}")

    lines += ["", "Factory status:"]
    for fac in world["factories"].values():
        uptime = world.get("factory_uptime", {}).get(fac["id"], {})
        lines.append(
            f"  {fac['id']}: {fac['status'].upper():8s} | "
            f"today {fac['units_produced_today']:3d} | "
            f"total {fac['total_units_produced']:4d} | "
            f"uptime {uptime.get('days_running', 0)}d running / "
            f"{uptime.get('days_starved', 0)}d starved"
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
            tag = " [CONTRACT]" if s.get("guaranteed") else ""
            lines.append(
                f"  {s['units']}x {s['component']} -> {s['warehouse']} "
                f"arrives day {s['arrives_day']}{tag}"
            )

    contracts = world.get("contracts_secured", [])
    if contracts:
        lines += ["", "Secured contracts:"]
        for c in contracts:
            lines.append(
                f"  {c['supplier']}: {c['units']}x {c['component']} "
                f"@ ${c['locked_cost']:,.0f} -> {c['target_warehouse']} day {c['delivery_day']}"
            )

    return "\n".join(lines)


# ── Task 1 Grader ─────────────────────────────────────────────────────────────

def _compute_days_of_stock(disruption: Dict, world: Dict) -> int:
    affected = set(disruption.get("affected_components", []))
    if not affected:
        return 30

    min_days = float("inf")
    for fac in world["factories"].values():
        recipe = fac["recipe"]
        rate = fac["production_rate"]
        wh_stock = world["warehouses"][fac["served_by"]]["stock"]

        if not any(c in recipe for c in affected):
            continue

        for comp in affected:
            if comp in recipe:
                per_day = recipe[comp] * rate
                if per_day > 0:
                    days = wh_stock.get(comp, 0) // per_day
                    min_days = min(min_days, days)

    return int(min_days) if min_days != float("inf") else 30


def _compute_factories_at_risk(disruption: Dict, world: Dict) -> List[str]:
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
      factories_at_risk    20%  recall
    """
    from parser import parse_assessment, COMPONENT_MAP
    parsed = parse_assessment(response)

    true_comps = set(disruption.get("affected_components", []))
    true_sev = disruption["severity"].lower()
    true_days = _compute_days_of_stock(disruption, world)
    true_facs = set(_compute_factories_at_risk(disruption, world))

    # affected_components F1
    agent_comps = set(parsed["affected_components"])
    if true_comps or agent_comps:
        tp = len(agent_comps & true_comps)
        prec = tp / len(agent_comps) if agent_comps else 0.0
        rec = tp / len(true_comps) if true_comps else 1.0
        comp_score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    else:
        comp_score = 1.0

    # severity
    sev_levels = ["low", "medium", "high", "critical"]
    agent_sev = (parsed["severity"] or "").lower()
    if agent_sev == true_sev:
        sev_score = 1.0
    elif agent_sev in sev_levels and true_sev in sev_levels:
        diff = abs(sev_levels.index(agent_sev) - sev_levels.index(true_sev))
        sev_score = max(0.0, 1.0 - diff * 0.4)
    else:
        sev_score = 0.0

    # days_of_stock
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

    # factories_at_risk (recall)
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
    total = float(max(0.001, min(0.999, total)))

    return total, {
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


# ── Task 2 Grader ─────────────────────────────────────────────────────────────

def _grade_task2(world: Dict) -> Tuple[float, Dict]:
    """
    Composite grader for resolve_disruption.

    budget_compliance        20%
    priority_orders_protected 35%
    action_diversity         25%
    efficiency               20%
    """
    budget = _TASK_BUDGET["resolve_disruption"]
    spent = world["total_cost"]

    budget_score = 1.0 if spent <= budget else max(0.0, 1.0 - (spent - budget) / budget)

    total_stock: Dict[str, int] = {}
    for wh in world["warehouses"].values():
        for comp, qty in wh["stock"].items():
            total_stock[comp] = total_stock.get(comp, 0) + qty

    min_run_days = float("inf")
    for fac in world["factories"].values():
        wh_stock = world["warehouses"][fac["served_by"]]["stock"]
        rate = fac["production_rate"]
        for comp, per_unit in fac["recipe"].items():
            per_day = per_unit * rate
            if per_day > 0:
                d = wh_stock.get(comp, 0) / per_day
                min_run_days = min(min_run_days, d)
    if min_run_days == float("inf"):
        min_run_days = 0.0

    producible = min(min_run_days, 6.0) * (80 + 60 + 50)
    pending = sum(s["units"] for s in world["pending_shipments"])
    total_supply = producible + pending

    hi_demand = sum(
        o["units_required"] - o["units_fulfilled"]
        for o in world["orders"].values()
        if o["priority"] in ("CRITICAL", "HIGH") and o["status"] != "fulfilled"
    )
    priority_score = min(0.999, total_supply / max(hi_demand, 1.0)) if hi_demand > 0 else 1.0

    useful = {
        "reroute_supplier", "expedite_shipping", "reallocate_stock",
        "activate_emergency_supplier", "notify_client",
    }
    diversity = len(world["action_types_used"] & useful) / 5.0

    if spent <= 0:
        eff = 0.1
    elif spent < budget * 0.05:
        eff = 0.3
    elif spent <= budget:
        eff = 0.5 + 0.5 * (spent / budget)
    else:
        eff = max(0.0, 1.0 - (spent - budget) / budget)

    total = (
        budget_score * 0.20
        + priority_score * 0.35
        + diversity * 0.25
        + eff * 0.20
    )
    total = float(max(0.001, min(0.999, total)))

    return total, {
        "budget_score": budget_score,
        "priority_score": priority_score,
        "diversity_score": diversity,
        "efficiency_score": eff,
        "spent": spent,
        "action_types_used": list(world["action_types_used"]),
    }


# ── Task 3 Grader (improved) ──────────────────────────────────────────────────

def _grade_task3(world: Dict) -> Tuple[float, Dict]:
    """
    Improved composite grader for cascade_management.

    order_fulfillment_rate   30%  total fulfilled / total required
    weighted_on_time_rate    20%  priority-weighted on-time delivery
    financial_score          25%  1 - penalties / max_possible_penalties
    factory_uptime_score     15%  (running days) / (running + starved days)
    budget_efficiency        10%  active spend within budget
    action_diversity_bonus    5%  additive boost for diverse action use
    """
    orders = list(world["orders"].values())
    total_required = sum(o["units_required"] for o in orders)
    total_fulfilled = sum(o["units_fulfilled"] for o in orders)
    fulfillment_rate = total_fulfilled / max(total_required, 1)

    # Priority-weighted on-time rate
    priority_weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    total_weight = sum(priority_weights.get(o["priority"], 1) for o in orders)
    weighted_on_time = sum(
        priority_weights.get(o["priority"], 1)
        for o in orders
        if o["status"] == "fulfilled" and o["days_late"] == 0
    ) / max(total_weight, 1)

    # Financial score
    day = world["day"]
    max_poss = sum(
        o["late_penalty_per_day"] * max(0, day - o["due_day"])
        for o in orders
    )
    actual_pen = world["total_penalties"]
    financial_score = max(0.0, 1.0 - actual_pen / max(max_poss, 1.0))

    # Factory uptime score (only count running vs. starved — exclude paused)
    uptime = world.get("factory_uptime", {})
    total_running = sum(v.get("days_running", 0) for v in uptime.values())
    total_starved = sum(v.get("days_starved", 0) for v in uptime.values())
    productive_total = total_running + total_starved
    uptime_score = total_running / max(productive_total, 1)

    # Budget efficiency
    budget = _TASK_BUDGET["cascade_management"]
    spent = world["total_cost"]
    if spent <= 0:
        budget_eff = 0.1
    elif spent <= budget:
        budget_eff = min(0.999, 0.4 + 0.6 * (spent / budget))
    else:
        budget_eff = max(0.0, 1.0 - (spent - budget) / budget)

    # Action diversity bonus (up to 5% additive boost)
    useful = {"reroute_supplier", "expedite_shipping", "reallocate_stock", "activate_emergency_supplier"}
    diversity_ratio = len(world["action_types_used"] & useful) / 4.0

    base_score = (
        fulfillment_rate   * 0.30
        + weighted_on_time * 0.20
        + financial_score  * 0.25
        + uptime_score     * 0.15
        + budget_eff       * 0.10
    )
    # Diversity bonus: up to 5% boost
    total = base_score * (1.0 + diversity_ratio * 0.05)
    total = float(max(0.001, min(0.999, total)))

    return total, {
        "fulfillment_rate": fulfillment_rate,
        "weighted_on_time_rate": weighted_on_time,
        "financial_score": financial_score,
        "factory_uptime_score": uptime_score,
        "budget_efficiency": budget_eff,
        "diversity_bonus": diversity_ratio * 0.05,
        "total_penalties": actual_pen,
        "max_possible_penalties": max_poss,
        "days_running": total_running,
        "days_starved": total_starved,
    }


# ── Task 4 Grader ─────────────────────────────────────────────────────────────

def _grade_task4(world: Dict) -> Tuple[float, Dict]:
    """
    Grader for budget_optimization (30-day horizon).

    cost_per_unit_score      30%  lower cost per unit produced = higher score
    order_fulfillment_rate   25%  units fulfilled / required
    financial_score          25%  1 - penalties / max_possible
    budget_efficiency        20%  balanced spend within $500k
    """
    orders = list(world["orders"].values())
    total_required = sum(o["units_required"] for o in orders)
    total_fulfilled = sum(o["units_fulfilled"] for o in orders)
    fulfillment_rate = total_fulfilled / max(total_required, 1)

    # Cost per unit efficiency: compare actual spend to a "reasonable" baseline
    total_produced = sum(f["total_units_produced"] for f in world["factories"].values())
    spent = world["total_cost"]
    if total_produced > 0:
        # Benchmark: ~$100/unit is a reasonable average given recipe costs
        benchmark_cost_per_unit = 100.0
        actual_cpu = spent / total_produced
        # Score: 1.0 at benchmark, declines above it
        cpu_score = max(0.0, min(1.0, benchmark_cost_per_unit / max(actual_cpu, 1.0)))
    else:
        cpu_score = 0.0 if spent > 0 else 0.5

    # Financial score
    day = world["day"]
    max_poss = sum(
        o["late_penalty_per_day"] * max(0, day - o["due_day"])
        for o in orders
    )
    actual_pen = world["total_penalties"]
    financial_score = max(0.0, 1.0 - actual_pen / max(max_poss, 1.0))

    # Budget efficiency
    budget = world["budget_remaining"] + spent  # original budget
    if spent <= 0:
        budget_eff = 0.1
    elif spent <= budget:
        budget_eff = min(0.999, 0.3 + 0.7 * (spent / budget))
    else:
        budget_eff = max(0.0, 1.0 - (spent - budget) / budget)

    total = (
        cpu_score        * 0.30
        + fulfillment_rate * 0.25
        + financial_score  * 0.25
        + budget_eff       * 0.20
    )
    total = float(max(0.001, min(0.999, total)))

    return total, {
        "cost_per_unit_score": cpu_score,
        "fulfillment_rate": fulfillment_rate,
        "financial_score": financial_score,
        "budget_efficiency": budget_eff,
        "total_produced": total_produced,
        "actual_spend": spent,
        "total_penalties": actual_pen,
    }


# ── Task 5 Grader ─────────────────────────────────────────────────────────────

def _grade_task5(world: Dict) -> Tuple[float, Dict]:
    """
    Grader for supplier_negotiation.

    supply_security_score    40%  contract units cover disruption shortfall
    cost_efficiency_score    35%  money saved vs. spot-market alternative
    diversification_score    25%  unique suppliers contracted
    """
    contracts = world.get("contracts_secured", [])

    # Supply security: do contracts cover the critical component needs?
    disruption = DISRUPTIONS.get(_T5_ANNOUNCED_DISRUPTION, {})
    affected_comps = set(disruption.get("affected_components", []))

    contracted_by_comp: Dict[str, int] = {}
    for c in contracts:
        comp = c["component"]
        contracted_by_comp[comp] = contracted_by_comp.get(comp, 0) + c["units"]

    # Target: each factory needs at least lead_time_days * daily_consumption per affected comp
    # Use 8 days (disruption duration) as the coverage target
    disruption_days = disruption.get("delay_days", 8)
    total_coverage_needed = 0
    total_coverage_secured = 0
    for fac in world["factories"].values():
        for comp in affected_comps:
            if comp in fac["recipe"]:
                needed = fac["recipe"][comp] * fac["production_rate"] * disruption_days
                total_coverage_needed += needed
                total_coverage_secured += min(
                    contracted_by_comp.get(comp, 0), needed
                )

    supply_security = (
        total_coverage_secured / max(total_coverage_needed, 1)
        if total_coverage_needed > 0 else (1.0 if contracts else 0.0)
    )
    supply_security = min(1.0, supply_security)

    # Cost efficiency: compare locked cost vs. spot market equivalent
    total_locked_cost = sum(c["locked_cost"] for c in contracts)
    spot_market_cost = 0.0
    for c in contracts:
        sup_id = c["supplier"]
        cpu = world["suppliers"].get(sup_id, {}).get("cost_per_unit", 100.0)
        spot_market_cost += c["units"] * cpu * 1.3  # spot premium

    if spot_market_cost > 0:
        savings_ratio = (spot_market_cost - total_locked_cost) / spot_market_cost
        cost_eff = max(0.0, min(1.0, 0.5 + savings_ratio))
    else:
        cost_eff = 0.3 if not contracts else 0.5

    # Diversification: unique suppliers used in contracts
    unique_suppliers = len({c["supplier"] for c in contracts})
    total_suppliers = len(world["suppliers"])
    div_score = unique_suppliers / max(total_suppliers, 1)

    total = (
        supply_security * 0.40
        + cost_eff      * 0.35
        + div_score     * 0.25
    )
    total = float(max(0.001, min(0.999, total)))

    return total, {
        "supply_security_score": supply_security,
        "cost_efficiency_score": cost_eff,
        "diversification_score": div_score,
        "contracts_count": len(contracts),
        "unique_suppliers": unique_suppliers,
        "total_locked_cost": total_locked_cost,
        "spot_market_equivalent": spot_market_cost,
        "coverage_needed": total_coverage_needed,
        "coverage_secured": total_coverage_secured,
    }


# ── Main Environment Class ────────────────────────────────────────────────────

class SupplyChainEnv:
    """
    OpenEnv-compliant supply chain disruption management environment.

    Usage:
        env = SupplyChainEnv()
        obs = env.reset(task="cascade_management", seed=42, difficulty=3)
        obs, reward, done, info = env.step(SCAction(command="assess situation"))
        state = env.state()
    """

    def __init__(self) -> None:
        self._world: Optional[Dict] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        task: str = "assess_disruption",
        seed: int = 42,
        difficulty: int = 2,
    ) -> SCObservation:
        """
        Initialise a new episode. difficulty 1 (easiest) – 5 (hardest).
        Same (task, seed, difficulty) always produces the same world.
        """
        if task not in _TASK_MAX_STEPS:
            raise ValueError(f"Unknown task '{task}'. Choose from {list(_TASK_MAX_STEPS)}")
        if difficulty not in _DIFFICULTY_SCALE:
            raise ValueError(f"difficulty must be 1–5, got {difficulty}")

        self._world = _init_world(task, difficulty)

        # Apply initial disruptions and extra difficulty disruptions
        if task == "assess_disruption":
            _apply_disruption(self._world, "port_strike_asia")
            self._world["_assessment_disruption_id"] = "port_strike_asia"

        elif task == "resolve_disruption":
            _apply_disruption(self._world, "factory_fire_sup_b")
            _apply_disruption(self._world, "logistics_collapse_americas")

        elif task == "cascade_management":
            _apply_disruption(self._world, "dual_disruption")

        elif task == "budget_optimization":
            _apply_disruption(self._world, "quality_recall_sup_e")

        elif task == "supplier_negotiation":
            pass  # Disruption announced but not yet applied

        # Extra difficulty disruptions (level 4 & 5)
        for dis_id in _DIFFICULTY_EXTRA_DISRUPTIONS.get(difficulty, []):
            if dis_id in DISRUPTIONS:
                _apply_disruption(self._world, dis_id)

        upcoming = ""
        if task == "supplier_negotiation":
            d = DISRUPTIONS[_T5_ANNOUNCED_DISRUPTION]
            upcoming = (
                f"INTEL: {d['name']} expected on step {_T5_DISRUPTION_STEP}! "
                f"Affects: {d['affected_components']}. "
                f"Negotiate contracts NOW before prices spike."
            )

        self._world["last_action_result"] = (
            f"[{task.upper()}] Difficulty {difficulty}/5 — Episode reset. "
            "Assess the situation and take action."
        )
        return _build_observation(self._world, upcoming_disruption=upcoming)

    def step(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        """Take one step. Returns (observation, reward, done, info)."""
        if self._world is None:
            raise RuntimeError("Call reset() before step().")
        if self._world["done"]:
            raise RuntimeError("Episode is done — call reset() to start a new episode.")

        task = self._world["task"]
        dispatch = {
            "assess_disruption":   self._step_task1,
            "resolve_disruption":  self._step_task2,
            "cascade_management":  self._step_task3,
            "budget_optimization": self._step_task4,
            "supplier_negotiation": self._step_task5,
        }
        if task not in dispatch:
            raise RuntimeError(f"Unknown task '{task}'.")
        return dispatch[task](action)

    def state(self) -> SCState:
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

        reward = float(max(0.001, min(0.999, reward)))
        w["cumulative_reward"] += reward
        w["score_so_far"] = reward
        w["done"] = True

        w["last_action_result"] = (
            f"Assessment graded: {reward:.3f}\n"
            f"  components    {grade_info['component_score']:.2f} "
            f"(got {grade_info['agent_components']}, expected {grade_info['true_components']})\n"
            f"  severity      {grade_info['severity_score']:.2f} "
            f"(got '{grade_info['agent_severity']}', expected '{grade_info['true_severity']}')\n"
            f"  days_of_stock {grade_info['days_score']:.2f} "
            f"(got {grade_info['agent_days']}, expected {grade_info['true_days']})\n"
            f"  factories     {grade_info['factory_score']:.2f} "
            f"(got {grade_info['agent_factories']}, expected {grade_info['true_factories']})"
        )

        return _build_observation(w), reward, True, {"grade": grade_info}

    # ── Task 2: resolve_disruption ────────────────────────────────────────────

    def _step_task2(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        w = self._world
        w["step_count"] += 1
        max_steps = _TASK_MAX_STEPS["resolve_disruption"]

        parsed = parse_action(action.command)
        step_reward, msg, success = _execute_action(w, parsed)
        step_reward = float(max(0.001, min(0.999, step_reward)))

        done = w["step_count"] >= max_steps

        if done:
            final_reward, grade_info = _grade_task2(w)
            final_reward = float(max(0.001, min(0.999, final_reward)))
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

        new_disruption_msg = ""
        if w["step_count"] == 4 and "factory_fire_sup_b" not in w["active_disruptions"]:
            _apply_disruption(w, "factory_fire_sup_b")
            new_disruption_msg = (
                "\n\n*** NEW DISRUPTION: Factory Fire at BetaCraft Germany (SUP_B)! "
                "Motors and sensors halted 12 days! ***"
            )

        parsed = parse_action(action.command)
        action_reward, msg, success = _execute_action(w, parsed)
        action_reward = float(max(0.001, min(0.999, action_reward)))

        daily_penalty = _simulate_day(w)

        # Shaped per-step reward: action quality + penalty avoidance
        penalty_reward = float(max(0.0, 1.0 - daily_penalty / MAX_DAILY_PENALTY))

        # Partial fulfillment bonus: incremental units fulfilled this step
        prev_fulfilled = sum(o.get("_prev_fulfilled", o["units_fulfilled"]) for o in w["orders"].values())
        curr_fulfilled = sum(o["units_fulfilled"] for o in w["orders"].values())
        for o in w["orders"].values():
            o["_prev_fulfilled"] = o["units_fulfilled"]
        fulfillment_delta = max(0, curr_fulfilled - prev_fulfilled)
        total_required = max(1, sum(o["units_required"] for o in w["orders"].values()))
        fulfillment_bonus = fulfillment_delta / total_required

        step_reward = float(max(0.001, min(0.999,
            action_reward   * 0.20
            + penalty_reward * 0.60
            + fulfillment_bonus * 0.20
        )))

        w["last_action_result"] = msg + new_disruption_msg
        done = w["step_count"] >= max_steps

        if done:
            final_reward, grade_info = _grade_task3(w)
            final_reward = float(max(0.0001, min(0.999, final_reward)))
            w["score_so_far"] = final_reward
            w["cumulative_reward"] += final_reward
            w["done"] = True
            w["last_action_result"] += (
                f"\n\n[FINAL SCORE: {final_reward:.3f}] "
                f"Fulfillment:{grade_info['fulfillment_rate']:.2f} "
                f"OnTime:{grade_info['weighted_on_time_rate']:.2f} "
                f"Financial:{grade_info['financial_score']:.2f} "
                f"Uptime:{grade_info['factory_uptime_score']:.2f} "
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

    # ── Task 4: budget_optimization ───────────────────────────────────────────

    def _step_task4(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        w = self._world
        w["step_count"] += 1
        max_steps = _TASK_MAX_STEPS["budget_optimization"]

        # Inject scheduled disruptions at specified steps
        new_disruption_msg = ""
        for step_trigger, dis_id in BUDGET_OPT_DISRUPTION_SCHEDULE:
            if w["step_count"] == step_trigger and dis_id not in w["active_disruptions"]:
                _apply_disruption(w, dis_id)
                d = DISRUPTIONS[dis_id]
                new_disruption_msg += (
                    f"\n\n*** NEW DISRUPTION at step {step_trigger}: {d['name']}! "
                    f"[{d['severity'].upper()}] Affects: {d['affected_components']} ***"
                )

        parsed = parse_action(action.command)
        action_reward, msg, success = _execute_action(w, parsed)
        action_reward = float(max(0.001, min(0.999, action_reward)))

        daily_penalty = _simulate_day(w)

        # Shaped reward: action quality + penalty avoidance + cost efficiency
        penalty_reward = float(max(0.0, 1.0 - daily_penalty / MAX_DAILY_PENALTY_T4))
        spent_ratio = w["total_cost"] / max(w["total_cost"] + w["budget_remaining"], 1)
        cost_eff_reward = max(0.0, 1.0 - spent_ratio)

        step_reward = float(max(0.001, min(0.999,
            action_reward    * 0.15
            + penalty_reward * 0.55
            + cost_eff_reward * 0.30
        )))

        w["last_action_result"] = msg + new_disruption_msg
        done = w["step_count"] >= max_steps

        if done:
            final_reward, grade_info = _grade_task4(w)
            final_reward = float(max(0.0001, min(0.999, final_reward)))
            w["score_so_far"] = final_reward
            w["cumulative_reward"] += final_reward
            w["done"] = True
            w["last_action_result"] += (
                f"\n\n[FINAL SCORE: {final_reward:.3f}] "
                f"CostPerUnit:{grade_info['cost_per_unit_score']:.2f} "
                f"Fulfillment:{grade_info['fulfillment_rate']:.2f} "
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

    # ── Task 5: supplier_negotiation ──────────────────────────────────────────

    def _step_task5(self, action: SCAction) -> Tuple[SCObservation, float, bool, dict]:
        w = self._world
        w["step_count"] += 1
        max_steps = _TASK_MAX_STEPS["supplier_negotiation"]

        # Apply announced disruption at the designated step
        disruption_msg = ""
        if w["step_count"] == _T5_DISRUPTION_STEP:
            _apply_disruption(w, _T5_ANNOUNCED_DISRUPTION)
            d = DISRUPTIONS[_T5_ANNOUNCED_DISRUPTION]
            disruption_msg = (
                f"\n\n*** DISRUPTION ARRIVED: {d['name']}! "
                f"[{d['severity'].upper()}] {d['affected_suppliers']} halted "
                f"{d['delay_days']} days — "
                f"components: {d['affected_components']} ***"
            )

        parsed = parse_action(action.command)
        action_reward, msg, success = _execute_action(w, parsed)

        # Bonus reward for negotiate_contract actions
        if parsed.get("type") == "negotiate_contract" and success:
            action_reward = min(0.999, action_reward + 0.15)

        action_reward = float(max(0.001, min(0.999, action_reward)))

        daily_penalty = _simulate_day(w)
        penalty_reward = float(max(0.0, 1.0 - daily_penalty / MAX_DAILY_PENALTY))

        step_reward = float(max(0.001, min(0.999,
            action_reward   * 0.50
            + penalty_reward * 0.50
        )))

        # Still announce upcoming disruption after it's been applied
        upcoming = ""
        if w["step_count"] < _T5_DISRUPTION_STEP:
            d = DISRUPTIONS[_T5_ANNOUNCED_DISRUPTION]
            steps_left = _T5_DISRUPTION_STEP - w["step_count"]
            upcoming = (
                f"INTEL: {d['name']} arrives in {steps_left} more step(s). "
                f"Affects: {d['affected_components']}. Negotiate contracts now!"
            )

        w["last_action_result"] = msg + disruption_msg
        done = w["step_count"] >= max_steps

        if done:
            final_reward, grade_info = _grade_task5(w)
            final_reward = float(max(0.0001, min(0.999, final_reward)))
            w["score_so_far"] = final_reward
            w["cumulative_reward"] += final_reward
            w["done"] = True
            w["last_action_result"] += (
                f"\n\n[FINAL SCORE: {final_reward:.3f}] "
                f"Security:{grade_info['supply_security_score']:.2f} "
                f"CostEff:{grade_info['cost_efficiency_score']:.2f} "
                f"Diversity:{grade_info['diversification_score']:.2f} "
                f"Contracts:{grade_info['contracts_count']}"
            )
            return _build_observation(w, upcoming_disruption=""), final_reward, True, {
                "parsed_action": {k: v for k, v in parsed.items() if k != "raw"},
                "success": success,
                "daily_penalty": daily_penalty,
                "grade": grade_info,
            }

        w["cumulative_reward"] += step_reward
        w["score_so_far"] = w["cumulative_reward"] / w["step_count"]

        return _build_observation(w, upcoming_disruption=upcoming), step_reward, False, {
            "parsed_action": {k: v for k, v in parsed.items() if k != "raw"},
            "success": success,
            "daily_penalty": daily_penalty,
        }
