"""
world.py — Seed data: suppliers, warehouses, factories, orders, disruptions.
All values are read-only templates; SupplyChainEnv deep-copies them each episode.
Field names match the Pydantic models in supply_chain_env.py.
"""

SUPPLIERS = {
    "SUP_A": {
        "id": "SUP_A",
        "name": "AlphaSource Taiwan",
        "location": "Taiwan",
        "components": ["chip", "sensor"],
        "reliability": 0.92,
        "lead_time_days": 7,
        "cost_per_unit": 50.0,
    },
    "SUP_B": {
        "id": "SUP_B",
        "name": "BetaCraft Germany",
        "location": "Germany",
        "components": ["motor", "sensor"],
        "reliability": 0.88,
        "lead_time_days": 5,
        "cost_per_unit": 80.0,
    },
    "SUP_C": {
        "id": "SUP_C",
        "name": "CasingCo Vietnam",
        "location": "Vietnam",
        "components": ["casing"],
        "reliability": 0.95,
        "lead_time_days": 4,
        "cost_per_unit": 30.0,
    },
    "SUP_D": {
        "id": "SUP_D",
        "name": "DeltaPower Mexico",
        "location": "Mexico",
        "components": ["battery", "motor"],
        "reliability": 0.85,
        "lead_time_days": 3,
        "cost_per_unit": 60.0,
    },
    "SUP_E": {
        "id": "SUP_E",
        "name": "EchoTech South Korea",
        "location": "South Korea",
        "components": ["chip", "battery"],
        "reliability": 0.90,
        "lead_time_days": 6,
        "cost_per_unit": 55.0,
    },
}

WAREHOUSES = {
    "WH_NORTH": {
        "id": "WH_NORTH",
        "name": "North Warehouse",
        "location": "Chicago",
        "stock": {
            "chip": 400,
            "sensor": 350,
            "casing": 600,
            "motor": 200,
            "battery": 300,
        },
    },
    "WH_SOUTH": {
        "id": "WH_SOUTH",
        "name": "South Warehouse",
        "location": "Dallas",
        "stock": {
            "chip": 250,
            "sensor": 500,
            "casing": 300,
            "motor": 400,
            "battery": 150,
        },
    },
    "WH_WEST": {
        "id": "WH_WEST",
        "name": "West Warehouse",
        "location": "Los Angeles",
        "stock": {
            "chip": 180,
            "sensor": 220,
            "casing": 450,
            "motor": 300,
            "battery": 500,
        },
    },
}

FACTORIES = {
    "FAC_ALPHA": {
        "id": "FAC_ALPHA",
        "name": "Alpha Factory",
        "location": "Detroit",
        "served_by": "WH_NORTH",
        "production_rate": 80,
        "recipe": {
            "chip": 2,
            "sensor": 1,
            "casing": 1,
            "motor": 1,
            "battery": 1,
        },
    },
    "FAC_BETA": {
        "id": "FAC_BETA",
        "name": "Beta Factory",
        "location": "Houston",
        "served_by": "WH_SOUTH",
        "production_rate": 60,
        "recipe": {
            "chip": 1,
            "sensor": 2,
            "motor": 2,
            "battery": 1,
            "casing": 1,
        },
    },
    "FAC_GAMMA": {
        "id": "FAC_GAMMA",
        "name": "Gamma Factory",
        "location": "Seattle",
        "served_by": "WH_WEST",
        "production_rate": 50,
        "recipe": {
            "chip": 1,
            "sensor": 1,
            "casing": 2,
            "motor": 1,
            "battery": 2,
        },
    },
}

ORDERS = {
    "ORD_001": {
        "id": "ORD_001",
        "customer": "TechRetail Inc.",
        "units": 400,
        "due_day": 6,
        "late_penalty_per_day": 12000.0,
        "priority": "CRITICAL",
    },
    "ORD_002": {
        "id": "ORD_002",
        "customer": "AutoParts Co.",
        "units": 300,
        "due_day": 9,
        "late_penalty_per_day": 8000.0,
        "priority": "HIGH",
    },
    "ORD_003": {
        "id": "ORD_003",
        "customer": "MedDevice Ltd.",
        "units": 200,
        "due_day": 12,
        "late_penalty_per_day": 5000.0,
        "priority": "MEDIUM",
    },
    "ORD_004": {
        "id": "ORD_004",
        "customer": "ConsumerGoods Corp.",
        "units": 150,
        "due_day": 15,
        "late_penalty_per_day": 3000.0,
        "priority": "LOW",
    },
}

# Maximum possible daily penalty (all 4 orders simultaneously late)
MAX_DAILY_PENALTY = 28000.0
# Over 10-day cascade window
MAX_TOTAL_PENALTY = MAX_DAILY_PENALTY * 10

COMPONENTS = ["chip", "sensor", "casing", "motor", "battery"]

# Priority order for order fulfillment allocation
PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# ── Disruption Library ────────────────────────────────────────────────────────
# stock_impact : {wh_id: {component: delta}}  (negative = removal)
# order_impact : {order_id: multiplier}        (multiplied against units_required)
# delay_days   : how long affected suppliers are halted (0 = supplier not halted)

DISRUPTIONS = {
    "port_strike_asia": {
        "id": "port_strike_asia",
        "name": "Port Strike Asia",
        "description": (
            "Major port strike across Asia-Pacific disrupts all shipments from "
            "SUP_A (Taiwan) and SUP_E (South Korea) for 8 days."
        ),
        "severity": "critical",
        "affected_suppliers": ["SUP_A", "SUP_E"],
        "delay_days": 8,
        "affected_components": ["chip", "sensor", "battery"],
        "stock_impact": {},
        "order_impact": {},
    },
    "factory_fire_sup_b": {
        "id": "factory_fire_sup_b",
        "name": "Factory Fire SUP_B",
        "description": (
            "Devastating fire at BetaCraft Germany factory halts motor and "
            "sensor production for 12 days."
        ),
        "severity": "high",
        "affected_suppliers": ["SUP_B"],
        "delay_days": 12,
        "affected_components": ["motor", "sensor"],
        "stock_impact": {},
        "order_impact": {},
    },
    "demand_spike": {
        "id": "demand_spike",
        "name": "Demand Spike ORD_001",
        "description": (
            "Unexpected viral product launch doubles TechRetail's unit "
            "requirement for ORD_001."
        ),
        "severity": "medium",
        "affected_suppliers": [],
        "delay_days": 0,
        "affected_components": [],
        "stock_impact": {},
        "order_impact": {"ORD_001": 2.0},
    },
    "logistics_collapse_americas": {
        "id": "logistics_collapse_americas",
        "name": "Logistics Collapse Americas",
        "description": (
            "Widespread logistics network failure delays all SUP_D (Mexico) "
            "shipments by 5 days."
        ),
        "severity": "high",
        "affected_suppliers": ["SUP_D"],
        "delay_days": 5,
        "affected_components": ["battery", "motor"],
        "stock_impact": {},
        "order_impact": {},
    },
    "quality_recall_sup_e": {
        "id": "quality_recall_sup_e",
        "name": "Quality Recall SUP_E Chips",
        "description": (
            "Defective chip batch from EchoTech Korea forces immediate recall: "
            "150 chips removed from WH_NORTH, 100 chips from WH_WEST."
        ),
        "severity": "critical",
        "affected_suppliers": ["SUP_E"],
        "delay_days": 0,
        "affected_components": ["chip"],
        "stock_impact": {
            "WH_NORTH": {"chip": -150},
            "WH_WEST": {"chip": -100},
        },
        "order_impact": {},
    },
    "dual_disruption": {
        "id": "dual_disruption",
        "name": "Dual Disruption (SUP_A + SUP_D)",
        "description": (
            "Simultaneous crises: Taiwan port strike halts SUP_A, and Mexico "
            "logistics collapse grounds SUP_D — chips, sensors, batteries, and "
            "motors all at risk for 7 days."
        ),
        "severity": "critical",
        "affected_suppliers": ["SUP_A", "SUP_D"],
        "delay_days": 7,
        "affected_components": ["chip", "sensor", "battery", "motor"],
        "stock_impact": {},
        "order_impact": {},
    },
}
