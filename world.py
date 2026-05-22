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

# Extended order set for the 30-day budget_optimization task
ORDERS_T4 = {
    "ORD_A01": {
        "id": "ORD_A01",
        "customer": "MegaTech Corp.",
        "units": 600,
        "due_day": 8,
        "late_penalty_per_day": 15000.0,
        "priority": "CRITICAL",
    },
    "ORD_A02": {
        "id": "ORD_A02",
        "customer": "AutoDrive Inc.",
        "units": 500,
        "due_day": 12,
        "late_penalty_per_day": 10000.0,
        "priority": "HIGH",
    },
    "ORD_A03": {
        "id": "ORD_A03",
        "customer": "MedSupply Ltd.",
        "units": 400,
        "due_day": 16,
        "late_penalty_per_day": 8000.0,
        "priority": "HIGH",
    },
    "ORD_A04": {
        "id": "ORD_A04",
        "customer": "RetailMax Co.",
        "units": 350,
        "due_day": 20,
        "late_penalty_per_day": 5000.0,
        "priority": "MEDIUM",
    },
    "ORD_A05": {
        "id": "ORD_A05",
        "customer": "IndustrialParts LLC.",
        "units": 250,
        "due_day": 25,
        "late_penalty_per_day": 3500.0,
        "priority": "MEDIUM",
    },
    "ORD_A06": {
        "id": "ORD_A06",
        "customer": "BudgetGoods Co.",
        "units": 200,
        "due_day": 28,
        "late_penalty_per_day": 2000.0,
        "priority": "LOW",
    },
}

# Maximum possible daily penalty (all 4 orders simultaneously late)
MAX_DAILY_PENALTY = 28000.0
# Over 10-day cascade window
MAX_TOTAL_PENALTY = MAX_DAILY_PENALTY * 10

# Maximum for T4 orders
MAX_DAILY_PENALTY_T4 = 43500.0
MAX_TOTAL_PENALTY_T4 = MAX_DAILY_PENALTY_T4 * 30

COMPONENTS = ["chip", "sensor", "casing", "motor", "battery"]

# Priority order for order fulfillment allocation
PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# Negotiation discounts per supplier (fraction off normal price)
SUPPLIER_NEGOTIATION_DISCOUNTS = {
    "SUP_A": 0.12,  # 12% discount for bulk pre-commitment
    "SUP_B": 0.10,
    "SUP_C": 0.18,  # CasingCo eager for volume
    "SUP_D": 0.15,
    "SUP_E": 0.08,  # Premium supplier, less flexible
}

# Disruption schedule for budget_optimization task: (step_number, disruption_id)
BUDGET_OPT_DISRUPTION_SCHEDULE = [
    (1,  "quality_recall_sup_e"),
    (10, "logistics_collapse_americas"),
    (20, "severe_weather_midwest"),
]

# ── Disruption Library ────────────────────────────────────────────────────────
# stock_impact : {wh_id: {component: delta}}  (negative = removal)
# order_impact : {order_id: multiplier}        (multiplied against units_required)
# delay_days   : how long affected suppliers are halted (0 = supplier not halted)

DISRUPTIONS = {
    # ── Original 6 ──────────────────────────────────────────────────────────
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

    # ── 10 New Realistic Scenarios ───────────────────────────────────────────
    "earthquake_taiwan": {
        "id": "earthquake_taiwan",
        "name": "Taiwan Earthquake",
        "description": (
            "Major 7.4 magnitude earthquake devastates Taiwan's semiconductor "
            "manufacturing hub. SUP_A halted 14 days; 200 chips and 100 sensors "
            "destroyed in WH_NORTH transit damage."
        ),
        "severity": "critical",
        "affected_suppliers": ["SUP_A"],
        "delay_days": 14,
        "affected_components": ["chip", "sensor"],
        "stock_impact": {
            "WH_NORTH": {"chip": -200, "sensor": -100},
        },
        "order_impact": {},
    },
    "cyber_attack_logistics": {
        "id": "cyber_attack_logistics",
        "name": "Cyber Attack on Logistics Systems",
        "description": (
            "Ransomware attack on a shared logistics provider locks down "
            "shipment routing for SUP_B and SUP_D for 6 days. "
            "Motor and battery supply frozen."
        ),
        "severity": "high",
        "affected_suppliers": ["SUP_B", "SUP_D"],
        "delay_days": 6,
        "affected_components": ["motor", "battery"],
        "stock_impact": {},
        "order_impact": {},
    },
    "global_chip_shortage": {
        "id": "global_chip_shortage",
        "name": "Global Chip Shortage",
        "description": (
            "Industry-wide chip shortage — new export restrictions and fab "
            "capacity crunch halt SUP_A and SUP_E for 10 days. "
            "Spot recalls remove chips across all warehouses."
        ),
        "severity": "critical",
        "affected_suppliers": ["SUP_A", "SUP_E"],
        "delay_days": 10,
        "affected_components": ["chip", "battery"],
        "stock_impact": {
            "WH_NORTH": {"chip": -100},
            "WH_SOUTH": {"chip": -80},
            "WH_WEST":  {"chip": -60},
        },
        "order_impact": {},
    },
    "severe_weather_midwest": {
        "id": "severe_weather_midwest",
        "name": "Polar Vortex — Midwest",
        "description": (
            "Extreme polar vortex temporarily closes WH_NORTH and WH_SOUTH. "
            "Freezing causes 20% spoilage of temperature-sensitive components "
            "across both warehouses."
        ),
        "severity": "high",
        "affected_suppliers": [],
        "delay_days": 0,
        "affected_components": ["chip", "sensor", "motor"],
        "stock_impact": {
            "WH_NORTH": {"chip": -80, "sensor": -70, "motor": -40},
            "WH_SOUTH": {"chip": -50, "sensor": -100, "motor": -80},
        },
        "order_impact": {},
    },
    "supplier_bankruptcy_sup_c": {
        "id": "supplier_bankruptcy_sup_c",
        "name": "Supplier Bankruptcy — CasingCo Vietnam",
        "description": (
            "CasingCo Vietnam (SUP_C) files for bankruptcy, immediately halting "
            "all casing shipments for 15 days while a receiver takes over operations."
        ),
        "severity": "critical",
        "affected_suppliers": ["SUP_C"],
        "delay_days": 15,
        "affected_components": ["casing"],
        "stock_impact": {},
        "order_impact": {},
    },
    "raw_material_shortage": {
        "id": "raw_material_shortage",
        "name": "Lithium Raw Material Shortage",
        "description": (
            "Global lithium carbonate shortage halts battery production at "
            "SUP_D and SUP_E for 9 days. 150 batteries also recalled from WH_WEST."
        ),
        "severity": "high",
        "affected_suppliers": ["SUP_D", "SUP_E"],
        "delay_days": 9,
        "affected_components": ["battery"],
        "stock_impact": {
            "WH_WEST": {"battery": -150},
        },
        "order_impact": {},
    },
    "regulatory_halt_sup_d": {
        "id": "regulatory_halt_sup_d",
        "name": "Regulatory Shutdown — DeltaPower Mexico",
        "description": (
            "Mexican environmental regulators issue an emergency shutdown order "
            "against DeltaPower's facility. SUP_D halted 11 days; battery and "
            "motor supply from Mexico frozen."
        ),
        "severity": "high",
        "affected_suppliers": ["SUP_D"],
        "delay_days": 11,
        "affected_components": ["battery", "motor"],
        "stock_impact": {},
        "order_impact": {},
    },
    "transportation_strike": {
        "id": "transportation_strike",
        "name": "Truckers' Strike — Americas",
        "description": (
            "Nationwide truckers' strike in North America delays all ground "
            "freight. SUP_C and SUP_D shipments held 7 days; warehouses lose "
            "in-transit stock already loaded before the strike."
        ),
        "severity": "medium",
        "affected_suppliers": ["SUP_C", "SUP_D"],
        "delay_days": 7,
        "affected_components": ["casing", "battery", "motor"],
        "stock_impact": {
            "WH_SOUTH": {"casing": -60, "battery": -50},
            "WH_WEST":  {"motor": -60},
        },
        "order_impact": {},
    },
    "triple_disruption": {
        "id": "triple_disruption",
        "name": "Triple Crisis (SUP_A + SUP_B + SUP_E)",
        "description": (
            "Three simultaneous crises: Taiwan port strike, German factory fire, "
            "and South Korea quality recall. All chip, sensor, motor, and battery "
            "supply chains affected for 8 days. Immediate stock losses across "
            "all warehouses."
        ),
        "severity": "critical",
        "affected_suppliers": ["SUP_A", "SUP_B", "SUP_E"],
        "delay_days": 8,
        "affected_components": ["chip", "sensor", "motor", "battery"],
        "stock_impact": {
            "WH_NORTH": {"chip": -100, "sensor": -80},
            "WH_SOUTH": {"motor": -80},
            "WH_WEST":  {"battery": -100},
        },
        "order_impact": {},
    },
    "demand_surge_all_orders": {
        "id": "demand_surge_all_orders",
        "name": "Holiday Demand Surge",
        "description": (
            "Unexpected holiday season order surge: all customers increase "
            "their requirements by 50%. All four orders now require more units "
            "under the same deadlines."
        ),
        "severity": "medium",
        "affected_suppliers": [],
        "delay_days": 0,
        "affected_components": [],
        "stock_impact": {},
        "order_impact": {
            "ORD_001": 1.5,
            "ORD_002": 1.5,
            "ORD_003": 1.5,
            "ORD_004": 1.5,
        },
    },
}
