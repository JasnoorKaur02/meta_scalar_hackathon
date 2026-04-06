"""
parser.py — Natural language → structured action parser.
Uses keyword matching ONLY (no LLM).
Handles plural component names and returns uppercase IDs.
"""

import re
from typing import Optional, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPLIER_IDS: List[str] = ["SUP_A", "SUP_B", "SUP_C", "SUP_D", "SUP_E"]
WAREHOUSE_IDS: List[str] = ["WH_NORTH", "WH_SOUTH", "WH_WEST"]
FACTORY_IDS: List[str] = ["FAC_ALPHA", "FAC_BETA", "FAC_GAMMA"]
ORDER_IDS: List[str] = ["ORD_001", "ORD_002", "ORD_003", "ORD_004"]

# Plural → singular normalization (longest match first)
COMPONENT_MAP = {
    "batteries": "battery",
    "casings": "casing",
    "sensors": "sensor",
    "motors": "motor",
    "chips": "chip",
    "battery": "battery",
    "casing": "casing",
    "sensor": "sensor",
    "motor": "motor",
    "chip": "chip",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[int]:
    """Return first integer found in text, else None."""
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None


def _extract_component(text: str) -> Optional[str]:
    """Return normalized component name (singular) found in text, else None."""
    text_lower = text.lower()
    for raw in COMPONENT_MAP:
        if re.search(r"\b" + raw + r"\b", text_lower):
            return COMPONENT_MAP[raw]
    return None


def _extract_ids_in_text_order(text: str, id_list: List[str]) -> List[str]:
    """
    Return IDs from id_list that appear in text, ordered by position in text.
    All comparisons are case-insensitive (text is uppercased).
    """
    upper = text.upper()
    hits = []
    for id_ in id_list:
        pos = upper.find(id_)
        if pos != -1:
            hits.append((pos, id_))
    hits.sort(key=lambda x: x[0])
    return [id_ for _, id_ in hits]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_action(command: str) -> dict:
    """
    Parse a natural language command into a structured action dict.

    The returned dict always has a 'type' key.
    Unknown commands return {'type': 'unknown', 'raw': command}.
    All entity IDs in the output are uppercase.
    """
    cmd = command.strip()
    low = cmd.lower()

    # ------------------------------------------------------------------
    # 1. reroute_supplier
    #    Keywords: "reroute", "switch supplier", "alternate supplier"
    # ------------------------------------------------------------------
    if "reroute" in low or "switch supplier" in low or "alternate supplier" in low:
        suppliers = _extract_ids_in_text_order(cmd, SUPPLIER_IDS)
        return {
            "type": "reroute_supplier",
            "from_supplier": suppliers[0] if len(suppliers) > 0 else None,
            "to_supplier": suppliers[1] if len(suppliers) > 1 else None,
            "component": _extract_component(low),
            "units": _extract_number(cmd) or 100,
        }

    # ------------------------------------------------------------------
    # 2. expedite_shipping
    #    Keywords: "expedite", "airfreight", "urgent ship", "rush"
    # ------------------------------------------------------------------
    if "expedite" in low or "airfreight" in low or "urgent ship" in low or "rush" in low:
        warehouses = _extract_ids_in_text_order(cmd, WAREHOUSE_IDS)
        factories = _extract_ids_in_text_order(cmd, FACTORY_IDS)
        return {
            "type": "expedite_shipping",
            "from_warehouse": warehouses[0] if warehouses else None,
            "to_factory": factories[0] if factories else None,
            "component": _extract_component(low),
            "units": _extract_number(cmd) or 100,
        }

    # ------------------------------------------------------------------
    # 3. reallocate_stock
    #    Keywords: "reallocate", "transfer", "move stock", "redistribute"
    # ------------------------------------------------------------------
    if (
        "reallocate" in low
        or "transfer" in low
        or "move stock" in low
        or "redistribute" in low
    ):
        # Extract warehouses IN TEXT ORDER so from/to direction is preserved
        warehouses = _extract_ids_in_text_order(cmd, WAREHOUSE_IDS)
        return {
            "type": "reallocate_stock",
            "from_warehouse": warehouses[0] if len(warehouses) > 0 else None,
            "to_warehouse": warehouses[1] if len(warehouses) > 1 else None,
            "component": _extract_component(low),
            "units": _extract_number(cmd) or 100,
        }

    # ------------------------------------------------------------------
    # 4. pause_factory
    #    Keywords: "pause", "halt", "stop factory", "idle factory"
    # ------------------------------------------------------------------
    if "pause" in low or "halt" in low or "stop factory" in low or "idle factory" in low:
        factories = _extract_ids_in_text_order(cmd, FACTORY_IDS)
        return {
            "type": "pause_factory",
            "factory": factories[0] if factories else None,
            "days": _extract_number(cmd) or 1,
        }

    # ------------------------------------------------------------------
    # 5. activate_emergency_supplier
    #    Keywords: "emergency supplier", "spot market", "emergency order"
    #    (checked before notify to avoid "emergency" false-matching "notify")
    # ------------------------------------------------------------------
    if "emergency supplier" in low or "spot market" in low or "emergency order" in low:
        warehouses = _extract_ids_in_text_order(cmd, WAREHOUSE_IDS)
        return {
            "type": "activate_emergency_supplier",
            "component": _extract_component(low),
            "units": _extract_number(cmd) or 100,
            "target_warehouse": warehouses[0] if warehouses else None,
        }

    # ------------------------------------------------------------------
    # 6. notify_client
    #    Keywords: "notify", "inform client", "warn client", "alert client"
    # ------------------------------------------------------------------
    if (
        "notify" in low
        or "inform client" in low
        or "warn client" in low
        or "alert client" in low
    ):
        orders = _extract_ids_in_text_order(cmd, ORDER_IDS)
        return {
            "type": "notify_client",
            "order_id": orders[0] if orders else None,
            "expected_delay_days": _extract_number(cmd) or 1,
        }

    # ------------------------------------------------------------------
    # 7. assess_situation
    #    Keywords: "assess", "status", "report", "situation", "overview"
    #    (Broadest match — must be last)
    # ------------------------------------------------------------------
    if (
        "assess" in low
        or "status" in low
        or "report" in low
        or "situation" in low
        or "overview" in low
    ):
        return {"type": "assess_situation", "raw": cmd}

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    return {"type": "unknown", "raw": cmd}


# ---------------------------------------------------------------------------
# Task 1 Assessment Parser
# ---------------------------------------------------------------------------

def parse_assessment(command: str) -> dict:
    """
    Extract structured fields from a Task 1 assessment response.

    Expected (flexible) format:
      assess situation: affected_components=[chip,sensor] severity=critical
                        days_of_stock=4 factories_at_risk=[FAC_ALPHA,FAC_BETA]

    Missing fields default to [] / None.
    """
    result: dict = {
        "affected_components": [],
        "severity": None,
        "days_of_stock": None,
        "factories_at_risk": [],
    }

    m = re.search(r"affected_components\s*=\s*\[([^\]]*)\]", command, re.IGNORECASE)
    if m:
        raw = [c.strip().lower() for c in m.group(1).split(",") if c.strip()]
        result["affected_components"] = [COMPONENT_MAP.get(c, c) for c in raw]

    m = re.search(r"severity\s*=\s*(\w+)", command, re.IGNORECASE)
    if m:
        result["severity"] = m.group(1).lower()

    m = re.search(r"days_of_stock\s*=\s*(\d+)", command, re.IGNORECASE)
    if m:
        result["days_of_stock"] = int(m.group(1))

    m = re.search(r"factories_at_risk\s*=\s*\[([^\]]*)\]", command, re.IGNORECASE)
    if m:
        result["factories_at_risk"] = [
            f.strip().upper() for f in m.group(1).split(",") if f.strip()
        ]

    return result
