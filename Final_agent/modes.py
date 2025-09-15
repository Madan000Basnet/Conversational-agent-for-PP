"""
modes.py

Detects user intent:
- WEB_SEARCH  → queries with web hints (news, prices, weather, etc.)
- PP          → production planning / optimization queries
- NORMAL      → everything else

Also provides helpers for handling PP objective scoping:
- needs_objective(pp_state) → True if agent must ask cost vs profit
- parse_objective(user_text) → returns "PROFIT" or "COST" if detected, else None
"""

import re
from typing import Dict, Optional

# --------------------------------------------------------------------
# Hints/lexicons
# --------------------------------------------------------------------

WEB_HINTS = [
    "news", "latest", "recent", "headline", "today", "weather",
    "price", "prices", "stock", "stocks", "exchange rate", "fx", "currency",
    "schedule", "flight", "train", "restaurant", "book a table", "review",
    "near me", "map", "traffic", "concert", "movie times", "ticket"
]

PP_HINTS = [
    "optimize", "optimization", "production plan", "production planning",
    "factory", "capacity", "setup cost", "setup time", "overtime", "demand",
    "inventory", "backlog", "machine hours", "labor hours", "materials",
    "bill of materials", "bom", "lot size", "min batch", "max batch", "big m",
    "profit", "maximize profit", "cost", "minimize cost", "revenue", "price",
    "period", "week", "month", "quarter", "products", "product mix", "milp",
    "mixed integer", "integer program", "linear program", "pulp",
]

# --------------------------------------------------------------------
# Objective parsing
# --------------------------------------------------------------------
PP_OBJECTIVE_REGEX = re.compile(
    r"\b("
    # Profit-oriented
    r"(profit\s*(max(?:imization)?|maximize|max)|maximize\s*profit)"
    r"|"
    r"(increase\s*profit(s)?|grow\s*profit(s)?|improve\s*margin(s)?|higher\s*margins?)"
    r"|"
    r"(earn\s*more|make\s*more\s*money|more\s*revenue|increase\s*revenue)"
    r"|"
    # Cost-oriented
    r"(cost\s*(min(?:imization)?|minimize|min)|minimize\s*cost)"
    r"|"
    r"(reduce\s*cost(s)?|cut\s*cost(s)?|lower\s*cost(s)?|lower\s*expense(s)?|reduce\s*expense(s)?|save\s*money)"
    r")\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------
# Mode detection
# --------------------------------------------------------------------

def _contains_any(text: str, hints: list) -> bool:
    t = text.lower()
    return any(h in t for h in hints)

def detect_mode(user_text: str, state: Optional[Dict] = None) -> str:
    """
    Return one of: "WEB_SEARCH", "PP", "NORMAL".
    - If text has PP hints OR previous state indicates PP → PP
    - Else if text has obvious web hints → WEB_SEARCH
    - Else NORMAL

    (We slightly prefer PP if both are present, to follow the user’s “PP router flow” design.)
    """
    text = (user_text or "").strip()
    if _contains_any(text, PP_HINTS):
        return "PP"
    if _contains_any(text, WEB_HINTS):
        return "WEB_SEARCH"

    # If a previous turn was PP, keep user in PP unless text clearly asks for web
    if state and isinstance(state, dict):
        prev_mode = (state.get("mode") or "").upper()
        if prev_mode == "PP":
            return "PP"

    return "NORMAL"

# --------------------------------------------------------------------
# Objective helpers
# --------------------------------------------------------------------

def parse_objective(user_text: str) -> Optional[str]:
    """
    Returns "PROFIT" or "COST" if clearly expressed, else None.

    Examples caught (non-exhaustive):
    - "profit maximization", "maximize profit", "increase profit", "improve margins"
    - "cost minimization", "minimize cost", "reduce costs", "save money"
    """
    if not user_text:
        return None

    # Regex first
    if PP_OBJECTIVE_REGEX.search(user_text):
        t = user_text.lower()
        # Heuristic tie-breaker: decide which side appears more
        profit_hits = sum(kw in t for kw in [
            "profit", "maximize profit", "increase profit", "grow profit",
            "improve margin", "higher margin", "earn more", "make more money",
            "more revenue", "increase revenue",
        ])
        cost_hits = sum(kw in t for kw in [
            "cost", "minimize cost", "reduce cost", "cut cost",
            "lower cost", "lower expense", "reduce expense", "save money",
        ])
        if profit_hits >= cost_hits and profit_hits > 0:
            return "PROFIT"
        if cost_hits > 0:
            return "COST"

        # Fallback to keyword buckets if regex matched but counts tied
        # (very conservative)
        if "profit" in t or "margin" in t or "revenue" in t:
            return "PROFIT"
        if "cost" in t or "expense" in t or "save money" in t:
            return "COST"

    # Lightweight keyword fallback (handles super short inputs)
    text = user_text.lower()
    if any(k in text for k in [
        "maximize profit", "profit maximization", "profit max",
        "increase profit", "grow profit", "improve margin", "higher margin",
        "earn more", "make more money", "more revenue", "increase revenue"
    ]):
        return "PROFIT"

    if any(k in text for k in [
        "minimize cost", "cost minimization", "cost min",
        "reduce cost", "cut cost", "lower cost",
        "lower expense", "reduce expense", "save money"
    ]):
        return "COST"

    # Single-token fallbacks
    if "profit" in text:
        return "PROFIT"
    if "cost" in text or "expense" in text:
        return "COST"

    return None


def needs_objective(pp_state: Dict) -> bool:
    """
    True if the PP objective is missing or invalid.
    We only accept lowercase 'profit' or 'cost' for solver compatibility.
    """
    if not isinstance(pp_state, dict):
        return True
    obj = (pp_state.get("objective") or "").strip().lower()
    return obj not in {"profit", "cost"}
