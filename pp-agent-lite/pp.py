# pp.py
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from services import chat_llm
from extract import (
    try_parse_list, try_parse_bool, try_parse_number_map, try_parse_nested_number_map
)

# Optional LLM struct extractor (safe fallback if absent)
try:
    from structured_extract import extract_pp_fields
except Exception:  # pragma: no cover
    def extract_pp_fields(text: str, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {}

# Centralized solver dispatcher (profit ILP or cost MILP).
from solvers import optimize

KG_PATH = Path("kg/pp_schema.json")


# =========================== KG / Schema helpers ===========================
def load_schema() -> Dict[str, Any]:
    try:
        return json.loads(Path(KG_PATH).read_text(encoding="utf-8"))
    except Exception:
        return {"variables": []}

def _schema_hint(schema: Dict[str, Any], name: str) -> str:
    for v in schema.get("variables", []):
        if v.get("name") == name:
            return v.get("hint", "")
    return ""

def _ask_one_question(missing_item: Dict[str, Any], context: Dict[str, Any]) -> str:
    sys = open("prompts/system.txt", "r", encoding="utf-8").read()
    qp = open("prompts/pp_questioner.txt", "r", encoding="utf-8").read()
    messages = [
        {"role": "system", "content": sys},
        {"role": "system", "content": qp},
        {"role": "user", "content": json.dumps({
            "missing_item": {k: missing_item[k] for k in ["name", "type", "hint"] if k in missing_item},
            "context": {k: context.get(k) for k in ["products", "periods", "resources"] if k in context}
        }, ensure_ascii=False)}
    ]
    question = chat_llm(messages, temperature=0.1)
    return question.strip()


# =========================== Periods parsing ===========================
_WORD2NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
}

def _parse_periods(text: str) -> List[str]:
    t = text.strip()
    tokens = re.findall(r"\b(?:W\d+|M\d+|D\d+|\d{4}-\d{2}-\d{2})\b", t, flags=re.IGNORECASE)
    if tokens:
        return [tok.upper() for tok in tokens]
    m = re.search(r"\b(\d+)\s*(week|month|day)s?\b", t, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1)); unit = m.group(2).lower()
        prefix = {"week":"W","month":"M","day":"D"}[unit]
        return [f"{prefix}{i+1}" for i in range(n)]
    m2 = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(week|month|day)s?\b", t, flags=re.IGNORECASE)
    if m2:
        n = _WORD2NUM[m2.group(1).lower()]; unit = m2.group(2).lower()
        prefix = {"week":"W","month":"M","day":"D"}[unit]
        return [f"{prefix}{i+1}" for i in range(n)]
    m3 = re.fullmatch(r"\s*(\d+)\s*", t)
    if m3:
        n = int(m3.group(1)); return [f"W{i+1}" for i in range(n)]
    if re.search(r"\bnext\s+week\b", t, flags=re.IGNORECASE):
        return ["W1"]
    return []


# ====================== Opportunistic extractors ======================
_RES_CANDIDATES = ["oven", "labor", "mixer", "packaging", "machine", "line", "worker", "staff"]

def _extract_products_from_text(text: str) -> List[str]:
    t = text.strip()
    m = re.search(r"Products?\s*:\s*(.+?)(?:\n\s*\n|^Constraints?:|\Z)",
                  t, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    segment = m.group(1).strip() if m else t
    segment = segment.replace("•", "\n")
    lines = re.split(r"[;\n]+", segment)
    names: List[str] = []
    for line in lines:
        line = line.strip(" .;:,")
        if not line: continue
        cand = line.split(":", 1)[0].strip()
        parts = re.split(r"\s*(?:,| and | & |/|\|)\s*", cand, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip(" .;:,")
            if p and len(p.split()) <= 4 and not re.search(r"\b(profit|cost|batches?|produce|setup)\b", p, re.I):
                names.append(p)
    out, seen = [], set()
    for n in names:
        nl = n.lower()
        if nl not in seen: seen.add(nl); out.append(n)
    if not out:
        title_chunks = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", t)
        blacklist = {"A Local Bakery","The Owner","Production Plan","Next Week","Total Profit",
                     "Setup Costs","Oven Time","Ingredient Limits","Minimum Batch Size"}
        out = [c.strip(" .,:;") for c in title_chunks if c not in blacklist]
        if len(out) < 2: return []
    return out

def _extract_periods_from_text(text: str) -> List[str]:
    return _parse_periods(text)

def _extract_resources_from_text(text: str) -> List[str]:
    caps = re.findall(r"([A-Z][a-zA-Z]+)\s+Time\s*:", text)
    hits = {c.lower() for c in caps}
    t = text.lower()
    for tok in _RES_CANDIDATES:
        if re.search(rf"\b{re.escape(tok)}\b", t):
            hits.add(tok)
    vocab = set(_RES_CANDIDATES)
    norm = set()
    for h in hits:
        if h.endswith("s") and h[:-1] in vocab: norm.add(h[:-1])
        else: norm.add(h)
    out = [x for x in norm if len(x) <= 20 and len(x.split()) <= 2]
    canon = ["oven","labor","mixer","packaging","line","machine","worker","staff"]
    ordered = [r for r in canon if r in out]
    for r in out:
        if r not in ordered: ordered.append(r)
    return ordered

def _extract_profit_from_text(text: str, products: List[str]) -> Dict[str, float]:
    prof: Dict[str, float] = {}
    for p in products:
        for m in re.finditer(re.escape(p), text, flags=re.IGNORECASE):
            window = text[m.start(): m.start()+200]
            mnum = re.search(r"profit[^0-9$]{0,40}\$?\s*([0-9]+(?:\.[0-9]+)?)", window, flags=re.IGNORECASE)
            if not mnum:
                mnum = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?)?[^A-Za-z0-9]{0,20}profit", window, flags=re.IGNORECASE)
            if mnum:
                prof[p] = float(mnum.group(1)); break
    return prof

def _extract_usage_from_text(text: str, products: List[str], resources: List[str]) -> Dict[str, Dict[str, float]]:
    usage: Dict[str, Dict[str, float]] = {}
    for p in products:
        for r in resources:
            patt = rf"(?:each|per)\s+(?:batch|unit|box)\s+of\s+{re.escape(p)}.*?(?:needs|required|uses)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:hours?|units?).*?\b{re.escape(r)}\b"
            m = re.search(patt, text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                usage.setdefault(p, {})[r] = float(m.group(1))
            else:
                patt2 = rf"{re.escape(p)}.*?([0-9]+(?:\.[0-9]+)?)\s*(?:hours?|units?).*?\b{re.escape(r)}\b"
                m2 = re.search(patt2, text, flags=re.IGNORECASE | re.DOTALL)
                if m2: usage.setdefault(p, {})[r] = float(m2.group(1))
    return usage

def _extract_capacity_from_text(text: str, resources: List[str], periods: List[str]) -> Dict[str, Dict[str, float]]:
    cap: Dict[str, Dict[str, float]] = {}
    if not periods: periods = ["W1"]
    for r in resources:
        patt = rf"\b{re.escape(r)}\b.*?(?:available|capacity|limit).*?([0-9]+(?:\.[0-9]+)?)\s*(?:hours?|units?)"
        m = re.search(patt, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            val = float(m.group(1))
            for t in periods:
                cap.setdefault(r, {})[t] = val
    return cap

def _extract_max_batches_from_text(text: str, products: List[str]) -> Dict[str, float]:
    """
    Detect phrases like:
      - "maximum of 40 batches of Classic Cookies"
      - "at most 25 batches of Gourmet Brownies"
      - "can produce at most 25 batches of Gourmet Brownies"
    """
    mx: Dict[str, float] = {}
    for p in products:
        patt1 = rf"maximum\s+of\s+([0-9]+)\s+batches?\s+of\s+{re.escape(p)}"
        m1 = re.search(patt1, text, flags=re.IGNORECASE | re.DOTALL)
        if m1:
            mx[p] = float(m1.group(1))
            continue
        patt2 = rf"at\s+most\s+([0-9]+)\s+batches?\s+of\s+{re.escape(p)}"
        m2 = re.search(patt2, text, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            mx[p] = float(m2.group(1))
            continue
        patt3 = rf"produce\s+at\s+most\s+([0-9]+)\s+batches?\s+of\s+{re.escape(p)}"
        m3 = re.search(patt3, text, flags=re.IGNORECASE | re.DOTALL)
        if m3:
            mx[p] = float(m3.group(1))
            continue
        near = re.search(rf"{re.escape(p)}.*?([0-9]+)\s+batches?", text, flags=re.IGNORECASE | re.DOTALL)
        if near and re.search(r"(maximum|at\s+most|limit|enough\s+ingredients)", text[near.start()-40:near.end()+40], flags=re.IGNORECASE):
            mx[p] = float(near.group(1))
    return mx

def _extract_setup_cost_from_text(text: str, products: List[str]) -> Dict[str, float]:
    sc: Dict[str, float] = {}
    for p in products:
        m = re.search(rf"setup\s+cost.*?\b{re.escape(p)}\b.*?([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE | re.DOTALL)
        if m: sc[p] = float(m.group(1))
    return sc

def _extract_min_batch_from_text(text: str, products: List[str]) -> Dict[str, float]:
    mb: Dict[str, float] = {}
    for p in products:
        m = re.search(rf"(?:if\s+any\s+)?{re.escape(p)}.*?at\s+least\s+([0-9]+)\s+batches?", text, flags=re.IGNORECASE | re.DOTALL)
        if m: mb[p] = float(m.group(1))
    return mb


# ============================== Normalizers ==============================
def _clean_products(items: List[str]) -> List[str]:
    cleaned=[]; seen=set()
    for it in items or []:
        it = it.strip(" .;:,")
        if 1 <= len(it.split()) <= 4 and not re.search(r"\b(after|profit|setup|produce|batches?|costs?)\b", it, flags=re.IGNORECASE):
            xl = it.lower()
            if xl not in seen: seen.add(xl); cleaned.append(it)
    return cleaned

def _clean_resources(items: List[str]) -> List[str]:
    if not items: return []
    vocab = set(_RES_CANDIDATES)
    out=[]
    for it in items:
        tok = it.strip().lower()
        if tok.endswith("s") and tok[:-1] in vocab: tok = tok[:-1]
        if tok in vocab and len(tok) <= 20 and len(tok.split()) <= 2:
            out.append(tok)
    canon = ["oven","labor","mixer","packaging","line","machine","worker","staff"]
    ordered = [r for r in canon if r in out]
    for r in out:
        if r not in ordered: ordered.append(r)
    return ordered

def _normalize_pp_state(pp: Dict[str, Any]) -> Dict[str, Any]:
    if "products" in pp:
        pp["products"] = _clean_products(pp["products"])
    if "resources" in pp:
        pp["resources"] = _clean_resources(pp["resources"])
    return pp


# ====================== LLM-first opportunistic ingestion ======================
def _opportunistic_ingest(user_text: str, pp_state: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(pp_state)

    hints = {k: out.get(k) for k in ("products","periods","resources") if out.get(k)}
    llm = extract_pp_fields(user_text, hints=hints)

    def set_if_missing(key):
        if key in llm and (key not in out or not out[key]):
            out[key] = llm[key]

    for k in ["products","periods","resources","profit","usage","capacity",
              "demand","allow_backlog","max_batches","setup_cost","min_batch",
              "initial_inventory","regular_cost","overtime_cost","material_cost",
              "holding_cost","shortage_cost","labor_hours","labor_budget","material_budget",
              "revenue","setup_link_bigM"]:
        set_if_missing(k)

    # Heuristics
    if ("products" not in out) or (not out["products"]):
        prods = _extract_products_from_text(user_text)
        if prods: out["products"] = prods

    if ("periods" not in out) or (not out["periods"]):
        per = _extract_periods_from_text(user_text)
        if per: out["periods"] = per

    if ("resources" not in out) or (not out["resources"]):
        res = _extract_resources_from_text(user_text)
        if res: out["resources"] = res

    prods = out.get("products") or []
    res   = out.get("resources") or []
    per   = out.get("periods") or []

    if prods and ("profit" not in out or not out["profit"]):
        pf = _extract_profit_from_text(user_text, prods)
        if pf: out["profit"] = pf

    if prods and res and ("usage" not in out or not out["usage"]):
        ug = _extract_usage_from_text(user_text, prods, res)
        if ug: out["usage"] = ug

    if res and ("capacity" not in out or not out["capacity"]):
        cp = _extract_capacity_from_text(user_text, res, per)
        if cp: out["capacity"] = cp

    if prods and ("max_batches" not in out or not out["max_batches"]):
        mbx = _extract_max_batches_from_text(user_text, prods)
        if mbx: out["max_batches"] = mbx

    if prods and ("setup_cost" not in out or not out["setup_cost"]):
        sc = _extract_setup_cost_from_text(user_text, prods)
        if sc: out["setup_cost"] = sc

    if prods and ("min_batch" not in out or not out["min_batch"]):
        mb = _extract_min_batch_from_text(user_text, prods)
        if mb: out["min_batch"] = mb

    return _normalize_pp_state(out)


# ================== Ingestion for the awaited variable ==================
def _ingest_answer(name: str, vtype: str, user_text: str, context: Optional[Dict[str, Any]] = None):
    # Prefer LLM structured extraction for the requested field
    llm = extract_pp_fields(user_text)
    if name in llm and llm[name] not in (None, "", [], {}):
        return llm[name]

    # Fallback parsers (with relaxed behavior via context)
    if vtype == "list[string]":
        if name == "periods":
            got = _parse_periods(user_text)
            if got: return got
        if name == "resources":
            return _clean_resources(_extract_resources_from_text(user_text))
        if name == "products":
            return _clean_products(_extract_products_from_text(user_text))
        return try_parse_list(user_text, keyword=name.rstrip("s"))

    if vtype == "boolean":
        return try_parse_bool(user_text)

    if vtype == "map[string, number]":
        got = try_parse_number_map(user_text)

        # Guard: avoid mis-filling initial_inventory from "ingredient limits / maximum ..." prose
        if name == "initial_inventory":
            prods = (context or {}).get("products") or []
            if got and not any(k in prods for k in got.keys()):
                got = {}
            if re.search(r"\b(maximum|at\s+most|limit|ingredient\s+limit)\b", user_text, flags=re.IGNORECASE):
                got = {}

        if got:
            return got

        keys = []
        if context:
            if name in ("profit", "setup_cost", "min_batch", "max_batches", "shortage_cost", "initial_inventory"):
                keys = context.get("products") or []
            elif name in ("labor_budget", "material_budget"):
                keys = context.get("periods") or []
        if keys:
            from extract import try_parse_number_map_relaxed
            relaxed = try_parse_number_map_relaxed(user_text, keys=keys, keyword=name)

            # For initial_inventory, ensure keys are real products
            if name == "initial_inventory" and relaxed and not any(k in keys for k in relaxed.keys()):
                relaxed = {}

            if relaxed:
                return relaxed
        return {}

    if vtype == "map[string, map[string, number]]":
        got = try_parse_nested_number_map(user_text)
        if got:
            return got
        if context:
            prods = context.get("products") or []
            res   = context.get("resources") or []
            per   = context.get("periods") or []
            if name == "usage" and prods and res:
                ug = _extract_usage_from_text(user_text, prods, res)
                if ug:
                    return ug
            if name == "capacity" and res:
                cp = _extract_capacity_from_text(user_text, res, per or ["W1"])
                if cp:
                    return cp
        return {}

    return user_text


# ========================= Solver-driven readiness =========================
PROFIT_CORE = ["products", "periods", "profit", "usage", "capacity"]

COST_ANY_OBJECTIVE_KEYS = {
    "regular_cost", "overtime_cost", "material_cost",
    "holding_cost", "setup_cost", "shortage_cost", "revenue"
}

TYPE_HINTS = {
    "products": "list[string]",
    "periods": "list[string]",
    "resources": "list[string]",
    "profit": "map[string, number]",
    "usage": "map[string, map[string, number]]",
    "capacity": "map[string, map[string, number]]",
    "demand": "map[string, map[string, number]]",
    "allow_backlog": "boolean",
    "max_batches": "map[string, number]",
    "setup_cost": "map[string, number]",
    "min_batch": "map[string, number]",
    "initial_inventory": "map[string, number]",
    "regular_cost": "map[string, map[string, number]]",
    "overtime_cost": "map[string, map[string, number]]",
    "material_cost": "map[string, map[string, number]]",
    "holding_cost": "map[string, map[string, number]]",
    "shortage_cost": "map[string, map[string, number]]",
    "labor_hours": "map[string, map[string, number]]",
    "labor_budget": "map[string, number]",
    "material_budget": "map[string, number]",
    "revenue": "map[string, map[string, number]]",
    "setup_link_bigM": "map[string, map[string, number]]",
}

def _missing_for_profit(pp: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints_map = {
        "products": "Distinct product names (e.g., Cookies, Brownies).",
        "periods": "Time buckets (e.g., W1, W2, or dates).",
        "profit": "Per-unit profit per product (e.g., Cookies=5, Brownies=7).",
        "usage": "usage[product][resource] per unit (e.g., Cookies: oven=2).",
        "capacity": "capacity[resource][period] (e.g., oven: W1=100).",
    }
    missing = []
    for k in PROFIT_CORE:
        if not pp.get(k):
            missing.append({"name": k, "type": TYPE_HINTS.get(k, "string"), "hint": hints_map.get(k, "")})
    return missing

def _missing_for_cost(pp: Dict[str, Any]) -> List[Dict[str, Any]]:
    missing = []
    for k in ["products", "periods"]:
        if not pp.get(k):
            missing.append({"name": k, "type": TYPE_HINTS[k], "hint": "Provide products and periods."})
    has_objective = any(pp.get(k) for k in COST_ANY_OBJECTIVE_KEYS)
    if not has_objective:
        missing.append({"name": "regular_cost", "type": TYPE_HINTS["regular_cost"],
                        "hint": "Provide at least one of: regular/overtime/material/holding/setup/shortage cost, or revenue."})
    bounded = False
    if pp.get("labor_budget") or pp.get("material_budget") or pp.get("max_batches"):
        bounded = True
    elif pp.get("demand") and (pp.get("allow_backlog") is False):
        bounded = True
    if not bounded:
        missing.append({"name": "labor_budget", "type": TYPE_HINTS["labor_budget"],
                        "hint": "Give a budget/limit per period (e.g., W1=80), OR set demand with allow_backlog=false, OR provide max_batches."})
    return missing

def _profit_ready(pp: Dict[str, Any]) -> bool:
    return len(_missing_for_profit(pp)) == 0

def _cost_ready(pp: Dict[str, Any]) -> bool:
    return len(_missing_for_cost(pp)) == 0

def _solver_mode(pp: Dict[str, Any]) -> str:
    """
    Prefer PROFIT if PROFIT is ready but COST is not.
    This prevents a single cost field (e.g., setup_cost) from blocking progress.
    """
    if pp.get("_force_profit"):
        return "PROFIT"
    if _cost_ready(pp):
        return "COST"
    if _profit_ready(pp):
        return "PROFIT"
    if any(pp.get(k) for k in COST_ANY_OBJECTIVE_KEYS):
        return "COST"
    return "PROFIT"

def _next_missing_item(pp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mode = _solver_mode(pp)
    needs = _missing_for_cost(pp) if mode == "COST" else _missing_for_profit(pp)
    return needs[0] if needs else None

def _solver_ready(pp: Dict[str, Any]) -> bool:
    mode = _solver_mode(pp)
    return _cost_ready(pp) if mode == "COST" else _profit_ready(pp)


# === Enrichment flow (ask optional fields one-by-one after core is ready) ===
OPTIONAL_BY_MODE = {
    "PROFIT": [
        "resources",
        "setup_cost",
        "min_batch",
        "max_batches",
        "demand",
        "allow_backlog",
    ],
    "COST": [
        "initial_inventory",
        "regular_cost",
        "overtime_cost",
        "material_cost",
        "holding_cost",
        "shortage_cost",
        "labor_hours",
        "labor_budget",
        "material_budget",
        "revenue",
        "setup_link_bigM",
        "max_batches",
        "min_batch",
        "demand",
        "allow_backlog",
    ],
}

def _build_optional_queue(pp: Dict[str, Any]) -> List[str]:
    mode = _solver_mode(pp)
    candidates = OPTIONAL_BY_MODE["COST"] if mode == "COST" else OPTIONAL_BY_MODE["PROFIT"]
    return [k for k in candidates if not pp.get(k)]

def _is_negative(text: str) -> bool:
    t = text.strip().lower().strip("\"'“”‘’").replace(".", "")
    negatives = {
        "no", "nope", "none", "skip", "n/a", "na", "not available",
        "dont have", "do not have", "nothing else", "thats all", "that is all",
        "no thanks", "no thank you", "stop", "end", "done"
    }
    return any(tok in t for tok in negatives)


# ========================= Pretty solution formatter =========================
def _round_qty(x):
    try:
        xf = float(x)
        return int(round(xf)) if abs(xf - round(xf)) < 1e-6 else round(xf, 2)
    except Exception:
        return x

def _format_solution_plain(meta: Dict[str, Any], pp: Dict[str, Any]) -> str:
    """
    Make a friendly, human-readable summary from solver output.
    Expects meta to contain at least: status, objective, and one of:
      - solution: {product: {period: qty}}
      - totals / x / y / z / v (cost model)
    """
    status = meta.get("status", "Unknown")
    objective = meta.get("objective")
    periods = pp.get("periods", [])
    products = pp.get("products", [])

    # Determine objective wording
    cost_keys = ["regular_cost","overtime_cost","material_cost","holding_cost","shortage_cost","revenue"]
    is_profit_mode = pp.get("_force_profit") or ("profit" in pp and not any(pp.get(k) for k in cost_keys))
    if is_profit_mode and pp.get("setup_cost"):
        obj_line = f"Estimated profit (after setup): {objective}"
    elif is_profit_mode:
        obj_line = f"Estimated profit: {objective}"
    else:
        obj_line = f"Minimum total cost (net): {objective}"

    # Prefer "solution" if present (profit model / simple ILP)
    plan = meta.get("solution")
    lines = []
    if isinstance(plan, dict):
        # plan: {product: {period: qty}}
        by_period = {t: [] for t in (periods or [])}
        if not by_period:
            all_t = set()
            for p, m in plan.items():
                if isinstance(m, dict):
                    all_t |= set(m.keys())
            periods = sorted(all_t)
            by_period = {t: [] for t in periods}
        # fill
        keys_products = products or list(plan.keys())
        for p in keys_products:
            for t, qty in (plan.get(p, {}) or {}).items():
                q = _round_qty(qty)
                if q and q != 0:
                    by_period.setdefault(t, []).append((p, q))
        for t in periods:
            items = by_period.get(t, [])
            if not items:
                lines.append(f"In **{t}**, do not produce any batches.")
                continue
            parts = [f"{q} of {p}" for (p, q) in items]
            if len(parts) == 1:
                lines.append(f"In **{t}**, produce {parts[0]}.")
            else:
                lines.append(f"In **{t}**, produce " + ", ".join(parts[:-1]) + f", and {parts[-1]}.")
    else:
        # cost model (MILP) often returns totals/x/y/z/v in meta
        totals = meta.get("totals") or meta.get("x") or {}
        if isinstance(totals, dict):
            for t in (periods or ["W1"]):
                row = []
                for p in (products or totals.keys()):
                    qty = None
                    if isinstance(totals.get(p), dict):
                        qty = totals[p].get(t)
                    elif isinstance(totals.get(p), (int, float)):
                        qty = totals[p]
                    if qty is not None and float(qty) > 0:
                        row.append(f"{_round_qty(qty)} of {p}")
                if row:
                    if len(row) == 1:
                        lines.append(f"In **{t}**, produce {row[0]}.")
                    else:
                        lines.append(f"In **{t}**, produce " + ", ".join(row[:-1]) + f", and {row[-1]}.")
                else:
                    lines.append(f"In **{t}**, do not produce any batches.")
        else:
            lines.append("No production recommended in the current plan.")

    header = f"**Optimal plan** (status: {status})\n- {obj_line}"
    body = "\n".join(f"- {s}" for s in lines)
    tail = "\n\nIf you'd like to adjust anything (profits, capacities, min/max, demand, etc.), say it and I'll re-run the optimization."
    return f"{header}\n{body}{tail}"


# ========================= Summarize =========================
def _summarize_pp(pp: Dict[str, Any]) -> str:
    parts = []
    if "products" in pp:       parts.append(f"Products: {pp['products']}")
    if "periods" in pp:        parts.append(f"Periods: {pp['periods']}")
    if "resources" in pp:      parts.append(f"Resources: {pp['resources']}")
    if "profit" in pp:         parts.append(f"Profit per product: {pp['profit']}")
    if "usage" in pp:          parts.append(f"Usage[product][resource]: {pp['usage']}")
    if "capacity" in pp:       parts.append(f"Capacity[resource][period]: {pp['capacity']}")
    if "max_batches" in pp:    parts.append(f"Max per product: {pp['max_batches']}")
    if "setup_cost" in pp:     parts.append(f"Setup cost: {pp['setup_cost']}")
    if "min_batch" in pp:      parts.append(f"Min batch: {pp['min_batch']}")
    if "demand" in pp:         parts.append(f"Demand[product][period]: {pp['demand']}")
    if "allow_backlog" in pp:  parts.append(f"Allow backlog: {pp['allow_backlog']}")
    if "initial_inventory" in pp: parts.append(f"Initial inventory: {pp['initial_inventory']}")
    if "regular_cost" in pp:      parts.append(f"Regular cost c[p][t]: {pp['regular_cost']}")
    if "overtime_cost" in pp:     parts.append(f"Overtime cost o[p][t]: {pp['overtime_cost']}")
    if "material_cost" in pp:     parts.append(f"Material cost m[p][t]: {pp['material_cost']}")
    if "holding_cost" in pp:      parts.append(f"Holding cost h[p][t]: {pp['holding_cost']}")
    if "shortage_cost" in pp:     parts.append(f"Shortage cost g[p][t]: {pp['shortage_cost']}")
    if "labor_hours" in pp:       parts.append(f"Labor hours per unit: {pp['labor_hours']}")
    if "labor_budget" in pp:      parts.append(f"Labor budget UB^L[t]: {pp['labor_budget']}")
    if "material_budget" in pp:   parts.append(f"Material budget UB^M[t]: {pp['material_budget']}")
    if "revenue" in pp:           parts.append(f"Revenue price[p][t]: {pp['revenue']}")
    return "Current PP setup:\n- " + "\n- ".join(parts)


# ========================= Main PP turn handler =========================
def handle_pp_turn(user_text: str, pp_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    schema = load_schema()

    # quick reset
    if user_text.strip().lower() in {"reset pp", "clear pp"}:
        return "Cleared PP context. Let's start again—what products do you want to plan for?", {}

    # Opportunistic ingest (LLM-first) + normalize
    pp_state = _opportunistic_ingest(user_text, pp_state)
    pp_state = _normalize_pp_state(pp_state)

    # Global "optimize" command works at any time (but only runs if ready)
    if any(tok in user_text.lower() for tok in ["optimize", "run solver", "solve", "go ahead"]):
        if _solver_ready(pp_state):
            # If profit is ready but cost isn't, favor profit path
            if _profit_ready(pp_state) and not _cost_ready(pp_state):
                pp_state["_force_profit"] = True
            plan, meta = optimize(pp_state)
            pp_state.update(meta)
            pp_state["_awaiting"] = None
            pp_state["_enriching"] = None
            # Pretty, plain-language result
            pretty = _format_solution_plain(meta, pp_state)
            return pretty, pp_state
        # Not ready: fall through to ask next missing item

    awaiting = pp_state.get("_awaiting")
    enriching = pp_state.get("_enriching")  # {"queue": [...], "idx": int}

    # ------------------------ AWAITING BRANCH ------------------------
    if awaiting:
        name = awaiting["name"]; vtype = awaiting["type"]

        # Enrichment entry question: do you have more details?
        if name == "_any_more":
            yn = try_parse_bool(user_text)
            if yn is False or _is_negative(user_text):
                pp_state["_awaiting"] = None
                summary = _summarize_pp(pp_state)
                return summary + "\n\nIf you want changes, tell me. Otherwise say “optimize”.", pp_state

            queue = _build_optional_queue(pp_state)
            if not queue:
                pp_state["_awaiting"] = None
                summary = _summarize_pp(pp_state)
                return summary + "\n\nIf you want changes, tell me. Otherwise say “optimize”.", pp_state

            pp_state["_enriching"] = {"queue": queue, "idx": 0}
            nxt_name = queue[0]
            pp_state["_awaiting"] = {"name": nxt_name, "type": TYPE_HINTS.get(nxt_name, "string")}
            question = _ask_one_question(
                {"name": nxt_name, "type": TYPE_HINTS.get(nxt_name, "string"), "hint": _schema_hint(schema, nxt_name)},
                pp_state
            )
            return question, pp_state

        # Normal awaited variable ingestion
        val = _ingest_answer(name, vtype, user_text, context=pp_state)

        # Treat explicit skip/none as skip (just move to next optional when enriching)
        if val in (None, "", [], {}) and _is_negative(user_text):
            val = None  # skip this field only

        # Special: if we're in COST mode asking for COST-only required (e.g., labor_budget)
        # and user skipped, fall back to PROFIT mode (keep cost extras but don't block).
        if val in (None, "", [], {}) and _solver_mode(pp_state) == "COST" and name in {"labor_budget", "material_budget"}:
            pp_state["_force_profit"] = True
            pp_state["_awaiting"] = None
            summary = _summarize_pp(pp_state)
            return (
                "Got it — skipping budgets. I’ll proceed with profit-based optimization and ignore incomplete cost constraints. "
                "You can still add budgets or demand later.\n\n" +
                summary + "\n\nIf you want changes, tell me. Otherwise say “optimize”.",
                pp_state
            )

        if val not in (None, "", [], {}):
            pp_state[name] = val
            pp_state["_awaiting"] = None
            # Mine again (same turn may include other info)
            pp_state = _opportunistic_ingest(user_text, pp_state)
            pp_state = _normalize_pp_state(pp_state)
        else:
            # couldn't parse; if enriching and user said "skip", move on silently
            if not (enriching and _is_negative(user_text)):
                hint = _schema_hint(schema, name) or {
                    "products": "Tell me the product names (natural text is fine).",
                    "periods": "Tell me the time buckets (e.g., a week → W1).",
                    "profit": "Tell me the per-unit profit for each product in your own words.",
                    "usage": "Describe resource usage per product (I'll parse it).",
                    "capacity": "Describe resource availability per period (I'll parse it)."
                }.get(name, "")
                return f"Sorry, I couldn't read that. **{name}** — {hint}. You can also say 'skip'.", pp_state

        # If enriching, advance queue (skip only this field; continue to next)
        enriching = pp_state.get("_enriching")
        if enriching:
            queue = enriching.get("queue", [])
            idx = enriching.get("idx", 0) + 1
            # auto-skip items that already got populated opportunistically
            while idx < len(queue) and pp_state.get(queue[idx]):
                idx += 1
            if idx >= len(queue):
                pp_state["_enriching"] = None
                summary = _summarize_pp(pp_state)
                return summary + "\n\nIf you want changes, tell me. Otherwise say “optimize”.", pp_state

            pp_state["_enriching"]["idx"] = idx
            nxt_name = queue[idx]
            pp_state["_awaiting"] = {"name": nxt_name, "type": TYPE_HINTS.get(nxt_name, "string")}
            question = _ask_one_question(
                {"name": nxt_name, "type": TYPE_HINTS.get(nxt_name, "string"), "hint": _schema_hint(schema, nxt_name)},
                pp_state
            )
            return question, pp_state

        # Not enriching: if core is now ready, invite enrichment (not auto-optimize)
        if _solver_ready(pp_state):
            pp_state["_awaiting"] = {"name": "_any_more", "type": "boolean"}
            return (
                "I can optimize now. Do you want to include any other details "
                "(e.g., setup_cost, min_batch, max_batches, demand & allow_backlog, budgets)? "
                "You can answer some and **skip** others. Say **no** to skip enrichment entirely.",
                pp_state
            )

        # Otherwise, ask the next required item
        nxt = _next_missing_item(pp_state)
        pp_state["_awaiting"] = {"name": nxt["name"], "type": nxt["type"]}
        question = _ask_one_question(nxt, pp_state)
        return question, pp_state

    # ------------------------ NOT AWAITING BRANCH ------------------------

    # Optimize if ready and the user explicitly asks
    if _solver_ready(pp_state) and any(k in user_text.lower() for k in ["optimize", "run solver", "solve", "go ahead"]):
        plan, meta = optimize(pp_state)
        pp_state.update(meta)
        pretty = _format_solution_plain(meta, pp_state)
        return pretty, pp_state

    # If model is ready but user didn't say optimize → invite enrichment
    if _solver_ready(pp_state) and not pp_state.get("_enriching"):
        pp_state["_awaiting"] = {"name": "_any_more", "type": "boolean"}
        return (
            "I can optimize now. Do you want to include any other details "
            "(e.g., setup_cost, min_batch, max_batches, demand & allow_backlog, budgets)? "
            "You can answer some and **skip** others. Say **no** to skip enrichment entirely.",
            pp_state
        )

    # Not ready yet → ask next required item
    nxt = _next_missing_item(pp_state)
    if nxt is None:
        summary = _summarize_pp(pp_state)
        return summary + "\n\nTell me what’s missing, or say “optimize”.", pp_state

    if not nxt.get("hint"):
        nxt["hint"] = _schema_hint(schema, nxt["name"])
    pp_state["_awaiting"] = {"name": nxt["name"], "type": nxt["type"]}
    question = _ask_one_question(nxt, pp_state)
    return question, pp_state
