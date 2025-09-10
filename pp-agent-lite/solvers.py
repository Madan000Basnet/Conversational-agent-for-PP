from typing import Dict, Any, Tuple, List, Optional
from pulp import (
    LpProblem, LpVariable, LpMaximize, LpMinimize, LpStatus,
    LpInteger, LpBinary, lpSum, value
)

# ---------- Utilities shared by solvers ----------

def _compute_bigM_profit(pp: Dict[str, Any], p: str) -> float:
    periods = pp.get("periods", [])
    usage_p = (pp.get("usage") or {}).get(p, {})
    cap = pp.get("capacity") or {}

    resource_keys = set(usage_p.keys()) | set(cap.keys())

    if usage_p and cap and resource_keys:
        total = 0.0
        for t in periods:
            per_t_bounds = []
            for r in resource_keys:
                if r in usage_p and r in cap and t in cap[r] and usage_p[r] > 0:
                    per_t_bounds.append(cap[r][t] / usage_p[r])
            total += min(per_t_bounds) if per_t_bounds else 0.0
        if total > 0:
            return total

    if "max_batches" in pp and p in pp["max_batches"]:
        return max(pp["max_batches"][p], 1.0) * max(1, len(periods) or 1)

    return 1e6

def _get_pt(mm: Dict[str, Any], p: str, t: str, default: float = 0.0) -> float:
    if not isinstance(mm, dict):
        return default
    v = mm.get(p)
    if isinstance(v, dict):
        try: return float(v.get(t, default))
        except: return default
    if isinstance(v, (int, float, str)):
        try: return float(v)
        except: return default
    if t in mm and isinstance(mm[t], (int, float, str)):
        try: return float(mm[t])
        except: return default
    return default

def _period_prev(periods: List[str], t: str) -> Optional[str]:
    try:
        i = periods.index(t)
        return periods[i-1] if i > 0 else None
    except ValueError:
        return None

def _bigM_prod(pp: Dict[str, Any], p: str, t: str) -> float:
    # explicit override
    M_override = _get_pt(pp.get("setup_link_bigM") or {}, p, t, default=None)
    if M_override is not None and M_override != 0:
        return float(M_override)

    candidates = []

    # from labor budget / hours
    L_t = (pp.get("labor_budget") or {}).get(t)
    h_pt = _get_pt(pp.get("labor_hours") or {}, p, t, default=0.0)
    if L_t is not None and h_pt > 0:
        candidates.append(float(L_t) / float(h_pt))

    # from material budget / material "cost" per unit
    M_t = (pp.get("material_budget") or {}).get(t)
    m_pt = _get_pt(pp.get("material_cost") or {}, p, t, default=0.0)
    if M_t is not None and m_pt > 0:
        candidates.append(float(M_t) / float(m_pt))

    # per-product max
    mx = (pp.get("max_batches") or {}).get(p)
    if mx:
        candidates.append(float(mx))

    return max(1.0, min(candidates) if candidates else 1e6)

# ---------- Solver 1: Profit − Setup ILP (your original) ----------

def _optimize_profit_ilp(pp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    products  = pp["products"]
    periods   = pp["periods"]
    profit    = pp["profit"]
    usage     = pp.get("usage") or {}
    capacity  = pp.get("capacity") or {}

    demand        = pp.get("demand")
    allow_backlog = bool(pp.get("allow_backlog", False))
    max_batches   = pp.get("max_batches")
    setup_cost    = pp.get("setup_cost") or {}
    min_batch     = pp.get("min_batch") or {}

    m = LpProblem("PP_MaxProfit_ILP", LpMaximize)

    x = {(p, t): LpVariable(f"x_{p}_{t}", lowBound=0, cat=LpInteger)
         for p in products for t in periods}
    y = {p: LpVariable(f"y_{p}", lowBound=0, upBound=1, cat=LpBinary)
         for p in products}

    m += (
        lpSum(profit[p] * x[(p, t)] for p in products for t in periods)
        - lpSum((setup_cost.get(p, 0.0)) * y[p] for p in products)
    )

    # capacity constraints only where capacity exists
    for r, cap_by_t in capacity.items():
        for t, cap_val in cap_by_t.items():
            m += lpSum((usage.get(p, {}).get(r, 0.0)) * x[(p, t)] for p in products) <= cap_val, f"cap_{r}_{t}"

    if demand and not allow_backlog:
        for p in products:
            for t in periods:
                if p in demand and t in demand[p]:
                    m += x[(p, t)] >= demand[p][t], f"demand_{p}_{t}"

    for p in products:
        M = _compute_bigM_profit(pp, p)
        m += lpSum(x[(p, t)] for t in periods) <= M * y[p], f"link_{p}"
        if max_batches and p in max_batches:
            m += lpSum(x[(p, t)] for t in periods) <= max_batches[p], f"max_{p}"
        if p in min_batch:
            m += lpSum(x[(p, t)] for t in periods) >= min_batch[p] * y[p], f"min_{p}"

    m.solve()

    status = LpStatus[m.status]
    sol = {p: {t: int(round(value(x[(p, t)]) or 0)) for t in periods} for p in products}
    obj = round(value(m.objective) or 0, 4)

    summary = (
        "Solver: PuLP (ILP, maximize profit−setup)\n"
        f"Status: {status}\n"
        f"Objective: {obj}\n"
        f"Plan (integer units): {sol}"
    )
    return summary, {"solution": sol, "objective": obj, "status": status}

# ---------- Solver 2: Multi-period cost MILP (inventory, OT, shortages, budgets) ----------

def _optimize_cost_milp(pp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    products = pp["products"]
    periods  = pp["periods"]

    regular_cost   = pp.get("regular_cost")   or {}
    overtime_cost  = pp.get("overtime_cost")  or {}
    material_cost  = pp.get("material_cost")  or {}
    holding_cost   = pp.get("holding_cost")   or {}
    setup_cost     = pp.get("setup_cost")     or {}
    shortage_cost  = pp.get("shortage_cost")  or {}
    revenue        = pp.get("revenue")        or {}

    labor_hours    = pp.get("labor_hours")    or {}
    labor_budget   = pp.get("labor_budget")   or {}
    material_budget= pp.get("material_budget")or {}

    demand         = pp.get("demand")         or {}
    I0             = pp.get("initial_inventory") or {}
    min_batch      = pp.get("min_batch")      or {}
    max_batches    = pp.get("max_batches")    or {}
    allow_backlog  = bool(pp.get("allow_backlog", True))

    m = LpProblem("PP_MinTotalCost_MILP", LpMinimize)

    x = {(p,t): LpVariable(f"x_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}  # regular
    y = {(p,t): LpVariable(f"y_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}  # overtime
    z = {(p,t): LpVariable(f"z_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}  # inventory
    v = {(p,t): LpVariable(f"v_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}  # shortage
    u = {(p,t): LpVariable(f"u_{p}_{t}", lowBound=0, upBound=1, cat=LpBinary) for p in products for t in periods}  # setup

    # Objective: total cost (regular + OT + material + holding + setup + shortage) - revenue
    obj_terms = []
    for p in products:
        for t in periods:
            cx = _get_pt(regular_cost, p, t, 0.0)
            cy = _get_pt(overtime_cost, p, t, 0.0)
            cm = _get_pt(material_cost, p, t, 0.0)
            ch = _get_pt(holding_cost, p, t, 0.0)
            cs = _get_pt(setup_cost, p, t, 0.0)
            cg = _get_pt(shortage_cost, p, t, 0.0)
            rev= _get_pt(revenue, p, t, 0.0)
            obj_terms += [
                cx * x[(p,t)],
                cy * y[(p,t)],
                cm * (x[(p,t)] + y[(p,t)]),
                ch * z[(p,t)],
                cs * u[(p,t)],
                cg * v[(p,t)],
                - rev * (x[(p,t)] + y[(p,t)])
            ]
    m += lpSum(obj_terms)

    # Inventory balance
    for p in products:
        for idx, t in enumerate(periods):
            d_pt = _get_pt(demand, p, t, 0.0)
            if idx == 0:
                I0_p = float(I0.get(p, 0.0))
                m += z[(p,t)] == I0_p + x[(p,t)] + y[(p,t)] - d_pt - v[(p,t)], f"inv_{p}_{t}"
            else:
                t_prev = periods[idx-1]
                m += z[(p,t)] == z[(p,t_prev)] + x[(p,t)] + y[(p,t)] - d_pt - v[(p,t)], f"inv_{p}_{t}"

    # Budgets (optional)
    for t in periods:
        if t in labor_budget:
            m += lpSum(_get_pt(labor_hours, p, t, 0.0) * (x[(p,t)] + y[(p,t)]) for p in products) <= float(labor_budget[t]), f"labor_{t}"
        if t in material_budget:
            m += lpSum(_get_pt(material_cost, p, t, 0.0) * (x[(p,t)] + y[(p,t)]) for p in products) <= float(material_budget[t]), f"mat_{t}"

    # Setup linking + min/max per period
    for p in products:
        for t in periods:
            Mpt = _bigM_prod(pp, p, t)
            m += x[(p,t)] + y[(p,t)] <= Mpt * u[(p,t)], f"link_{p}_{t}"
            if p in min_batch and min_batch[p] > 0:
                m += x[(p,t)] + y[(p,t)] >= float(min_batch[p]) * u[(p,t)], f"min_{p}_{t}"
            if p in max_batches and max_batches[p] > 0:
                m += x[(p,t)] + y[(p,t)] <= float(max_batches[p]), f"max_{p}_{t}"
            if not allow_backlog:
                m += v[(p,t)] == 0, f"nobacklog_{p}_{t}"

    m.solve()

    status = LpStatus[m.status]
    sol_total = {p: {t: int(round((value(x[(p,t)]) or 0) + (value(y[(p,t)]) or 0))) for t in periods} for p in products}
    sol_reg   = {p: {t: int(round(value(x[(p,t)]) or 0)) for t in periods} for p in products}
    sol_ot    = {p: {t: int(round(value(y[(p,t)]) or 0)) for t in periods} for p in products}
    sol_inv   = {p: {t: int(round(value(z[(p,t)]) or 0)) for t in periods} for p in products}
    sol_short = {p: {t: int(round(value(v[(p,t)]) or 0)) for t in periods} for p in products}
    sol_setup = {p: {t: int(round(value(u[(p,t)]) or 0)) for t in periods} for p in products}

    obj_val = round(value(m.objective) or 0, 4)

    summary = (
        "Solver: PuLP (MILP, minimize total cost)\n"
        f"Status: {status}\n"
        f"Objective (total cost − revenue): {obj_val}\n"
        f"Totals (x+y): {sol_total}\n"
        f"Regular (x): {sol_reg}\n"
        f"Overtime (y): {sol_ot}\n"
        f"Inventory (z): {sol_inv}\n"
        f"Shortage (v): {sol_short}\n"
        f"Setup (u): {sol_setup}"
    )
    meta = {
        "solution_total": sol_total,
        "solution_regular": sol_reg,
        "solution_overtime": sol_ot,
        "inventory": sol_inv,
        "shortage": sol_short,
        "setup": sol_setup,
        "objective": obj_val,
        "status": status
    }
    return summary, meta

# ---------- Dispatcher ----------

def optimize(pp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Auto-selects the appropriate model:
      - If cost-model keys exist → MILP with inventory/OT/shortage/budgets.
      - Else → profit−setup ILP (your original).
    """
    cost_keys = {
        "regular_cost","overtime_cost","material_cost","holding_cost","shortage_cost",
        "labor_hours","labor_budget","material_budget","initial_inventory","revenue","setup_link_bigM"
    }
    if any(k in pp for k in cost_keys):
        return _optimize_cost_milp(pp)
    return _optimize_profit_ilp(pp)
