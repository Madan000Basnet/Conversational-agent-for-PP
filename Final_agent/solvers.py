#solvers.py

from typing import Dict, Any, Tuple, List, Optional
from pulp import (
    LpProblem, LpVariable, LpMaximize, LpMinimize, LpStatus,
    LpInteger, LpBinary, lpSum, value
)

# =========================
# Utilities
# =========================

def _get_pt(mm: Dict[str, Any], p: str, t: str, default: float = 0.0) -> float:
    """Safe getter for mm[p][t] or permissive fallbacks."""
    if not isinstance(mm, dict):
        return default
    v = mm.get(p)
    if isinstance(v, dict):
        try:
            return float(v.get(t, default))
        except Exception:
            return default
    if isinstance(v, (int, float, str)):
        try:
            return float(v)
        except Exception:
            return default
    if t in mm and isinstance(mm[t], (int, float, str)):
        try:
            return float(mm[t])
        except Exception:
            return default
    return default


def _bigM_prod(pp: Dict[str, Any], p: str, t: str) -> float:
    """Big-M for setup linking, with safe upper bounds from budgets if available."""
    # explicit override
    override = _get_pt(pp.get("setup_link_bigM") or {}, p, t, default=None)
    if override is not None and override != 0:
        return float(override)

    candidates: List[float] = []

    # labor-budget-derived bound
    Lt = (pp.get("labor_budget") or {}).get(t)
    hpt = _get_pt(pp.get("labor_hours") or {}, p, t, 0.0)
    if Lt is not None and hpt > 0:
        candidates.append(float(Lt) / float(hpt))

    # material-budget-derived bound
    Mb = (pp.get("material_budget") or {}).get(t)
    mpt = _get_pt(pp.get("material_cost") or {}, p, t, 0.0)
    if Mb is not None and mpt > 0:
        candidates.append(float(Mb) / float(mpt))

    # per-product max if present
    mx = (pp.get("max_batches") or {}).get(p)
    if mx:
        candidates.append(float(mx))

    return max(1.0, min(candidates) if candidates else 1e6)


# =========================
# Shared variable builders
# =========================

def _make_vars_single(products: List[str], periods: List[str]):
    x = {(p, t): LpVariable(f"x_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}
    y = {(p, t): LpVariable(f"y_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}
    z = {(p, t): LpVariable(f"z_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}
    v = {(p, t): LpVariable(f"v_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}
    u = {(p, t): LpVariable(f"u_{p}_{t}", lowBound=0, upBound=1, cat=LpBinary) for p in products for t in periods}
    return x, y, z, v, u


def _make_vars_multi(products: List[str], periods: List[str], plants: List[str]):
    x = {(p, t, s): LpVariable(f"x_{p}_{t}_{s}", lowBound=0, cat=LpInteger) for p in products for t in periods for s in plants}
    y = {(p, t, s): LpVariable(f"y_{p}_{t}_{s}", lowBound=0, cat=LpInteger) for p in products for t in periods for s in plants}
    u = {(p, t, s): LpVariable(f"u_{p}_{t}_{s}", lowBound=0, upBound=1, cat=LpBinary) for p in products for t in periods for s in plants}
    z = {(p, t): LpVariable(f"z_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}
    v = {(p, t): LpVariable(f"v_{p}_{t}", lowBound=0, cat=LpInteger) for p in products for t in periods}
    return x, y, z, v, u


# =========================
# Shared constraints
# =========================

def _add_inventory_balance(m: LpProblem, products, periods, x, y, z, v, I0, demand, plants=None):
    def prod_sum(p, t):
        if plants:
            return lpSum(x[(p, t, s)] + y[(p, t, s)] for s in plants)
        return x[(p, t)] + y[(p, t)]

    for p in products:
        for i, t in enumerate(periods):
            d = _get_pt(demand, p, t, 0.0)
            if i == 0:
                m += z[(p, t)] == float(I0.get(p, 0.0)) + prod_sum(p, t) - d - v[(p, t)]
            else:
                tprev = periods[i - 1]
                m += z[(p, t)] == z[(p, tprev)] + prod_sum(p, t) - d - v[(p, t)]


def _add_budgets(m: LpProblem, products, periods, x, y,
                 labor_hours, labor_budget, material_cost, material_budget,
                 plants=None, labor_budget_plant=None):
    if plants:
        # plant labor budgets
        for s in plants:
            for t in periods:
                cap = float(((labor_budget_plant or {}).get(s) or {}).get(t, 0.0))
                if cap > 0:
                    m += lpSum(_get_pt(labor_hours, p, t, 0.0) * (x[(p, t, s)] + y[(p, t, s)]) for p in products) <= cap
        # period material budgets
        for t in periods:
            Mb = float((material_budget or {}).get(t, 0.0))
            if Mb > 0:
                m += lpSum(_get_pt(material_cost, p, t, 0.0) * (x[(p, t, s)] + y[(p, t, s)]) for p in products for s in plants) <= Mb
    else:
        for t in periods:
            Lt = float((labor_budget or {}).get(t, 0.0))
            if Lt > 0:
                m += lpSum(_get_pt(labor_hours, p, t, 0.0) * (x[(p, t)] + y[(p, t)]) for p in products) <= Lt
            Mb = float((material_budget or {}).get(t, 0.0))
            if Mb > 0:
                m += lpSum(_get_pt(material_cost, p, t, 0.0) * (x[(p, t)] + y[(p, t)]) for p in products) <= Mb


def _add_setup_linking(m: LpProblem, pp: Dict[str, Any], products, periods, x, y, u,
                       min_batch, max_batches, allow_backlog, v, plants=None):
    if plants:
        for p in products:
            for t in periods:
                M = _bigM_prod(pp, p, t)
                for s in plants:
                    m += x[(p, t, s)] + y[(p, t, s)] <= M * u[(p, t, s)]
                    if p in min_batch and min_batch[p] > 0:
                        m += x[(p, t, s)] + y[(p, t, s)] >= float(min_batch[p]) * u[(p, t, s)]
                    if p in max_batches and max_batches[p] > 0:
                        m += x[(p, t, s)] + y[(p, t, s)] <= float(max_batches[p])
                if not allow_backlog:
                    m += v[(p, t)] == 0
    else:
        for p in products:
            for t in periods:
                M = _bigM_prod(pp, p, t)
                m += x[(p, t)] + y[(p, t)] <= M * u[(p, t)]
                if p in min_batch and min_batch[p] > 0:
                    m += x[(p, t)] + y[(p, t)] >= float(min_batch[p]) * u[(p, t)]
                if p in max_batches and max_batches[p] > 0:
                    m += x[(p, t)] + y[(p, t)] <= float(max_batches[p])
                if not allow_backlog:
                    m += v[(p, t)] == 0


def _add_sustainability(m: LpProblem, products, periods, x, y,
                        water_per_unit, water_cap, co2_per_unit, co2_cap, plants=None):
    if water_per_unit and water_cap:
        for t in periods:
            if t in water_cap:
                if plants:
                    m += lpSum(_get_pt(water_per_unit, p, t, 0.0) *
                               lpSum(x[(p, t, s)] + y[(p, t, s)] for s in plants)
                               for p in products) <= float(water_cap[t])
                else:
                    m += lpSum(_get_pt(water_per_unit, p, t, 0.0) *
                               (x[(p, t)] + y[(p, t)]) for p in products) <= float(water_cap[t])
    if co2_per_unit and co2_cap:
        for t in periods:
            if t in co2_cap:
                if plants:
                    m += lpSum(_get_pt(co2_per_unit, p, t, 0.0) *
                               lpSum(x[(p, t, s)] + y[(p, t, s)] for s in plants)
                               for p in products) <= float(co2_cap[t])
                else:
                    m += lpSum(_get_pt(co2_per_unit, p, t, 0.0) *
                               (x[(p, t)] + y[(p, t)]) for p in products) <= float(co2_cap[t])


def _add_shelf_life(m: LpProblem, products, periods, x, y, z, shelf_life, plants=None):
    if not shelf_life:
        return
    idx = {t: i for i, t in enumerate(periods)}
    for p in products:
        if p in shelf_life:
            L = max(0, int(shelf_life[p]))
            for t in periods:
                i = idx[t]
                start = max(0, i - L + 1)
                if plants:
                    window = [lpSum(x[(p, periods[j], s)] + y[(p, periods[j], s)] for s in plants)
                              for j in range(start, i + 1)]
                else:
                    window = [x[(p, periods[j])] + y[(p, periods[j])]
                              for j in range(start, i + 1)]
                if window:
                    m += z[(p, t)] <= lpSum(window)
                else:
                    m += z[(p, t)] == 0


# =========================
# Objectives (only difference)
# =========================

def _build_objective_cost(products, periods, plants, x, y, z, v, u, params):
    # minimize costs + penalties − revenue
    get2 = lambda D, p, t, default=0.0: _get_pt(params.get(D, {}) or {}, p, t, default)
    terms = []
    if plants:
        transport_cost = params.get("transport_cost") or {}
        for p in products:
            for t in periods:
                prod = lpSum(x[(p, t, s)] + y[(p, t, s)] for s in plants)
                terms += [
                    get2("holding_cost", p, t, 0.0) * z[(p, t)],
                    get2("shortage_cost", p, t, 0.0) * v[(p, t)],
                    - get2("revenue", p, t, 0.0) * prod,
                ]
                for s in plants:
                    ctrans = float(((transport_cost.get(p) or {}).get(s) or {}).get(t, 0.0))
                    terms += [
                        get2("regular_cost", p, t, 0.0) * x[(p, t, s)],
                        get2("overtime_cost", p, t, 0.0) * y[(p, t, s)],
                        get2("material_cost", p, t, 0.0) * (x[(p, t, s)] + y[(p, t, s)]),
                        get2("setup_cost", p, t, 0.0) * u[(p, t, s)],
                        ctrans * (x[(p, t, s)] + y[(p, t, s)]),
                    ]
    else:
        for p in products:
            for t in periods:
                terms += [
                    get2("regular_cost", p, t, 0.0) * x[(p, t)],
                    get2("overtime_cost", p, t, 0.0) * y[(p, t)],
                    get2("material_cost", p, t, 0.0) * (x[(p, t)] + y[(p, t)]),
                    get2("holding_cost", p, t, 0.0) * z[(p, t)],
                    get2("setup_cost", p, t, 0.0) * u[(p, t)],
                    get2("shortage_cost", p, t, 0.0) * v[(p, t)],
                    - get2("revenue", p, t, 0.0) * (x[(p, t)] + y[(p, t)]),
                ]
    return lpSum(terms)


def _build_objective_profit(products, periods, plants, x, y, z, v, u, params):
    # maximize revenue − costs − penalties
    get2 = lambda D, p, t, default=0.0: _get_pt(params.get(D, {}) or {}, p, t, default)
    terms = []
    if plants:
        transport_cost = params.get("transport_cost") or {}
        for p in products:
            for t in periods:
                prod = lpSum(x[(p, t, s)] + y[(p, t, s)] for s in plants)
                terms += [
                    get2("revenue", p, t, 0.0) * prod,
                    - get2("holding_cost", p, t, 0.0) * z[(p, t)],
                    - get2("shortage_cost", p, t, 0.0) * v[(p, t)],
                ]
                for s in plants:
                    ctrans = float(((transport_cost.get(p) or {}).get(s) or {}).get(t, 0.0))
                    terms += [
                        - get2("regular_cost", p, t, 0.0) * x[(p, t, s)],
                        - get2("overtime_cost", p, t, 0.0) * y[(p, t, s)],
                        - get2("material_cost", p, t, 0.0) * (x[(p, t, s)] + y[(p, t, s)]),
                        - get2("setup_cost", p, t, 0.0) * u[(p, t, s)],
                        - ctrans * (x[(p, t, s)] + y[(p, t, s)]),
                    ]
    else:
        for p in products:
            for t in periods:
                terms += [
                    get2("revenue", p, t, 0.0) * (x[(p, t)] + y[(p, t)]),
                    - get2("regular_cost", p, t, 0.0) * x[(p, t)],
                    - get2("overtime_cost", p, t, 0.0) * y[(p, t)],
                    - get2("material_cost", p, t, 0.0) * (x[(p, t)] + y[(p, t)]),
                    - get2("holding_cost", p, t, 0.0) * z[(p, t)],
                    - get2("setup_cost", p, t, 0.0) * u[(p, t)],
                    - get2("shortage_cost", p, t, 0.0) * v[(p, t)],
                ]
    return lpSum(terms)


# =========================
# Profit maximization MILP
# =========================

def _optimize_profit_milp(pp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    products = pp.get("products", [])
    periods  = pp.get("periods", [])
    plants   = pp.get("plants", [])

    # decisions
    if plants:
        x, y, z, v, u = _make_vars_multi(products, periods, plants)
        m = LpProblem("PP_Profit_MILP_MultiPlant", LpMaximize)
    else:
        x, y, z, v, u = _make_vars_single(products, periods)
        m = LpProblem("PP_Profit_MILP", LpMaximize)

    # constraints (shared)
    _add_inventory_balance(m, products, periods, x, y, z, v,
                           pp.get("initial_inventory") or {},
                           pp.get("demand") or {}, plants if plants else None)

    _add_budgets(m, products, periods, x, y,
                 pp.get("labor_hours") or {},
                 pp.get("labor_budget") or {},
                 pp.get("material_cost") or {},
                 pp.get("material_budget") or {},
                 plants if plants else None,
                 pp.get("labor_budget_plant") or {})

    _add_setup_linking(m, pp, products, periods, x, y, u,
                       pp.get("min_batch") or {},
                       pp.get("max_batches") or {},
                       bool(pp.get("allow_backlog", True)),
                       v, plants if plants else None)

    _add_sustainability(m, products, periods, x, y,
                        pp.get("water_per_unit") or {},
                        pp.get("water_cap") or {},
                        pp.get("co2_per_unit") or {},
                        pp.get("co2_cap") or {},
                        plants if plants else None)

    _add_shelf_life(m, products, periods, x, y, z, pp.get("shelf_life") or {}, plants if plants else None)

    # objective (profit)
    m += _build_objective_profit(products, periods, plants if plants else None, x, y, z, v, u, pp)

    # solve & summarize
    m.solve()
    status = LpStatus[m.status]
    if plants:
        sol_total = {p: {t: {s: int(round((value(x[(p, t, s)]) or 0) + (value(y[(p, t, s)]) or 0))) for s in plants}
                         for t in periods} for p in products}
    else:
        sol_total = {p: {t: int(round((value(x[(p, t)]) or 0) + (value(y[(p, t)]) or 0))) for t in periods}
                     for p in products}
    sol_inv   = {p: {t: int(round(value(z[(p, t)]) or 0)) for t in periods} for p in products}
    sol_short = {p: {t: int(round(value(v[(p, t)]) or 0)) for t in periods} for p in products}
    obj_val = round(value(m.objective) or 0, 4)

    summary = (
        f"Solver: PuLP (Profit MILP{' , multi-plant' if plants else ''})\n"
        f"Status: {status}\n"
        f"Objective (revenue − all costs): {obj_val}\n"
        f"Totals (x+y): {sol_total}\n"
        f"Inventory (z): {sol_inv}\n"
        f"Shortage (v): {sol_short}"
    )
    meta = {"solution_total": sol_total, "inventory": sol_inv, "shortage": sol_short,
            "objective": obj_val, "status": status}
    return summary, meta


# =========================
# Cost minimization MILP
# =========================

def _optimize_cost_milp(pp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    products = pp.get("products", [])
    periods  = pp.get("periods", [])
    plants   = pp.get("plants", [])

    # decisions
    if plants:
        x, y, z, v, u = _make_vars_multi(products, periods, plants)
        m = LpProblem("PP_Cost_MILP_MultiPlant", LpMinimize)
    else:
        x, y, z, v, u = _make_vars_single(products, periods)
        m = LpProblem("PP_Cost_MILP", LpMinimize)

    # constraints (shared)
    _add_inventory_balance(m, products, periods, x, y, z, v,
                           pp.get("initial_inventory") or {},
                           pp.get("demand") or {}, plants if plants else None)

    _add_budgets(m, products, periods, x, y,
                 pp.get("labor_hours") or {},
                 pp.get("labor_budget") or {},
                 pp.get("material_cost") or {},
                 pp.get("material_budget") or {},
                 plants if plants else None,
                 pp.get("labor_budget_plant") or {})

    _add_setup_linking(m, pp, products, periods, x, y, u,
                       pp.get("min_batch") or {},
                       pp.get("max_batches") or {},
                       bool(pp.get("allow_backlog", True)),
                       v, plants if plants else None)

    _add_sustainability(m, products, periods, x, y,
                        pp.get("water_per_unit") or {},
                        pp.get("water_cap") or {},
                        pp.get("co2_per_unit") or {},
                        pp.get("co2_cap") or {},
                        plants if plants else None)

    _add_shelf_life(m, products, periods, x, y, z, pp.get("shelf_life") or {}, plants if plants else None)

    # objective (cost)
    m += _build_objective_cost(products, periods, plants if plants else None, x, y, z, v, u, pp)

    # solve & summarize
    m.solve()
    status = LpStatus[m.status]
    if plants:
        sol_total = {p: {t: {s: int(round((value(x[(p, t, s)]) or 0) + (value(y[(p, t, s)]) or 0))) for s in plants}
                         for t in periods} for p in products}
    else:
        sol_total = {p: {t: int(round((value(x[(p, t)]) or 0) + (value(y[(p, t)]) or 0))) for t in periods}
                     for p in products}
    if not plants:
    # Single-site: give breakdowns
        sol_reg = {p: {t: int(round(value(x[(p, t)]) or 0)) for t in periods} for p in products}
        sol_ot  = {p: {t: int(round(value(y[(p, t)]) or 0)) for t in periods} for p in products}
        sol_setup = {p: {t: int(round(value(u[(p, t)]) or 0)) for t in periods} for p in products}
    else:
    # Multi-plant: not meaningful
        sol_reg, sol_ot, sol_setup = None, None, None

    sol_inv   = {p: {t: int(round(value(z[(p, t)]) or 0)) for t in periods} for p in products}
    sol_short = {p: {t: int(round(value(v[(p, t)]) or 0)) for t in periods} for p in products}
    sol_setup = None if plants else {p: {t: int(round(value(u[(p, t)]) or 0)) for t in periods} for p in products}
    obj_val = round(value(m.objective) or 0, 4)

    summary = (
        f"Solver: PuLP (Cost MILP{' , multi-plant' if plants else ''})\n"
        f"Status: {status}\n"
        f"Objective (total cost − revenue): {obj_val}\n"
        f"Totals (x+y): {sol_total}\n"
        f"Inventory (z): {sol_inv}\n"
        f"Shortage (v): {sol_short}"
        + ("" if plants else f"\nSetup (u): {sol_setup}\nRegular (x): { {p:{t:int(round(value(x[(p,t)]) or 0)) for t in periods} for p in products} }\nOvertime (y): { {p:{t:int(round(value(y[(p,t)]) or 0)) for t in periods} for p in products} }")
    )
    meta = {
        "solution_total": sol_total,
        "inventory": sol_inv,
        "shortage": sol_short,
        "objective": obj_val,
        "status": status
    }
    if not plants:
        meta.update({
             "solution_regular": sol_reg,
             "solution_overtime": sol_ot,
             "setup": sol_setup
        })
    return summary, meta


# =========================
# Dispatcher
# =========================

def optimize(pp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Select: Profit MILP if scope/objective says 'profit'; else Cost MILP.
    """
    active = set(pp.get("_active_modules") or [])
    obj = (pp.get("objective") or "").lower()
    if "profit" in active or obj == "profit":
        return _optimize_profit_milp(pp)
    return _optimize_cost_milp(pp)
