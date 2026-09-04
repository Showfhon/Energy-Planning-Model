"""Solver backends.

The built-in simplex (:mod:`energyplan.simplex`) is always available and needs
no third-party package.  When SciPy (HiGHS) or PuLP (CBC) happen to be
installed they are used instead for large models, because they are one to two
orders of magnitude faster.  All backends return the same :class:`Solution`,
including dual values, so results are interchangeable and can be cross-checked
against each other -- see ``tests/test_backends.py``.
"""

from __future__ import annotations

import time
from typing import List

from .lp import EQ, GE, INF, LE, LpProblem, Solution

__all__ = ["solve_problem", "available_solvers", "SolverNotAvailable"]


class SolverNotAvailable(RuntimeError):
    pass


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def available_solvers() -> List[str]:
    """Backends usable in this interpreter, fastest first."""
    out = []
    if _has("scipy.optimize"):
        out.append("highs")
    if _has("pulp"):
        out.append("cbc")
    out.append("simplex")
    return out


def solve_problem(problem: LpProblem, solver: str = "auto", verbose: bool = False, **options) -> Solution:
    """Solve ``problem`` with the requested (or best available) backend."""
    choices = available_solvers()
    if solver == "auto":
        chosen = choices[0]
    else:
        if solver not in ("highs", "cbc", "simplex"):
            raise SolverNotAvailable(f"unknown solver {solver!r}")
        if solver not in choices:
            raise SolverNotAvailable(
                f"solver {solver!r} is not installed; available: {', '.join(choices)}"
            )
        chosen = solver

    started = time.perf_counter()
    if chosen == "highs":
        solution = _solve_highs(problem, verbose=verbose, **options)
    elif chosen == "cbc":
        solution = _solve_cbc(problem, verbose=verbose, **options)
    else:
        from .simplex import solve_simplex

        solution = solve_simplex(problem, **options)
    solution.solver = chosen
    if verbose:
        elapsed = time.perf_counter() - started
        print(
            f"[solver] {chosen}: {solution.status} "
            f"objective={solution.objective:,.4f} in {elapsed:.2f}s"
        )
    return solution


# ---------------------------------------------------------------------------
# SciPy / HiGHS
# ---------------------------------------------------------------------------


def _solve_highs(problem: LpProblem, verbose: bool = False, **options) -> Solution:
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    n = problem.num_vars
    direction = 1.0 if problem.sense == "min" else -1.0
    c = np.zeros(n)
    for idx, coef in problem.objective.coeffs.items():
        c[idx] = direction * coef

    ub_data: List[float] = []
    ub_rows: List[int] = []
    ub_cols: List[int] = []
    ub_rhs: List[float] = []
    eq_data: List[float] = []
    eq_rows: List[int] = []
    eq_cols: List[int] = []
    eq_rhs: List[float] = []
    # Map each original row to (block, position) so duals can be reassembled.
    row_map: List[tuple] = []

    for con in problem.constraints:
        if con.sense == EQ:
            r = len(eq_rhs)
            for idx, coef in con.expr.coeffs.items():
                eq_rows.append(r)
                eq_cols.append(idx)
                eq_data.append(coef)
            eq_rhs.append(con.rhs)
            row_map.append(("eq", r, 1.0))
        else:
            r = len(ub_rhs)
            sign = 1.0 if con.sense == LE else -1.0
            for idx, coef in con.expr.coeffs.items():
                ub_rows.append(r)
                ub_cols.append(idx)
                ub_data.append(sign * coef)
            ub_rhs.append(sign * con.rhs)
            row_map.append(("ub", r, sign))

    a_ub = csr_matrix((ub_data, (ub_rows, ub_cols)), shape=(len(ub_rhs), n)) if ub_rhs else None
    a_eq = csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(len(eq_rhs), n)) if eq_rhs else None
    bounds = [
        (None if v.lb == -INF else v.lb, None if v.ub == INF else v.ub)
        for v in problem.variables
    ]

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.array(ub_rhs) if ub_rhs else None,
        A_eq=a_eq,
        b_eq=np.array(eq_rhs) if eq_rhs else None,
        bounds=bounds,
        method="highs",
        options={"disp": bool(verbose)},
    )

    status = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded"}.get(
        result.status, "error"
    )
    if status != "optimal":
        return Solution(status, float("nan"), [0.0] * n, message=result.message)

    x = [float(v) for v in result.x]
    duals = [0.0] * problem.num_constraints
    marginals = getattr(result, "ineqlin", None), getattr(result, "eqlin", None)
    ineq_m = marginals[0].marginals if marginals[0] is not None else []
    eq_m = marginals[1].marginals if marginals[1] is not None else []
    for i, (block, pos, sign) in enumerate(row_map):
        source = ineq_m if block == "ub" else eq_m
        if pos < len(source):
            # SciPy reports dObjective/dRHS for the internal minimisation.
            duals[i] = float(direction * sign * source[pos])

    return Solution("optimal", float(problem.objective.value(x)), x, duals,
                    iterations=int(getattr(result, "nit", 0) or 0))


# ---------------------------------------------------------------------------
# PuLP / CBC
# ---------------------------------------------------------------------------


def _solve_cbc(problem: LpProblem, verbose: bool = False, **options) -> Solution:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _solve_cbc_inner(problem, verbose=verbose, **options)


def _solve_cbc_inner(problem: LpProblem, verbose: bool = False, **options) -> Solution:
    import pulp

    model = pulp.LpProblem(
        problem.name, pulp.LpMinimize if problem.sense == "min" else pulp.LpMaximize
    )
    pvars = [
        pulp.LpVariable(
            f"v{v.index}",
            lowBound=None if v.lb == -INF else v.lb,
            upBound=None if v.ub == INF else v.ub,
        )
        for v in problem.variables
    ]
    model += pulp.lpSum(coef * pvars[i] for i, coef in problem.objective.coeffs.items()) + problem.objective.const

    handles = []
    for con in problem.constraints:
        expr = pulp.lpSum(coef * pvars[i] for i, coef in con.expr.coeffs.items())
        if con.sense == LE:
            row = expr <= con.rhs
        elif con.sense == GE:
            row = expr >= con.rhs
        else:
            row = expr == con.rhs
        name = f"r{con.index}"
        model += (row, name)
        handles.append(name)

    model.solve(pulp.PULP_CBC_CMD(msg=1 if verbose else 0))
    status = {
        pulp.LpStatusOptimal: "optimal",
        pulp.LpStatusInfeasible: "infeasible",
        pulp.LpStatusUnbounded: "unbounded",
        pulp.LpStatusNotSolved: "error",
        pulp.LpStatusUndefined: "error",
    }.get(model.status, "error")
    if status != "optimal":
        return Solution(status, float("nan"), [0.0] * problem.num_vars)

    x = [float(v.value() or 0.0) for v in pvars]
    duals = [0.0] * problem.num_constraints
    for i, name in enumerate(handles):
        row = model.constraints.get(name)
        if row is not None and row.pi is not None:
            duals[i] = float(row.pi)
    return Solution("optimal", float(problem.objective.value(x)), x, duals)
