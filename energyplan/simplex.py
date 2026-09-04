"""A two-phase simplex solver written from scratch.

The solver takes an :class:`~energyplan.lp.LpProblem` and returns primal
values, the optimal objective and dual values (shadow prices) for every
original row.  It needs nothing but the standard library; if NumPy happens to
be importable the dense pivot is vectorised, which is roughly two orders of
magnitude faster on medium problems.

Algorithm
---------
1. **Standardise.**  Every variable is shifted/split so that it is
   non-negative, finite upper bounds become explicit rows, and every row
   becomes an equality with a slack of coefficient ``+1`` (``<=``) or ``-1``
   (``>=``).  Rows with a negative right-hand side are negated.
2. **Phase 1.**  Artificial variables are added to rows that do not already
   own a unit basis column, and ``sum(artificials)`` is minimised.  A positive
   optimum means the problem is infeasible.
3. **Phase 2.**  The real cost row is restored and the simplex continues with
   artificial columns locked out of the basis.

Pricing uses Dantzig's rule with a most-negative-reduced-cost choice and a
largest-pivot tie-break in the ratio test.  If progress stalls (degeneracy)
the solver switches to Bland's rule, which guarantees termination.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .lp import EQ, GE, INF, LE, LpProblem, Solution

try:  # pragma: no cover - exercised implicitly by whichever branch runs
    import numpy as _np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False

__all__ = ["solve_simplex", "SimplexError", "StandardForm"]

# Numerical tolerances.  ``PIVOT_TOL`` guards the ratio test, ``COST_TOL`` the
# optimality test and ``FEAS_TOL`` the phase-1 verdict.
PIVOT_TOL = 1e-9
COST_TOL = 1e-9
FEAS_TOL = 1e-7
ZERO_TOL = 1e-11


class SimplexError(RuntimeError):
    """Raised when the solver cannot make progress for a structural reason."""


# ---------------------------------------------------------------------------
# Standard form
# ---------------------------------------------------------------------------


class StandardForm:
    """``min c'z`` s.t. ``Az = b``, ``z >= 0`` built from an ``LpProblem``.

    Attributes
    ----------
    rows / rhs / cost
        The constraint matrix (list of dicts), right-hand side and cost row.
    recover
        For each original variable, ``(kind, i, j, shift)`` describing how to
        rebuild its value from ``z``.
    row_source
        For each standard row, the index of the originating problem row or
        ``-1`` for rows introduced by a finite upper bound.
    row_sign
        ``+1`` or ``-1``: whether the originating row was negated.
    """

    def __init__(self, problem: LpProblem):
        self.problem = problem
        self.rows: List[dict] = []
        self.rhs: List[float] = []
        self.cost: List[float] = []
        self.recover: List[Tuple[str, int, int, float]] = []
        self.row_source: List[int] = []
        self.row_sign: List[float] = []
        self.slack_col: List[int] = []
        self.slack_sign: List[float] = []
        self.artificial_col: List[int] = []
        self.n_structural = 0
        self._build()

    # -- helpers -------------------------------------------------------------
    def _new_column(self, cost: float) -> int:
        self.cost.append(cost)
        return len(self.cost) - 1

    def _build(self) -> None:
        problem = self.problem
        obj = problem.objective
        direction = 1.0 if problem.sense == "min" else -1.0

        # Map every original variable onto one or two non-negative columns.
        # ``sub[j]`` holds the substitution needed to rewrite a row coefficient.
        sub: List[Tuple[str, int, int, float]] = []
        for var in problem.variables:
            lb, ub = var.lb, var.ub
            obj_c = direction * obj.coeffs.get(var.index, 0.0)
            if lb == -INF and ub == INF:
                pos = self._new_column(obj_c)
                neg = self._new_column(-obj_c)
                sub.append(("free", pos, neg, 0.0))
            elif lb == -INF:
                # x = ub - z, z >= 0
                col = self._new_column(-obj_c)
                sub.append(("flip", col, -1, ub))
            else:
                # x = lb + z, 0 <= z <= ub - lb
                col = self._new_column(obj_c)
                sub.append(("shift", col, -1, lb))
        self.recover = sub
        self.n_structural = len(self.cost)

        # Objective constant is tracked separately and added back at the end.
        self.obj_offset = direction * obj.const + sum(
            direction * obj.coeffs.get(v.index, 0.0) * s[3]
            for v, s in zip(problem.variables, sub)
            if s[0] in ("shift", "flip")
        )

        # Original rows.
        for con in problem.constraints:
            coeffs, offset = self._translate(con.expr.coeffs, sub)
            self._append_row(coeffs, con.sense, con.rhs - offset, con.index)

        # Finite upper bounds become explicit rows (``shift`` and ``flip`` only;
        # the split ``free`` case has no finite bound by construction).
        for var, s in zip(problem.variables, sub):
            kind, col, _, shift = s
            if kind == "shift" and var.ub != INF:
                self._append_row({col: 1.0}, LE, var.ub - var.lb, -1)
            elif kind == "flip" and var.lb == -INF and var.ub != INF:
                pass  # z >= 0 already encodes x <= ub

    def _translate(self, coeffs: dict, sub) -> Tuple[dict, float]:
        """Rewrite original-variable coefficients in terms of the ``z`` columns."""
        out: dict = {}
        offset = 0.0
        for idx, coef in coeffs.items():
            kind, a, b, shift = sub[idx]
            if kind == "free":
                out[a] = out.get(a, 0.0) + coef
                out[b] = out.get(b, 0.0) - coef
            elif kind == "shift":
                out[a] = out.get(a, 0.0) + coef
                offset += coef * shift
            else:  # flip: x = ub - z
                out[a] = out.get(a, 0.0) - coef
                offset += coef * shift
        return {k: v for k, v in out.items() if abs(v) > ZERO_TOL}, offset

    def _append_row(self, coeffs: dict, sense: str, rhs: float, source: int) -> None:
        row = dict(coeffs)
        value = float(rhs)
        sign = 1.0
        if value < 0.0:
            row = {k: -v for k, v in row.items()}
            value = -value
            sign = -1.0
            sense = {LE: GE, GE: LE, EQ: EQ}[sense]
        self.rows.append(row)
        self.rhs.append(value)
        self.row_source.append(source)
        self.row_sign.append(sign)
        self.slack_col.append(-1)
        self.slack_sign.append(0.0)
        self.artificial_col.append(-1)
        self._pending_sense = getattr(self, "_pending_sense", [])
        self._pending_sense.append(sense)

    def finalise(self) -> Tuple[List[int], int]:
        """Add slack and artificial columns.  Returns ``(basis, n_before_art)``."""
        senses = self._pending_sense
        # Slacks first so that their columns stay contiguous.
        for i, sense in enumerate(senses):
            if sense == LE:
                col = self._new_column(0.0)
                self.rows[i][col] = 1.0
                self.slack_col[i] = col
                self.slack_sign[i] = 1.0
            elif sense == GE:
                col = self._new_column(0.0)
                self.rows[i][col] = -1.0
                self.slack_col[i] = col
                self.slack_sign[i] = -1.0
        n_before_art = len(self.cost)

        basis: List[int] = []
        for i, sense in enumerate(senses):
            if sense == LE:
                basis.append(self.slack_col[i])
            else:
                col = self._new_column(0.0)
                self.rows[i][col] = 1.0
                self.artificial_col[i] = col
                basis.append(col)
        return basis, n_before_art


# ---------------------------------------------------------------------------
# Tableau operations
# ---------------------------------------------------------------------------


def _dense_tableau(std: StandardForm, ncols: int):
    m = len(std.rows)
    if _HAS_NUMPY:
        table = _np.zeros((m + 1, ncols + 1), dtype=_np.float64)
        for i, row in enumerate(std.rows):
            if row:
                idx = list(row.keys())
                table[i, idx] = list(row.values())
            table[i, ncols] = std.rhs[i]
        return table
    table = [[0.0] * (ncols + 1) for _ in range(m + 1)]
    for i, row in enumerate(std.rows):
        target = table[i]
        for col, val in row.items():
            target[col] = val
        target[ncols] = std.rhs[i]
    return table


def _set_cost_row(table, cost: Sequence[float], basis: Sequence[int], ncols: int) -> None:
    """Install ``cost`` as the reduced-cost row and price out the basis."""
    obj = table[-1]
    if _HAS_NUMPY:
        obj[:ncols] = cost
        obj[ncols] = 0.0
        for i, bvar in enumerate(basis):
            factor = obj[bvar]
            if factor:
                obj -= factor * table[i]
    else:
        for j in range(ncols):
            obj[j] = cost[j]
        obj[ncols] = 0.0
        for i, bvar in enumerate(basis):
            factor = obj[bvar]
            if not factor:
                continue
            row = table[i]
            for j in range(ncols + 1):
                v = row[j]
                if v:
                    obj[j] -= factor * v


def _pivot(table, row_index: int, col: int, ncols: int) -> None:
    pivot_row = table[row_index]
    pivot_value = pivot_row[col]
    if _HAS_NUMPY:
        pivot_row /= pivot_value
        column = table[:, col].copy()
        column[row_index] = 0.0
        nz = _np.nonzero(column)[0]
        if nz.size:
            table[nz] -= _np.outer(column[nz], pivot_row)
        table[:, col] = 0.0
        table[row_index, col] = 1.0
    else:
        inv = 1.0 / pivot_value
        for j in range(ncols + 1):
            if pivot_row[j]:
                pivot_row[j] *= inv
        pivot_row[col] = 1.0
        for i, row in enumerate(table):
            if i == row_index:
                continue
            factor = row[col]
            if not factor:
                continue
            for j in range(ncols + 1):
                v = pivot_row[j]
                if v:
                    row[j] -= factor * v
            row[col] = 0.0


def _choose_entering(table, ncols: int, locked, locked_np, bland: bool) -> int:
    """Index of the entering column, or ``-1`` when no reduced cost is negative."""
    obj = table[-1]
    if bland:
        if _HAS_NUMPY:
            negative = _np.nonzero((obj[:ncols] < -COST_TOL) & ~locked_np)[0]
            return int(negative[0]) if negative.size else -1
        for j in range(ncols):
            if obj[j] < -COST_TOL and not locked[j]:
                return j
        return -1
    if _HAS_NUMPY:
        costs = _np.where(locked_np, 0.0, obj[:ncols])
        j = int(_np.argmin(costs))
        return j if costs[j] < -COST_TOL else -1
    best, best_value = -1, -COST_TOL
    for j in range(ncols):
        value = obj[j]
        if value < best_value and not locked[j]:
            best_value, best = value, j
    return best


def _ratio_test(table, m: int, col: int, ncols: int, basis, basis_np, bland: bool) -> int:
    """Return the leaving row, or ``-1`` when the problem is unbounded."""
    if _HAS_NUMPY:
        column = table[:m, col]
        eligible = column > PIVOT_TOL
        if not eligible.any():
            return -1
        ratios = _np.where(eligible, table[:m, ncols] / _np.where(eligible, column, 1.0), _np.inf)
        best_ratio = float(ratios.min())
        tied = _np.nonzero(ratios <= best_ratio + 1e-12)[0]
        if tied.size == 1:
            return int(tied[0])
        if bland:
            # Smallest basic-variable index among the ties keeps Bland's rule valid.
            return int(tied[_np.argmin(basis_np[tied])])
        # Otherwise prefer the largest pivot element for numerical stability.
        return int(tied[_np.argmax(column[tied])])

    best_row, best_ratio, best_pivot = -1, INF, 0.0
    for i in range(m):
        row = table[i]
        a = row[col]
        if a <= PIVOT_TOL:
            continue
        ratio = row[ncols] / a
        if ratio < best_ratio - 1e-12:
            best_row, best_ratio, best_pivot = i, ratio, a
        elif ratio <= best_ratio + 1e-12:
            if bland:
                if best_row < 0 or basis[i] < basis[best_row]:
                    best_row, best_ratio, best_pivot = i, min(ratio, best_ratio), a
            elif a > best_pivot:
                best_row, best_ratio, best_pivot = i, min(ratio, best_ratio), a
    return best_row


def _iterate(table, m: int, ncols: int, basis: List[int], locked, max_iter: int) -> Tuple[str, int]:
    """Run simplex iterations until optimal, unbounded or out of iterations."""
    locked_np = _np.array(locked, dtype=bool) if _HAS_NUMPY else None
    basis_np = _np.array(basis, dtype=_np.int64) if _HAS_NUMPY else None
    iterations = 0
    bland = False
    stall = 0
    last_obj = INF
    while iterations < max_iter:
        col = _choose_entering(table, ncols, locked, locked_np, bland)
        if col < 0:
            return "optimal", iterations
        row = _ratio_test(table, m, col, ncols, basis, basis_np, bland)
        if row < 0:
            return "unbounded", iterations
        _pivot(table, row, col, ncols)
        basis[row] = col
        if _HAS_NUMPY:
            basis_np[row] = col
        iterations += 1

        current = float(table[-1][ncols])
        if abs(current - last_obj) <= 1e-12 * max(1.0, abs(current)):
            stall += 1
            if stall > 40:
                bland = True          # degenerate: fall back to a rule that cannot cycle
        else:
            stall = 0
            bland = False             # progress resumed, go back to Dantzig pricing
        last_obj = current
    return "iteration_limit", iterations


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def solve_simplex(
    problem: LpProblem,
    max_iter: Optional[int] = None,
    max_cells: int = 60_000_000,
    **_,
) -> Solution:
    """Solve ``problem`` with the built-in two-phase simplex.

    ``max_cells`` caps the size of the dense tableau (default 60 million
    entries, about 480 MB in float64).  Beyond that a dense method is the wrong
    tool, and :class:`SimplexError` is raised with a pointer to HiGHS.
    """
    std = StandardForm(problem)
    basis, n_before_art = std.finalise()
    ncols = len(std.cost)
    m = len(std.rows)

    if m == 0:
        return _solve_unconstrained(problem, std)

    if max_iter is None:
        max_iter = max(5000, 40 * (m + ncols))

    _check_size(m, ncols, max_cells)
    table = _dense_tableau(std, ncols)

    # ---- Phase 1 ----------------------------------------------------------
    has_artificial = any(col >= 0 for col in std.artificial_col)
    total_iterations = 0
    if has_artificial:
        phase1_cost = [0.0] * ncols
        for col in std.artificial_col:
            if col >= 0:
                phase1_cost[col] = 1.0
        _set_cost_row(table, phase1_cost, basis, ncols)
        locked = [False] * ncols
        status, iters = _iterate(table, m, ncols, basis, locked, max_iter)
        total_iterations += iters
        if status == "iteration_limit":
            return Solution("iteration_limit", float("nan"), [0.0] * problem.num_vars,
                            message="phase 1 hit the iteration limit",
                            iterations=total_iterations, solver="simplex")
        infeasibility = -table[-1][ncols]
        if infeasibility > FEAS_TOL * max(1.0, max(std.rhs) if std.rhs else 1.0):
            return Solution("infeasible", float("inf"), [0.0] * problem.num_vars,
                            message=f"phase 1 residual {infeasibility:.3e}",
                            iterations=total_iterations, solver="simplex")
        _drive_out_artificials(table, m, ncols, basis, n_before_art)

    # ---- Phase 2 ----------------------------------------------------------
    _set_cost_row(table, std.cost, basis, ncols)
    locked = [j >= n_before_art for j in range(ncols)]
    status, iters = _iterate(table, m, ncols, basis, locked, max_iter)
    total_iterations += iters
    if status == "unbounded":
        return Solution("unbounded", -INF, [0.0] * problem.num_vars,
                        message="objective is unbounded below",
                        iterations=total_iterations, solver="simplex")
    if status == "iteration_limit":
        return Solution("iteration_limit", float("nan"), [0.0] * problem.num_vars,
                        message="phase 2 hit the iteration limit",
                        iterations=total_iterations, solver="simplex")

    z = [0.0] * ncols
    for i, col in enumerate(basis):
        value = float(table[i][ncols])
        z[col] = value if abs(value) > ZERO_TOL else 0.0

    x = _recover_primal(problem, std, z)
    duals = _recover_duals(problem, std, table, ncols)

    objective = float(problem.objective.value(x))
    return Solution("optimal", objective, x, duals,
                    message="", iterations=total_iterations, solver="simplex")


def _check_size(rows: int, cols: int, max_cells: int) -> None:
    """Refuse problems too large for a dense tableau, with a useful message.

    The tableau is stored densely, which is fine up to a few thousand rows and
    columns but grows as their product.  Failing loudly beats exhausting memory
    an hour into a run.
    """
    cells = (rows + 1) * (cols + 1)
    if cells <= max_cells:
        return
    gigabytes = cells * 8 / 1024 ** 3
    raise SimplexError(
        f"this model needs a dense tableau of {rows + 1:,} x {cols + 1:,} "
        f"({gigabytes:,.1f} GB), beyond the built-in solver's limit.\n"
        "The built-in simplex is meant for small and medium models. For one "
        "this size, install a sparse solver:\n"
        "    pip install scipy      # HiGHS, picked up automatically\n"
        "or shrink the problem with fewer representative days or hours, "
        "for example --days 4 --hours 6."
    )


def _solve_unconstrained(problem: LpProblem, std: StandardForm) -> Solution:
    """Handle a model with no rows: each variable goes to its best bound."""
    x = []
    direction = 1.0 if problem.sense == "min" else -1.0
    for var in problem.variables:
        c = direction * problem.objective.coeffs.get(var.index, 0.0)
        if c > 0:
            value = var.lb
        elif c < 0:
            value = var.ub
        else:
            value = var.lb if var.lb != -INF else (var.ub if var.ub != INF else 0.0)
        if value in (INF, -INF):
            return Solution("unbounded", -INF, [0.0] * problem.num_vars,
                            message="unbounded variable with no constraints",
                            solver="simplex")
        x.append(value)
    return Solution("optimal", problem.objective.value(x), x, [], solver="simplex")


def _drive_out_artificials(table, m: int, ncols: int, basis: List[int], n_before_art: int) -> None:
    """Pivot artificial variables out of the basis, dropping redundant rows."""
    for i in range(m):
        if basis[i] < n_before_art:
            continue
        row = table[i]
        replacement = -1
        for j in range(n_before_art):
            if abs(row[j]) > PIVOT_TOL:
                replacement = j
                break
        if replacement >= 0:
            _pivot(table, i, replacement, ncols)
            basis[i] = replacement
        # Otherwise the row is redundant: it stays basic at value zero, which
        # is harmless because artificial columns are locked out of phase 2.


def _recover_primal(problem: LpProblem, std: StandardForm, z: Sequence[float]) -> List[float]:
    x = []
    for var, (kind, a, b, shift) in zip(problem.variables, std.recover):
        if kind == "free":
            value = z[a] - z[b]
        elif kind == "shift":
            value = shift + z[a]
        else:  # flip
            value = shift - z[a]
        # Clip tiny numerical excursions back inside the declared bounds.
        if var.lb != -INF and value < var.lb:
            value = var.lb
        if var.ub != INF and value > var.ub:
            value = var.ub
        x.append(float(value))
    return x


def _recover_duals(problem: LpProblem, std: StandardForm, table, ncols: int) -> List[float]:
    """Shadow prices ``dObjective / dRHS`` for each original constraint."""
    obj = table[-1]
    duals = [0.0] * problem.num_constraints
    direction = 1.0 if problem.sense == "min" else -1.0
    for i, source in enumerate(std.row_source):
        if source < 0:
            continue  # bound row, not an original constraint
        slack = std.slack_col[i]
        if slack >= 0:
            # reduced cost of the slack is -y * slack_sign
            y = -obj[slack] / std.slack_sign[i]
        else:
            art = std.artificial_col[i]
            y = -obj[art] if art >= 0 else 0.0
        duals[source] = float(direction * std.row_sign[i] * y)
    return duals
