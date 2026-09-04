"""A small linear-programming modelling layer, written from scratch.

This module deliberately depends on nothing outside the Python standard
library.  It provides just enough algebra to express a capacity-expansion
model comfortably::

    p = LpProblem("plan", sense="min")
    x = p.add_var("x", lb=0, ub=10)
    y = p.add_var("y", lb=0)
    p.add(x + 2 * y <= 14, name="resource")
    p.add(3 * x - y >= 0)
    p.set_objective(-x - 2 * y)
    sol = p.solve()

The objects are intentionally lightweight: a ``LinExpr`` is a dictionary from
variable index to coefficient plus a constant term, and a ``Var`` is a
``LinExpr`` with exactly one unit coefficient.  Building a model with a few
hundred thousand non-zeros stays comfortably fast.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

INF = float("inf")

Number = Union[int, float]
Operand = Union["LinExpr", Number]

LE = "<="
GE = ">="
EQ = "=="

__all__ = [
    "INF",
    "LE",
    "GE",
    "EQ",
    "LinExpr",
    "Var",
    "Constraint",
    "LpProblem",
    "Solution",
    "lpsum",
    "lpdot",
]


class LinExpr:
    """An affine expression ``sum(coeff * var) + const``."""

    __slots__ = ("coeffs", "const")

    def __init__(self, coeffs: Optional[Dict[int, float]] = None, const: Number = 0.0):
        self.coeffs: Dict[int, float] = dict(coeffs) if coeffs else {}
        self.const: float = float(const)

    # -- construction helpers ------------------------------------------------
    def copy(self) -> "LinExpr":
        return LinExpr(self.coeffs, self.const)

    def add_term(self, index: int, coeff: Number) -> "LinExpr":
        """Add ``coeff * x_index`` in place.  The fast path for model building."""
        if coeff:
            c = self.coeffs
            new = c.get(index, 0.0) + coeff
            if new:
                c[index] = new
            else:
                c.pop(index, None)
        return self

    def add_expr(self, other: "LinExpr", scale: Number = 1.0) -> "LinExpr":
        """Add ``scale * other`` in place."""
        if scale:
            c = self.coeffs
            for idx, coef in other.coeffs.items():
                new = c.get(idx, 0.0) + coef * scale
                if new:
                    c[idx] = new
                else:
                    c.pop(idx, None)
            self.const += other.const * scale
        return self

    def is_constant(self) -> bool:
        return not self.coeffs

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other: Operand) -> "LinExpr":
        out = self.copy()
        if isinstance(other, LinExpr):
            out.add_expr(other)
        else:
            out.const += float(other)
        return out

    __radd__ = __add__

    def __sub__(self, other: Operand) -> "LinExpr":
        out = self.copy()
        if isinstance(other, LinExpr):
            out.add_expr(other, -1.0)
        else:
            out.const -= float(other)
        return out

    def __rsub__(self, other: Operand) -> "LinExpr":
        out = self.__neg__()
        if isinstance(other, LinExpr):
            out.add_expr(other)
        else:
            out.const += float(other)
        return out

    def __neg__(self) -> "LinExpr":
        return LinExpr({i: -c for i, c in self.coeffs.items()}, -self.const)

    def __mul__(self, other: Number) -> "LinExpr":
        if isinstance(other, LinExpr):
            if other.is_constant():
                other = other.const
            elif self.is_constant():
                return other * self.const
            else:
                raise TypeError("cannot multiply two non-constant expressions")
        factor = float(other)
        if factor == 0.0:
            return LinExpr()
        return LinExpr({i: c * factor for i, c in self.coeffs.items()}, self.const * factor)

    __rmul__ = __mul__

    def __truediv__(self, other: Number) -> "LinExpr":
        return self.__mul__(1.0 / float(other))

    # -- constraint sugar ----------------------------------------------------
    def __le__(self, other: Operand) -> "Constraint":
        return _make_constraint(self, other, LE)

    def __ge__(self, other: Operand) -> "Constraint":
        return _make_constraint(self, other, GE)

    def __eq__(self, other: Operand) -> "Constraint":  # type: ignore[override]
        return _make_constraint(self, other, EQ)

    __hash__ = object.__hash__

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        terms = " + ".join(f"{c:g}*x{i}" for i, c in sorted(self.coeffs.items()))
        if self.const:
            terms = f"{terms} + {self.const:g}" if terms else f"{self.const:g}"
        return f"<LinExpr {terms or '0'}>"

    def value(self, x: Sequence[float]) -> float:
        total = self.const
        for idx, coef in self.coeffs.items():
            total += coef * x[idx]
        return total


class Var(LinExpr):
    """A decision variable.  Behaves as a one-term :class:`LinExpr`."""

    __slots__ = ("index", "name", "lb", "ub")

    def __init__(self, index: int, name: str, lb: float, ub: float):
        LinExpr.__init__(self, {index: 1.0}, 0.0)
        self.index = index
        self.name = name
        self.lb = lb
        self.ub = ub

    __hash__ = object.__hash__

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Var {self.name}>"


def _make_constraint(lhs: LinExpr, rhs: Operand, sense: str) -> "Constraint":
    expr = lhs.copy()
    if isinstance(rhs, LinExpr):
        expr.add_expr(rhs, -1.0)
    else:
        expr.const -= float(rhs)
    rhs_value = -expr.const
    expr.const = 0.0
    return Constraint(expr, sense, rhs_value)


class Constraint:
    """``expr sense rhs`` with ``expr`` carrying no constant term."""

    __slots__ = ("expr", "sense", "rhs", "name", "index")

    def __init__(self, expr: LinExpr, sense: str, rhs: float, name: str = ""):
        if sense not in (LE, GE, EQ):
            raise ValueError(f"unknown sense {sense!r}")
        self.expr = expr
        self.sense = sense
        self.rhs = float(rhs)
        self.name = name
        self.index = -1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Constraint {self.name or self.index}: {self.expr!r} {self.sense} {self.rhs:g}>"


def lpsum(terms: Iterable[Operand]) -> LinExpr:
    """Sum expressions efficiently (avoids the quadratic cost of ``sum``)."""
    out = LinExpr()
    for term in terms:
        if isinstance(term, LinExpr):
            out.add_expr(term)
        else:
            out.const += float(term)
    return out


def lpdot(coeffs: Iterable[Number], terms: Iterable[Operand]) -> LinExpr:
    """Weighted sum ``sum(a_i * e_i)``."""
    out = LinExpr()
    for coeff, term in zip(coeffs, terms):
        if not coeff:
            continue
        if isinstance(term, LinExpr):
            out.add_expr(term, coeff)
        else:
            out.const += float(coeff) * float(term)
    return out


class Solution:
    """Result of solving an :class:`LpProblem`."""

    __slots__ = ("status", "objective", "x", "duals", "message", "iterations", "solver")

    def __init__(
        self,
        status: str,
        objective: float,
        x: Sequence[float],
        duals: Optional[Sequence[float]] = None,
        message: str = "",
        iterations: int = 0,
        solver: str = "",
    ):
        self.status = status
        self.objective = objective
        self.x = list(x)
        self.duals = list(duals) if duals is not None else None
        self.message = message
        self.iterations = iterations
        self.solver = solver

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Solution {self.status} obj={self.objective:.6g} solver={self.solver}>"


class LpProblem:
    """A linear program in the form ``min/max c'x`` subject to rows and bounds."""

    def __init__(self, name: str = "lp", sense: str = "min"):
        if sense not in ("min", "max"):
            raise ValueError("sense must be 'min' or 'max'")
        self.name = name
        self.sense = sense
        self.variables: List[Var] = []
        self.constraints: List[Constraint] = []
        self.objective = LinExpr()
        self._names: Dict[str, int] = {}

    # -- model building ------------------------------------------------------
    def add_var(self, name: str, lb: float = 0.0, ub: float = INF) -> Var:
        if lb > ub:
            raise ValueError(f"variable {name}: lb {lb} exceeds ub {ub}")
        index = len(self.variables)
        if name in self._names:
            name = f"{name}#{index}"
        var = Var(index, name, float(lb), float(ub))
        self.variables.append(var)
        self._names[name] = index
        return var

    def add_vars(
        self,
        name: str,
        keys: Iterable[Tuple],
        lb: float = 0.0,
        ub: float = INF,
    ) -> Dict[Tuple, Var]:
        """Create a family of variables indexed by ``keys``."""
        out: Dict[Tuple, Var] = {}
        for key in keys:
            label = key if isinstance(key, tuple) else (key,)
            suffix = ",".join(str(k) for k in label)
            out[key] = self.add_var(f"{name}[{suffix}]", lb=lb, ub=ub)
        return out

    def add(self, constraint: Constraint, name: str = "") -> Constraint:
        if not isinstance(constraint, Constraint):
            raise TypeError(
                "add() expects a constraint such as 'x + y <= 3'; "
                f"got {type(constraint).__name__}"
            )
        if name:
            constraint.name = name
        elif not constraint.name:
            constraint.name = f"c{len(self.constraints)}"
        constraint.index = len(self.constraints)
        self.constraints.append(constraint)
        return constraint

    def set_objective(self, expr: Operand) -> None:
        if isinstance(expr, LinExpr):
            self.objective = expr.copy()
        else:
            self.objective = LinExpr(const=float(expr))

    # -- introspection -------------------------------------------------------
    @property
    def num_vars(self) -> int:
        return len(self.variables)

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    @property
    def num_nonzeros(self) -> int:
        return sum(len(c.expr.coeffs) for c in self.constraints)

    def stats(self) -> Dict[str, int]:
        return {
            "variables": self.num_vars,
            "constraints": self.num_constraints,
            "nonzeros": self.num_nonzeros,
        }

    # -- solving -------------------------------------------------------------
    def solve(self, solver: str = "auto", **options) -> Solution:
        from .solvers import solve_problem

        return solve_problem(self, solver=solver, **options)

    # -- export --------------------------------------------------------------
    def to_lp_string(self) -> str:
        """Serialise to CPLEX LP format (handy for debugging with other tools)."""
        lines = ["\\Problem: " + self.name, "Minimize" if self.sense == "min" else "Maximize"]
        lines.append(" obj: " + _terms_to_string(self.objective, self.variables))
        lines.append("Subject To")
        op = {LE: "<=", GE: ">=", EQ: "="}
        for con in self.constraints:
            lines.append(
                f" {con.name}: {_terms_to_string(con.expr, self.variables)} "
                f"{op[con.sense]} {con.rhs:.10g}"
            )
        lines.append("Bounds")
        for var in self.variables:
            lo = "-inf" if var.lb == -INF else f"{var.lb:.10g}"
            hi = "+inf" if var.ub == INF else f"{var.ub:.10g}"
            lines.append(f" {lo} <= {var.name} <= {hi}")
        lines.append("End")
        return "\n".join(lines)


def _terms_to_string(expr: LinExpr, variables: Sequence[Var]) -> str:
    parts = []
    for idx, coef in sorted(expr.coeffs.items()):
        sign = "+" if coef >= 0 else "-"
        parts.append(f"{sign} {abs(coef):.10g} {variables[idx].name}")
    if not parts:
        return "0"
    text = " ".join(parts)
    return text[2:] if text.startswith("+ ") else text
