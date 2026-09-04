"""Tests for the built-in simplex solver, including cross-checks against
whichever third-party backends are installed."""

import random
import unittest

from energyplan.lp import INF, LpProblem
from energyplan.simplex import SimplexError, solve_simplex
from energyplan.solvers import available_solvers, solve_problem


def wyndor():
    """A textbook LP with a known optimum of 36 at (2, 6) and duals (0, 1.5, 1)."""
    p = LpProblem(sense="max")
    x = p.add_var("x")
    y = p.add_var("y")
    p.add(x <= 4)
    p.add(2 * y <= 12)
    p.add(3 * x + 2 * y <= 18)
    p.set_objective(3 * x + 5 * y)
    return p


class TestSimplex(unittest.TestCase):
    def test_known_optimum_and_duals(self):
        solution = solve_simplex(wyndor())
        self.assertEqual(solution.status, "optimal")
        self.assertAlmostEqual(solution.objective, 36.0, places=7)
        self.assertAlmostEqual(solution.x[0], 2.0, places=7)
        self.assertAlmostEqual(solution.x[1], 6.0, places=7)
        self.assertAlmostEqual(solution.duals[0], 0.0, places=7)
        self.assertAlmostEqual(solution.duals[1], 1.5, places=7)
        self.assertAlmostEqual(solution.duals[2], 1.0, places=7)

    def test_minimisation_with_equality_and_free_variable(self):
        p = LpProblem(sense="min")
        a = p.add_var("a", lb=-INF)
        b = p.add_var("b")
        p.add(a + b == 5)
        p.add(a - b >= 1)
        p.set_objective(2 * a + 3 * b)
        solution = solve_simplex(p)
        self.assertEqual(solution.status, "optimal")
        self.assertAlmostEqual(solution.objective, 10.0, places=7)
        self.assertAlmostEqual(solution.x[0], 5.0, places=7)

    def test_variable_bounds_are_respected(self):
        p = LpProblem(sense="max")
        x = p.add_var("x", lb=1.0, ub=3.0)
        p.add(x <= 10)
        p.set_objective(x)
        solution = solve_simplex(p)
        self.assertAlmostEqual(solution.x[0], 3.0, places=9)

        p2 = LpProblem(sense="min")
        y = p2.add_var("y", lb=2.0, ub=8.0)
        p2.add(y >= 0)
        p2.set_objective(y)
        self.assertAlmostEqual(solve_simplex(p2).x[0], 2.0, places=9)

    def test_negative_lower_bound(self):
        p = LpProblem(sense="min")
        x = p.add_var("x", lb=-5.0, ub=5.0)
        p.add(x >= -3.0)
        p.set_objective(x)
        solution = solve_simplex(p)
        self.assertAlmostEqual(solution.objective, -3.0, places=7)

    def test_infeasible(self):
        p = LpProblem()
        x = p.add_var("x")
        p.add(x >= 5)
        p.add(x <= 2)
        p.set_objective(x)
        self.assertEqual(solve_simplex(p).status, "infeasible")

    def test_unbounded(self):
        p = LpProblem(sense="max")
        x = p.add_var("x")
        p.add(x >= 1)
        p.set_objective(x)
        self.assertEqual(solve_simplex(p).status, "unbounded")

    def test_degenerate_problem_terminates(self):
        # Degenerate vertices are where a naive simplex can cycle.
        p = LpProblem(sense="min")
        v = [p.add_var(f"x{i}") for i in range(4)]
        p.add(v[0] + v[1] + v[2] + v[3] == 1)
        p.add(v[0] - v[1] == 0)
        p.add(v[2] - v[3] == 0)
        p.add(v[0] + v[2] <= 0.5)
        p.set_objective(-v[0] - 2 * v[2])
        solution = solve_simplex(p)
        self.assertEqual(solution.status, "optimal")

    def test_oversized_model_is_refused_with_advice(self):
        """A dense tableau has a hard size ceiling; say so instead of thrashing."""
        p = LpProblem(sense="min")
        variables = [p.add_var(f"x{i}", ub=1.0) for i in range(60)]
        for i in range(60):
            p.add(variables[i] >= 0.1)
        p.set_objective(sum(variables))
        with self.assertRaises(SimplexError) as caught:
            solve_simplex(p, max_cells=100)
        message = str(caught.exception)
        self.assertIn("dense tableau", message)
        self.assertIn("scipy", message)

    def test_no_constraints(self):
        p = LpProblem(sense="min")
        x = p.add_var("x", lb=1.0, ub=4.0)
        p.set_objective(x)
        solution = solve_simplex(p)
        self.assertEqual(solution.status, "optimal")
        self.assertAlmostEqual(solution.objective, 1.0)


class TestBackendAgreement(unittest.TestCase):
    """Every installed backend must return the same optimum and duals."""

    def _random_problem(self, seed):
        rng = random.Random(seed)
        n, m = 6, 5
        p = LpProblem(sense="min")
        variables = [p.add_var(f"x{i}", ub=rng.uniform(5, 20)) for i in range(n)]
        for _ in range(m):
            coeffs = [rng.uniform(0.5, 4.0) for _ in range(n)]
            expr = sum(c * v for c, v in zip(coeffs, variables))
            p.add(expr >= rng.uniform(10, 40))
        p.set_objective(sum(rng.uniform(1, 9) * v for v in variables))
        return p

    def test_backends_agree(self):
        """Objectives must match. CBC's default tolerance is around 1e-8
        relative, so the comparison is relative rather than absolute."""
        backends = available_solvers()
        if len(backends) < 2:
            self.skipTest("only one backend installed")
        for seed in range(12):
            problem = self._random_problem(seed)
            solutions = {b: solve_problem(problem, solver=b) for b in backends}
            statuses = {s.status for s in solutions.values()}
            self.assertEqual(len(statuses), 1, f"seed {seed}: statuses {statuses}")
            if "optimal" not in statuses:
                continue
            objectives = [s.objective for s in solutions.values()]
            spread = max(objectives) - min(objectives)
            scale = max(1.0, abs(objectives[0]))
            self.assertLess(
                spread / scale, 1e-6,
                f"seed {seed}: objectives disagree {objectives}",
            )

    def test_duals_are_shadow_prices(self):
        """For every backend, each dual must predict the objective's response
        to a small change in that constraint's right-hand side.

        This is a stronger check than comparing duals between solvers: at a
        degenerate optimum several dual solutions are valid, so agreement is
        not required, but every reported dual must still be a valid derivative.
        """
        delta = 1e-5
        for backend in available_solvers():
            problem = self._random_problem(3)
            base = solve_problem(problem, solver=backend)
            self.assertEqual(base.status, "optimal", backend)
            self.assertTrue(base.duals, f"{backend} returned no duals")
            for row in range(len(problem.constraints)):
                original = problem.constraints[row].rhs
                problem.constraints[row].rhs = original + delta
                bumped = solve_problem(problem, solver=backend)
                problem.constraints[row].rhs = original
                predicted = base.objective + base.duals[row] * delta
                error = abs(bumped.objective - predicted) / max(1.0, abs(predicted))
                self.assertLess(
                    error, 1e-7,
                    f"{backend}: dual {row} does not predict the objective change "
                    f"({bumped.objective} vs {predicted})",
                )


if __name__ == "__main__":
    unittest.main()
