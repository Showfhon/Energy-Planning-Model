"""Tests for the modelling layer."""

import unittest

from energyplan.lp import LE, LpProblem, lpdot, lpsum


class TestExpressions(unittest.TestCase):
    def setUp(self):
        self.p = LpProblem()
        self.x = self.p.add_var("x")
        self.y = self.p.add_var("y")

    def test_arithmetic(self):
        e = 2 * self.x + 3 * self.y - 4
        self.assertEqual(e.coeffs[self.x.index], 2.0)
        self.assertEqual(e.coeffs[self.y.index], 3.0)
        self.assertEqual(e.const, -4.0)

    def test_subtraction_and_negation(self):
        e = 5 - (self.x - 2 * self.y)
        self.assertEqual(e.coeffs[self.x.index], -1.0)
        self.assertEqual(e.coeffs[self.y.index], 2.0)
        self.assertEqual(e.const, 5.0)

    def test_zero_coefficients_are_dropped(self):
        e = self.x - self.x
        self.assertEqual(e.coeffs, {})

    def test_multiplying_two_variables_is_rejected(self):
        with self.assertRaises(TypeError):
            _ = self.x * self.y

    def test_lpsum_and_lpdot(self):
        total = lpsum([self.x, self.y, 3])
        self.assertEqual(total.const, 3.0)
        weighted = lpdot([2, 5], [self.x, self.y])
        self.assertEqual(weighted.coeffs[self.y.index], 5.0)

    def test_constraint_moves_constants_to_rhs(self):
        con = self.x + 3 <= 2 * self.y + 10
        self.assertEqual(con.sense, LE)
        self.assertEqual(con.rhs, 7.0)
        self.assertEqual(con.expr.coeffs[self.y.index], -2.0)
        self.assertEqual(con.expr.const, 0.0)

    def test_add_rejects_non_constraints(self):
        with self.assertRaises(TypeError):
            self.p.add(self.x)

    def test_bounds_validated(self):
        with self.assertRaises(ValueError):
            self.p.add_var("bad", lb=5, ub=1)

    def test_stats_and_lp_export(self):
        self.p.add(self.x + self.y <= 4, name="cap")
        self.p.set_objective(self.x)
        stats = self.p.stats()
        self.assertEqual(stats["variables"], 2)
        self.assertEqual(stats["constraints"], 1)
        self.assertEqual(stats["nonzeros"], 2)
        text = self.p.to_lp_string()
        self.assertIn("cap:", text)
        self.assertIn("Minimize", text)

    def test_duplicate_names_are_disambiguated(self):
        a = self.p.add_var("x")
        self.assertNotEqual(a.name, self.x.name)


if __name__ == "__main__":
    unittest.main()
