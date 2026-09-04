"""Tests for the scenario schema, its validation and its economics helpers."""

import unittest

from energyplan.data import (
    Scenario,
    ScenarioError,
    Technology,
    _crf,
    _year_value,
    load_scenario,
)


BASE = {
    "years": [2025, 2030],
    "regions": [{"name": "r", "demand_twh": 10}],
    "fuels": {"gas": {"price": 30, "co2": 0.2}},
    "technologies": [
        {"name": "ccgt", "efficiency": 0.5, "fuel": "gas", "capex": 900},
    ],
}


class TestYearValues(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(_year_value(42, 2030), 42.0)

    def test_interpolates_between_given_years(self):
        spec = {2020: 0.0, 2030: 100.0}
        self.assertEqual(_year_value(spec, 2025), 50.0)

    def test_holds_flat_outside_the_range(self):
        spec = {2020: 5.0, 2030: 9.0}
        self.assertEqual(_year_value(spec, 2000), 5.0)
        self.assertEqual(_year_value(spec, 2100), 9.0)

    def test_missing_uses_default(self):
        self.assertEqual(_year_value(None, 2030, 7.0), 7.0)


class TestEconomics(unittest.TestCase):
    def test_capital_recovery_factor(self):
        # 6% over 25 years is a well-known 0.07823.
        self.assertAlmostEqual(_crf(0.06, 25), 0.0782267, places=6)

    def test_zero_rate_is_straight_line(self):
        self.assertAlmostEqual(_crf(0.0, 20), 0.05)

    def test_annuity_uses_the_build_year_capex(self):
        tech = Technology(name="solar", capex={2025: 1000, 2035: 500}, lifetime=25)
        early = tech.annuity(0.06, 2025)
        late = tech.annuity(0.06, 2035)
        self.assertAlmostEqual(early, 1000 * 1000 * _crf(0.06, 25), places=3)
        self.assertAlmostEqual(late, early / 2.0, places=3)

    def test_technology_discount_rate_overrides_the_system_rate(self):
        tech = Technology(name="t", capex=1000, lifetime=20, discount_rate=0.10)
        self.assertAlmostEqual(tech.annuity(0.03, 2025), 1000 * 1000 * _crf(0.10, 20))


class TestExistingCapacity(unittest.TestCase):
    def test_constant_when_no_retirement_given(self):
        tech = Technology(name="t", existing={"r": 500})
        self.assertEqual(tech.existing_capacity("r", 2040, 2025), 500)

    def test_linear_retirement(self):
        tech = Technology(name="t", existing={"r": 400}, retire_by=2045)
        self.assertEqual(tech.existing_capacity("r", 2025, 2025), 400)
        self.assertEqual(tech.existing_capacity("r", 2035, 2025), 200)
        self.assertEqual(tech.existing_capacity("r", 2045, 2025), 0)
        self.assertEqual(tech.existing_capacity("r", 2050, 2025), 0)

    def test_unknown_region_has_none(self):
        tech = Technology(name="t", existing={"r": 400})
        self.assertEqual(tech.existing_capacity("other", 2025, 2025), 0.0)


class TestPeriodWeighting(unittest.TestCase):
    def test_gap_between_milestones(self):
        s = load_scenario(dict(BASE, years=[2025, 2030, 2040], horizon_end=2049))
        self.assertEqual(s.period_weight(2025), 5)
        self.assertEqual(s.period_weight(2030), 10)
        self.assertEqual(s.period_weight(2040), 10)

    def test_discounting(self):
        s = load_scenario(BASE)
        self.assertEqual(s.discount_factor(2025), 1.0)
        self.assertAlmostEqual(s.discount_factor(2030), 1 / 1.06 ** 5)


class TestValidation(unittest.TestCase):
    def _expect(self, changes, fragment):
        spec = {**BASE, **changes}
        with self.assertRaises(ScenarioError) as caught:
            load_scenario(spec)
        self.assertIn(fragment, str(caught.exception))

    def test_years_must_increase(self):
        self._expect({"years": [2030, 2025]}, "increasing order")

    def test_unknown_fuel(self):
        self._expect(
            {"technologies": [{"name": "t", "fuel": "diesel"}]}, "unknown fuel"
        )

    def test_vre_needs_a_profile(self):
        self._expect(
            {"technologies": [{"name": "pv", "kind": "vre"}]}, "need a 'profile'"
        )

    def test_efficiency_range(self):
        self._expect(
            {"technologies": [{"name": "t", "efficiency": 1.4}]}, "efficiency"
        )

    def test_duplicate_technology_names(self):
        self._expect(
            {"technologies": [{"name": "t"}, {"name": "t"}]}, "unique"
        )

    def test_existing_capacity_in_unknown_region(self):
        self._expect(
            {"technologies": [{"name": "t", "existing": {"nowhere": 100}}]},
            "unknown region",
        )

    def test_line_endpoints_must_exist(self):
        self._expect(
            {"lines": [{"name": "l", "from_region": "r", "to_region": "ghost"}]},
            "unknown region",
        )

    def test_hours_per_day_must_divide_24(self):
        self._expect({"hours_per_day": 7}, "hours_per_day")

    def test_missing_demand(self):
        self._expect({"regions": [{"name": "r"}]}, "demand_twh")

    def test_unknown_field_is_rejected(self):
        self._expect(
            {"technologies": [{"name": "t", "capexx": 100}]}, "unknown field"
        )

    def test_valid_scenario_loads(self):
        scenario = load_scenario(BASE)
        self.assertIsInstance(scenario, Scenario)
        self.assertEqual(scenario.region_names(), ["r"])


if __name__ == "__main__":
    unittest.main()
