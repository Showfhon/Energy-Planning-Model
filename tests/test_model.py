"""Tests for the capacity-expansion model itself.

The first case is solved by hand so the model's arithmetic can be checked
against a closed-form answer.  The rest verify that each constraint family
does what it claims, and that the built-in solver reproduces the same plan as
the third-party backends.
"""

import unittest

from energyplan.data import _crf, load_scenario
from energyplan.model import CapacityExpansionModel
from energyplan.solvers import available_solvers


def flat_profiles():
    """Perfectly flat profiles, so a hand calculation is possible."""
    return {
        "demand": [1.0] * 8760,
        "flat": [1.0] * 8760,
        "solar": [1.0 if 6 <= (h % 24) < 18 else 0.0 for h in range(8760)],
    }


ANALYTIC = {
    "name": "analytic",
    "years": [2025],
    "horizon_end": 2025,
    "discount_rate": 0.05,
    "representative_days": 1,
    "hours_per_day": 1,
    "regions": [{"name": "r", "demand_twh": 8.76, "peak_mw": 1000.0}],
    "fuels": {"gas": {"price": 30.0, "co2": 0.2}},
    "technologies": [
        {
            "name": "ccgt", "kind": "thermal", "capex": 1000.0, "fom": 0.0, "vom": 0.0,
            "efficiency": 0.5, "fuel": "gas", "lifetime": 20, "availability": 1.0,
            "firm_factor": 1.0,
        },
    ],
    "policy": {"reserve_margin": 0.15, "voll": 10000.0},
}


class TestAnalyticCase(unittest.TestCase):
    """One technology, flat demand: every number can be derived by hand."""

    @classmethod
    def setUpClass(cls):
        scenario = load_scenario(ANALYTIC)
        cls.result = CapacityExpansionModel(scenario, profiles=flat_profiles()).solve()
        cls.summary = cls.result.years[0]
        cls.annuity = 1000.0 * 1000.0 * _crf(0.05, 20)

    def test_solved(self):
        self.assertTrue(self.result.optimal)

    def test_capacity_is_set_by_the_reserve_margin(self):
        # 1000 MW peak x 1.15, all of it firm.
        self.assertAlmostEqual(self.summary.capacity_mw["ccgt"], 1150.0, places=4)

    def test_generation_equals_demand(self):
        self.assertAlmostEqual(self.summary.generation_mwh["ccgt"], 8.76e6, places=1)
        self.assertAlmostEqual(self.summary.unserved_mwh, 0.0, places=6)

    def test_total_cost_matches_the_hand_calculation(self):
        expected_capital = self.annuity * 1150.0
        expected_fuel = (30.0 / 0.5) * 8.76e6
        self.assertAlmostEqual(self.summary.cost["capital"], expected_capital, places=2)
        self.assertAlmostEqual(self.summary.cost["fuel"], expected_fuel, places=2)
        self.assertAlmostEqual(
            self.result.objective, expected_capital + expected_fuel, places=2
        )

    def test_emissions_match_the_fuel_burn(self):
        expected = 0.2 / 0.5 * 8.76e6
        self.assertAlmostEqual(self.summary.emissions_t, expected, places=1)

    def test_marginal_price_is_the_running_cost(self):
        # Capacity is set by the reserve margin, not by the energy balance, so
        # one extra MWh costs only fuel: 30 / 0.5 = 60 USD/MWh.
        self.assertAlmostEqual(self.summary.marginal_price, 60.0, places=4)

    def test_capacity_price_is_the_annuity(self):
        # One more MW of firm requirement costs one more MW of plant.
        self.assertAlmostEqual(
            self.summary.capacity_price["r"], self.annuity, places=2
        )

    def test_lcoe_matches_cost_over_energy(self):
        self.assertAlmostEqual(
            self.result.lcoe(), self.summary.total_cost / 8.76e6, places=6
        )

    def test_audit_is_clean(self):
        for name, residual in self.result.audit().items():
            self.assertLess(residual, 1e-9, name)


def build(**changes):
    spec = {
        "name": "t",
        "years": [2025, 2030],
        "horizon_end": 2034,
        "discount_rate": 0.05,
        "representative_days": 2,
        "hours_per_day": 6,
        "regions": [{"name": "r", "demand_twh": {2025: 20, 2030: 24}}],
        "fuels": {"gas": {"price": 30.0, "co2": 0.2}, "coal": {"price": 12.0, "co2": 0.34}},
        "technologies": [
            {"name": "coal", "kind": "thermal", "capex": 1800, "fom": 50, "efficiency": 0.4,
             "fuel": "coal", "lifetime": 30, "availability": 0.9},
            {"name": "ccgt", "kind": "thermal", "capex": 900, "fom": 20, "efficiency": 0.55,
             "fuel": "gas", "lifetime": 25, "availability": 0.9},
            {"name": "solar", "kind": "vre", "capex": 700, "fom": 12, "profile": "solar",
             "lifetime": 25, "renewable": True},
        ],
        "policy": {"reserve_margin": 0.15, "voll": 8000},
    }
    for key, value in changes.items():
        if key == "policy":
            spec["policy"] = {**spec["policy"], **value}
        elif key == "technologies_extra":
            spec["technologies"] = spec["technologies"] + value
        else:
            spec[key] = value
    return CapacityExpansionModel(load_scenario(spec))


class TestConstraintFamilies(unittest.TestCase):
    def test_carbon_cap_binds(self):
        model = build(policy={"carbon_cap": {2025: 8.0, 2030: 3.0}})
        result = model.solve()
        self.assertTrue(result.optimal)
        self.assertLessEqual(result.year(2025).emissions_t, 8.0e6 + 1.0)
        self.assertLessEqual(result.year(2030).emissions_t, 3.0e6 + 1.0)
        # A binding cap must produce a positive shadow carbon price.
        self.assertGreater(result.year(2030).carbon_shadow_price, 0.0)

    def test_renewable_share_binds(self):
        model = build(policy={"renewable_share": {2025: 0.4, 2030: 0.6}})
        result = model.solve()
        self.assertTrue(result.optimal)
        for year, target in ((2025, 0.4), (2030, 0.6)):
            summary = result.year(year)
            renewable = summary.generation_mwh.get("solar", 0.0)
            self.assertGreaterEqual(renewable / summary.demand_mwh, target - 1e-6)

    def test_build_rate_limit_is_respected(self):
        model = build(technologies_extra=[
            {"name": "capped", "kind": "thermal", "capex": 10, "efficiency": 0.5,
             "fuel": "gas", "lifetime": 30, "max_build_per_year": 100},
        ])
        result = model.solve()
        schedule = result.build_schedule().get("capped", {})
        # 100 MW/yr over a five-year period is at most 500 MW.
        self.assertLessEqual(schedule.get(2025, 0.0), 500.0 + 1e-6)

    def test_lead_time_blocks_early_builds(self):
        model = build(technologies_extra=[
            {"name": "slow", "kind": "thermal", "capex": 5, "efficiency": 0.6,
             "fuel": "gas", "lifetime": 40, "lead_time": 8},
        ])
        result = model.solve()
        schedule = result.build_schedule().get("slow", {})
        self.assertAlmostEqual(schedule.get(2025, 0.0), 0.0, places=6)

    def test_potential_limit_is_system_wide(self):
        spec_model = build(
            regions=[{"name": "a", "demand_twh": 10}, {"name": "b", "demand_twh": 10}],
            lines=[{"name": "ab", "from_region": "a", "to_region": "b",
                    "existing_mw": 2000, "expandable": False}],
        )
        # Re-declare solar with a potential limit and solve.
        scenario = spec_model.scenario
        for tech in scenario.technologies:
            if tech.name == "solar":
                tech.max_total_capacity = 500.0
        result = CapacityExpansionModel(scenario).solve()
        self.assertTrue(result.optimal)
        for summary in result.years:
            self.assertLessEqual(summary.capacity_mw.get("solar", 0.0), 500.0 + 1e-6)

    def test_impossible_scenario_is_priced_rather_than_infeasible(self):
        """With too little buildable capacity the model must still solve, and
        report the reliability and policy shortfalls explicitly."""
        model = build(
            policy={"voll": 900.0, "renewable_share": {2025: 0.5, 2030: 0.5}},
            technologies=[
                {"name": "tiny", "kind": "thermal", "capex": 900, "efficiency": 0.5,
                 "fuel": "gas", "lifetime": 25, "max_total_capacity": 100.0},
            ],
        )
        result = model.solve()
        self.assertTrue(result.optimal, "a hopeless scenario must not be infeasible")
        summary = result.year(2025)
        self.assertGreater(summary.unserved_mwh, 0.0)
        self.assertGreater(sum(summary.capacity_shortfall_mw.values()), 0.0)
        self.assertGreater(sum(summary.share_deficit_mwh.values()), 0.0)
        # The penalties must be carried in the cost breakdown, or the objective
        # reconciliation in audit() would not close.
        self.assertGreater(summary.cost["penalty"], 0.0)
        for name, residual in result.audit().items():
            self.assertLess(residual, 1e-9, name)

    def test_carbon_cap_overshoot_is_reported(self):
        model = build(policy={"carbon_cap": {2025: 0.0, 2030: 0.0}},
                      technologies=[
                          {"name": "onlygas", "kind": "thermal", "capex": 900,
                           "efficiency": 0.5, "fuel": "gas", "lifetime": 25},
                      ])
        result = model.solve()
        self.assertTrue(result.optimal)
        self.assertGreater(result.year(2025).emissions_overshoot_t, 0.0)

    def test_annual_energy_budget_limits_output(self):
        model = build(technologies_extra=[
            {"name": "run_of_river", "kind": "hydro", "capex": 1, "efficiency": 1.0,
             "lifetime": 50, "max_capacity_factor": 0.25, "renewable": True,
             "max_total_capacity": 1000.0},
        ])
        result = model.solve()
        summary = result.year(2025)
        capacity = summary.capacity_mw.get("run_of_river", 0.0)
        generated = summary.generation_mwh.get("run_of_river", 0.0)
        self.assertLessEqual(generated, 0.25 * 8760 * capacity + 1.0)


class TestStorage(unittest.TestCase):
    """Storage must conserve energy: it cannot be a source."""

    @classmethod
    def setUpClass(cls):
        spec = {
            "name": "storage",
            "years": [2030],
            "horizon_end": 2034,
            "discount_rate": 0.05,
            "representative_days": 3,
            "hours_per_day": 24,
            "regions": [{"name": "r", "demand_twh": 20}],
            "fuels": {"gas": {"price": 90.0, "co2": 0.2}},
            "technologies": [
                {"name": "peaker", "kind": "thermal", "capex": 600, "fom": 10,
                 "efficiency": 0.4, "fuel": "gas", "lifetime": 25, "availability": 0.95},
                {"name": "solar", "kind": "vre", "capex": 500, "fom": 10,
                 "profile": "solar", "lifetime": 25, "renewable": True},
            ],
            "storage": [
                {"name": "battery", "capex_power": 100, "capex_energy": 120, "fom": 5,
                 "lifetime": 15, "efficiency_charge": 0.9, "efficiency_discharge": 0.9,
                 "min_duration": 1, "max_duration": 8},
            ],
            "policy": {"reserve_margin": 0.15, "voll": 8000,
                       "renewable_share": {2030: 0.55}},
        }
        cls.model = CapacityExpansionModel(load_scenario(spec))
        cls.result = cls.model.solve()

    def test_solved_and_storage_was_built(self):
        self.assertTrue(self.result.optimal)
        self.assertGreater(self.result.years[0].storage_power_mw.get("battery", 0.0), 0.0)

    def test_discharge_never_exceeds_charge_times_round_trip_efficiency(self):
        index = self.model.index
        variables = self.model.vars
        x = self.result.solution.x
        store = self.model.scenario.storage[0]
        charged = sum(
            index.step_weight[d] * x[variables["charge"][(store.name, "r", 2030, d, h)].index]
            for d, h in index.steps()
        )
        discharged = sum(
            index.step_weight[d] * x[variables["discharge"][(store.name, "r", 2030, d, h)].index]
            for d, h in index.steps()
        )
        self.assertGreater(charged, 0.0)
        self.assertLessEqual(
            discharged, charged * store.round_trip_efficiency + 1e-3,
            "storage discharged more energy than it stored",
        )

    def test_state_of_charge_stays_within_the_energy_capacity(self):
        index = self.model.index
        variables = self.model.vars
        x = self.result.solution.x
        energy_cap = self.result.years[0].storage_energy_mwh["battery"]
        worst = max(
            x[variables["soc"][("battery", "r", 2030, d, h)].index]
            for d, h in index.steps()
        )
        self.assertLessEqual(worst, energy_cap + 1e-6)

    def test_duration_bounds_hold(self):
        summary = self.result.years[0]
        power = summary.storage_power_mw["battery"]
        energy = summary.storage_energy_mwh["battery"]
        self.assertGreaterEqual(energy, 1.0 * power - 1e-6)
        self.assertLessEqual(energy, 8.0 * power + 1e-6)


class TestTransmission(unittest.TestCase):
    def test_power_flows_from_the_cheap_region_and_loses_energy(self):
        spec = {
            "name": "grid",
            "years": [2030],
            "horizon_end": 2034,
            "discount_rate": 0.05,
            "representative_days": 2,
            "hours_per_day": 6,
            "regions": [
                {"name": "cheap", "demand_twh": 4},
                {"name": "dear", "demand_twh": 8},
            ],
            "fuels": {"cheapgas": {"price": 10.0, "co2": 0.2},
                      "deargas": {"price": 120.0, "co2": 0.2}},
            "technologies": [
                {"name": "cheapgen", "kind": "thermal", "capex": 700, "efficiency": 0.5,
                 "fuel": "cheapgas", "lifetime": 25, "regions": ["cheap"],
                 "availability": 1.0, "firm_factor": 1.0},
                {"name": "deargen", "kind": "thermal", "capex": 700, "efficiency": 0.5,
                 "fuel": "deargas", "lifetime": 25, "regions": ["dear"],
                 "availability": 1.0, "firm_factor": 1.0},
            ],
            "lines": [{"name": "link", "from_region": "cheap", "to_region": "dear",
                       "existing_mw": 5000, "capex": 500, "loss": 0.05,
                       "expandable": False}],
            "policy": {"reserve_margin": 0.1, "voll": 9000},
        }
        model = CapacityExpansionModel(load_scenario(spec))
        result = model.solve()
        self.assertTrue(result.optimal)
        summary = result.years[0]
        self.assertGreater(summary.transmission_losses_mwh, 0.0)
        # The cheap region should be exporting, so it generates more than it uses.
        self.assertGreater(summary.generation_mwh["cheapgen"], 4e6)
        for name, residual in result.audit().items():
            self.assertLess(residual, 1e-9, name)


class TestSolverAgreementOnAPlan(unittest.TestCase):
    def test_builtin_simplex_reproduces_the_third_party_plan(self):
        backends = available_solvers()
        if "simplex" not in backends or len(backends) < 2:
            self.skipTest("need a third-party backend to compare against")
        spec = dict(ANALYTIC)
        spec = {**spec, "representative_days": 2, "hours_per_day": 3,
                "regions": [{"name": "r", "demand_twh": 8.76}]}
        scenario = load_scenario(spec)
        objectives = {}
        for backend in backends:
            model = CapacityExpansionModel(scenario, profiles=flat_profiles())
            result = model.solve(solver=backend)
            self.assertTrue(result.optimal, backend)
            objectives[backend] = result.objective
        values = list(objectives.values())
        spread = (max(values) - min(values)) / max(1.0, abs(values[0]))
        self.assertLess(spread, 1e-6, f"backends disagree: {objectives}")


if __name__ == "__main__":
    unittest.main()
