"""End-to-end tests: bundled examples, reporting and the command line."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

from energyplan import CapacityExpansionModel, load_scenario
from energyplan.cli import main
from energyplan.examples import get_example
from energyplan.report import html_report, text_report, write_csv, write_json
from energyplan.study import apply_override, compare, run_sensitivity


class TestExamples(unittest.TestCase):
    def test_minimal_example_solves_and_audits_clean(self):
        result = CapacityExpansionModel(load_scenario(get_example("minimal"))).solve()
        self.assertTrue(result.optimal)
        for name, residual in result.audit().items():
            self.assertLess(residual, 1e-8, name)

    def test_island_example_solves_and_audits_clean(self):
        result = CapacityExpansionModel(load_scenario(get_example("island"))).solve()
        self.assertTrue(result.optimal)
        for name, residual in result.audit().items():
            self.assertLess(residual, 1e-8, name)

    def test_island_meets_its_carbon_trajectory(self):
        scenario = load_scenario(get_example("island"))
        result = CapacityExpansionModel(scenario).solve()
        final = result.years[-1]
        self.assertLess(final.emissions_t, 1e4, "2050 should be effectively zero-carbon")
        self.assertEqual(final.emissions_overshoot_t, 0.0)
        self.assertEqual(final.capacity_shortfall_mw, {})

    def test_unknown_example_name(self):
        with self.assertRaises(KeyError):
            get_example("nope")


class TestReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = CapacityExpansionModel(load_scenario(get_example("minimal"))).solve()

    def test_text_report_contains_the_headline_sections(self):
        text = text_report(self.result)
        for heading in ("INSTALLED CAPACITY", "GENERATION", "ANNUAL INDICATORS",
                        "ANNUAL COST BY COMPONENT", "CONSISTENCY CHECKS"):
            self.assertIn(heading, text)
        self.assertIn("All checks pass.", text)

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_csv(self.result, directory)
            names = {os.path.basename(p) for p in paths}
            self.assertEqual(
                names,
                {"capacity.csv", "generation.csv", "costs.csv",
                 "indicators.csv", "dispatch.csv"},
            )
            for path in paths:
                self.assertGreater(os.path.getsize(path), 0, path)

    def test_json_export_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(self.result, os.path.join(directory, "plan.json"))
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        self.assertEqual(data["status"], "optimal")
        self.assertIn("years", data)
        self.assertGreater(data["npv_total_cost_usd"], 0.0)

    def test_html_report_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = html_report(self.result, os.path.join(directory, "plan.html"))
            with open(path, encoding="utf-8") as handle:
                page = handle.read()
        self.assertIn("<svg", page)
        self.assertIn("Installed capacity", page)
        self.assertIn("prefers-color-scheme", page)
        # No external requests: charts are inline SVG, styles are inline CSS.
        self.assertNotIn("http://", page)
        self.assertNotIn("<script", page)


class TestStudyHelpers(unittest.TestCase):
    def test_override_by_name_and_multiplier(self):
        spec = get_example("minimal")
        changed = apply_override(spec, "technologies.solar.capex", 400)
        solar = next(t for t in changed["technologies"] if t["name"] == "solar")
        self.assertEqual(solar["capex"], 400)

        doubled = apply_override(spec, "technologies.solar.capex*", 2.0)
        solar2 = next(t for t in doubled["technologies"] if t["name"] == "solar")
        self.assertEqual(solar2["capex"], 1400)
        # The original document must not be mutated.
        original = next(t for t in spec["technologies"] if t["name"] == "solar")
        self.assertEqual(original["capex"], 700)

    def test_override_creates_nested_keys(self):
        spec = apply_override(get_example("minimal"), "policy.carbon_cap", {"2035": 4.0})
        self.assertEqual(spec["policy"]["carbon_cap"], {"2035": 4.0})

    def test_unknown_technology_name_is_rejected(self):
        with self.assertRaises(KeyError):
            apply_override(get_example("minimal"), "technologies.fusion.capex", 1)

    def test_sensitivity_sweep_moves_the_answer(self):
        spec = get_example("minimal")
        rows = run_sensitivity(spec, "technologies.solar.capex", [400, 1600])
        self.assertEqual(len(rows), 2)
        cheap, dear = rows
        self.assertLess(cheap["npv_bn_usd"], dear["npv_bn_usd"])
        self.assertGreaterEqual(
            cheap.get("cap_solar_mw", 0.0), dear.get("cap_solar_mw", 0.0)
        )
        table = compare(rows)
        self.assertIn("npv bn usd", table)


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "s.json")
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(get_example("minimal"), handle)

    def tearDown(self):
        self.directory.cleanup()

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_solvers_command(self):
        code, out, _ = self._run(["solvers"])
        self.assertEqual(code, 0)
        self.assertIn("simplex", out)

    def test_solve_command_writes_every_output(self):
        html = os.path.join(self.directory.name, "plan.html")
        csv_dir = os.path.join(self.directory.name, "csv")
        js = os.path.join(self.directory.name, "plan.json")
        code, out, _ = self._run(
            ["solve", self.path, "--html", html, "--csv", csv_dir, "--json", js]
        )
        self.assertEqual(code, 0)
        self.assertIn("OPTIMAL ENERGY INVESTMENT PLAN", out)
        self.assertTrue(os.path.exists(html))
        self.assertTrue(os.path.exists(js))
        self.assertTrue(os.path.isdir(csv_dir))

    def test_solve_honours_overrides_and_time_resolution(self):
        code, out, _ = self._run(
            ["solve", self.path, "--set", "technologies.solar.capex=250",
             "--days", "2", "--hours", "3", "--quiet"]
        )
        self.assertEqual(code, 0)

    def test_bad_scenario_reports_cleanly(self):
        broken = os.path.join(self.directory.name, "broken.json")
        with open(broken, "w", encoding="utf-8") as handle:
            json.dump({"years": [], "regions": [], "technologies": []}, handle)
        code, _, err = self._run(["solve", broken])
        self.assertEqual(code, 1)
        self.assertIn("Scenario error", err)

    def test_missing_file(self):
        code, _, err = self._run(["solve", "/no/such/file.json"])
        self.assertEqual(code, 1)
        self.assertIn("not found", err.lower())

    def test_example_command_writes_a_runnable_scenario(self):
        target = os.path.join(self.directory.name, "written.json")
        code, out, _ = self._run(["example", target, "--name", "minimal"])
        self.assertEqual(code, 0)
        scenario = load_scenario(target)
        self.assertEqual(scenario.name, "minimal")

    def test_sensitivity_command(self):
        code, out, _ = self._run(
            ["sensitivity", self.path, "--vary", "technologies.solar.capex",
             "--values", "400,1200", "--days", "2", "--hours", "3"]
        )
        self.assertEqual(code, 0)
        self.assertIn("SENSITIVITY", out)

    def test_compare_command(self):
        code, out, _ = self._run(
            ["compare", self.path, self.path, "--days", "2", "--hours", "3"]
        )
        self.assertEqual(code, 0)
        self.assertIn("npv bn usd", out)


if __name__ == "__main__":
    unittest.main()
