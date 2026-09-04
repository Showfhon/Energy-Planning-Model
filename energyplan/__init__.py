"""energyplan - least-cost energy investment planning, built from scratch.

The package is layered so each piece is useful on its own:

``energyplan.lp``          a small linear-programming modelling layer
``energyplan.simplex``     a two-phase simplex solver (no dependencies)
``energyplan.solvers``     backend dispatch: HiGHS, CBC, or the built-in simplex
``energyplan.timeseries``  hourly profiles and representative-day clustering
``energyplan.data``        the scenario schema and its validation
``energyplan.model``       the capacity-expansion linear program
``energyplan.results``     plan extraction, KPIs, prices and consistency audits
``energyplan.report``      terminal, CSV and HTML reporting
``energyplan.study``       overrides, sensitivity sweeps and comparisons

Quick start::

    from energyplan import load_scenario, CapacityExpansionModel, text_report

    scenario = load_scenario("examples/island.yaml")
    result = CapacityExpansionModel(scenario).solve()
    print(text_report(result))
"""

__version__ = "1.0.0"

from .data import (
    Policy,
    Region,
    Scenario,
    ScenarioError,
    StorageTechnology,
    Technology,
    TransmissionLine,
    load_scenario,
)
from .lp import LpProblem, lpsum
from .model import CapacityExpansionModel
from .report import html_report, text_report, write_csv, write_json
from .results import PlanResult
from .solvers import available_solvers
from .study import apply_override, compare, run_sensitivity
from .timeseries import cluster_days, load_profiles_csv, synthesise_profiles

__all__ = [
    "__version__",
    "CapacityExpansionModel",
    "LpProblem",
    "PlanResult",
    "Policy",
    "Region",
    "Scenario",
    "ScenarioError",
    "StorageTechnology",
    "Technology",
    "TransmissionLine",
    "apply_override",
    "available_solvers",
    "cluster_days",
    "compare",
    "html_report",
    "load_profiles_csv",
    "load_scenario",
    "lpsum",
    "run_sensitivity",
    "synthesise_profiles",
    "text_report",
    "write_csv",
    "write_json",
]
