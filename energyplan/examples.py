"""Bundled example scenarios.

``island`` is a full-size study: a two-region island power system with an
ageing coal fleet, a nuclear phase-out, offshore wind potential, batteries,
pumped hydro, an internal transmission link and a legislated net-zero
trajectory.  ``minimal`` is a three-technology single-region toy that solves in
under a second with the built-in simplex, for testing and teaching.

Costs are illustrative but drawn from the range typical of recent IEA/NREL
technology assumptions.  Replace them with your own before drawing conclusions.
"""

from __future__ import annotations

import json
import os
from typing import Dict

__all__ = ["ISLAND", "MINIMAL", "get_example", "write_example"]


MINIMAL: dict = {
    "name": "minimal",
    "description": "One region, three technologies, two planning years.",
    "years": [2025, 2035],
    "horizon_end": 2044,
    "discount_rate": 0.06,
    "representative_days": 3,
    "hours_per_day": 4,
    "regions": [
        {"name": "grid", "demand_twh": {2025: 40, 2035: 52}},
    ],
    "fuels": {
        "gas": {"price": 32, "co2": 0.202},
    },
    "technologies": [
        {
            "name": "ccgt", "kind": "thermal", "capex": 950, "fom": 25, "vom": 3,
            "efficiency": 0.58, "fuel": "gas", "lifetime": 25, "availability": 0.92,
            "existing": {"grid": 3000},
        },
        {
            "name": "solar", "kind": "vre", "capex": 700, "fom": 14, "profile": "solar",
            "lifetime": 25, "renewable": True, "max_total_capacity": 20000,
        },
        {
            "name": "wind", "kind": "vre", "capex": 1500, "fom": 40,
            "profile": "wind_onshore", "lifetime": 25, "renewable": True,
            "max_total_capacity": 8000,
        },
    ],
    "storage": [
        {
            "name": "battery", "capex_power": 130, "capex_energy": 190, "fom": 7,
            "lifetime": 15, "min_duration": 2, "max_duration": 6,
        },
    ],
    "policy": {
        "reserve_margin": 0.15,
        "voll": 5000,
        "carbon_price": {2025: 25, 2035: 90},
    },
}


ISLAND: dict = {
    "name": "island-2050",
    "description": (
        "A two-region island system decarbonising to net zero by 2050: coal "
        "retires, nuclear is phased out, offshore wind and solar scale up "
        "behind build-rate limits, and storage plus a north-south link keep "
        "the lights on."
    ),
    "years": [2025, 2030, 2035, 2040, 2045, 2050],
    "horizon_end": 2059,
    "discount_rate": 0.055,
    "representative_days": 8,
    "hours_per_day": 12,
    "weather_seed": 20250101,
    "regions": [
        {
            "name": "north",
            "demand_twh": {2025: 170, 2030: 190, 2035: 208, 2040: 225, 2045: 238, 2050: 248},
            "reserve_margin": 0.15,
        },
        {
            "name": "south",
            "demand_twh": {2025: 110, 2030: 124, 2035: 138, 2040: 152, 2045: 164, 2050: 174},
            "reserve_margin": 0.15,
        },
    ],
    "fuels": {
        "coal": {"price": {2025: 14, 2050: 18}, "co2": 0.34},
        "gas": {"price": {2025: 34, 2030: 32, 2050: 30}, "co2": 0.202},
        "uranium": {"price": 4.5, "co2": 0.0},
        "hydrogen": {"price": {2030: 190, 2040: 120, 2050: 85}, "co2": 0.0},
    },
    "technologies": [
        {
            "name": "coal", "kind": "thermal", "capex": 2200, "fom": 65, "vom": 4.5,
            "efficiency": 0.40, "fuel": "coal", "lifetime": 40, "availability": 0.85,
            "existing": {"north": 6000, "south": 4000}, "retire_by": 2045,
            "max_build_per_year": 0, "ramp_limit": 0.4, "allow_retirement": True,
        },
        {
            "name": "ccgt", "kind": "thermal", "capex": 950, "fom": 26, "vom": 3.2,
            "efficiency": 0.58, "fuel": "gas", "lifetime": 25, "availability": 0.92,
            "existing": {"north": 9000, "south": 6000},
            "max_build_per_year": 2000, "ramp_limit": 0.6,
        },
        {
            "name": "ocgt", "kind": "thermal", "capex": 560, "fom": 14, "vom": 6.0,
            "efficiency": 0.38, "fuel": "gas", "lifetime": 25, "availability": 0.92,
            "max_build_per_year": 1500,
        },
        {
            "name": "h2_turbine", "kind": "thermal", "capex": 1100, "fom": 30, "vom": 5.0,
            "efficiency": 0.42, "fuel": "hydrogen", "lifetime": 25, "availability": 0.90,
            "clean": True, "lead_time": 5, "max_build_per_year": 1200,
        },
        {
            "name": "nuclear", "kind": "thermal", "capex": 6800, "fom": 130, "vom": 2.2,
            "efficiency": 0.33, "fuel": "uranium", "lifetime": 60, "availability": 0.90,
            "clean": True, "existing": {"north": 2600}, "retire_by": 2035,
            "lead_time": 10, "max_build_per_year": 600, "max_total_capacity": 6000,
            "min_load_factor": 0.5,
        },
        {
            "name": "hydro", "kind": "hydro", "capex": 3000, "fom": 45, "vom": 1.0,
            "efficiency": 1.0, "lifetime": 60, "availability": 0.95,
            "existing": {"north": 2100, "south": 600}, "renewable": True,
            "max_capacity_factor": 0.30, "max_build_per_year": 60,
            "max_total_capacity": 3500,
        },
        {
            "name": "solar", "kind": "vre", "profile": "solar", "fom": 13,
            "capex": {2025: 780, 2030: 640, 2035: 560, 2040: 500, 2045: 470, 2050: 450},
            "lifetime": 25, "renewable": True,
            "max_build_per_year": 3500, "max_total_capacity": 55000,
        },
        {
            "name": "wind_onshore", "kind": "vre", "profile": "wind_onshore", "fom": 38,
            "capex": {2025: 1450, 2035: 1300, 2050: 1200},
            "lifetime": 25, "renewable": True,
            "max_build_per_year": 500, "max_total_capacity": 6000,
        },
        {
            "name": "wind_offshore", "kind": "vre", "profile": "wind_offshore", "fom": 80,
            "capex": {2025: 3100, 2030: 2600, 2035: 2300, 2040: 2100, 2050: 1950},
            "lifetime": 25, "renewable": True, "lead_time": 5,
            "max_build_per_year": 1800, "max_total_capacity": 40000,
        },
    ],
    "storage": [
        {
            "name": "battery",
            "capex_power": {2025: 150, 2030: 115, 2035: 95, 2040: 85, 2050: 75},
            "capex_energy": {2025: 210, 2030: 150, 2035: 120, 2040: 100, 2050: 88},
            "fom": 8, "vom": 0.5, "lifetime": 15,
            "efficiency_charge": 0.93, "efficiency_discharge": 0.93,
            "min_duration": 2, "max_duration": 8, "firm_factor": 0.9,
            "max_build_power_per_year": 4000,
        },
        {
            "name": "pumped_hydro", "capex_power": 1500, "capex_energy": 40, "fom": 18,
            "lifetime": 50, "efficiency_charge": 0.88, "efficiency_discharge": 0.88,
            "min_duration": 6, "max_duration": 12, "firm_factor": 0.95,
            "existing_power": {"north": 2600}, "existing_energy": {"north": 20800},
            "max_build_power_per_year": 200, "max_total_power": 5000,
            "regions": ["north"], "lead_time": 10,
        },
    ],
    "lines": [
        {
            "name": "north_south", "from_region": "north", "to_region": "south",
            "existing_mw": 4000, "capex": 700, "fom": 12, "loss": 0.035,
            "lifetime": 40, "max_total_mw": 12000,
        },
    ],
    "policy": {
        "reserve_margin": 0.15,
        "voll": 6000,
        "carbon_price": {2025: 25, 2030: 55, 2035: 90, 2040: 130, 2045: 170, 2050: 210},
        "carbon_cap": {2025: 110, 2030: 82, 2035: 55, 2040: 33, 2045: 14, 2050: 0.0},
        "renewable_share": {2025: 0.15, 2030: 0.30, 2035: 0.45, 2040: 0.60, 2050: 0.75},
        "max_unserved_share": 0.0002,
    },
}


_EXAMPLES: Dict[str, dict] = {"island": ISLAND, "minimal": MINIMAL}


def get_example(name: str) -> dict:
    try:
        return json.loads(json.dumps(_EXAMPLES[name]))
    except KeyError:
        raise KeyError(
            f"unknown example {name!r}; choose from {', '.join(sorted(_EXAMPLES))}"
        ) from None


def write_example(name: str, path: str) -> str:
    """Write a bundled example to ``path`` as YAML (or JSON if PyYAML is absent)."""
    spec = get_example(name)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            path = os.path.splitext(path)[0] + ".json"
        else:
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle, sort_keys=False, allow_unicode=True,
                               default_flow_style=False, width=88)
            return path
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
    return path
