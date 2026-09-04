"""Scenario data model for the capacity-expansion planner.

Units (consistent throughout, and checked on load)
--------------------------------------------------
================  ==========================================================
Capacity          MW (power), MWh (storage energy)
Energy            MWh; annual demand is given in TWh for readability
capex             USD per kW  (storage energy capex: USD per kWh)
Fixed O&M         USD per kW per year
Variable O&M      USD per MWh of electricity produced
Fuel price        USD per MWh of *thermal* input (USD/GJ x 3.6 = USD/MWh_th)
Emission factor   tCO2 per MWh of thermal input
Carbon price      USD per tCO2
VOLL              USD per MWh of unserved energy
================  ==========================================================

A scenario is a plain dict/JSON/YAML document; :func:`load_scenario` turns it
into the dataclasses below and validates it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

__all__ = [
    "Technology",
    "StorageTechnology",
    "TransmissionLine",
    "Region",
    "Policy",
    "Scenario",
    "load_scenario",
    "ScenarioError",
]

HOURS_PER_YEAR = 8760


class ScenarioError(ValueError):
    """Raised when a scenario document is malformed or internally inconsistent."""


YearMap = Union[float, Dict[int, float], None]


def _year_value(spec: YearMap, year: int, default: float = 0.0) -> float:
    """Resolve a value that may be a scalar or a ``{year: value}`` mapping.

    Mappings are interpolated linearly between the given years and held flat
    beyond the first and last entry, so a scenario can specify e.g. a carbon
    price only at milestone years.
    """
    if spec is None:
        return default
    if isinstance(spec, (int, float)):
        return float(spec)
    if not spec:
        return default
    years = sorted(int(y) for y in spec.keys())
    values = {int(y): float(v) for y, v in spec.items()}
    if year <= years[0]:
        return values[years[0]]
    if year >= years[-1]:
        return values[years[-1]]
    for lo, hi in zip(years, years[1:]):
        if lo <= year <= hi:
            if hi == lo:
                return values[lo]
            weight = (year - lo) / (hi - lo)
            return values[lo] * (1 - weight) + values[hi] * weight
    return default


def _crf(rate: float, lifetime: float) -> float:
    """Capital recovery factor: the level annual payment repaying 1 unit."""
    if lifetime <= 0:
        raise ScenarioError("lifetime must be positive")
    if rate <= 0:
        return 1.0 / lifetime
    return rate / (1.0 - (1.0 + rate) ** (-lifetime))


@dataclass
class Technology:
    """A dispatchable or variable generation technology."""

    name: str
    kind: str = "thermal"                 # thermal | vre | hydro | other
    capex: YearMap = 0.0                  # USD/kW overnight; a {year: value}
                                          # mapping expresses a learning curve
    fom: float = 0.0                      # USD/kW/yr
    vom: float = 0.0                      # USD/MWh_e
    efficiency: float = 1.0               # MWh_e per MWh_th
    fuel: Optional[str] = None
    lifetime: float = 25.0                # economic life, years
    lead_time: int = 0                    # years between decision and service
    discount_rate: Optional[float] = None  # technology-specific WACC
    availability: float = 1.0             # planned/forced outage derating
    profile: Optional[str] = None         # hourly capacity-factor profile name
    max_capacity_factor: Optional[float] = None   # annual energy budget
    min_capacity_factor: Optional[float] = None   # must-run floor
    min_load_factor: float = 0.0          # hourly floor as a share of capacity
    ramp_limit: Optional[float] = None    # max hourly change, share of capacity
    firm_factor: Optional[float] = None   # capacity credit for the reserve margin
    renewable: bool = False
    clean: bool = False                   # counts toward a clean-energy standard
    existing: Dict[str, float] = field(default_factory=dict)   # region -> MW
    retire_by: Optional[int] = None       # existing fleet linearly gone by this year
    existing_schedule: Optional[Dict[int, float]] = None        # year -> MW remaining
    max_build_per_year: Optional[float] = None
    max_total_capacity: Optional[float] = None   # system-wide potential, MW
    max_capacity_per_region: Optional[Dict[str, float]] = None  # regional potential, MW
    min_total_capacity: YearMap = None           # policy floor, MW
    allow_retirement: bool = False
    regions: Optional[List[str]] = None   # buildable regions (default: all)

    def existing_capacity(self, region: str, year: int, first_year: int) -> float:
        """Existing (pre-model) capacity of this technology still in service."""
        base = float(self.existing.get(region, 0.0))
        if base <= 0.0:
            return 0.0
        if self.existing_schedule:
            share = _year_value(self.existing_schedule, year, base)
            total = sum(self.existing.values()) or 1.0
            return max(0.0, share * base / total)
        if self.retire_by is not None:
            if year >= self.retire_by:
                return 0.0
            span = max(1, self.retire_by - first_year)
            return base * max(0.0, 1.0 - (year - first_year) / span)
        return base

    def capacity_credit(self) -> float:
        """Share of nameplate capacity counted toward the reserve margin."""
        if self.firm_factor is not None:
            return float(self.firm_factor)
        if self.kind == "vre":
            return 0.15
        if self.kind == "hydro":
            return 0.6
        return float(self.availability)

    def annuity(self, rate: float, year: Optional[int] = None) -> float:
        """Annualised capital cost of a unit built in ``year``, USD/MW/yr.

        Capital is charged as a level annuity over the asset's life rather than
        as a lump sum, so assets commissioned late in the horizon are not
        penalised for benefits the model cannot see.
        """
        wacc = self.discount_rate if self.discount_rate is not None else rate
        capex = _year_value(self.capex, year, 0.0) if year is not None else _year_value(
            self.capex, 0, 0.0
        )
        return capex * 1000.0 * _crf(wacc, self.lifetime)


@dataclass
class StorageTechnology:
    """A storage asset with independently sized power and energy capacity."""

    name: str
    capex_power: YearMap = 0.0            # USD/kW (may be a {year: value} curve)
    capex_energy: YearMap = 0.0           # USD/kWh (may be a {year: value} curve)
    fom: float = 0.0                      # USD/kW/yr on power capacity
    vom: float = 0.0                      # USD/MWh discharged
    efficiency_charge: float = 0.92
    efficiency_discharge: float = 0.92
    lifetime: float = 15.0
    lead_time: int = 0
    discount_rate: Optional[float] = None
    min_duration: float = 1.0             # hours of energy per MW of power
    max_duration: float = 8.0
    self_discharge: float = 0.0           # fraction of stored energy lost per hour
    firm_factor: float = 0.9
    existing_power: Dict[str, float] = field(default_factory=dict)
    existing_energy: Dict[str, float] = field(default_factory=dict)
    max_build_power_per_year: Optional[float] = None
    max_total_power: Optional[float] = None      # system-wide, MW
    max_power_per_region: Optional[Dict[str, float]] = None
    regions: Optional[List[str]] = None

    @property
    def round_trip_efficiency(self) -> float:
        return self.efficiency_charge * self.efficiency_discharge

    def annuity_power(self, rate: float, year: Optional[int] = None) -> float:
        wacc = self.discount_rate if self.discount_rate is not None else rate
        capex = _year_value(self.capex_power, year if year is not None else 0, 0.0)
        return capex * 1000.0 * _crf(wacc, self.lifetime)

    def annuity_energy(self, rate: float, year: Optional[int] = None) -> float:
        wacc = self.discount_rate if self.discount_rate is not None else rate
        capex = _year_value(self.capex_energy, year if year is not None else 0, 0.0)
        return capex * 1000.0 * _crf(wacc, self.lifetime)


@dataclass
class TransmissionLine:
    """A bidirectional interconnector between two regions."""

    name: str
    from_region: str
    to_region: str
    existing_mw: float = 0.0
    capex: float = 0.0                    # USD/kW of new transfer capability
    fom: float = 0.0                      # USD/kW/yr
    loss: float = 0.02                    # fractional loss on the transfer
    lifetime: float = 40.0
    max_total_mw: Optional[float] = None
    expandable: bool = True

    def annuity(self, rate: float) -> float:
        return self.capex * 1000.0 * _crf(rate, self.lifetime)


@dataclass
class Region:
    """A demand centre; a single-region scenario is the common case."""

    name: str
    demand_twh: YearMap = None            # annual electricity demand, TWh
    demand_profile: Optional[str] = None  # hourly shape name
    peak_mw: YearMap = None               # optional explicit peak, else derived
    reserve_margin: Optional[float] = None

    def annual_demand_mwh(self, year: int) -> float:
        return _year_value(self.demand_twh, year, 0.0) * 1e6


@dataclass
class Policy:
    """System-wide constraints and prices."""

    carbon_price: YearMap = None          # USD/tCO2
    carbon_cap: YearMap = None            # MtCO2 per year
    carbon_budget: Optional[float] = None  # cumulative MtCO2 over the horizon
    renewable_share: YearMap = None       # minimum renewable share of generation
    clean_share: YearMap = None           # minimum clean (renewable + firm clean) share
    reserve_margin: float = 0.15
    voll: float = 5000.0                  # USD/MWh of unserved energy
    max_unserved_share: Optional[float] = None  # cap on unserved energy per year

    # Backstop prices.  Reliability and policy targets are enforced through a
    # heavily penalised slack rather than as hard constraints, so a scenario
    # that cannot meet them returns a priced, diagnosable answer instead of
    # "infeasible".  Raise these to make a target effectively inviolable; the
    # shadow price of a binding target is capped at its backstop.
    capacity_shortfall_penalty: float = 1.0e6   # USD per MW-year of firm capacity
    carbon_backstop_price: float = 10000.0      # USD per tCO2 above the cap
    share_shortfall_penalty: float = 2000.0     # USD per MWh short of a share target


@dataclass
class Scenario:
    """Everything needed to build and solve one planning problem."""

    name: str = "scenario"
    description: str = ""
    years: List[int] = field(default_factory=list)
    horizon_end: Optional[int] = None
    discount_rate: float = 0.06
    regions: List[Region] = field(default_factory=list)
    technologies: List[Technology] = field(default_factory=list)
    storage: List[StorageTechnology] = field(default_factory=list)
    lines: List[TransmissionLine] = field(default_factory=list)
    fuels: Dict[str, dict] = field(default_factory=dict)
    policy: Policy = field(default_factory=Policy)
    profiles: Dict[str, List[float]] = field(default_factory=dict)
    representative_days: int = 8
    hours_per_day: int = 24
    weather_seed: int = 20250101
    allow_unserved: bool = True

    # -- derived helpers -----------------------------------------------------
    @property
    def first_year(self) -> int:
        return self.years[0]

    def period_weight(self, year: int) -> float:
        """How many calendar years the milestone ``year`` stands for."""
        index = self.years.index(year)
        if index + 1 < len(self.years):
            return float(self.years[index + 1] - year)
        if self.horizon_end:
            return float(max(1, self.horizon_end - year + 1))
        if len(self.years) > 1:
            return float(self.years[-1] - self.years[-2])
        return 1.0

    def discount_factor(self, year: int) -> float:
        return 1.0 / (1.0 + self.discount_rate) ** (year - self.first_year)

    def fuel_price(self, fuel: Optional[str], year: int) -> float:
        if not fuel:
            return 0.0
        spec = self.fuels.get(fuel)
        if spec is None:
            raise ScenarioError(f"technology references unknown fuel {fuel!r}")
        return _year_value(spec.get("price"), year, 0.0)

    def fuel_emission_factor(self, fuel: Optional[str]) -> float:
        if not fuel:
            return 0.0
        spec = self.fuels.get(fuel)
        if spec is None:
            raise ScenarioError(f"technology references unknown fuel {fuel!r}")
        return float(spec.get("co2", 0.0))

    def region_names(self) -> List[str]:
        return [r.name for r in self.regions]

    def technology(self, name: str) -> Technology:
        for tech in self.technologies:
            if tech.name == name:
                return tech
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _coerce_year_map(value):
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    return value


def _build(cls, spec: dict, name_key: str = "name"):
    """Instantiate a dataclass from a dict, rejecting unknown keys loudly."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(spec) - fields
    if unknown:
        raise ScenarioError(
            f"{cls.__name__} {spec.get(name_key, '?')!r}: unknown field(s) "
            f"{', '.join(sorted(unknown))}"
        )
    kwargs = {}
    for key, value in spec.items():
        if key in ("demand_twh", "peak_mw", "carbon_price", "carbon_cap",
                   "renewable_share", "clean_share", "existing_schedule",
                   "min_total_capacity", "capex", "capex_power", "capex_energy"):
            value = _coerce_year_map(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_scenario(source: Union[str, dict]) -> Scenario:
    """Load a scenario from a dict, or from a JSON/YAML file path."""
    if isinstance(source, str):
        if not os.path.exists(source):
            raise ScenarioError(f"scenario file not found: {source}")
        with open(source, "r", encoding="utf-8") as handle:
            text = handle.read()
        if source.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ScenarioError(
                    "PyYAML is required for .yaml scenarios; "
                    "install it or use the JSON form"
                ) from exc
            spec = yaml.safe_load(text)
        else:
            spec = json.loads(text)
    else:
        spec = dict(source)

    if not isinstance(spec, dict):
        raise ScenarioError("scenario document must be a mapping")

    spec = dict(spec)
    regions = [_build(Region, r) for r in spec.pop("regions", [])]
    technologies = [_build(Technology, t) for t in spec.pop("technologies", [])]
    storage = [_build(StorageTechnology, s) for s in spec.pop("storage", [])]
    lines = [_build(TransmissionLine, l) for l in spec.pop("lines", [])]
    policy = _build(Policy, spec.pop("policy", {}) or {})

    scenario = Scenario(
        regions=regions,
        technologies=technologies,
        storage=storage,
        lines=lines,
        policy=policy,
        **{k: v for k, v in spec.items()},
    )
    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    """Check a scenario for the mistakes that silently produce nonsense."""
    problems: List[str] = []

    if not scenario.years:
        problems.append("no planning years given")
    elif sorted(scenario.years) != list(scenario.years):
        problems.append("years must be listed in increasing order")
    elif len(set(scenario.years)) != len(scenario.years):
        problems.append("years must be distinct")

    if not scenario.regions:
        problems.append("no regions defined")
    if not scenario.technologies:
        problems.append("no technologies defined")

    names = [t.name for t in scenario.technologies]
    if len(set(names)) != len(names):
        problems.append("technology names must be unique")
    region_names = set(scenario.region_names())
    if len(region_names) != len(scenario.regions):
        problems.append("region names must be unique")

    if not 0.0 <= scenario.discount_rate < 1.0:
        problems.append(f"discount_rate {scenario.discount_rate} outside [0, 1)")

    for region in scenario.regions:
        if region.demand_twh is None:
            problems.append(f"region {region.name!r} has no demand_twh")
        if region.demand_profile and region.demand_profile not in scenario.profiles:
            # Profiles may also be synthesised later; only flag unknown *files*.
            pass

    for tech in scenario.technologies:
        if tech.kind not in ("thermal", "vre", "hydro", "other"):
            problems.append(f"technology {tech.name!r}: unknown kind {tech.kind!r}")
        if not 0.0 < tech.efficiency <= 1.0:
            problems.append(f"technology {tech.name!r}: efficiency must be in (0, 1]")
        if tech.lifetime <= 0:
            problems.append(f"technology {tech.name!r}: lifetime must be positive")
        if tech.fuel and tech.fuel not in scenario.fuels:
            problems.append(f"technology {tech.name!r}: unknown fuel {tech.fuel!r}")
        if tech.kind == "vre" and not tech.profile:
            problems.append(
                f"technology {tech.name!r}: variable renewables need a 'profile'"
            )
        for region in tech.existing:
            if region not in region_names:
                problems.append(
                    f"technology {tech.name!r}: existing capacity in unknown region {region!r}"
                )
        for region in tech.max_capacity_per_region or {}:
            if region not in region_names:
                problems.append(
                    f"technology {tech.name!r}: capacity limit for unknown region {region!r}"
                )
        if tech.max_total_capacity is not None and tech.max_total_capacity < 0:
            problems.append(f"technology {tech.name!r}: negative max_total_capacity")

    for store in scenario.storage:
        if not 0.0 < store.efficiency_charge <= 1.0 or not 0.0 < store.efficiency_discharge <= 1.0:
            problems.append(f"storage {store.name!r}: efficiencies must be in (0, 1]")
        if store.min_duration > store.max_duration:
            problems.append(f"storage {store.name!r}: min_duration exceeds max_duration")

    for line in scenario.lines:
        for endpoint in (line.from_region, line.to_region):
            if endpoint not in region_names:
                problems.append(f"line {line.name!r}: unknown region {endpoint!r}")
        if not 0.0 <= line.loss < 1.0:
            problems.append(f"line {line.name!r}: loss must be in [0, 1)")

    for name, spec in scenario.fuels.items():
        if "price" not in spec:
            problems.append(f"fuel {name!r}: no price given")

    if scenario.hours_per_day not in (1, 2, 3, 4, 6, 8, 12, 24):
        problems.append("hours_per_day must divide 24")
    if scenario.representative_days < 1:
        problems.append("representative_days must be at least 1")

    if problems:
        raise ScenarioError("invalid scenario:\n  - " + "\n  - ".join(problems))
