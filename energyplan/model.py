"""The capacity-expansion linear program.

The model chooses how much of each generation, storage and transmission asset
to build in each planning year, and how to dispatch the resulting fleet across
a set of representative days, so as to minimise the discounted cost of serving
demand subject to reliability, emission and policy constraints.

Objective (minimised)::

    sum over years y of  discount(y) * period_length(y) * [
          annuity(tech)      * new capacity in service
        + fixed O&M          * total capacity
        + (VOM + fuel/eff + CO2 price * emission rate / eff) * generation
        + storage annuities and O&M
        + transmission annuities
        + value of lost load * unserved energy
    ]

Capital cost enters as a level annuity over the asset's economic life rather
than as a lump sum in the build year.  That is the standard way to avoid
"end effects", where assets built late in the horizon look artificially
expensive because the model cannot see the benefit they deliver after the
last modelled year.

Every constraint below is named, so the dual values returned by the solver can
be read as economically meaningful prices: the dual of the energy balance is
the marginal cost of electricity in that hour, the dual of the emission cap is
the shadow carbon price, and the dual of the reserve-margin row is the
marginal value of firm capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .data import HOURS_PER_YEAR, Scenario, ScenarioError
from .lp import INF, LpProblem, Var, lpsum
from .timeseries import RepresentativeDays, cluster_days, synthesise_profiles

__all__ = ["CapacityExpansionModel", "ModelIndex"]


@dataclass
class ModelIndex:
    """The index sets and reduced profiles the LP is built over."""

    years: List[int]
    regions: List[str]
    rep: RepresentativeDays
    step_weight: List[float]                     # hours/year represented per step of day d
    demand: Dict[Tuple[str, int], List[List[float]]] = field(default_factory=dict)
    peak_mw: Dict[Tuple[str, int], float] = field(default_factory=dict)
    capacity_factor: Dict[str, List[List[float]]] = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return self.rep.n_days

    @property
    def n_hours(self) -> int:
        return self.rep.hours_per_day

    def steps(self):
        for day in range(self.n_days):
            for hour in range(self.n_hours):
                yield day, hour


class CapacityExpansionModel:
    """Builds and solves the planning LP for one :class:`Scenario`."""

    def __init__(
        self,
        scenario: Scenario,
        profiles: Optional[Dict[str, Sequence[float]]] = None,
        representative_days: Optional[RepresentativeDays] = None,
        preserve_annual_means: bool = True,
    ):
        self.scenario = scenario
        self.profiles = dict(profiles) if profiles else synthesise_profiles(scenario.weather_seed)
        self.profiles.update(scenario.profiles or {})
        self.preserve_annual_means = preserve_annual_means
        self.index = self._build_index(representative_days)
        self.problem: Optional[LpProblem] = None
        self.vars: Dict[str, Dict[Tuple, Var]] = {}
        self._row_lookup: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_index(self, rep: Optional[RepresentativeDays]) -> ModelIndex:
        scenario = self.scenario
        needed = {"demand"}
        for tech in scenario.technologies:
            if tech.profile:
                needed.add(tech.profile)
        missing = needed - set(self.profiles)
        if missing:
            raise ScenarioError(
                "missing hourly profile(s): " + ", ".join(sorted(missing))
            )

        for region in scenario.regions:
            if region.demand_profile and region.demand_profile not in self.profiles:
                raise ScenarioError(
                    f"region {region.name!r} references unknown profile "
                    f"{region.demand_profile!r}"
                )

        if rep is None:
            rep = cluster_days(
                {k: v for k, v in self.profiles.items() if len(v) == HOURS_PER_YEAR},
                n_days=scenario.representative_days,
                hours_per_day=scenario.hours_per_day,
                seed=scenario.weather_seed,
                feature_keys=sorted(needed),
                net_load=self._net_load_signal(),
            )

        step_weight = [w * rep.hours_per_step for w in rep.weights]
        index = ModelIndex(
            years=list(scenario.years),
            regions=scenario.region_names(),
            rep=rep,
            step_weight=step_weight,
        )

        # ---- capacity factors, rescaled to keep annual means honest -------
        for tech in scenario.technologies:
            if not tech.profile:
                continue
            reduced = rep.slice_profile(self.profiles[tech.profile])
            reduced = self._rescale(reduced, self.profiles[tech.profile], step_weight)
            index.capacity_factor[tech.name] = reduced

        # ---- demand in MW per region-year ---------------------------------
        for region in scenario.regions:
            shape_name = region.demand_profile or "demand"
            raw = self.profiles[shape_name]
            reduced = rep.slice_profile(raw)
            reduced = self._rescale(reduced, raw, step_weight)
            full_peak = max(raw)
            for year in scenario.years:
                annual = region.annual_demand_mwh(year)
                mean_mw = annual / HOURS_PER_YEAR
                index.demand[(region.name, year)] = [
                    [mean_mw * value for value in day] for day in reduced
                ]
                # Peak comes from the *full* 8760 series: clustering must not
                # be allowed to shrink the capacity requirement.
                from .data import _year_value

                explicit = _year_value(region.peak_mw, year, 0.0)
                index.peak_mw[(region.name, year)] = explicit or mean_mw * full_peak
        return index

    def _net_load_signal(self) -> List[float]:
        """Demand minus a rough renewable build-out, used to find the peak day."""
        demand = self.profiles.get("demand")
        if demand is None:
            return []
        vre = [name for name in ("solar", "wind_onshore", "wind_offshore") if name in self.profiles]
        if not vre:
            return list(demand)
        share = 0.4 / len(vre)
        return [
            demand[h] - share * sum(self.profiles[name][h] for name in vre)
            for h in range(len(demand))
        ]

    def _rescale(
        self,
        reduced: List[List[float]],
        original: Sequence[float],
        step_weight: Sequence[float],
    ) -> List[List[float]]:
        """Scale a clustered profile so its weighted annual mean matches the original."""
        if not self.preserve_annual_means:
            return reduced
        target = sum(original) / len(original)
        weighted = sum(
            step_weight[d] * value
            for d, day in enumerate(reduced)
            for value in day
        )
        if weighted <= 1e-12:
            return reduced
        factor = target * HOURS_PER_YEAR / weighted
        return [[value * factor for value in day] for day in reduced]

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def build(self) -> LpProblem:
        scenario = self.scenario
        index = self.index
        problem = LpProblem(scenario.name, sense="min")
        self.problem = problem
        years = index.years
        regions = index.regions
        wacc = scenario.discount_rate   # never rebind: annuities below depend on it

        build_v: Dict[Tuple, Var] = {}
        retire_v: Dict[Tuple, Var] = {}
        cap_v: Dict[Tuple, Var] = {}
        newcap_v: Dict[Tuple, Var] = {}
        gen_v: Dict[Tuple, Var] = {}

        # ---- generation capacity ------------------------------------------
        for tech in scenario.technologies:
            buildable = set(tech.regions) if tech.regions else set(regions)
            for region in regions:
                for year in years:
                    can_build = region in buildable and year >= scenario.first_year + tech.lead_time
                    limit = INF
                    if not can_build:
                        limit = 0.0
                    elif tech.max_build_per_year is not None:
                        limit = tech.max_build_per_year * scenario.period_weight(year)
                    build_v[(tech.name, region, year)] = problem.add_var(
                        f"build[{tech.name},{region},{year}]", lb=0.0, ub=limit
                    )
                    regional_cap = INF
                    if tech.max_capacity_per_region:
                        regional_cap = tech.max_capacity_per_region.get(region, INF)
                    elif len(regions) == 1 and tech.max_total_capacity is not None:
                        regional_cap = tech.max_total_capacity
                    cap_v[(tech.name, region, year)] = problem.add_var(
                        f"cap[{tech.name},{region},{year}]", lb=0.0, ub=regional_cap
                    )
                    newcap_v[(tech.name, region, year)] = problem.add_var(
                        f"newcap[{tech.name},{region},{year}]", lb=0.0
                    )
                    if tech.allow_retirement:
                        existing = tech.existing_capacity(region, year, scenario.first_year)
                        retire_v[(tech.name, region, year)] = problem.add_var(
                            f"retire[{tech.name},{region},{year}]", lb=0.0, ub=max(0.0, existing)
                        )

        # ---- capacity accounting ------------------------------------------
        for tech in scenario.technologies:
            for region in regions:
                for year in years:
                    # New capacity still inside its economic life.
                    vintage = lpsum(
                        build_v[(tech.name, region, other)]
                        for other in years
                        if other <= year and (year - other) < tech.lifetime
                    )
                    problem.add(
                        newcap_v[(tech.name, region, year)] - vintage == 0.0,
                        name=f"newcap_def[{tech.name},{region},{year}]",
                    )
                    existing = tech.existing_capacity(region, year, scenario.first_year)
                    retired = lpsum(
                        retire_v[(tech.name, region, other)]
                        for other in years
                        if other <= year and (tech.name, region, other) in retire_v
                    )
                    problem.add(
                        cap_v[(tech.name, region, year)]
                        - newcap_v[(tech.name, region, year)]
                        + retired
                        == existing,
                        name=f"cap_def[{tech.name},{region},{year}]",
                    )
            # Resource potential is a limit on the whole system, not per region.
            if tech.max_total_capacity is not None and len(regions) > 1:
                for year in years:
                    problem.add(
                        lpsum(cap_v[(tech.name, r, year)] for r in regions)
                        <= tech.max_total_capacity,
                        name=f"cap_potential[{tech.name},{year}]",
                    )

            # System-wide capacity floor (e.g. a legislated renewable target).
            if tech.min_total_capacity is not None:
                from .data import _year_value

                for year in years:
                    floor = _year_value(tech.min_total_capacity, year, 0.0)
                    if floor > 0:
                        problem.add(
                            lpsum(cap_v[(tech.name, r, year)] for r in regions) >= floor,
                            name=f"cap_floor[{tech.name},{year}]",
                        )

        # ---- dispatch ------------------------------------------------------
        for tech in scenario.technologies:
            profile = index.capacity_factor.get(tech.name)
            for region in regions:
                for year in years:
                    cap = cap_v[(tech.name, region, year)]
                    for day, hour in index.steps():
                        gen_v[(tech.name, region, year, day, hour)] = problem.add_var(
                            f"gen[{tech.name},{region},{year},{day},{hour}]", lb=0.0
                        )
                    for day, hour in index.steps():
                        available = (
                            profile[day][hour] if profile is not None else tech.availability
                        )
                        problem.add(
                            gen_v[(tech.name, region, year, day, hour)]
                            - available * cap <= 0.0,
                            name=f"dispatch[{tech.name},{region},{year},{day},{hour}]",
                        )
                        if tech.min_load_factor > 0 and profile is None:
                            problem.add(
                                gen_v[(tech.name, region, year, day, hour)]
                                - tech.min_load_factor * cap >= 0.0,
                                name=f"minload[{tech.name},{region},{year},{day},{hour}]",
                            )
                    # Annual energy budgets (hydro inflow, fuel contracts, ...).
                    annual = lpsum(
                        index.step_weight[day] * gen_v[(tech.name, region, year, day, hour)]
                        for day, hour in index.steps()
                    )
                    if tech.max_capacity_factor is not None:
                        problem.add(
                            annual - tech.max_capacity_factor * HOURS_PER_YEAR * cap <= 0.0,
                            name=f"energy_max[{tech.name},{region},{year}]",
                        )
                    if tech.min_capacity_factor is not None:
                        problem.add(
                            annual - tech.min_capacity_factor * HOURS_PER_YEAR * cap >= 0.0,
                            name=f"energy_min[{tech.name},{region},{year}]",
                        )
                    # Ramping between consecutive hours of a representative day.
                    if tech.ramp_limit is not None:
                        for day in range(index.n_days):
                            for hour in range(1, index.n_hours):
                                now = gen_v[(tech.name, region, year, day, hour)]
                                before = gen_v[(tech.name, region, year, day, hour - 1)]
                                problem.add(
                                    now - before - tech.ramp_limit * cap <= 0.0,
                                    name=f"rampup[{tech.name},{region},{year},{day},{hour}]",
                                )
                                problem.add(
                                    before - now - tech.ramp_limit * cap <= 0.0,
                                    name=f"rampdn[{tech.name},{region},{year},{day},{hour}]",
                                )

        # ---- storage --------------------------------------------------------
        sbuild_p, sbuild_e, scap_p, scap_e = {}, {}, {}, {}
        charge_v, discharge_v, soc_v = {}, {}, {}
        for store in scenario.storage:
            buildable = set(store.regions) if store.regions else set(regions)
            for region in regions:
                for year in years:
                    can_build = region in buildable and year >= scenario.first_year + store.lead_time
                    limit = INF
                    if not can_build:
                        limit = 0.0
                    elif store.max_build_power_per_year is not None:
                        limit = store.max_build_power_per_year * scenario.period_weight(year)
                    sbuild_p[(store.name, region, year)] = problem.add_var(
                        f"sbuild_p[{store.name},{region},{year}]", lb=0.0, ub=limit
                    )
                    sbuild_e[(store.name, region, year)] = problem.add_var(
                        f"sbuild_e[{store.name},{region},{year}]", lb=0.0,
                        ub=INF if can_build else 0.0,
                    )
                    regional_power = INF
                    if store.max_power_per_region:
                        regional_power = store.max_power_per_region.get(region, INF)
                    elif len(regions) == 1 and store.max_total_power is not None:
                        regional_power = store.max_total_power
                    scap_p[(store.name, region, year)] = problem.add_var(
                        f"scap_p[{store.name},{region},{year}]", lb=0.0, ub=regional_power
                    )
                    scap_e[(store.name, region, year)] = problem.add_var(
                        f"scap_e[{store.name},{region},{year}]", lb=0.0
                    )
                    for day, hour in index.steps():
                        key = (store.name, region, year, day, hour)
                        charge_v[key] = problem.add_var(f"charge[{store.name},{region},{year},{day},{hour}]", lb=0.0)
                        discharge_v[key] = problem.add_var(f"discharge[{store.name},{region},{year},{day},{hour}]", lb=0.0)
                        soc_v[key] = problem.add_var(f"soc[{store.name},{region},{year},{day},{hour}]", lb=0.0)

        for store in scenario.storage:
            for region in regions:
                existing_p = float(store.existing_power.get(region, 0.0))
                existing_e = float(store.existing_energy.get(region, existing_p * store.min_duration))
                for year in years:
                    live_p = lpsum(
                        sbuild_p[(store.name, region, other)]
                        for other in years
                        if other <= year and (year - other) < store.lifetime
                    )
                    live_e = lpsum(
                        sbuild_e[(store.name, region, other)]
                        for other in years
                        if other <= year and (year - other) < store.lifetime
                    )
                    problem.add(
                        scap_p[(store.name, region, year)] - live_p == existing_p,
                        name=f"scap_p_def[{store.name},{region},{year}]",
                    )
                    problem.add(
                        scap_e[(store.name, region, year)] - live_e == existing_e,
                        name=f"scap_e_def[{store.name},{region},{year}]",
                    )
                    problem.add(
                        scap_e[(store.name, region, year)]
                        - store.min_duration * scap_p[(store.name, region, year)] >= 0.0,
                        name=f"duration_min[{store.name},{region},{year}]",
                    )
                    problem.add(
                        scap_e[(store.name, region, year)]
                        - store.max_duration * scap_p[(store.name, region, year)] <= 0.0,
                        name=f"duration_max[{store.name},{region},{year}]",
                    )
                    step_hours = index.rep.hours_per_step
                    keep = (1.0 - store.self_discharge) ** step_hours
                    for day in range(index.n_days):
                        for hour in range(index.n_hours):
                            key = (store.name, region, year, day, hour)
                            previous = (store.name, region, year, day,
                                        (hour - 1) % index.n_hours)
                            problem.add(
                                soc_v[key]
                                - keep * soc_v[previous]
                                - store.efficiency_charge * step_hours * charge_v[key]
                                + (step_hours / store.efficiency_discharge) * discharge_v[key]
                                == 0.0,
                                name=f"soc[{store.name},{region},{year},{day},{hour}]",
                            )
                            problem.add(
                                soc_v[key] - scap_e[(store.name, region, year)] <= 0.0,
                                name=f"soc_cap[{store.name},{region},{year},{day},{hour}]",
                            )
                            problem.add(
                                charge_v[key] - scap_p[(store.name, region, year)] <= 0.0,
                                name=f"charge_cap[{store.name},{region},{year},{day},{hour}]",
                            )
                            problem.add(
                                discharge_v[key] - scap_p[(store.name, region, year)] <= 0.0,
                                name=f"discharge_cap[{store.name},{region},{year},{day},{hour}]",
                            )
            # Resource potential applies to the whole system, not to each region.
            if store.max_total_power is not None and len(regions) > 1:
                for year in years:
                    problem.add(
                        lpsum(scap_p[(store.name, r, year)] for r in regions)
                        <= store.max_total_power,
                        name=f"storage_potential[{store.name},{year}]",
                    )

        # ---- transmission ----------------------------------------------------
        line_cap, line_build, flow_f, flow_b = {}, {}, {}, {}
        for line in scenario.lines:
            for year in years:
                line_build[(line.name, year)] = problem.add_var(
                    f"line_build[{line.name},{year}]", lb=0.0,
                    ub=INF if line.expandable else 0.0,
                )
                line_cap[(line.name, year)] = problem.add_var(
                    f"line_cap[{line.name},{year}]", lb=0.0,
                    ub=line.max_total_mw if line.max_total_mw is not None else INF,
                )
                for day, hour in index.steps():
                    flow_f[(line.name, year, day, hour)] = problem.add_var(
                        f"flow_fwd[{line.name},{year},{day},{hour}]", lb=0.0
                    )
                    flow_b[(line.name, year, day, hour)] = problem.add_var(
                        f"flow_bwd[{line.name},{year},{day},{hour}]", lb=0.0
                    )
        for line in scenario.lines:
            for year in years:
                live = lpsum(
                    line_build[(line.name, other)]
                    for other in years
                    if other <= year and (year - other) < line.lifetime
                )
                problem.add(
                    line_cap[(line.name, year)] - live == line.existing_mw,
                    name=f"line_cap_def[{line.name},{year}]",
                )
                for day, hour in index.steps():
                    problem.add(
                        flow_f[(line.name, year, day, hour)] - line_cap[(line.name, year)] <= 0.0,
                        name=f"line_fwd[{line.name},{year},{day},{hour}]",
                    )
                    problem.add(
                        flow_b[(line.name, year, day, hour)] - line_cap[(line.name, year)] <= 0.0,
                        name=f"line_bwd[{line.name},{year},{day},{hour}]",
                    )

        # ---- unserved energy --------------------------------------------------
        unserved_v = {}
        for region in regions:
            for year in years:
                for day, hour in index.steps():
                    unserved_v[(region, year, day, hour)] = problem.add_var(
                        f"unserved[{region},{year},{day},{hour}]", lb=0.0,
                        ub=INF if scenario.allow_unserved else 0.0,
                    )

        # ---- energy balance ---------------------------------------------------
        for region in regions:
            for year in years:
                demand = index.demand[(region, year)]
                for day, hour in index.steps():
                    terms = lpsum(
                        gen_v[(tech.name, region, year, day, hour)]
                        for tech in scenario.technologies
                    )
                    for store in scenario.storage:
                        key = (store.name, region, year, day, hour)
                        terms.add_expr(discharge_v[key], 1.0)
                        terms.add_expr(charge_v[key], -1.0)
                    for line in scenario.lines:
                        key = (line.name, year, day, hour)
                        if line.to_region == region:
                            terms.add_expr(flow_f[key], 1.0 - line.loss)
                            terms.add_expr(flow_b[key], -1.0)
                        elif line.from_region == region:
                            terms.add_expr(flow_b[key], 1.0 - line.loss)
                            terms.add_expr(flow_f[key], -1.0)
                    terms.add_expr(unserved_v[(region, year, day, hour)], 1.0)
                    problem.add(
                        terms == demand[day][hour],
                        name=f"balance[{region},{year},{day},{hour}]",
                    )

        # ---- reserve margin ----------------------------------------------------
        # A penalised slack keeps the model solvable when no buildable portfolio
        # can meet the requirement; the report flags any shortfall loudly.
        policy = scenario.policy
        shortfall_v: Dict[Tuple, Var] = {}
        for year in years:
            for region in scenario.regions:
                margin = region.reserve_margin
                if margin is None:
                    margin = policy.reserve_margin
                if margin is None:
                    continue
                firm = lpsum(
                    tech.capacity_credit() * cap_v[(tech.name, region.name, year)]
                    for tech in scenario.technologies
                )
                for store in scenario.storage:
                    firm.add_expr(scap_p[(store.name, region.name, year)], store.firm_factor)
                for line in scenario.lines:
                    if region.name in (line.from_region, line.to_region):
                        firm.add_expr(line_cap[(line.name, year)], 0.75 * (1.0 - line.loss))
                requirement = index.peak_mw[(region.name, year)] * (1.0 + margin)
                shortfall = problem.add_var(
                    f"capacity_shortfall[{region.name},{year}]", lb=0.0
                )
                shortfall_v[(region.name, year)] = shortfall
                firm.add_expr(shortfall, 1.0)
                problem.add(
                    firm >= requirement,
                    name=f"reserve[{region.name},{year}]",
                )

        # ---- emissions ----------------------------------------------------------
        from .data import _year_value

        emissions_by_year: Dict[int, "object"] = {}
        overshoot_v: Dict[int, Var] = {}
        for year in years:
            expr = lpsum([])
            for tech in scenario.technologies:
                tonnes_per_mwh = scenario.fuel_emission_factor(tech.fuel) / tech.efficiency
                if tonnes_per_mwh <= 0:
                    continue
                for region in regions:
                    for day, hour in index.steps():
                        expr.add_term(
                            gen_v[(tech.name, region, year, day, hour)].index,
                            tonnes_per_mwh * index.step_weight[day],
                        )
            emissions_by_year[year] = expr
            cap_mt = _year_value(policy.carbon_cap, year, None) if policy.carbon_cap is not None else None
            if cap_mt is not None:
                overshoot = problem.add_var(f"co2_overshoot[{year}]", lb=0.0)
                overshoot_v[year] = overshoot
                problem.add(
                    expr - overshoot <= cap_mt * 1e6, name=f"co2_cap[{year}]"
                )

        if policy.carbon_budget is not None:
            total = lpsum([])
            for year in years:
                total.add_expr(emissions_by_year[year], scenario.period_weight(year))
            problem.add(total <= policy.carbon_budget * 1e6, name="co2_budget")

        # ---- renewable / clean energy shares --------------------------------------
        share_deficit_v: Dict[Tuple, Var] = {}
        for label, spec, predicate in (
            ("rps", policy.renewable_share, lambda t: t.renewable),
            ("ces", policy.clean_share, lambda t: t.clean or t.renewable),
        ):
            if spec is None:
                continue
            for year in years:
                share = _year_value(spec, year, 0.0)
                if share <= 0:
                    continue
                qualifying = lpsum([])
                for tech in scenario.technologies:
                    if not predicate(tech):
                        continue
                    for region in regions:
                        for day, hour in index.steps():
                            qualifying.add_term(
                                gen_v[(tech.name, region, year, day, hour)].index,
                                index.step_weight[day],
                            )
                total_demand = sum(
                    r.annual_demand_mwh(year) for r in scenario.regions
                )
                deficit = problem.add_var(f"{label}_deficit[{year}]", lb=0.0)
                share_deficit_v[(label, year)] = deficit
                qualifying.add_expr(deficit, 1.0)
                problem.add(
                    qualifying >= share * total_demand,
                    name=f"{label}[{year}]",
                )

        # ---- unserved-energy ceiling ------------------------------------------------
        if policy.max_unserved_share is not None:
            for year in years:
                total_unserved = lpsum([])
                for region in regions:
                    for day, hour in index.steps():
                        total_unserved.add_term(
                            unserved_v[(region, year, day, hour)].index,
                            index.step_weight[day],
                        )
                allowed = policy.max_unserved_share * sum(
                    r.annual_demand_mwh(year) for r in scenario.regions
                )
                problem.add(total_unserved <= allowed, name=f"unserved_cap[{year}]")

        # ---- objective ------------------------------------------------------------
        objective = lpsum([])

        # Capital is charged on the *build* variable of each vintage, summed over
        # every modelled year in which that vintage is still in service.  This
        # is equivalent to charging the annuity on installed new capacity, but
        # it also lets capex follow a learning curve: a plant built in 2045 pays
        # the 2045 price for its whole life.
        def service_weight(build_year: int, lifetime: float) -> float:
            return sum(
                scenario.discount_factor(y) * scenario.period_weight(y)
                for y in years
                if build_year <= y and (y - build_year) < lifetime
            )

        for tech in scenario.technologies:
            for region in regions:
                for build_year in years:
                    coefficient = tech.annuity(wacc, build_year) * service_weight(
                        build_year, tech.lifetime
                    )
                    objective.add_expr(build_v[(tech.name, region, build_year)], coefficient)
        for store in scenario.storage:
            for region in regions:
                for build_year in years:
                    span = service_weight(build_year, store.lifetime)
                    objective.add_expr(
                        sbuild_p[(store.name, region, build_year)],
                        store.annuity_power(wacc, build_year) * span,
                    )
                    objective.add_expr(
                        sbuild_e[(store.name, region, build_year)],
                        store.annuity_energy(wacc, build_year) * span,
                    )
        for line in scenario.lines:
            for build_year in years:
                objective.add_expr(
                    line_build[(line.name, build_year)],
                    line.annuity(wacc) * service_weight(build_year, line.lifetime),
                )

        for year in years:
            weight = scenario.discount_factor(year) * scenario.period_weight(year)
            carbon_price = _year_value(policy.carbon_price, year, 0.0)
            for tech in scenario.technologies:
                fom = tech.fom * 1000.0
                fuel_price = scenario.fuel_price(tech.fuel, year)
                emission_rate = scenario.fuel_emission_factor(tech.fuel)
                marginal = (
                    tech.vom
                    + fuel_price / tech.efficiency
                    + carbon_price * emission_rate / tech.efficiency
                )
                for region in regions:
                    objective.add_expr(cap_v[(tech.name, region, year)], weight * fom)
                    if marginal:
                        for day, hour in index.steps():
                            objective.add_term(
                                gen_v[(tech.name, region, year, day, hour)].index,
                                weight * marginal * index.step_weight[day],
                            )
            for store in scenario.storage:
                fom = store.fom * 1000.0
                for region in regions:
                    objective.add_expr(scap_p[(store.name, region, year)], weight * fom)
                    if store.vom:
                        for day, hour in index.steps():
                            objective.add_term(
                                discharge_v[(store.name, region, year, day, hour)].index,
                                weight * store.vom * index.step_weight[day],
                            )
            for line in scenario.lines:
                objective.add_expr(line_cap[(line.name, year)], weight * line.fom * 1000.0)
            for region in regions:
                for day, hour in index.steps():
                    objective.add_term(
                        unserved_v[(region, year, day, hour)].index,
                        weight * policy.voll * index.step_weight[day],
                    )
            for region in regions:
                slack = shortfall_v.get((region, year))
                if slack is not None:
                    objective.add_expr(
                        slack, weight * policy.capacity_shortfall_penalty
                    )
            if year in overshoot_v:
                objective.add_expr(
                    overshoot_v[year], weight * policy.carbon_backstop_price
                )
            for label in ("rps", "ces"):
                deficit = share_deficit_v.get((label, year))
                if deficit is not None:
                    objective.add_expr(
                        deficit, weight * policy.share_shortfall_penalty
                    )
        problem.set_objective(objective)

        self.vars = {
            "build": build_v,
            "retire": retire_v,
            "cap": cap_v,
            "newcap": newcap_v,
            "gen": gen_v,
            "sbuild_p": sbuild_p,
            "sbuild_e": sbuild_e,
            "scap_p": scap_p,
            "scap_e": scap_e,
            "charge": charge_v,
            "discharge": discharge_v,
            "soc": soc_v,
            "line_cap": line_cap,
            "line_build": line_build,
            "flow_fwd": flow_f,
            "flow_bwd": flow_b,
            "unserved": unserved_v,
            "capacity_shortfall": shortfall_v,
            "co2_overshoot": overshoot_v,
            "share_deficit": share_deficit_v,
        }
        # Name -> row position, so dual values can be looked up by meaning.
        self._row_lookup = {con.name: con.index for con in problem.constraints}
        return problem

    def row_index(self, name: str) -> Optional[int]:
        """Position of a named constraint, for reading its dual value."""
        return self._row_lookup.get(name)

    # ------------------------------------------------------------------
    def solve(self, solver: str = "auto", verbose: bool = False, **options):
        from .results import PlanResult

        if self.problem is None:
            self.build()
        solution = self.problem.solve(solver=solver, verbose=verbose, **options)
        return PlanResult(self, solution)
