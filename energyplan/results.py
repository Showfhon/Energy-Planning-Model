"""Turning a solved LP back into planning answers.

:class:`PlanResult` reads the raw solution vector and exposes the quantities a
planner actually cares about: the build schedule, the annual generation mix,
system cost broken down by component, emissions, reliability, the levelised
cost of electricity, and the marginal prices implied by the dual values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .data import _year_value

__all__ = ["PlanResult", "YearSummary"]


@dataclass
class YearSummary:
    """Everything the model says about one planning year."""

    year: int
    demand_mwh: float = 0.0
    generation_mwh: Dict[str, float] = field(default_factory=dict)
    curtailment_mwh: Dict[str, float] = field(default_factory=dict)
    capacity_mw: Dict[str, float] = field(default_factory=dict)
    builds_mw: Dict[str, float] = field(default_factory=dict)
    storage_power_mw: Dict[str, float] = field(default_factory=dict)
    storage_energy_mwh: Dict[str, float] = field(default_factory=dict)
    storage_throughput_mwh: Dict[str, float] = field(default_factory=dict)
    unserved_mwh: float = 0.0
    transmission_losses_mwh: float = 0.0
    capacity_shortfall_mw: Dict[str, float] = field(default_factory=dict)
    emissions_overshoot_t: float = 0.0
    share_deficit_mwh: Dict[str, float] = field(default_factory=dict)
    emissions_t: float = 0.0
    cost: Dict[str, float] = field(default_factory=dict)
    marginal_price: float = 0.0          # load-weighted, USD/MWh
    carbon_shadow_price: float = 0.0     # USD/tCO2 from the emission cap
    carbon_price_is_degenerate: bool = False  # dual sits at the backstop: read as ">="

    capacity_price: Dict[str, float] = field(default_factory=dict)  # USD/MW-yr
    _renewable: float = 0.0

    @property
    def total_cost(self) -> float:
        return sum(self.cost.values())

    @property
    def renewable_share(self) -> float:
        total = sum(self.generation_mwh.values())
        return 0.0 if total <= 0 else self._renewable / total

    @property
    def average_cost(self) -> float:
        """Undiscounted system cost per MWh of demand served."""
        return self.total_cost / self.demand_mwh if self.demand_mwh > 0 else 0.0


class PlanResult:
    """Post-processed output of :meth:`CapacityExpansionModel.solve`."""

    def __init__(self, model, solution):
        self.model = model
        self.solution = solution
        self.scenario = model.scenario
        self.index = model.index
        self.years: List[YearSummary] = []
        if solution.optimal:
            self._extract()

    # ------------------------------------------------------------------
    @property
    def status(self) -> str:
        return self.solution.status

    @property
    def optimal(self) -> bool:
        return self.solution.optimal

    @property
    def objective(self) -> float:
        """Net present value of total system cost, USD."""
        return self.solution.objective

    def year(self, year: int) -> YearSummary:
        for summary in self.years:
            if summary.year == year:
                return summary
        raise KeyError(year)

    # ------------------------------------------------------------------
    def _value(self, var) -> float:
        return self.solution.x[var.index] if var is not None else 0.0

    def _dual(self, name: str) -> float:
        duals = self.solution.duals
        if not duals:
            return 0.0
        row = self.model.row_index(name)
        return duals[row] if row is not None and row < len(duals) else 0.0

    def _extract(self) -> None:
        scenario = self.scenario
        index = self.index
        v = self.model.vars
        regions = index.regions
        renewables = {t.name for t in scenario.technologies if t.renewable}

        for year in index.years:
            summary = YearSummary(year=year)
            summary.demand_mwh = sum(r.annual_demand_mwh(year) for r in scenario.regions)

            # ---- capacity, builds, generation ---------------------------
            for tech in scenario.technologies:
                cap = sum(self._value(v["cap"][(tech.name, r, year)]) for r in regions)
                built = sum(self._value(v["build"][(tech.name, r, year)]) for r in regions)
                if cap > 1e-6:
                    summary.capacity_mw[tech.name] = cap
                if built > 1e-6:
                    summary.builds_mw[tech.name] = built

                produced = 0.0
                potential = 0.0
                profile = index.capacity_factor.get(tech.name)
                for region in regions:
                    cap_r = self._value(v["cap"][(tech.name, region, year)])
                    for day, hour in index.steps():
                        weight = index.step_weight[day]
                        produced += weight * self._value(
                            v["gen"][(tech.name, region, year, day, hour)]
                        )
                        if profile is not None:
                            potential += weight * profile[day][hour] * cap_r
                if produced > 1e-6:
                    summary.generation_mwh[tech.name] = produced
                spilled = potential - produced
                if profile is not None and spilled > 1e-6:
                    summary.curtailment_mwh[tech.name] = spilled

            # ---- storage --------------------------------------------------
            for store in scenario.storage:
                power = sum(self._value(v["scap_p"][(store.name, r, year)]) for r in regions)
                energy = sum(self._value(v["scap_e"][(store.name, r, year)]) for r in regions)
                if power > 1e-6:
                    summary.storage_power_mw[store.name] = power
                    summary.storage_energy_mwh[store.name] = energy
                throughput = 0.0
                for region in regions:
                    for day, hour in index.steps():
                        throughput += index.step_weight[day] * self._value(
                            v["discharge"][(store.name, region, year, day, hour)]
                        )
                if throughput > 1e-6:
                    summary.storage_throughput_mwh[store.name] = throughput

            # ---- reliability and emissions --------------------------------
            unserved = 0.0
            for region in regions:
                for day, hour in index.steps():
                    unserved += index.step_weight[day] * self._value(
                        v["unserved"][(region, year, day, hour)]
                    )
            summary.unserved_mwh = unserved
            summary.transmission_losses_mwh = sum(
                index.step_weight[day] * line.loss * (
                    self._value(v["flow_fwd"][(line.name, year, day, hour)])
                    + self._value(v["flow_bwd"][(line.name, year, day, hour)])
                )
                for line in scenario.lines
                for day, hour in index.steps()
            )

            emissions = 0.0
            for tech in scenario.technologies:
                rate = scenario.fuel_emission_factor(tech.fuel) / tech.efficiency
                if rate > 0:
                    emissions += rate * summary.generation_mwh.get(tech.name, 0.0)
            summary.emissions_t = emissions

            # ---- penalised slacks: reliability and policy shortfalls -------
            for region in regions:
                slack = v["capacity_shortfall"].get((region, year))
                value = self._value(slack) if slack is not None else 0.0
                if value > 1e-6:
                    summary.capacity_shortfall_mw[region] = value
            overshoot = v["co2_overshoot"].get(year)
            if overshoot is not None:
                summary.emissions_overshoot_t = max(0.0, self._value(overshoot))
            for label in ("rps", "ces"):
                deficit = v["share_deficit"].get((label, year))
                if deficit is not None and self._value(deficit) > 1e-6:
                    summary.share_deficit_mwh[label] = self._value(deficit)

            summary.cost = self._year_costs(year, summary)
            summary._renewable = sum(
                mwh for name, mwh in summary.generation_mwh.items() if name in renewables
            )

            # ---- prices from duals ----------------------------------------
            # Duals come back in objective units, i.e. discounted and multiplied
            # by the number of calendar years the milestone stands for.  Divide
            # that weight out to get a price a planner can read directly.
            annual = scenario.discount_factor(year) * scenario.period_weight(year)
            summary.marginal_price = self._load_weighted_price(year)
            # Relaxing an emission cap lowers cost, so its dual is negative; the
            # shadow carbon price is its magnitude.
            summary.carbon_shadow_price = (
                -self._dual(f"co2_cap[{year}]") / annual if annual else 0.0
            )
            # A cap of exactly zero puts the optimum on a degenerate vertex,
            # where the dual is not unique and the solver may return the
            # backstop price rather than the true marginal abatement cost.
            # Flag it instead of reporting a misleading number as fact; use
            # ``empirical_marginal_carbon_cost`` for the reliable figure.
            backstop = scenario.policy.carbon_backstop_price
            summary.carbon_price_is_degenerate = bool(
                backstop
                and summary.emissions_overshoot_t <= 1e-6
                and summary.carbon_shadow_price >= backstop * (1.0 - 1e-6)
            )
            # Tightening a reserve requirement raises cost, so this dual is
            # already positive: it is the marginal value of firm capacity.
            for region in regions:
                summary.capacity_price[region] = (
                    self._dual(f"reserve[{region},{year}]") / annual if annual else 0.0
                )

            self.years.append(summary)

    # ------------------------------------------------------------------
    def _year_costs(self, year: int, summary: YearSummary) -> Dict[str, float]:
        """Undiscounted annual cost of the year, split by component (USD)."""
        scenario = self.scenario
        index = self.index
        v = self.model.vars
        rate = scenario.discount_rate
        carbon_price = _year_value(scenario.policy.carbon_price, year, 0.0)
        costs = {
            "capital": 0.0,
            "fixed_om": 0.0,
            "variable_om": 0.0,
            "fuel": 0.0,
            "carbon": 0.0,
            "storage": 0.0,
            "transmission": 0.0,
            "unserved": 0.0,
            "penalty": 0.0,
        }
        for tech in scenario.technologies:
            cap = summary.capacity_mw.get(tech.name, 0.0)
            gen = summary.generation_mwh.get(tech.name, 0.0)
            # Each vintage still in service pays the annuity of its own build year.
            for region in index.regions:
                for vintage in index.years:
                    if vintage > year or (year - vintage) >= tech.lifetime:
                        continue
                    costs["capital"] += tech.annuity(rate, vintage) * self._value(
                        v["build"][(tech.name, region, vintage)]
                    )
            costs["fixed_om"] += tech.fom * 1000.0 * cap
            costs["variable_om"] += tech.vom * gen
            fuel_price = scenario.fuel_price(tech.fuel, year)
            costs["fuel"] += fuel_price / tech.efficiency * gen
            emission_rate = scenario.fuel_emission_factor(tech.fuel) / tech.efficiency
            costs["carbon"] += carbon_price * emission_rate * gen

        for store in scenario.storage:
            for region in index.regions:
                for vintage in index.years:
                    if vintage <= year and (year - vintage) < store.lifetime:
                        costs["storage"] += store.annuity_power(rate, vintage) * self._value(
                            v["sbuild_p"][(store.name, region, vintage)]
                        )
                        costs["storage"] += store.annuity_energy(rate, vintage) * self._value(
                            v["sbuild_e"][(store.name, region, vintage)]
                        )
                costs["storage"] += store.fom * 1000.0 * self._value(
                    v["scap_p"][(store.name, region, year)]
                )
            costs["storage"] += store.vom * summary.storage_throughput_mwh.get(store.name, 0.0)

        for line in scenario.lines:
            live = sum(
                self._value(v["line_build"][(line.name, other)])
                for other in index.years
                if other <= year and (year - other) < line.lifetime
            )
            costs["transmission"] += line.annuity(rate) * live
            costs["transmission"] += line.fom * 1000.0 * self._value(
                v["line_cap"][(line.name, year)]
            )

        costs["unserved"] = scenario.policy.voll * summary.unserved_mwh
        policy = scenario.policy
        costs["penalty"] = (
            policy.capacity_shortfall_penalty * sum(summary.capacity_shortfall_mw.values())
            + policy.carbon_backstop_price * summary.emissions_overshoot_t
            + policy.share_shortfall_penalty * sum(summary.share_deficit_mwh.values())
        )
        return costs

    def _load_weighted_price(self, year: int) -> float:
        """Demand-weighted marginal cost of energy from the balance duals."""
        index = self.index
        numerator = 0.0
        denominator = 0.0
        for region in index.regions:
            demand = index.demand[(region, year)]
            for day, hour in index.steps():
                dual = self._dual(f"balance[{region},{year},{day},{hour}]")
                load = demand[day][hour]
                weight = index.step_weight[day] * load
                # The balance row is an equality whose RHS is demand, so its
                # dual is the marginal cost of one more MW in that hour --
                # divided by the step weight it carries in the objective.
                price = dual / index.step_weight[day] if index.step_weight[day] else 0.0
                discount = self.scenario.discount_factor(year) * self.scenario.period_weight(year)
                numerator += weight * price / discount if discount else 0.0
                denominator += weight
        return numerator / denominator if denominator > 0 else 0.0

    def hourly_prices(self, year: int, region: Optional[str] = None) -> List[List[float]]:
        """Marginal electricity price by representative day and hour (USD/MWh)."""
        index = self.index
        region = region or index.regions[0]
        discount = self.scenario.discount_factor(year) * self.scenario.period_weight(year)
        out = []
        for day in range(index.n_days):
            row = []
            for hour in range(index.n_hours):
                dual = self._dual(f"balance[{region},{year},{day},{hour}]")
                weight = index.step_weight[day] * discount
                row.append(dual / weight if weight else 0.0)
            out.append(row)
        return out

    def dispatch(self, year: int, region: Optional[str] = None) -> Dict[str, List[List[float]]]:
        """Dispatch in MW by technology, representative day and hour."""
        index = self.index
        v = self.model.vars
        region = region or index.regions[0]
        out: Dict[str, List[List[float]]] = {}
        for tech in self.scenario.technologies:
            grid = [
                [
                    self._value(v["gen"][(tech.name, region, year, day, hour)])
                    for hour in range(index.n_hours)
                ]
                for day in range(index.n_days)
            ]
            if any(any(row) for row in grid):
                out[tech.name] = grid
        for store in self.scenario.storage:
            discharge = [
                [
                    self._value(v["discharge"][(store.name, region, year, day, hour)])
                    - self._value(v["charge"][(store.name, region, year, day, hour)])
                    for hour in range(index.n_hours)
                ]
                for day in range(index.n_days)
            ]
            if any(any(abs(x) > 1e-6 for x in row) for row in discharge):
                out[f"{store.name} (net)"] = discharge
        out["demand"] = [
            list(day) for day in index.demand[(region, year)]
        ]
        return out

    # ------------------------------------------------------------------
    def lcoe(self) -> float:
        """System levelised cost: discounted cost over discounted energy served."""
        cost = 0.0
        energy = 0.0
        for summary in self.years:
            weight = (
                self.scenario.discount_factor(summary.year)
                * self.scenario.period_weight(summary.year)
            )
            cost += weight * summary.total_cost
            energy += weight * summary.demand_mwh
        return cost / energy if energy > 0 else 0.0

    def total_emissions_t(self) -> float:
        return sum(
            s.emissions_t * self.scenario.period_weight(s.year) for s in self.years
        )

    def build_schedule(self) -> Dict[str, Dict[int, float]]:
        """New capacity by technology and year, MW."""
        out: Dict[str, Dict[int, float]] = {}
        for summary in self.years:
            for name, mw in summary.builds_mw.items():
                out.setdefault(name, {})[summary.year] = mw
        for store in self.scenario.storage:
            for summary in self.years:
                built = sum(
                    self._value(self.model.vars["sbuild_p"][(store.name, r, summary.year)])
                    for r in self.index.regions
                )
                if built > 1e-6:
                    out.setdefault(f"{store.name} (power)", {})[summary.year] = built
        return out

    def empirical_marginal_carbon_cost(
        self, year: int, delta_mt: float = 0.5, solver: str = "auto"
    ) -> float:
        """Marginal cost of the emission cap in ``year``, by re-solving.

        The LP dual is the cheap answer, but at a cap of exactly zero the
        optimum is degenerate and the dual is not unique.  This method relaxes
        the cap by ``delta_mt`` megatonnes, re-solves, and returns the
        resulting cost saving per tonne -- a finite difference that is well
        defined whether or not the vertex is degenerate.

        Returns 0.0 when the scenario has no cap for that year.
        """
        import copy

        from .model import CapacityExpansionModel

        scenario = self.scenario
        if scenario.policy.carbon_cap is None:
            return 0.0
        relaxed = copy.deepcopy(scenario)
        cap = dict(relaxed.policy.carbon_cap) if isinstance(
            relaxed.policy.carbon_cap, dict
        ) else {y: float(relaxed.policy.carbon_cap) for y in scenario.years}
        if year not in cap:
            from .data import _year_value

            cap[year] = _year_value(scenario.policy.carbon_cap, year, 0.0)
        cap[year] = cap[year] + delta_mt
        relaxed.policy.carbon_cap = cap
        other = CapacityExpansionModel(
            relaxed,
            profiles=self.model.profiles,
            representative_days=self.index.rep,
        ).solve(solver=solver)
        if not other.optimal:
            return float("nan")
        annual = scenario.discount_factor(year) * scenario.period_weight(year)
        npv_saving = self.objective - other.objective
        return npv_saving / (delta_mt * 1e6) / annual if annual else 0.0

    def audit(self) -> Dict[str, float]:
        """Independent consistency checks on the solved plan.

        Each entry is a *relative* residual that should be at solver
        tolerance.  The checks re-derive quantities from the extracted
        results rather than trusting the LP, so they catch mistakes in the
        model *and* in this post-processing layer:

        ``objective``
            NPV rebuilt from the annual cost breakdown versus the solver's
            objective value.
        ``energy_balance``
            Generation + discharge - charge + unserved - demand, per year.
        ``reserve_margin``
            Firm capacity shortfall against the requirement, per year.
        ``capacity_limits``
            Capacity above a technology's stated potential.
        """
        checks: Dict[str, float] = {}
        scenario = self.scenario
        index = self.index
        v = self.model.vars

        npv = sum(
            scenario.discount_factor(s.year) * scenario.period_weight(s.year) * s.total_cost
            for s in self.years
        )
        scale = max(abs(self.objective), 1.0)
        checks["objective"] = abs(npv - self.objective) / scale

        worst_balance = 0.0
        worst_reserve = 0.0
        worst_capacity = 0.0
        for summary in self.years:
            year = summary.year
            charged = sum(
                index.step_weight[day] * self._value(
                    v["charge"][(store.name, region, year, day, hour)]
                )
                for store in scenario.storage
                for region in index.regions
                for day, hour in index.steps()
            )
            # Energy injected into the network must also cover transmission losses.
            losses = sum(
                index.step_weight[day] * line.loss * (
                    self._value(v["flow_fwd"][(line.name, year, day, hour)])
                    + self._value(v["flow_bwd"][(line.name, year, day, hour)])
                )
                for line in scenario.lines
                for day, hour in index.steps()
            )
            residual = (
                sum(summary.generation_mwh.values())
                + sum(summary.storage_throughput_mwh.values())
                - charged
                - losses
                + summary.unserved_mwh
                - summary.demand_mwh
            )
            if summary.demand_mwh > 0:
                worst_balance = max(worst_balance, abs(residual) / summary.demand_mwh)

            for region in scenario.regions:
                margin = region.reserve_margin
                if margin is None:
                    margin = scenario.policy.reserve_margin
                if margin is None:
                    continue
                firm = sum(
                    tech.capacity_credit() * self._value(v["cap"][(tech.name, region.name, year)])
                    for tech in scenario.technologies
                )
                firm += sum(
                    store.firm_factor * self._value(v["scap_p"][(store.name, region.name, year)])
                    for store in scenario.storage
                )
                for line in scenario.lines:
                    if region.name in (line.from_region, line.to_region):
                        firm += 0.75 * (1.0 - line.loss) * self._value(
                            v["line_cap"][(line.name, year)]
                        )
                slack = v["capacity_shortfall"].get((region.name, year))
                if slack is not None:
                    firm += self._value(slack)
                required = index.peak_mw[(region.name, year)] * (1.0 + margin)
                if required > 0:
                    worst_reserve = max(worst_reserve, max(0.0, required - firm) / required)

            for tech in scenario.technologies:
                if tech.max_total_capacity is None:
                    continue
                built = sum(
                    self._value(v["cap"][(tech.name, region, year)])
                    for region in index.regions
                )
                excess = built - tech.max_total_capacity
                if excess > 0:
                    worst_capacity = max(
                        worst_capacity, excess / max(tech.max_total_capacity, 1.0)
                    )

        checks["energy_balance"] = worst_balance
        checks["reserve_margin"] = worst_reserve
        checks["capacity_limits"] = worst_capacity
        return checks

    def to_dict(self) -> dict:
        """A JSON-serialisable summary of the plan."""
        return {
            "scenario": self.scenario.name,
            "status": self.status,
            "solver": self.solution.solver,
            "npv_total_cost_usd": self.objective,
            "system_lcoe_usd_per_mwh": self.lcoe(),
            "cumulative_emissions_tco2": self.total_emissions_t(),
            "years": [
                {
                    "year": s.year,
                    "demand_mwh": s.demand_mwh,
                    "capacity_mw": s.capacity_mw,
                    "builds_mw": s.builds_mw,
                    "generation_mwh": s.generation_mwh,
                    "curtailment_mwh": s.curtailment_mwh,
                    "storage_power_mw": s.storage_power_mw,
                    "storage_energy_mwh": s.storage_energy_mwh,
                    "unserved_mwh": s.unserved_mwh,
                    "transmission_losses_mwh": s.transmission_losses_mwh,
                    "capacity_shortfall_mw": s.capacity_shortfall_mw,
                    "emissions_overshoot_tco2": s.emissions_overshoot_t,
                    "share_deficit_mwh": s.share_deficit_mwh,
                    "emissions_tco2": s.emissions_t,
                    "renewable_share": s.renewable_share,
                    "cost_usd": s.cost,
                    "marginal_price_usd_per_mwh": s.marginal_price,
                    "carbon_shadow_price_usd_per_t": s.carbon_shadow_price,
                    "carbon_price_is_degenerate": s.carbon_price_is_degenerate,
                    "capacity_price_usd_per_mw_yr": s.capacity_price,
                }
                for s in self.years
            ],
        }
